"""Display-only controls and extraction of saved volume cells."""
from copy import deepcopy
import math

from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from turbo_moc.app import workflow as flow
from turbo_moc.meshing import inspection


DEFAULTS = {"mode": "surface", "method": "worst", "metric": "skewness", "n": 100,
    "threshold": .8, "layer": 1, "context": ["surfaces"]}
METRICS = inspection.METRIC_LABELS


def identifier(field, row):
    return {"type": "inspection-" + field, "row": row}


def stores():
    return [dcc.Store(id="mesh-inspection-" + name, data={} if name == "settings" else None)
        for name in ("settings", "request", "result")]


def controls(row, appearance, saved=None):
    settings = {**DEFAULTS, **(saved or {})}
    if "blade" in settings["context"]:
        settings["context"] = ["surfaces"]
    def dropdown(field, options):
        return dcc.Dropdown(id=identifier(field, row), options=[{"label": label, "value": value}
            for value, label in options], value=settings[field], clearable=False, searchable=False)
    def number(field, **kwargs):
        return dcc.Input(id=identifier(field, row), type="number", value=settings[field],
            debounce=True, **kwargs)
    def control(text, child, field=None):
        return html.Label([html.Span(text, **({"id": identifier(field + "-label", row)} if field else {})),
            child], className="control inspection-control")
    return [control("Display mode", dropdown("mode", [("surface", "Boundary surfaces"),
                ("filter", "Mesh quality filter"), ("layer", "Spanwise layer")])),
        html.Div(appearance, id=identifier("appearance", row),
            style={} if settings["mode"] == "surface" else {"display": "none"}),
        html.Div([
                control("Metric", dropdown("metric", list(METRICS.items()))),
                html.Div([
                    control("Select cells", dropdown("method", [("worst", "Worst N"), ("threshold", "Threshold")])),
                    html.Div(control("Number of cells", number("n", min=1, step=1)), id=identifier("n-fields", row)),
                    html.Div(control("Skewness ≥", number("threshold", min=0, max=1, step="any"), "threshold"),
                        id=identifier("threshold-fields", row)),
                ], id=identifier("filter-fields", row)),
                html.Div([
                    html.Div(id=identifier("layer-caption", row), className="inspection-caption"),
                    html.Div(dcc.Slider(id=identifier("layer", row), min=1, max=max(1, settings["layer"] or 1), step=1,
                        value=settings["layer"], marks={1: "1"}, updatemode="mouseup"), className="inspection-layer-slider"),
                    html.Div([html.Button("Previous", id=identifier("previous", row), n_clicks=0),
                        html.Button("Next", id=identifier("next", row), n_clicks=0)], className="inspection-step"),
                ], id=identifier("layer-fields", row)),
                dcc.Checklist(id=identifier("context", row), options=[{"label": "Faint boundary surfaces", "value": "surfaces"}],
                    value=settings["context"]),
                html.Small("Context follows your Boundary surfaces selection. Inspection shows one passage of this row."),
        ], id=identifier("active-fields", row),
            style={"display": "none"} if settings["mode"] == "surface" else {}, className="inspection-controls")]


def extract(request):
    """Read the requested artifact, independently of the current design settings."""
    if not request:
        return None
    identity = {key: request.get(key) for key in ("key", "row_id", "job", "context", "status", "label")}
    if request.get("error"):
        return {**identity, "error": request["error"], "note": request["error"]}
    try:
        options = request["options"]
        if options["mode"] not in ("worst", "threshold", "layer"):
            raise ValueError("Choose a quality filter or spanwise layer.")
        result = inspection.inspect_mesh(flow.directory(request["reference"]), **options)
        return {**result, **identity}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return {**identity, "error": str(exc), "note": "Cannot inspect mesh: " + str(exc)}


def summary(result, request):
    if not request:
        return []
    if request.get("error"):
        return html.Div(request["error"], className="message")
    if not result or result.get("key") != request["key"]:
        return html.Div("Updating mesh inspection…", className="inspection-caption")
    if result.get("error"):
        return html.Div(result["note"], className="message")
    lower, upper = result["color_range"]
    midpoint = math.sqrt(lower * upper) if result["logarithmic"] else (lower + upper) / 2
    ends = [upper, midpoint, lower] if result["metric"] == "orthogonality" else [lower, midpoint, upper]
    status = " · Outdated mesh" if result.get("status") == "Outdated" else ""
    scale = " · logarithmic" if result["logarithmic"] else ""
    return [html.Div([
        html.Div([html.Strong(f"{result.get('label', result['row_id'])} · {result['metric_label']}{status}"),
            html.Div(f"{result['selected_count']:,} of {result['total_count']:,} volume cells selected · one passage"),
            html.Small(result["note"])], className="inspection-count"),
        html.Div([html.Div("Better → Worse" + scale), html.Div(className="inspection-gradient"),
            html.Div([html.Span(f"{value:.4g}") for value in ends], className="inspection-ticks")],
            className="inspection-scale", title=result["definition"]),
    ], className="inspection-readout")]


def register(app):
    @app.callback(*[Output(identifier(name, MATCH), "style") for name in
        ("appearance", "active-fields", "filter-fields", "layer-fields", "n-fields", "threshold-fields")],
        Output(identifier("threshold-label", MATCH), "children"),
        Output(identifier("threshold", MATCH), "min"), Output(identifier("threshold", MATCH), "max"),
        Input(identifier("mode", MATCH), "value"), Input(identifier("method", MATCH), "value"),
        Input(identifier("metric", MATCH), "value"))
    def fields(mode, method, metric):
        visible = (mode == "surface", mode != "surface", mode == "filter", mode == "layer",
            method == "worst", method == "threshold")
        return (*[{} if show else {"display": "none"} for show in visible],
            METRICS[metric] + (" ≤" if metric == "orthogonality" else " ≥"),
            1 if metric == "aspect_ratio" else 0, None if metric == "aspect_ratio" else 1)

    @app.callback(Output(identifier("threshold", MATCH), "value"),
        Input(identifier("metric", MATCH), "value"), prevent_initial_call=True)
    def threshold_default(metric):
        return {"skewness": .8, "orthogonality": .2, "aspect_ratio": 10}[metric]

    @app.callback(Output(identifier("layer", MATCH), "max"), Output(identifier("layer", MATCH), "value"),
        Output(identifier("layer", MATCH), "marks"), Output(identifier("layer-caption", MATCH), "children"),
        Output(identifier("previous", MATCH), "disabled"), Output(identifier("next", MATCH), "disabled"),
        Output(identifier("layer", MATCH), "disabled"),
        Input("volume-data", "data"), Input(identifier("layer", MATCH), "value"),
        Input(identifier("previous", MATCH), "n_clicks"), Input(identifier("next", MATCH), "n_clicks"),
        State(identifier("layer", MATCH), "id"))
    def layers(data, value, previous, next_, item):
        row = next((row for row in (data or {}).get("rows", []) if row["id"] == item["row"]), {})
        stations = row.get("span_stations", [])
        if len(stations) < 2:
            # A temporarily invalid design can hide a retained mesh. Keep this
            # row's chosen layer until an actual replacement artifact arrives.
            layer = max(1, int(value or 1))
            return layer, layer, {layer: str(layer)}, "Generate a 3D mesh to inspect its spanwise layers.", True, True, True
        count = len(stations) - 1
        layer = max(1, min(count, int(value or 1)))
        triggered = ctx.triggered_id
        if isinstance(triggered, dict):
            if triggered["type"] == "inspection-previous" and previous:
                layer = max(1, layer - 1)
            elif triggered["type"] == "inspection-next" and next_:
                layer = min(count, layer + 1)
        ends = ("Span start", "Span end") if row.get("is_linear") else ("Hub", "Shroud/tip")
        caption = f"Layer {layer} of {count} · η = {stations[layer-1]:.4g}–{stations[layer]:.4g}"
        marks = {1: ends[0], count: ends[1]} if count > 1 else {1: "Single layer"}
        return count, layer, marks, caption, layer <= 1, layer >= count, False

    names = list(DEFAULTS)
    @app.callback(Output("mesh-inspection-settings", "data"),
        *[Input(identifier(name, ALL), "value") for name in names],
        *[State(identifier(name, ALL), "id") for name in names], State("mesh-inspection-settings", "data"))
    def preferences(*args):
        result = deepcopy(args[-1] or {})
        for name, values, ids in zip(names, args[:len(names)], args[len(names):-1]):
            for value, item in zip(values or [], ids or []):
                result.setdefault(item["row"], deepcopy(DEFAULTS))[name] = value
        return no_update if result == args[-1] else result

    app.clientside_callback("function(settings, row, data, shown) { return window.turbomocInspection.request(settings, row, data, shown); }",
        Output("mesh-inspection-request", "data"), Input("mesh-inspection-settings", "data"),
        Input("mesh-display-row", "value"), Input("volume-data", "data"), Input("plotted-rows", "value"))
    app.callback(Output("mesh-inspection-result", "data"), Input("mesh-inspection-request", "data"))(extract)
    app.callback(Output("mesh-inspection-summary", "children"), Input("mesh-inspection-result", "data"),
        Input("mesh-inspection-request", "data"))(summary)
