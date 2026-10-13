"""Per-cascade visualization controls, independent of design configuration."""
from copy import deepcopy
import json
import re

from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from turbo_moc.configuration import project as projects
from turbo_moc.meshing.mesh_3d import BOUNDARIES


COLORS = {"blade": "#477db3", "hub": "#a6b3b8", "shroud": "#ccd1d6",
          "inlet": "#6699bf", "outlet": "#73a68c", "periodic_left": "#b399bf", "periodic_right": "#bfb585"}


def identifier(kind, pane, row, **extra):
    return {"type": "display-" + kind, "pane": pane, "row": row, **extra}


def label(row):
    return {"stator_1": "Stator", "rotor_1": "Rotor"}.get(row["id"], row["id"].replace("_", " ").title())


def limit(row, pane, data=None):
    if pane == "mesh":
        dataset = next((item for item in (data or {}).get("rows", []) if item["id"] == row["id"]), None)
        if dataset:
            return 25 if dataset.get("is_linear") else dataset["num_blades"]
    dimensions = row["geometry"]["meridional_channel"]
    return 25 if dimensions["type"] == "linear" else dimensions["num_blades"]


def defaults(row):
    colors = {**COLORS, "blade": "#ef8c26" if row["role"] == "rotor" else COLORS["blade"]}
    return {"copies": min(limit(row, "geometry"), row["mesh"]["matplotlib"]["num_blades"]),
            "surfaces": [name for name, chosen in row["mesh"]["pyvista"]["surfaces"].items() if chosen],
            "colors": colors, "edges": row["mesh"]["pyvista"]["show_mesh_edges"]}


def content(project, pane, preferences=None):
    editors = []
    for row in projects.active_rows(project):
        row_id = row["id"]
        settings = (preferences or {}).get(row_id, defaults(row))
        maximum = limit(row, pane)
        count = min(maximum, max(1, settings["copies"]))
        surfaces = []
        linear = project["machine"]["geometry_type"] == "linear"
        for name in BOUNDARIES:
            text = {"hub": "Span start", "shroud": "Span end"}.get(name, name.replace("_", " ").title()) if linear else name.replace("_", " ").title()
            color = settings["colors"][name]
            color_id = identifier("color", pane, row_id, surface=name)
            surfaces.append(html.Div([
                dcc.Checklist(id=identifier("surface", pane, row_id, surface=name),
                    options=[{"label": text, "value": name}], value=[name] if name in settings["surfaces"] else []),
                dcc.Store(id=color_id, data=color),
                html.Div(id=identifier("picker", pane, row_id, surface=name), className="surface-color-picker",
                    **{"data-color-id": json.dumps(color_id, sort_keys=True, separators=(",", ":")),
                       "data-color": color, "data-color-label": f"{label(row)} {text} color"}),
            ], className="display-surface-row"))
        editors.append(html.Section([
            html.Label([html.Span("Displayed blades"), dcc.Input(id=identifier("copies", pane, row_id),
                type="number", min=1, max=maximum, step=1, value=count, debounce=True)], className="control",
                title="Display copies only; the physical blade count and exported mesh are unchanged."),
            dcc.Slider(id=identifier("slider", pane, row_id), min=1, max=maximum, step=1,
                value=count, marks={1: "1", maximum: str(maximum)}, updatemode="mouseup"),
            html.Div([html.Button(text, id=identifier("preset", pane, row_id, action=action), n_clicks=0)
                for text, action in (("Blade only", "blade"), ("All surfaces", "surfaces"), ("Reset display", "reset"))],
                className="display-presets"),
            html.Div(surfaces, className="display-surfaces"),
            *([dcc.Checklist(id=identifier("edges", pane, row_id), options=[{"label": "Mesh edges", "value": "edges"}],
                value=["edges"] if settings["edges"] else [])]
              if pane == "mesh" else []),
        ], id=identifier("editor", pane, row_id), style={"display": "none"}))
    return editors


def view(project, pane, viewport):
    preferences = {row["id"]: defaults(row) for row in projects.active_rows(project)}
    rows = projects.active_rows(project)
    return [dcc.Store(id=f"{pane}-display", data=preferences), dcc.Store(id=f"{pane}-display-open", data=False),
        html.Div(html.Button("Configure display", id=f"{pane}-configure-display", n_clicks=0,
            title="Choose a blade row and configure its displayed blades, surfaces and colors."),
            className="cascade-display-toolbar"),
        html.Div([viewport, html.Aside([
            html.Div([html.Strong("Configure display"), html.Button("×", id=f"{pane}-close-display", n_clicks=0,
                title="Close display panel", **{"aria-label": "Close display panel"})], className="display-panel-header"),
            html.Label([html.Span("Blade row"), dcc.Dropdown(id=f"{pane}-display-row", clearable=False, searchable=False,
                value=rows[0]["id"], options=[{"label": label(row), "value": row["id"]} for row in rows])],
                className="control display-row-selector", title="Configure this row without changing which rows are shown."),
            html.Div(content(project, pane, preferences), id=f"{pane}-display-editors"),
        ], id=f"{pane}-display-panel", className="display-panel",
            style={"display": "none"})], className="display-viewport")]


def register(app, pane):
    @app.callback(Output(f"{pane}-display-editors", "children"), Output(f"{pane}-display-row", "options"),
        Output(f"{pane}-display-row", "value"), Input("base-project", "data"),
        State(f"{pane}-display", "data"), State(f"{pane}-display-row", "value"))
    def structure(project, preferences, selected):
        rows = projects.active_rows(project)
        selected = selected if selected in {row["id"] for row in rows} else rows[0]["id"]
        return content(project, pane, preferences), [{"label": label(row), "value": row["id"]} for row in rows], selected

    @app.callback(Output(identifier("editor", pane, ALL), "style"), Output(f"{pane}-display-panel", "style"),
        Output(f"{pane}-display-open", "data"), Input(f"{pane}-configure-display", "n_clicks"),
        Input(f"{pane}-close-display", "n_clicks"), Input(f"{pane}-display-row", "value"),
        Input(identifier("editor", pane, ALL), "id"), State(f"{pane}-display-open", "data"))
    def open_panel(opens, closes, selected, identifiers, opened):
        triggered = ctx.triggered_id
        if triggered == f"{pane}-configure-display" and opens:
            opened = True
        elif triggered == f"{pane}-close-display" and closes:
            opened = False
        opened = bool(opened and selected in {item["row"] for item in identifiers})
        return ([{} if item["row"] == selected else {"display": "none"} for item in identifiers],
                {} if opened else {"display": "none"}, opened)

    @app.callback(Output(f"{pane}-display", "data"),
        Input(identifier("copies", pane, ALL), "value"),
        Input(identifier("surface", pane, ALL, surface=ALL), "value"),
        Input(identifier("color", pane, ALL, surface=ALL), "data"),
        *([Input(identifier("edges", pane, ALL), "value")] if pane == "mesh" else []),
        State(identifier("copies", pane, ALL), "id"),
        State(identifier("surface", pane, ALL, surface=ALL), "id"), State(identifier("color", pane, ALL, surface=ALL), "id"),
        *([State(identifier("edges", pane, ALL), "id")] if pane == "mesh" else []),
        State(f"{pane}-display", "data"), State("base-project", "data"))
    def preferences(*args):
        size = 4 if pane == "mesh" else 3
        values, identifiers = args[:size], args[size:2 * size]
        saved, project = args[-2:]
        result = deepcopy(saved or {})
        for row in projects.active_rows(project):
            result.setdefault(row["id"], defaults(row))
        for group, ids, field in zip(values, identifiers, ("copies", "surfaces", "colors", "edges")):
            for value, item in zip(group or [], ids or []):
                settings = result.get(item["row"])
                if settings is None:
                    continue
                if field == "copies" and value is not None:
                    settings[field] = max(1, int(value))
                elif field == "edges":
                    settings[field] = bool(value)
                elif field == "surfaces":
                    selected = set(settings[field])
                    selected.discard(item["surface"])
                    if value:
                        selected.add(item["surface"])
                    settings[field] = [name for name in BOUNDARIES if name in selected]
                elif field == "colors" and isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                    settings[field][item["surface"]] = value
        return no_update if result == saved else result

    @app.callback(Output(identifier("copies", pane, MATCH), "value"), Output(identifier("copies", pane, MATCH), "max"),
        Output(identifier("slider", pane, MATCH), "value"), Output(identifier("slider", pane, MATCH), "max"),
        Output(identifier("slider", pane, MATCH), "marks"), Input(identifier("copies", pane, MATCH), "value"),
        Input(identifier("slider", pane, MATCH), "value"), Input("config", "data"), Input("volume-data", "data"),
        State(identifier("copies", pane, MATCH), "id"))
    def copies(number, slider, state, data, item):
        if not state or not state.get("project"):
            return (no_update,) * 5
        try:
            row = projects.get_row(state["project"], item["row"])
        except (KeyError, ValueError):
            return (no_update,) * 5
        maximum = limit(row, pane, data)
        triggered = ctx.triggered_id
        current = slider if isinstance(triggered, dict) and triggered["type"] == "display-slider" else number
        count = min(maximum, max(1, int(current or 1)))
        return count, maximum, count, maximum, {1: "1", maximum: str(maximum)}

    @app.callback(Output(identifier("picker", pane, MATCH, surface=MATCH), "data-color"),
        Input(identifier("color", pane, MATCH, surface=MATCH), "data"))
    def color(value):
        return value

    @app.callback(Output(identifier("surface", pane, MATCH, surface=ALL), "value"),
        Output(identifier("color", pane, MATCH, surface=ALL), "data"),
        Output(identifier("copies", pane, MATCH), "value", allow_duplicate=True),
        *([Output(identifier("edges", pane, MATCH), "value")] if pane == "mesh" else []),
        Input(identifier("preset", pane, MATCH, action=ALL), "n_clicks"),
        State(identifier("surface", pane, MATCH, surface=ALL), "id"),
        State(identifier("color", pane, MATCH, surface=ALL), "id"), State("config", "data"),
        State("volume-data", "data"), prevent_initial_call=True)
    def preset(clicks, surfaces, colors, state, data):
        item = ctx.triggered_id
        if not isinstance(item, dict) or not state or not state.get("project"):
            return (no_update,) * (4 if pane == "mesh" else 3)
        row = projects.get_row(state["project"], item["row"])
        settings = defaults(row)
        action = item["action"]
        selected = ["blade"] if action == "blade" else list(BOUNDARIES) if action == "surfaces" else settings["surfaces"]
        reset = action == "reset"
        return ([[i["surface"]] if i["surface"] in selected else [] for i in surfaces],
                [settings["colors"][i["surface"]] if reset else no_update for i in colors],
                settings["copies"] if reset else no_update,
                *([["edges"] if settings["edges"] else []] if pane == "mesh" and reset else [no_update] if pane == "mesh" else []))
