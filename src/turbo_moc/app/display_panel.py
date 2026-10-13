"""Per-cascade visualization controls, independent of design configuration."""
from copy import deepcopy
import json
import re

from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from turbo_moc.configuration import project as projects
from turbo_moc.meshing.mesh_3d import BOUNDARIES
from turbo_moc.app import inspection_panel


COLORS = {"blade": "#477db3", "hub": "#a6b3b8", "shroud": "#a6b3b8",
          "inlet": "#c94f4f", "outlet": "#c94f4f", "periodic_left": "#4d9966", "periodic_right": "#4d9966"}


def geometry_family(definition, fallback):
    """Choose a camera from saved geometry, including older untagged annular meshes."""
    if definition.get("type") in {"axial", "radial", "linear"}:
        return definition["type"]
    curves = [definition.get(boundary, {}).get("control_points_rz_mm", [])
              for boundary in ("hub", "shroud")]
    if not all(len(points) >= 2 for points in curves):
        return fallback
    spans = [[max(p[axis] for p in points) - min(p[axis] for p in points)
              for axis in (0, 1)] for points in curves]
    tolerance = max(1.0, *(abs(value) for points in curves for point in points for value in point)) * 1e-9
    if all(r <= tolerance and z > tolerance for r, z in spans):
        return "axial"
    if all(z <= tolerance and r > tolerance for r, z in spans):
        return "radial"
    return fallback


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
    # Use the physical role, not the row ID or stage number, for blade colors.
    colors = {**COLORS, "blade": "#ef8c26" if row["role"] == "rotor" else COLORS["blade"]}
    return {"copies": min(limit(row, "geometry"), row["mesh"]["matplotlib"]["num_blades"]),
            "surfaces": [name for name, chosen in row["mesh"]["pyvista"]["surfaces"].items() if chosen],
            "colors": colors, "edges": row["mesh"]["pyvista"]["show_mesh_edges"]}


def content(project, pane, preferences=None, inspection_preferences=None, selected=None):
    editors = []
    rows = projects.active_rows(project)
    if selected is not None:
        rows = [row for row in rows if row["id"] in selected][:1]
    for row in rows:
        row_id = row["id"]
        settings = (preferences or {}).get(row_id, defaults(row))
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
        appearance = [
            html.Div([html.Button(text, id=identifier("preset", pane, row_id, action=action), n_clicks=0)
                for text, action in (("Blade only", "blade"), ("All surfaces", "surfaces"), ("Reset display", "reset"))],
                className="display-presets"),
            html.Div(surfaces, className="display-surfaces"),
            *([dcc.Checklist(id=identifier("edges", pane, row_id), options=[{"label": "Mesh edges", "value": "edges"}],
                value=["edges"] if settings["edges"] else [])]
              if pane == "mesh" else []),
        ]
        children = inspection_panel.controls(row_id, appearance,
            (inspection_preferences or {}).get(row_id)) if pane == "mesh" else appearance
        editors.append(html.Section(children, id=identifier("editor", pane, row_id), style={}))
    return editors


def count_controls(project, pane, preferences=None, selected=None):
    controls = []
    for row in projects.active_rows(project):
        if selected is not None and row["id"] not in selected:
            continue
        maximum = limit(row, pane)
        count = min(maximum, max(1, (preferences or {}).get(row["id"], defaults(row))["copies"]))
        # Dash's number input retains stale local state after slider updates.
        # A controlled text box with a numeric keyboard stays synchronized;
        # the callback below enforces integer counts and the row's bounds.
        controls.append(html.Div([
            html.Label([html.Span(label(row)), dcc.Input(id=identifier("copies", pane, row["id"]),
                type="text", inputMode="numeric", pattern="[0-9]*", min=1, max=maximum,
                step=1, value=count, debounce=True)], className="control"),
            dcc.Slider(id=identifier("slider", pane, row["id"]), min=1, max=maximum, step=1,
                value=count, marks={1: "1", maximum: str(maximum)}, updatemode="mouseup"),
        ], className="display-row-count"))
    return controls


def view(project, pane, viewport):
    preferences = {row["id"]: defaults(row) for row in projects.active_rows(project)}
    rows = projects.active_rows(project)
    return [dcc.Store(id=f"{pane}-display", data=preferences),
        *(inspection_panel.stores() if pane == "mesh" else []),
        html.Div([viewport, html.Aside([
            html.Div(html.Strong("Configure display"), className="display-panel-header"),
            html.Details([html.Summary("Blade rows"),
                dcc.Checklist(id=f"{pane}-display-row", value=["all", *preferences],
                    options=[{"label": "All rows", "value": "all"},
                             *[{"label": label(row), "value": row["id"]} for row in rows]])],
                className="display-selection"),
            html.Details([html.Summary("Displayed blades"),
                html.Div(count_controls(project, pane, preferences), id=f"{pane}-display-counts")],
                className="display-selection display-count-selection"),
            html.Div(content(project, pane, preferences, selected=list(preferences)), id=f"{pane}-display-editors"),
            *([html.Div(id="mesh-inspection-summary", className="mesh-inspection-summary")]
              if pane == "mesh" else []),
        ], id=f"{pane}-display-panel", className="display-panel")], className="display-viewport")]


def selected_rows(chosen, previous, available):
    """Resolve the All rows checkbox, including unchecking one of its children."""
    chosen, previous = chosen or [], previous or []
    selected = [row for row in available if row in chosen]
    was_all = set(previous) == set(available)
    if "all" in chosen and not was_all:
        return list(available)
    if "all" not in chosen and was_all and selected == available:
        return []
    return selected


def register(app, pane):
    @app.callback(Output(f"{pane}-display-editors", "children"),
        Output(f"{pane}-display-counts", "children"),
        Input("base-project", "data"), Input(f"{pane}-display-row", "value"),
        State(f"{pane}-display", "data"),
        *([State("mesh-inspection-settings", "data")] if pane == "mesh" else []))
    def structure(project, selected, preferences, inspection_preferences=None):
        return (content(project, pane, preferences, inspection_preferences, selected or []),
                count_controls(project, pane, preferences, selected or []))

    @app.callback(Output(f"{pane}-display", "data"),
        Input(identifier("copies", pane, ALL), "value"),
        Input(identifier("surface", pane, ALL, surface=ALL), "value"),
        Input(identifier("color", pane, ALL, surface=ALL), "data"),
        *([Input(identifier("edges", pane, ALL), "value")] if pane == "mesh" else []),
        State(identifier("copies", pane, ALL), "id"),
        State(identifier("surface", pane, ALL, surface=ALL), "id"), State(identifier("color", pane, ALL, surface=ALL), "id"),
        *([State(identifier("edges", pane, ALL), "id")] if pane == "mesh" else []),
        State(f"{pane}-display", "data"), State("base-project", "data"),
        State(f"{pane}-display-row", "value"))
    def preferences(*args):
        size = 4 if pane == "mesh" else 3
        values, identifiers = args[:size], args[size:2 * size]
        saved, project, selected_rows = args[-3:]
        changed = list(ctx.triggered_prop_ids.values())
        result = deepcopy(saved or {})
        for row in projects.active_rows(project):
            result.setdefault(row["id"], defaults(row))
        for group, ids, field in zip(values, identifiers, ("copies", "surfaces", "colors", "edges")):
            for value, item in zip(group or [], ids or []):
                if item not in changed:
                    continue
                # Recreating an editor can fire its synchronized inputs. Only
                # an actual value change should propagate to the other rows.
                source = result.get(item["row"], {})
                current = (item["surface"] in source.get(field, []) if field == "surfaces" else
                    source.get(field, {}).get(item["surface"]) if field == "colors" else source.get(field))
                if current == (bool(value) if field in ("surfaces", "edges") else value):
                    continue
                targets = [item["row"]] if field == "copies" else selected_rows or []
                for row_id in targets:
                    settings = result.get(row_id)
                    if settings is None:
                        continue
                    if field == "copies" and value is not None:
                        try:
                            settings[field] = max(1, int(value))
                        except (TypeError, ValueError, OverflowError):
                            continue  # The synchronized input restores a valid count.
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
        try:
            count = min(maximum, max(1, int(current or 1)))
        except (TypeError, ValueError, OverflowError):
            count = 1
        return count, maximum, count, maximum, {1: "1", maximum: str(maximum)}

    @app.callback(Output(identifier("picker", pane, MATCH, surface=MATCH), "data-color"),
        Input(identifier("color", pane, MATCH, surface=MATCH), "data"))
    def color(value):
        return value

    @app.callback(Output(f"{pane}-display", "data", allow_duplicate=True),
        Output(f"{pane}-display-editors", "children", allow_duplicate=True),
        Output(f"{pane}-display-counts", "children", allow_duplicate=True),
        Input(identifier("preset", pane, ALL, action=ALL), "n_clicks"),
        State(f"{pane}-display-row", "value"), State("base-project", "data"),
        State(f"{pane}-display", "data"),
        *([State("mesh-inspection-settings", "data")] if pane == "mesh" else []), prevent_initial_call=True)
    def preset(clicks, selected, project, saved, inspection_preferences=None):
        item = ctx.triggered_id
        if not isinstance(item, dict) or not any(clicks or []):
            return no_update, no_update, no_update
        result = deepcopy(saved or {})
        for row in projects.active_rows(project):
            if row["id"] not in (selected or []):
                continue
            settings = result.setdefault(row["id"], defaults(row))
            if item["action"] == "reset":
                result[row["id"]] = defaults(row)
            else:
                settings["surfaces"] = ["blade"] if item["action"] == "blade" else list(BOUNDARIES)
        return (result, content(project, pane, result, inspection_preferences, selected),
                count_controls(project, pane, result, selected))
