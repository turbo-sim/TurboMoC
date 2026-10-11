"""Row-aware TurboMoC geometry and meshing workspaces."""
from copy import deepcopy
import base64
import binascii
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import time
import yaml
from uuid import uuid4

import dash_vtk  # Register the browser bundle before constructing scenes.
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from dash.exceptions import MissingCallbackContextException
from flask import abort, send_from_directory

from . import workflow as flow, plots, project_export
from ...meshing import meshing_3d as mesh
from ... import configuration_project as project_model
from ...configuration_rotor import default_axial_interspace_mm
from .preview import preview_geometry

ACTIONS = ("nozzle", "reference", "volume", "cancel")
ROTOR_INLET_FIELDS = {
    "fluid_name": "Fluid",
    "P0_rel": "Relative stagnation pressure [Pa]",
    "T0_rel": "Relative stagnation temperature [K]",
    "M_inlet": "Relative inlet Mach number",
    "beta_inlet": "Relative inlet flow angle [deg]",
}


def row_label(row):
    # A single stage needs concise labels; stable IDs still identify saved jobs
    # and allow independently named rows when multistage support is introduced.
    return {"stator_1": "Stator", "rotor_1": "Rotor"}.get(
        row["id"], row["id"].replace("_", " ").title())


def read_uploaded_project(contents):
    """Decode and validate a browser upload before changing any settings."""
    try:
        _, encoded = contents.split(",", 1)
        content = base64.b64decode(encoded, validate=True).decode("utf-8-sig")
        return project_model.validate_project(project_model.normalize_project(yaml.safe_load(content)))
    except (ValueError, TypeError, KeyError, UnicodeError, binascii.Error, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration: {exc}") from exc


def controls(value, prefix, row_id):
    """Stable row IDs separate field ownership from display order."""
    result = []
    for name, item in value.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            result.append(html.Details([html.Summary(name.replace("_", " ")),
                *controls(item, path, row_id)], open=True))
            continue
        if path == "meridional_channel.type":
            continue
        identifier = {"type": "parameter", "row": row_id, "path": path}
        if isinstance(item, bool):
            widget = dcc.Dropdown(id=identifier, options=[
                {"label": "Yes", "value": True}, {"label": "No", "value": False}],
                value=item, clearable=False)
        elif name in ("solver", "domain_shift_mode", "parametrization_type"):
            options = {"solver": ["conventional", "mln"],
                "domain_shift_mode": ["auto", "manual"],
                "parametrization_type": ["conventional", "vortex"]}[name]
            widget = dcc.Dropdown(id=identifier, options=options, value=item, clearable=False)
        else:
            widget = dcc.Input(id=identifier, value=item,
                type="number" if isinstance(item, (int, float)) else "text", debounce=True,
                step=1 if name in flow.INTEGER_KEYS else "any")
        result.append(html.Label([html.Span(name.replace("_", " ")), widget],
            title=f"{row_id}.{path}", className="control"))
    return result


def group(title, children):
    return html.Details([html.Summary(title), *children], open=True, className="field-group")


def rotor_inlet_controls(geometry, row_id, combined):
    """Keep manual values mounted while coupled values use separate read-only cells.

    This preserves the user's independent inlet state when toggling coupling,
    and prevents calculated values from feeding back into editable settings.
    """
    coupling = geometry["inlet_coupling"]
    coupled = coupling["mode"] == "stator_exit"
    manual = []
    for name, label in ROTOR_INLET_FIELDS.items():
        manual.append(html.Label([html.Span(label), dcc.Input(
            id={"type": "parameter", "row": row_id, "path": f"rotor_design.{name}"},
            value=geometry["rotor_design"][name],
            type="text" if name == "fluid_name" else "number", debounce=True,
            step="any")], className="control"))
    return group("Rotor inlet state", [
        html.Label([html.Span("Inlet conditions"), dcc.Dropdown(
            id={"type": "parameter", "row": row_id, "path": "inlet_coupling.mode"},
            options=[{"label": "Specify manually", "value": "manual"},
                {"label": "Couple to stator exit", "value": "stator_exit", "disabled": not combined}],
            value=coupling["mode"], clearable=False)], className="control"),
        html.Div(html.Label([html.Span("Rotational speed [rpm]"), dcc.Input(
            id={"type": "parameter", "row": row_id, "path": "inlet_coupling.rpm"},
            value=coupling["rpm"], type="number", debounce=True, step="any")], className="control"),
            id={"type": "rotor-coupling-speed", "row": row_id},
            style={} if coupled else {"display": "none"}),
        html.Div(manual, id={"type": "rotor-manual-inlet", "row": row_id},
            style={"display": "none"} if coupled else {}),
        html.Div(id={"type": "rotor-coupled-inlet", "row": row_id}, className="computed-inlet",
            style={} if coupled else {"display": "none"}),
        html.Small("Inlet quantities use the rotor-relative frame; the flow angle is measured from the meridional direction."),
    ])


def internal_domain_sides(project, row_id):
    """Identify internal boundaries without requiring a currently valid gap."""
    rows = project_model.active_rows(project)
    index = next(i for i, row in enumerate(rows) if row["id"] == row_id)
    return {side for side, internal in (("inlet", index > 0),
        ("outlet", index < len(rows) - 1)) if internal}


def flow_domain_controls(project, row):
    """Retain solitary-row values while shared boundaries derive their extent.

    Hidden controls remain mounted so switching to a single row restores its
    saved extension, instead of writing the derived assembly value into YAML.
    """
    internal = internal_domain_sides(project, row["id"])
    children = []
    for name, value in row["geometry"]["flow_domain"].items():
        side = name.removesuffix("_chords") if name.endswith("_chords") else None
        children.append(html.Div(controls({name: value}, "flow_domain", row["id"]),
            style={"display": "none"} if side in internal else {}))
    children.append(html.Div(id={"type": "shared-domain", "row": row["id"]},
        className="computed-inlet"))
    return group("Flow domain definition", children)


def shared_domain_readout(project, row_id):
    """Show physical distances, not the inactive conformal chord fractions."""
    assembly = project_model.row_assembly(project, row_id)
    if not assembly:
        return []
    row = project_model.get_row(project, row_id)
    channel = row["geometry"]["meridional_channel"]
    origin = assembly["placement"]["streamwise_offset_mm"]
    result = []
    for side, interface in assembly["interfaces"].items():
        coordinate = interface["coordinate_mm"]
        axial = interface["geometry_type"] == "axial"
        label = f"{side.title()} interface {'z' if axial else 'radius'} [mm]"
        result.append(html.Label([html.Span(label), dcc.Input(value=coordinate,
            type="number", disabled=True)], className="control computed-control"))
        distances = []
        for fraction, name in ((0, "Hub"), (0.5, "Midspan"), (1, "Tip")):
            if axial:
                chord = (1 - fraction) * channel["hub_chord"] + fraction * channel["tip_chord"]
                distance = origin - coordinate if side == "inlet" else coordinate - origin - chord
            else:
                distance = abs(coordinate - channel[f"{side}_radius"])
            distances.append(html.Label([html.Span(f"{name} extension [mm]"),
                dcc.Input(value=round(distance, 10), type="number", disabled=True)],
                className="control computed-control"))
        result.extend(distances)
    result.append(html.Small("Internal boundaries meet at the shared mid-gap interface. "
        "These physical extensions are derived; saved chord fractions apply when the row is used alone. "
        "External inlet/outlet extensions remain editable."))
    return result


def sidebar(project):
    result = []
    family = project["machine"]["geometry_type"]
    for row in project_model.active_rows(project):
        identifier, geometry = row["id"], row["geometry"]
        if row["role"] == "stator":
            fields = [group(title, controls(geometry[name], name, identifier)) for name, title in (
                ("method_of_characteristics", "Method of characteristics"),
                ("meridional_channel", "Linear cascade definition" if family == "linear"
                     else "Meridional channel definition"),
                ("blade_geometry", "Blade geometry definition"))]
            fields.append(flow_domain_controls(project, row))
        else:
            fields = [rotor_inlet_controls(geometry, identifier,
                    project["machine"]["design_mode"] == "stator_rotor"),
                group("Vortex blade design", controls({k: v for k, v in geometry["rotor_design"].items()
                    if k not in ROTOR_INLET_FIELDS}, "rotor_design", identifier)),
                group("Linear cascade definition" if family == "linear" else "Meridional channel definition",
                    controls(geometry["meridional_channel"], "meridional_channel", identifier)),
                flow_domain_controls(project, row)]
            if family == "axial" and project["machine"]["design_mode"] == "stator_rotor":
                fields.insert(0, group("Axial interspace", [*controls({"axial_interspace_mm":
                    geometry.get("axial_interspace_mm", default_axial_interspace_mm(
                        geometry["meridional_channel"]))}, "", identifier),
                    html.Small("Minimum axial clearance from the upstream trailing edge to this leading-edge plane [mm]. "
                        "New or missing values default to 10% of the rotor midspan axial chord; saved gaps remain unchanged.")]))
        placement = row.get("placement", {})
        visible = {k: v for k, v in placement.items() if
            (k == "streamwise_offset_mm" and family == "axial") or
            (k == "angular_phase_deg" and family != "linear")}
        if row["role"] == "rotor" and project["machine"]["design_mode"] == "stator_rotor":
            visible.pop("streamwise_offset_mm", None)
        if visible:
            title = "Assembly placement" if len(project_model.active_rows(project)) > 1 else "Assembly placement · display only"
            fields.append(group(title, controls(visible, "placement", identifier)))
        result.append(html.Details([html.Summary(row_label(row)), *fields],
            open=True, className="settings design row-settings", **{"data-row": identifier}))
        config = project_model.row_configuration(project, identifier)
        numerics = config["mesh"]["numerics"]
        section = {k: v for k, v in numerics.items() if k not in ("span_layers", "span_stretch")}
        mesh_fields = [
                group("2D mesh · physical mm" if family == "linear" else "2D mesh · reference conformal mm",
                      controls(section, "mesh.numerics", identifier)),
                group("Span discretization", controls({k: numerics[k] for k in
                    ("span_layers", "span_stretch")}, "mesh.numerics", identifier)),
                group("Mesh visualization", controls(config["mesh"]["visualization"],
                    "mesh.visualization", identifier)),
                html.Small("Edge widths are display pixels; they do not require remeshing."),
                group("CGNS mesh output", controls({k: v for k, v in config["mesh"]["cgns"].items()
                    if k != "enabled"}, "mesh.cgns", identifier)),
            ]
        result.append(html.Details([html.Summary(row_label(row)), *mesh_fields],
            open=True, className="settings mesh row-settings", **{"data-row": identifier}))
    return result


def empty(title):
    return plots.figure(title, "", "")


def viewport(name):
    return html.Div([html.Div(id=f"{name}-scene", className="vtk-host"),
        html.Small(id=f"{name}-note")], className="vtk-frame")


def surface_options(family):
    return [{"label": {"hub": "span start", "shroud": "span end"}.get(n, n.replace("_", " "))
             if family == "linear" else n.replace("_", " "), "value": n}
            for n in mesh.BOUNDARIES]


def view_controls(pane, config, *, mesh_view=False):
    family = config["meridional_channel"]["type"]
    count = 25 if family == "linear" else config["meridional_channel"]["num_blades"]
    defaults = config["mesh"]["pyvista"]
    result = [html.Div([html.Label("Displayed blades"), dcc.Slider(
        id=f"{pane}-copies", min=1, max=count, step=1,
        value=min(count, config["mesh"]["matplotlib"]["num_blades"]),
        marks={1: "1", count: str(count)}, updatemode="mouseup")], className="blade-count"),
        dcc.Checklist(id=f"{pane}-surfaces", options=surface_options(family),
            value=[n for n, chosen in defaults["surfaces"].items() if chosen],
            inline=True, className="surface-options")]
    if mesh_view:
        result.append(dcc.Checklist(id="mesh-edges", options=[{"label": "Mesh edges", "value": "edges"}],
            value=["edges"] if defaults["show_mesh_edges"] else []))
    return html.Div(result, className="view-controls")


@lru_cache(maxsize=8)
def reference_artifact(session, job):
    return mesh.load_section(flow.directory({"session": session, "job": job}))


@lru_cache(maxsize=8)
def volume_artifact(session, job):
    folder = flow.directory({"session": session, "job": job})
    return [json.loads((folder / name).read_text(encoding="utf-8")) for name in
        ("browser_surfaces.json", "mesh_summary.json", "mesh_report.json")]


def selected_references(state, registry):
    return (registry or {}).get((state or {}).get("row_id"), {})


def ready_rows(state, registry):
    """Yield current row profiles, resolving coupled inlet dependencies first."""
    for row in project_model.active_rows(state["project"]):
        try:
            config = flow.resolved_configuration(state["project"], row["id"], registry)
        except (ValueError, OSError, RuntimeError):
            continue
        reference = (registry or {}).get(row["id"], {}).get("nozzle")
        keys = flow.keys(config, reference)
        if flow.current(reference, keys["nozzle"]):
            yield row, config, reference, keys


# These are derived display datasets, never cached MoC solutions.
_PREVIEWS = {}
def cached_preview(key, design):
    if key not in _PREVIEWS:
        if len(_PREVIEWS) >= 8:
            _PREVIEWS.pop(next(iter(_PREVIEWS)))
        _PREVIEWS[key] = preview_geometry(design)
    return _PREVIEWS[key]


def report_panel(summary, report):
    cells = sum(summary["cells"].values())
    rows = [("Volume cells", f"{cells:,}"), ("Volume nodes", f"{summary['volume_nodes']:,}"),
            ("Span layers", len(summary["span_stations"]) - 1),
            ("Minimum SICN", f"{summary['quality']['min_sicn']:.4g}"),
            ("3D time [s]", f"{summary['elapsed_s']:.2f}")]
    table = html.Table([html.Tbody([html.Tr([html.Td(k), html.Td(str(v))]) for k, v in rows])])
    quality = html.Table([html.Thead(html.Tr([html.Th(v) for v in ("Metric", "Min", "Mean", "Max")])),
        html.Tbody([html.Tr([html.Td(n.replace("_", " ")), *[html.Td(f"{m[v]:.3g}")
            for v in ("min", "mean", "max")]]) for n, m in report["metrics"].items()])])
    result = [html.H3("Actual volume mesh"), table, quality]
    for name in ("skewness", "orthogonality", "aspect_ratio"):
        result += [dcc.Graph(figure=plots.histogram(name, report["metrics"][name]),
                             config={"displayModeBar": False}),
                   html.Small(report["definitions"][name])]
    result += [html.Details([html.Summary("Worst cells · original Gmsh IDs"),
        *[html.P(n + ": " + ", ".join(f"{c['cell_id']} ({c['value']:.3g})" for c in cells))
          for n, cells in report["worst_cells"].items()]])]
    return result


def create_app(config_file=None, output_dir=None):
    app = Dash(__name__, assets_folder=str(flow.APP_DIR / "assets"),
        title="TurboMoC", suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("{%favicon%}",
        f'<link rel="icon" type="image/svg+xml" href="{app.get_asset_url("turbomoc-balanced-atlantic.svg")}">')
    app.server.config["TURBO_MOC_OUTPUT_DIR"] = (
        Path(output_dir).resolve() if output_dir is not None else flow.OUTPUT)

    @app.server.route("/artifacts/<session>/<job>/<filename>")
    def download(session, job, filename):
        folder = flow.directory({"session": session, "job": job})
        if Path(filename).name != filename or Path(filename).suffix.lower() not in (
            ".json", ".yaml", ".cgns", ".log"):
            abort(404)
        if Path(filename).suffix.lower() == ".cgns":
            status_file = folder / "status.json"
            if not status_file.exists() or flow.read_json(status_file).get("state") != "complete":
                abort(404)
        return send_from_directory(folder, filename, as_attachment=True)

    def layout():
        project = project_model.load_project(config_file) if config_file is not None else project_model.default_project()
        row = project_model.active_rows(project)[0]
        config = project_model.row_configuration(project, row["id"])
        stores = [dcc.Store(id=n) for n in ("config", "job", "job-completion", "launch-busy",
            "geometry-data", "geometry-key", "volume-data", "mesh-width")]
        return html.Div([*stores, dcc.Store(id="session", data=uuid4().hex),
            dcc.Store(id="loaded-project"), dcc.Download(id="configuration-download"),
            dcc.Download(id="project-download"),
            dcc.Store(id="base-project", data=project), dcc.Store(id="row-references", data={}),
            dcc.Interval(id="poll", interval=750),
            html.Header([html.H2("TurboMoC"), html.A([
                html.Strong("Sustainable Thermal Power"),
                html.Span("Research group at DTU Construct")],
                href="https://thermalpower.dtu.dk/", target="_blank", rel="noopener noreferrer",
                className="authorship", id="authorship")]),
            html.Div([html.Aside([
                html.Div([
                    html.Div([
                        dcc.Upload(id="configuration-upload", accept=".yaml,.yml", multiple=False,
                            children=html.Button("Load configuration", id="load-configuration")),
                        html.Button("Save configuration", id="save-configuration", n_clicks=0),
                    ], className="configuration-actions"),
                    html.Small(id="configuration-message"),
                ], className="configuration-settings"),
                html.Div([
                    html.Label([html.Span("Design mode"), dcc.Dropdown(id="design-mode", clearable=False,
                        value=project["machine"]["design_mode"], options=[
                            {"label": "Stator only", "value": "stator"},
                            {"label": "Rotor only", "value": "rotor"},
                            {"label": "Stator + Rotor", "value": "stator_rotor"}])], className="control"),
                    html.Label([html.Span("Machine geometry"), dcc.Dropdown(id="machine-family", clearable=False,
                        value=project["machine"]["geometry_type"], options=[
                            {"label": "Linear cascade", "value": "linear"},
                            {"label": "Axial", "value": "axial"},
                            {"label": "Radial", "value": "radial"}])], className="control"),
                    html.Small(id="machine-note"),
                ], className="project-settings"),
                html.Div(id="settings", children=sidebar(project)),
                html.Small("Flow inputs: SI. Geometry: mm. Each field belongs to its named blade row."),
            ], id="sidebar"), html.Main([
                dcc.Tabs(id="workspace", value="design", children=[
                    dcc.Tab(label="Geometry generation", value="design"),
                    dcc.Tab(label="Mesh generation", value="mesh")]),
                html.Div([html.Label("Selected row"), dcc.Dropdown(id="selected-row", clearable=False,
                    value=row["id"], options=[{"label": row_label(r), "value": r["id"]}
                    for r in project_model.active_rows(project)]), html.Small(id="row-status")],
                    className="row-selector"),
                html.Div(id="validation", className="message"),
                html.Div(id="interspace-note", className="assembly-info"),
                html.Div([html.Div([html.Button(label, id=name, n_clicks=0,
                    disabled=name=="cancel") for name, label in zip(ACTIONS,
                    ("Compute rotor blade" if row["role"] == "rotor" else "Compute MoC",
                     "Generate 2D mesh", "Generate 3D mesh", "Cancel job"))],
                    className="actions"), html.Div([html.Span(id="spinner"), html.Span("Ready", id="job-status"),
                    html.Progress(id="job-progress", max=1, value=0)], className="job-strip"),
                    html.Button("Download project (.zip)", id="download-project", n_clicks=0,
                        disabled=True, title=project_export.NO_MESH)], className="action-bar"),
                html.Div(id="job-error", className="message"),
                html.Div(id="project-download-message", className="message"),
                html.Div([dcc.Loading(html.Div([
                    html.Section([
                        html.Details([html.Summary("Blade-to-blade and machine geometry"), html.Div([
                            html.Div([dcc.Graph(id="design-section", figure=empty("Selected blade row"),
                                className="engineering-plot"), html.Small(id="design-basis")], className="pane"),
                            html.Div([dcc.Graph(id="meridional_channel", figure=empty("Machine geometry"),
                                className="engineering-plot"), html.Small(id="assembly-basis")], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Three-dimensional machine geometry"),
                            view_controls("geometry", config), viewport("geometry")],
                            open=True, className="plot-group", id="geometry-details"),
                    ], id="design-panel"),
                    html.Section([
                        html.Details([html.Summary("Reference section and span discretization"), html.Div([
                            html.Div([dcc.Graph(id="reference-plot", figure=plots.section_mesh_placeholder("2D reference mesh"),
                                className="engineering-plot"), html.Small(id="reference-basis")], className="pane"),
                            html.Div([dcc.Graph(id="meridional-plot", figure=empty("Span grid projection"),
                                className="engineering-plot"), html.Small(id="meridional-basis")], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Three-dimensional mesh · selected-row quality"),
                            view_controls("mesh", config, mesh_view=True), html.Div(id="volume-status", className="message"),
                            html.Div([html.Div(viewport("mesh"), className="mesh-pane"),
                                html.Aside(id="report", children="Generate a 3D mesh to see its report.", className="report")],
                                className="split")], open=True, className="plot-group", id="mesh-details"),
                    ], id="mesh-panel"),
                ], id="views"), type="circle", delay_show=300)],
                    className="content-scroll"),
                html.Details([html.Summary("Terminal · live activity"), html.Pre(id="activity",
                    children="No background job yet.")], open=False, className="terminal", id="terminal"),
            ])], className="shell")], id="app")
    app.layout = layout

    @app.callback(Output("loaded-project", "data"), Output("configuration-download", "data"),
        Output("configuration-message", "children"),
        Output("row-references", "data", allow_duplicate=True), Output("session", "data"),
        Output("job", "data", allow_duplicate=True),
        Input("configuration-upload", "contents"), Input("save-configuration", "n_clicks"),
        State("config", "data"),
        State("base-project", "data"), State("job", "data"), State("job-completion", "data"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), prevent_initial_call=True)
    def configuration_file(contents, clicks, state, base, handle, completed, values, identifiers):
        unchanged = (no_update,) * 3
        try:
            if ctx.triggered_id == "configuration-upload":
                if not contents:
                    return (no_update,) * 6
                if handle and completed != handle["job"]:
                    raise ValueError("Wait for the active job or cancel it before loading a configuration.")
                project = read_uploaded_project(contents)
                return project, no_update, "", {}, uuid4().hex, None
            project = project_model.validate_project(flow.edited_project(
                state["project"] if state and state.get("project") else base, values, identifiers))
            if state and not state.get("valid"):
                raise ValueError("Correct the invalid settings before saving the configuration.")
            download = {"content": yaml.safe_dump(project, sort_keys=False, allow_unicode=True),
                "filename": f"turbo_moc_{datetime.now():%Y-%m-%d_%H-%M-%S}.yaml", "type": "text/yaml"}
            return no_update, download, "Configuration saved.", *unchanged
        except (ValueError, TypeError, KeyError) as exc:
            return no_update, no_update, str(exc), *unchanged

    @app.callback(Output("base-project", "data"), Output("settings", "children"),
        Output("selected-row", "options"), Output("selected-row", "value"),
        Output("machine-family", "options"), Output("machine-family", "value"),
        Output("design-mode", "value"), Output("machine-note", "children"),
        Input("design-mode", "value"), Input("machine-family", "value"),
        State("config", "data"), State("base-project", "data"), State("selected-row", "value"),
        Input("loaded-project", "data"))
    def structure(mode, family, state, base, selected, loaded=None):
        # Even an invalid edit state retains its last validated project. A mode
        # change must preserve those accepted edits when hiding a blade row.
        project = deepcopy(state["project"] if state and state.get("project") else base)
        if loaded is not None and ctx.triggered_id == "loaded-project":
            project = deepcopy(loaded)
            mode, family = project["machine"]["design_mode"], project["machine"]["geometry_type"]
            state = None
        note = ("Invalid entries discarded; last valid settings retained."
                if state and not state.get("valid") else "")
        if mode == "stator_rotor" and family == "linear":
            family = "axial"
            note += " Stator + Rotor uses an annular machine; Axial selected."
        # Change the family first: a combined project cannot validate as linear.
        if mode != "stator_rotor":
            project = project_model.set_design_mode(project, mode)
        project = project_model.set_machine_type(project, family)
        project = project_model.set_design_mode(project, mode)
        rows = project_model.active_rows(project)
        identifiers = [r["id"] for r in rows]
        selected = selected if selected in identifiers else identifiers[0]
        options = [{"label": label, "value": value, "disabled": mode=="stator_rotor" and value=="linear"}
            for label, value in (("Linear cascade", "linear"), ("Axial", "axial"), ("Radial", "radial"))]
        return project, sidebar(project), [{"label": row_label(r), "value": r["id"]} for r in rows], selected, options, family, mode, note

    @app.callback(Output("config", "data"), Output("validation", "children"),
        Input({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        Input("base-project", "data"), Input("row-references", "data"), Input("selected-row", "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), State("config", "data"))
    def edit(values, base, registry, row_id, identifiers, previous=None):
        try:
            # A loaded project replaces the controls. Values from the previous
            # sidebar may still accompany this update; retain the uploaded values.
            try:
                replaced = ctx.triggered_id == "base-project"
            except MissingCallbackContextException:
                replaced = False
            project = project_model.validate_project(
                deepcopy(base) if replaced else flow.edited_project(base, values, identifiers))
            row = project_model.get_row(project, row_id)
            state = {"project": project, "row_id": row_id, "role": row["role"], "valid": True,
                     "supported": True, "config": project_model.row_configuration(project, row_id)}
            try:
                config = flow.resolved_configuration(project, row_id, registry)
            except (ValueError, OSError, RuntimeError) as exc:
                # Missing or stale upstream results are a workflow dependency,
                # not an invalid project: retain editable settings and explain
                # why rotor execution is temporarily unavailable.
                state.update(supported=False, dependency_error=str(exc))
                return state, str(exc)
            state.update(config=config, **flow.keys(config, (registry or {}).get(row_id, {}).get("nozzle")))
            return state, ""
        except (ValueError, TypeError, KeyError) as exc:
            # Leave the raw input mounted so it can be corrected, but retain
            # the last accepted row settings for later structural changes.
            accepted = deepcopy(previous["project"] if previous and previous.get("project") else base)
            return {"project": accepted, "row_id": row_id, "valid": False, "supported": False}, str(exc)

    @app.callback(Output({"type": "shared-domain", "row": ALL}, "children"),
        Input("config", "data"), State({"type": "shared-domain", "row": ALL}, "id"))
    def shared_domain_display(state, identifiers):
        values = []
        for identifier in identifiers:
            try:
                if not state or not state.get("valid"):
                    values.append([])
                    continue
                values.append(shared_domain_readout(state["project"], identifier["row"]))
            except (ValueError, KeyError, TypeError) as exc:
                values.append(html.Small(f"Cannot derive the shared boundary: {exc}"))
        return values

    @app.callback(
        Output({"type": "rotor-manual-inlet", "row": ALL}, "style"),
        Output({"type": "rotor-coupled-inlet", "row": ALL}, "children"),
        Output({"type": "rotor-coupled-inlet", "row": ALL}, "style"),
        Output({"type": "rotor-coupling-speed", "row": ALL}, "style"),
        Input("config", "data"), Input("row-references", "data"),
        State({"type": "rotor-manual-inlet", "row": ALL}, "id"))
    def rotor_inlet_display(state, registry, identifiers):
        manual_styles, computed_values, computed_styles, speed_styles = [], [], [], []
        for identifier in identifiers:
            row_id = identifier["row"]
            try:
                row = project_model.get_row(state["project"], row_id)
            except (KeyError, TypeError, ValueError):
                manual_styles.append({})
                computed_values.append([])
                computed_styles.append({"display": "none"})
                speed_styles.append({"display": "none"})
                continue
            coupled = row["geometry"]["inlet_coupling"]["mode"] == "stator_exit"
            manual_styles.append({"display": "none"} if coupled else {})
            computed_styles.append({} if coupled else {"display": "none"})
            speed_styles.append({} if coupled else {"display": "none"})
            values, message = {}, (
                "Computed from the current stator exit and rotor speed. "
                "Ideal transfer of the stator exit state; no interspace-flow/loss model.")
            if coupled:
                try:
                    if not state.get("valid"):
                        raise ValueError("Correct the invalid settings to update the coupled inlet state.")
                    resolved = flow.resolved_configuration(state["project"], row_id, registry)
                    values = resolved["rotor_design"]
                    source = resolved["rotor_source"]
                    message += (f" Inlet midspan radius: {source['inlet_radius_mm']:.4g} mm; "
                                f"blade speed: {source['triangle']['U']:.4g} m/s.")
                    if values["M_inlet"] <= 1:
                        message += " The relative inlet is subsonic; the vortex method requires Mach > 1."
                except (ValueError, OSError, RuntimeError) as exc:
                    message = str(exc)
            computed_values.append([
                html.Label([html.Span(label), dcc.Input(value=values.get(name), disabled=True,
                    type="text" if name == "fluid_name" else "number", placeholder="Pending")],
                    className="control computed-control",
                    title="Calculated from stator exit; switch to manual inlet conditions to edit.")
                for name, label in ROTOR_INLET_FIELDS.items()] + [html.Small(message)])
        return manual_styles, computed_values, computed_styles, speed_styles

    @app.callback(Output("sidebar", "className"), Output("design-panel", "style"), Output("mesh-panel", "style"),
        Input("workspace", "value"))
    def workspace(name):
        return (f"active-{name}",
            {} if name == "design" else {"display": "none"}, {} if name == "mesh" else {"display": "none"})

    @app.callback(*[Output(f"{pane}-copies", prop) for pane in ("geometry", "mesh")
        for prop in ("max", "marks", "value")], Output("geometry-surfaces", "options"),
        Output("mesh-surfaces", "options"), Output("mesh-width", "data"),
        Input("config", "data"), State("geometry-copies", "value"), State("mesh-copies", "value"))
    def display_settings(state, geometry_count, mesh_count):
        if not state or not state.get("valid"):
            return (no_update,) * 9
        family = state["project"]["machine"]["geometry_type"]
        config = state.get("config")
        annular_counts = [r["geometry"]["meridional_channel"]["num_blades"]
            for r in project_model.active_rows(state["project"]) if family != "linear"]
        count = 25 if family == "linear" else max(annular_counts, default=1)
        values = tuple(value for current in (geometry_count, mesh_count)
            for value in (count, {1: "1", count: str(count)}, min(count, max(1, current or 1))))
        return (*values, surface_options(family), surface_options(family),
                config["mesh"]["visualization"]["volume_edge_width"] if config else 0.35)

    @app.callback(Output("geometry-data", "data"), Output("geometry-key", "data"),
        Output("design-section", "figure"), Output("meridional_channel", "figure"),
        Output("design-basis", "children"), Output("assembly-basis", "children"),
        Input("config", "data"), Input("row-references", "data"), State("geometry-key", "data"))
    def geometry(state, registry, previous_key):
        if not state or not state.get("valid"):
            return None, None, empty("Selected blade row"), empty("Machine geometry"), "", ""
        ready = list(ready_rows(state, registry))
        active = project_model.active_rows(state["project"])
        placements = {r["id"]: project_model.assembly_placement(state["project"], r["id"]) for r in active}
        planning = [(row_label(r), r["geometry"]["meridional_channel"], placements[r["id"]])
            for r in active if r["id"] not in {item[0]["id"] for item in ready}]
        try:
            interfaces = project_model.shared_interfaces(state["project"])
            interface_error = ""
        except ValueError as exc:
            interfaces, interface_error = [], str(exc)
        key = flow.digest([state["row_id"], planning, interfaces, interface_error, [(r["id"], k["geometry"], placements[r["id"]])
            for r, _, _, k in ready]])
        if key == previous_key:
            return (no_update,) * 6
        datasets, views, errors = [], [], []
        section, basis = empty("Selected blade row"), (
            "Compute the rotor blade to display its geometry." if state.get("role") == "rotor"
            else "Compute MoC for this row to display its geometry.")
        for row, config, reference, keys in ready:
            try:
                design = flow.get_design(config, reference)
                dataset = deepcopy(cached_preview(keys["geometry"], design))
                dataset.update(id=row["id"], label=row_label(row), placement=placements[row["id"]])
                datasets.append(dataset)
                views.append((row_label(row), design, placements[row["id"]]))
                if row["id"] == state["row_id"]:
                    section = plots.section_figure(design)
                    method = "vortex-method" if row["role"] == "rotor" else "MoC"
                    basis = (f"Physical linear cascade section [mm]; uniformly scaled {method} blade and pitch."
                        if dataset.get("is_linear") else
                        f"Reference conformal section: {method} coordinates scaled to the selected meridional chord.")
                    if row["role"] == "rotor":
                        pitch_ratio = design.blade["pitch"] / design.blade["natural_pitch"]
                        basis += (f" Relative outlet angle: {design.blade['metal_angle_out']:.3g} deg; "
                                  f"reference pitch: {design.blade['pitch']:.3g} mm "
                                  f"(vortex natural pitch: {design.blade['natural_pitch']:.3g} mm; "
                                  f"applied/natural ratio: {pitch_ratio:.3g}).")
                        if abs(pitch_ratio - 1) > 0.01:
                            basis += (" Blade count/chord change the vortex design pitch; the original cascade "
                                      "flow solution then serves as a profile design reference.")
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(f"{row_label(row)}: {exc}")
        if not state.get("supported"):
            basis = state.get("dependency_error", "Resolve the inlet conditions to compute this row.")
        note = ("Shared machine view. Adjacent row domains meet at the marked physical interface; "
                "3D CGNS meshes use this assembly placement. Each row remains a separate mesh; "
                "solver interface coupling is configured in the CFD solver." if interfaces else
                "Shared machine view. A single-row mesh uses its row-local coordinates.")
        if interface_error:
            note = f"Shared interface unavailable: {interface_error}"
        if planning:
            note += " Dashed envelopes show prescribed meridional channels for rows awaiting a current blade design."
        if errors:
            note += " " + " ".join(errors)
        return {"key": key, "rows": datasets}, key, section, plots.assembly_meridional_figure(
            views, planning, interfaces=interfaces), basis, note

    @app.callback(Output("reference-plot", "figure"), Output("meridional-plot", "figure"),
        Output("reference-basis", "children"), Output("meridional-basis", "children"),
        Input("config", "data"), Input("row-references", "data"), State("reference-plot", "figure"))
    def mesh_setup(state, registry, previous):
        if not state or not state.get("valid") or not state.get("supported"):
            return plots.section_mesh_placeholder("Selected-row 2D mesh"), empty("Span grid projection"), (
                (state or {}).get("dependency_error", "Select a valid blade row.")), ""
        refs, config = selected_references(state, registry), state["config"]
        reference = refs.get("reference")
        if not flow.current(reference, state["reference"]):
            return plots.section_mesh_placeholder("2D reference mesh"), empty("Span grid projection"), "Generate an up-to-date 2D mesh for this row.", ""
        numerics = config["mesh"]["numerics"]
        render_key = flow.digest([state["row_id"], reference["job"], state["reference"],
            numerics["span_layers"], numerics["span_stretch"]])
        widths = config["mesh"]["visualization"]
        meta = (previous or {}).get("layout", {}).get("meta", {})
        if meta.get("mesh_key") == render_key:
            if meta.get("edge_width") == widths["section_edge_width"]:
                return (no_update,) * 4
            section_patch, span_patch = Patch(), Patch()
            section_patch["data"][0]["line"]["width"] = widths["section_edge_width"]
            for index in range(1, numerics["span_layers"] + 3):
                span_patch["data"][index]["line"]["width"] = widths["section_edge_width"]
            section_patch["layout"]["meta"]["edge_width"] = widths["section_edge_width"]
            return section_patch, span_patch, no_update, no_update
        section, design = reference_artifact(reference["session"], reference["job"])
        section_plot = plots.section_mesh_figure(section, design, edge_width=widths["section_edge_width"])
        section_plot.update_layout(meta={"mesh_key": render_key, "edge_width": widths["section_edge_width"]})
        span_plot, span_data = plots.meridional_figure(design, numerics, section,
            edge_width=widths["section_edge_width"])
        basis = ("Actual physical 2D mesh [mm]; translational periodicity."
            if design.meridional_channel.is_linear else "Actual reference conformal mesh; annular 2D export uses this basis.")
        if numerics["inflation_layers"]["enabled"]:
            lower, upper = span_data["estimated_first_height_mm"]
            basis += (f" Sampled mapped first-layer height: {lower:.3g}–{upper:.3g} mm"
                      " in the blade surface; this is not a measured 3D wall-normal distance.")
        span_basis = ("Projected span nodes and grid lines; a preview of the sweep, not a separate volume mesh."
            f" {span_data['span_layers']} span layers; projected spacing"
            f" {span_data['min_span_spacing_mm']:.3g}–{span_data['max_span_spacing_mm']:.3g} mm.")
        return section_plot, span_plot, basis, span_basis

    @app.callback(Output("volume-data", "data"), Output("report", "children"),
        Input("config", "data"), Input("row-references", "data"))
    def mesh_view(state, registry):
        if not state or not state.get("valid"):
            return None, "Correct the configuration before inspecting the mesh."
        datasets, selected_report = [], "Generate a 3D mesh for the selected row to see its report."
        for row in project_model.active_rows(state["project"]):
            reference = (registry or {}).get(row["id"], {}).get("volume")
            if not reference:
                continue
            try:
                surfaces, summary, report = volume_artifact(reference["session"], reference["job"])
                definition = summary.get("geometry", {})
                linear = definition.get("type") == "linear"
                # Browser surfaces stay row-local. A saved assembly records
                # the one transform also used by its CGNS export. In particular,
                # do not reposition a stale mesh using newly edited gap values.
                assembly = summary.get("assembly") or {}
                placement = assembly.get("placement") or project_model.assembly_placement(
                    state["project"], row["id"])
                dataset = {"id": row["id"], "label": row_label(row), "surfaces": surfaces,
                    "is_linear": linear, "placement": placement,
                    "job": reference["job"]}
                if linear:
                    dataset["pitch_mm"] = definition["pitch_mm"]
                else:
                    cfg = project_model.row_configuration(state["project"], row["id"])
                    dataset["num_blades"] = definition.get("num_blades", cfg["meridional_channel"]["num_blades"])
                datasets.append(dataset)
                if row["id"] == state["row_id"]:
                    selected_report = report_panel(summary, report)
            except (OSError, ValueError, KeyError) as exc:
                if row["id"] == state["row_id"]:
                    selected_report = str(exc)
        key = flow.digest([(d["id"], d["job"], d["placement"]) for d in datasets])
        return {"key": key, "rows": datasets}, selected_report

    @app.callback(Output("job", "data"), Output("job-error", "children"),
        *[Input(action, "n_clicks") for action in ACTIONS],
        State("config", "data"), State("session", "data"), State("job", "data"),
        State("job-completion", "data"), State("row-references", "data"), prevent_initial_call=True,
        running=[(Output("launch-busy", "data"), True, False)])
    def launch(*arguments):
        state, session, handle, completed, registry = arguments[-5:]
        action = ctx.triggered_id
        running = bool(handle and completed != handle["job"])
        if action == "cancel":
            if running:
                flow.cancel_job(handle)
            return no_update, ""
        if running:
            return no_update, "Wait for the active row job or cancel it first."
        if not state or not state.get("valid") or not state.get("supported"):
            return no_update, (state or {}).get("dependency_error", "Select a valid blade row.")
        try:
            handle = flow.start_job(session, action, state["config"], selected_references(state, registry),
                row_id=state["row_id"], row_role=state["role"], project=state["project"], registry=registry)
            return handle, ""
        except (ValueError, OSError, RuntimeError) as exc:
            return no_update, str(exc)

    @app.callback(Output("row-references", "data"), Output("job-completion", "data"),
        Output("job-status", "children"), Output("job-progress", "value"), Output("activity", "children"),
        Output("spinner", "className"), Output("workspace", "value"),
        Output("selected-row", "value", allow_duplicate=True), Input("poll", "n_intervals"), Input("job", "data"),
        State("job-completion", "data"), State("row-references", "data"), State("base-project", "data"),
        prevent_initial_call=True)
    def poll(_, handle, completed, registry, project):
        if not handle or completed == handle["job"]:
            return (no_update,) * 8
        status, log = flow.poll_job(handle)
        done = status["state"] != "running"
        refs, destination, selected = no_update, no_update, no_update
        if status["state"] == "complete":
            result = json.loads((flow.directory(handle) / "result.json").read_text(encoding="utf-8"))
            refs = deepcopy(registry or {})
            row_id = handle.get("row_id", "stator_1")
            refs.setdefault(row_id, {})[handle["action"]] = result["reference"]
            if handle["action"] in ("reference", "volume") and row_id in {
                    r["id"] for r in project_model.active_rows(project)}:
                destination = "mesh"
                selected = row_id
        elapsed = time.time() - handle["started"]
        message = f"{status['message']} · {elapsed:.1f} s"
        return (refs, handle["job"] if done else no_update, message, status.get("fraction"), log,
                "spinner running" if not done else "", destination, selected)

    @app.callback(*[Output(action, "disabled") for action in ACTIONS],
        Output("volume-status", "children"), Output("row-status", "children"),
        Input("config", "data"), Input("row-references", "data"), Input("job", "data"),
        Input("job-completion", "data"), Input("launch-busy", "data"))
    def readiness(state, registry, handle, completed, launching):
        busy = bool(launching or (handle and completed != handle["job"]))
        supported = bool(state and state.get("valid") and state.get("supported"))
        refs = selected_references(state, registry)
        current = {stage: supported and flow.current(refs.get(stage), state[stage])
                   for stage in ("nozzle", "reference", "volume")}
        rotor = bool(state and state.get("role") == "rotor")
        supersonic = not rotor or not supported or state["config"]["rotor_design"]["M_inlet"] > 1
        disabled = (busy or not supported or not supersonic, busy or not current["nozzle"], busy or not current["reference"],
                    not busy)
        message = ""
        if refs.get("volume") and not current["volume"]:
            message = "Displayed mesh belongs to an earlier configuration; regenerate it to obtain an updated CGNS mesh."
        scope = (f"Actions apply to {row_label(project_model.get_row(state['project'], state['row_id']))}." if supported else
                 (state or {}).get("dependency_error", "Select a valid blade row."))
        if not supersonic:
            scope += " The vortex method requires a supersonic relative inlet (Mach > 1)."
        return (*disabled, message, scope)

    @app.callback(Output("nozzle", "children"), Input("config", "data"))
    def design_action_label(state):
        return "Compute rotor blade" if (state or {}).get("role") == "rotor" else "Compute MoC"

    @app.callback(Output("interspace-note", "children"), Input("config", "data"))
    def interspace_note(state):
        if not state or not state.get("valid"):
            return ""
        messages = []
        try:
            interfaces = {(item["upstream_id"], item["downstream_id"]): item
                for item in project_model.shared_interfaces(state["project"])}
        except ValueError as exc:
            return f"Shared interface unavailable: {exc}"
        for spacing in project_model.assembly_interspaces(state["project"]):
            upstream = row_label(project_model.get_row(state["project"], spacing["upstream_id"]))
            downstream = row_label(project_model.get_row(state["project"], spacing["downstream_id"]))
            gap = spacing["gap_mm"]
            if spacing["geometry_type"] == "axial":
                description = f"Minimum axial interspace: {gap:g} mm; prescribed on {downstream}."
            else:
                description = f"Radial interspace along the flow: {gap:g} mm; derived from outlet/inlet radii."
            interface = interfaces.get((spacing["upstream_id"], spacing["downstream_id"]))
            if interface:
                coordinate_name = "z" if spacing["geometry_type"] == "axial" else "r"
                description += f" Shared interface: {coordinate_name} = {interface['coordinate_mm']:g} mm (mid-gap)."
            messages.append(html.Span(f"{upstream} → {downstream} · {description}"))
        return messages

    @app.callback(Output("download-project", "disabled"), Output("download-project", "title"),
        Input("config", "data"), Input("row-references", "data"), Input("job", "data"),
        Input("job-completion", "data"), Input("launch-busy", "data"))
    def project_download_ready(state, registry, handle, completed, launching):
        if launching or (handle and completed != handle["job"]):
            return True, "Wait for the active job to finish before downloading the project."
        if not state or not state.get("valid"):
            return True, "Correct the configuration before downloading the project."
        try:
            meshes, omitted = project_export.collect_meshes(state["project"], registry)
        except (ValueError, TypeError, KeyError):
            return True, "Correct the configuration before downloading the project."
        if not meshes:
            return True, project_export.NO_MESH
        title = f"Download {len(meshes)} mesh file(s), the final YAML, README, mesh report and activity log."
        if omitted:
            title += " Unavailable or outdated meshes are listed in the README."
        return False, title

    @app.callback(Output("project-download", "data"), Output("project-download-message", "children"),
        Input("download-project", "n_clicks"), State("config", "data"), State("row-references", "data"),
        State("job", "data"), State("job-completion", "data"), State("launch-busy", "data"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), prevent_initial_call=True)
    def download_project(clicks, state, registry, handle, completed, launching, values, identifiers):
        try:
            if launching or (handle and completed != handle["job"]):
                raise ValueError("Wait for the active job to finish before downloading the project.")
            if not state or not state.get("valid"):
                raise ValueError("Correct the invalid settings before downloading the project.")
            project = project_model.validate_project(flow.edited_project(state["project"], values, identifiers))
            data, filename = project_export.build_archive(project, registry)
            return dcc.send_bytes(data, filename, type="application/zip"), ""
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return no_update, f"Cannot download project: {exc}"

    app.clientside_callback("function(data, copies, surfaces) { return window.dash_clientside.studio.render(data, copies, surfaces, [], 1, 'geometry'); }",
        Output("geometry-scene", "children"), Output("geometry-note", "children"), Input("geometry-data", "data"),
        Input("geometry-copies", "value"), Input("geometry-surfaces", "value"))
    app.clientside_callback("function(data, copies, surfaces, edges, width) { return window.dash_clientside.studio.render(data, copies, surfaces, edges, width, 'mesh'); }",
        Output("mesh-scene", "children"), Output("mesh-note", "children"), Input("volume-data", "data"), Input("mesh-copies", "value"),
        Input("mesh-surfaces", "value"), Input("mesh-edges", "value"), Input("mesh-width", "data"))
    return app


