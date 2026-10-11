"""Row-aware TurboMoC geometry and meshing workspaces."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import time
from uuid import uuid4

import dash_vtk  # Register the browser bundle before constructing scenes.
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from flask import abort, send_from_directory

from . import workflow as flow, plots
from ..meshing import meshing_3d as mesh
from .. import configuration_project as project_model
from .preview import preview_geometry

ACTIONS = ("nozzle", "reference", "volume", "export_2d", "export", "cancel")


def row_label(row):
    # A single stage needs concise labels; stable IDs still identify saved jobs
    # and allow independently named rows when multistage support is introduced.
    return {"stator_1": "Stator", "rotor_1": "Rotor"}.get(
        row["id"], row["id"].replace("_", " ").title())


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
                ("blade_geometry", "Blade geometry definition"),
                ("flow_domain", "Flow domain definition"))]
        else:
            fields = [html.P("Rotor blade geometry integration pending; channel dimensions are available for layout.",
                className="placeholder"),
                group("Rotor parametrization", controls({"parametrization_type": geometry["parametrization_type"]},
                    "", identifier)),
                group("Linear cascade definition" if family == "linear" else "Meridional channel definition",
                    controls(geometry["meridional_channel"], "meridional_channel", identifier))]
            if family == "axial" and project["machine"]["design_mode"] == "stator_rotor":
                fields.append(group("Axial interspace", controls({"axial_interspace_mm":
                    geometry.get("axial_interspace_mm", 0.0)}, "", identifier)))
                fields.append(html.Small("Minimum axial clearance from the upstream trailing edge to this leading-edge plane [mm]."))
        placement = row.get("placement", {})
        visible = {k: v for k, v in placement.items() if
            (k == "streamwise_offset_mm" and family == "axial") or
            (k == "angular_phase_deg" and family != "linear")}
        if row["role"] == "rotor" and project["machine"]["design_mode"] == "stator_rotor":
            visible.pop("streamwise_offset_mm", None)
        if visible:
            fields.append(group("Assembly placement · display only", controls(visible, "placement", identifier)))
        result.append(html.Details([html.Summary(row_label(row)), *fields],
            open=True, className="settings design row-settings", **{"data-row": identifier}))
        if row["role"] == "rotor":
            mesh_fields = [html.P("Rotor meshing becomes available after geometry integration.",
                                 className="placeholder")]
        else:
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
                group("CGNS export", controls({k: v for k, v in config["mesh"]["cgns"].items()
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
    """Yield independently current row designs, never inventing a pending rotor."""
    for row in project_model.active_rows(state["project"]):
        if row["role"] != "stator":
            continue
        config = project_model.row_configuration(state["project"], row["id"])
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


def create_app(config_file=None):
    app = Dash(__name__, assets_folder=str(flow.APP_DIR / "assets"),
        title="TurboMoC", suppress_callback_exceptions=True)

    @app.server.route("/artifacts/<session>/<job>/<filename>")
    def download(session, job, filename):
        folder = flow.directory({"session": session, "job": job})
        if Path(filename).name != filename or Path(filename).suffix not in (
            ".json", ".yaml", ".msh", ".cgns", ".npz", ".log"):
            abort(404)
        return send_from_directory(folder, filename, as_attachment=True)

    def layout():
        project = project_model.load_project(config_file or flow.CASES["axial_stator"])
        row = project_model.active_rows(project)[0]
        config = (project_model.row_configuration(project, row["id"]) if row["role"] == "stator" else
            project_model.row_configuration(project_model.default_project(project["machine"]["geometry_type"]), "stator_1"))
        stores = [dcc.Store(id=n) for n in ("config", "job", "job-completion", "launch-busy",
            "geometry-data", "geometry-key", "volume-data", "mesh-width")]
        return html.Div([*stores, dcc.Store(id="session", data=uuid4().hex),
            dcc.Store(id="base-project", data=project), dcc.Store(id="row-references", data={}),
            dcc.Interval(id="poll", interval=750),
            html.Header([html.H2("TurboMoC"), html.A([
                html.Strong("Sustainable Thermal Power"),
                html.Span("Research group at DTU Construct")],
                href="https://thermalpower.dtu.dk/", target="_blank", rel="noopener noreferrer",
                className="authorship", id="authorship")]),
            html.Div([html.Aside([
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
                html.Small("MoC inputs: SI. Geometry: mm. Each field belongs to its named blade row."),
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
                    ("Compute MoC", "Generate 2D mesh", "Generate 3D mesh", "Export 2D", "Export 3D", "Cancel job"))],
                    className="actions"), html.Div([html.Span(id="spinner"), html.Span("Ready", id="job-status"),
                    html.Progress(id="job-progress", max=1, value=0)], className="job-strip")], className="action-bar"),
                html.Div(id="job-error", className="message"),
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
                ], id="views"), type="circle", delay_show=300), html.Div(id="downloads", className="downloads")],
                    className="content-scroll"),
                html.Details([html.Summary("Terminal · live activity"), html.Pre(id="activity",
                    children="No background job yet.")], open=False, className="terminal", id="terminal"),
            ])], className="shell")], id="app")
    app.layout = layout

    @app.callback(Output("base-project", "data"), Output("settings", "children"),
        Output("selected-row", "options"), Output("selected-row", "value"),
        Output("machine-family", "options"), Output("machine-family", "value"), Output("machine-note", "children"),
        Input("design-mode", "value"), Input("machine-family", "value"),
        State("config", "data"), State("base-project", "data"), State("selected-row", "value"))
    def structure(mode, family, state, base, selected):
        # Even an invalid edit state retains its last validated project. A mode
        # change must preserve those accepted edits when hiding a blade row.
        project = deepcopy(state["project"] if state and state.get("project") else base)
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
        return project, sidebar(project), [{"label": row_label(r), "value": r["id"]} for r in rows], selected, options, family, note

    @app.callback(Output("config", "data"), Output("validation", "children"),
        Input({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        Input("base-project", "data"), Input("row-references", "data"), Input("selected-row", "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), State("config", "data"))
    def edit(values, base, registry, row_id, identifiers, previous=None):
        try:
            project = project_model.validate_project(flow.edited_project(base, values, identifiers))
            row = project_model.get_row(project, row_id)
            state = {"project": project, "row_id": row_id, "role": row["role"], "valid": True,
                     "supported": row["role"] == "stator", "config": None}
            if state["supported"]:
                config = project_model.row_configuration(project, row_id)
                state.update(config=config, **flow.keys(config, (registry or {}).get(row_id, {}).get("nozzle")))
            return state, ""
        except (ValueError, TypeError, KeyError) as exc:
            # Leave the raw input mounted so it can be corrected, but retain
            # the last accepted row settings for later structural changes.
            accepted = deepcopy(previous["project"] if previous and previous.get("project") else base)
            return {"project": accepted, "row_id": row_id, "valid": False, "supported": False}, str(exc)

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
            for r in project_model.active_rows(state["project"]) if r["role"] == "stator"
            and not family == "linear"]
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
            for r in active if r["role"] == "rotor"]
        key = flow.digest([state["row_id"], planning, [(r["id"], k["geometry"], placements[r["id"]])
            for r, _, _, k in ready]])
        if key == previous_key:
            return (no_update,) * 6
        datasets, views, errors = [], [], []
        section, basis = empty("Selected blade row"), "Compute MoC for this row to display its geometry."
        for row, config, reference, keys in ready:
            try:
                design = flow.get_design(config, reference)
                dataset = deepcopy(cached_preview(keys["geometry"], design))
                dataset.update(id=row["id"], label=row_label(row), placement=placements[row["id"]])
                datasets.append(dataset)
                views.append((row_label(row), design, placements[row["id"]]))
                if row["id"] == state["row_id"]:
                    section = plots.section_figure(design)
                    basis = ("Physical linear cascade section [mm]; uniformly scaled MoC blade and pitch."
                        if dataset.get("is_linear") else
                        "Reference conformal section: MoC coordinates scaled to the selected meridional chord.")
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(f"{row_label(row)}: {exc}")
        if not state.get("supported"):
            basis = "Rotor geometry integration pending; no rotor surface is synthesized."
        note = "Shared machine view. Row placement affects display only; these remain independent row domains."
        if any(r["role"] == "rotor" for r in project_model.active_rows(state["project"])):
            note += " Dashed rotor envelope shows its prescribed meridional channel; blade geometry is pending."
        if errors:
            note += " " + " ".join(errors)
        return {"key": key, "rows": datasets}, key, section, plots.assembly_meridional_figure(views, planning), basis, note

    @app.callback(Output("reference-plot", "figure"), Output("meridional-plot", "figure"),
        Output("reference-basis", "children"), Output("meridional-basis", "children"),
        Input("config", "data"), Input("row-references", "data"), State("reference-plot", "figure"))
    def mesh_setup(state, registry, previous):
        if not state or not state.get("valid") or not state.get("supported"):
            return plots.section_mesh_placeholder("Selected-row 2D mesh"), empty("Span grid projection"), "Select an implemented stator row.", ""
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
            if row["role"] != "stator":
                continue
            reference = (registry or {}).get(row["id"], {}).get("volume")
            if not reference:
                continue
            try:
                surfaces, summary, report = volume_artifact(reference["session"], reference["job"])
                definition = summary.get("geometry", {})
                linear = definition.get("type") == "linear"
                dataset = {"id": row["id"], "label": row_label(row), "surfaces": surfaces,
                    "is_linear": linear, "placement": project_model.assembly_placement(state["project"], row["id"]),
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
            return no_update, "Select a valid stator row; rotor execution is pending."
        try:
            handle = flow.start_job(session, action, state["config"], selected_references(state, registry),
                row_id=state["row_id"], row_role=state["role"], project=state["project"])
            return handle, ""
        except (ValueError, OSError) as exc:
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
        disabled = (busy or not supported, busy or not current["nozzle"], busy or not current["reference"],
                    busy or not current["reference"], busy or not current["volume"], not busy)
        message = ""
        if refs.get("volume") and not current["volume"]:
            message = "Displayed mesh belongs to an earlier configuration; regenerate it before export."
        scope = (f"Actions apply to {row_label(project_model.get_row(state['project'], state['row_id']))}." if supported else
                 "Rotor integration pending; its planning settings do not enable jobs.")
        return (*disabled, message, scope)

    @app.callback(Output("interspace-note", "children"), Input("config", "data"))
    def interspace_note(state):
        if not state or not state.get("valid"):
            return ""
        messages = []
        for spacing in project_model.assembly_interspaces(state["project"]):
            upstream = row_label(project_model.get_row(state["project"], spacing["upstream_id"]))
            downstream = row_label(project_model.get_row(state["project"], spacing["downstream_id"]))
            gap = spacing["gap_mm"]
            if spacing["geometry_type"] == "axial":
                description = f"Minimum axial interspace: {gap:g} mm; prescribed on {downstream}."
            else:
                description = f"Radial interspace along the flow: {gap:g} mm; derived from outlet/inlet radii."
                if gap < 0:
                    description += " Meridional row extents overlap; review the radii."
            messages.append(html.Span(f"{upstream} → {downstream} · {description}"))
        return messages

    @app.callback(Output("downloads", "children"), Input("row-references", "data"), Input("job", "data"))
    def downloads(registry, handle):
        links, seen = [], set()
        references = [ref for stages in (registry or {}).values() for ref in stages.values()]
        if handle:
            references.append(handle)
        for reference in references:
            if reference["job"] in seen:
                continue
            seen.add(reference["job"])
            folder = flow.directory(reference)
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in (".json", ".yaml", ".msh", ".cgns", ".npz", ".log"):
                    continue
                links.append(html.A(f"{reference.get('row_id', 'stator_1')} · {reference['action']} · {path.name}",
                    href=f"/artifacts/{reference['session']}/{reference['job']}/{path.name}"))
        return links

    app.clientside_callback("function(data, copies, surfaces) { return window.dash_clientside.studio.render(data, copies, surfaces, [], 1, 'geometry'); }",
        Output("geometry-scene", "children"), Output("geometry-note", "children"), Input("geometry-data", "data"),
        Input("geometry-copies", "value"), Input("geometry-surfaces", "value"))
    app.clientside_callback("function(data, copies, surfaces, edges, width) { return window.dash_clientside.studio.render(data, copies, surfaces, edges, width, 'mesh'); }",
        Output("mesh-scene", "children"), Output("mesh-note", "children"), Input("volume-data", "data"), Input("mesh-copies", "value"),
        Input("mesh-surfaces", "value"), Input("mesh-edges", "value"), Input("mesh-width", "data"))
    return app


