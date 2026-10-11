"""TurboMoC design and meshing app. Run from the repository root on port 8052."""
import json
from functools import lru_cache
from pathlib import Path
import time
from uuid import uuid4

import dash_vtk  # Registers the browser component bundle.
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from flask import abort, send_from_directory

from . import workflow as flow, geometry_source as geo, meshing_source as mesh, plots
from .configuration import validate_configuration
from .preview import preview_geometry


def controls(value, prefix):
    """Readable YAML paths make control scope explicit to technical users."""
    rows = []
    for name, item in value.items():
        path = f"{prefix}.{name}"
        if isinstance(item, dict):
            rows.append(html.Details([html.Summary(name.replace("_", " ")),
                                      *controls(item, path)], open=True))
            continue
        if path == "meridional_channel.type":
            continue
        identifier = {"type": "parameter", "path": path}
        if isinstance(item, bool):
            control = dcc.Dropdown(id=identifier, options=[{"label": "Yes", "value": True},
                {"label": "No", "value": False}], value=item, clearable=False)
        elif name in ("solver", "domain_shift_mode"):
            control = dcc.Dropdown(id=identifier, options=["conventional", "mln"]
                if name == "solver" else ["auto", "manual"], value=item, clearable=False)
        else:
            control = dcc.Input(id=identifier, value=item,
                type="number" if isinstance(item, (float, int)) else "text",
                debounce=True, step=1 if name in flow.INTEGER_KEYS else "any")
        rows.append(html.Label([html.Span(name.replace("_", " ")), control],
                              title=path, className="control"))
    return rows


def group(title, scope, children):
    return html.Details([html.Summary([title, html.Small(scope)]), *children],
                        open=True, className=f"settings {scope}")


def channel_fields(config):
    return controls(config["meridional_channel"], "meridional_channel")


def sidebar(config):
    numerics = config["mesh"]["numerics"]
    section = {k: v for k, v in numerics.items() if k not in ("span_layers", "span_stretch")}
    return [
        group("Method of characteristics", "design",
              controls(config["method_of_characteristics"], "method_of_characteristics")),
        group("Meridional channel definition", "design", [
            html.Label([html.Span("Machine type"), dcc.Dropdown(
                id="channel-family", options=[{"label": "Axial", "value": "axial"},
                    {"label": "Radial", "value": "radial"}],
                value=config["meridional_channel"]["type"], clearable=False)], className="control"),
            html.Div(id="channel-fields", children=channel_fields(config))]),
        group("Blade geometry definition", "design",
              controls(config["blade_geometry"], "blade_geometry")),
        group("Flow domain definition", "design",
              controls(config["flow_domain"], "flow_domain")),
        group("2D mesh · conformal mm", "setup", controls(section, "mesh.numerics")),
        group("Span discretization", "setup", controls(
            {k: numerics[k] for k in ("span_layers", "span_stretch")}, "mesh.numerics")),
        group("Mesh visualization", "visualization", [
            edge_width_control("section", config), edge_width_control("volume", config),
            html.Small("Display widths in pixels; changes do not require remeshing.")]),
        group("CGNS export", "volume", controls(
            {k: v for k, v in config["mesh"]["cgns"].items() if k != "enabled"}, "mesh.cgns")),
    ]


def view_controls(pane, config, *, mesh_view=False):
    """Each 3D view owns its selection; 2D workspaces have no unused controls."""
    count = config["meridional_channel"]["num_blades"]
    defaults = config["mesh"]["pyvista"]
    result = [
        html.Div([html.Label("Displayed blades"), dcc.Slider(
            id=f"{pane}-copies", min=1, max=count, step=1,
            value=min(count, config["mesh"]["matplotlib"]["num_blades"]),
            marks={1: "1", count: str(count)}, updatemode="mouseup",
            tooltip={"placement": "bottom"})], className="blade-count"),
        dcc.Checklist(id=f"{pane}-surfaces", options=[
            {"label": n.replace("_", " "), "value": n} for n in
            ("blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right")],
            value=[n for n, selected in defaults["surfaces"].items() if selected],
            inline=True, className="surface-options")]
    if mesh_view:
        result += [dcc.Checklist(id="mesh-edges",
            options=[{"label": "Mesh edges", "value": "edges"}],
            value=["edges"] if defaults["show_mesh_edges"] else [])]
    return html.Div(result, className="view-controls")


def edge_width_control(kind, config):
    name = f"{kind}_edge_width"
    return html.Label([html.Span("3D mesh edge width [px]" if kind == "volume" else "2D grid edge width [px]"),
        dcc.Input(id={"type": "parameter", "path": f"mesh.visualization.{name}"},
            type="number", min=0.05, max=3, step=0.05, debounce=True,
            value=config["mesh"]["visualization"][name])],
        className="control", title=f"mesh.visualization.{name}: display only; no remeshing.")


def empty(title):
    return plots.figure(title, "", "")


def viewport(name):
    return html.Div([html.Div(id=f"{name}-scene", className="vtk-host"),
                     html.Small(id=f"{name}-note")], className="vtk-frame")


@lru_cache(maxsize=8)
def reference_artifact(session, job):
    return mesh.load_section(flow.directory({"session": session, "job": job}))


@lru_cache(maxsize=8)
def volume_artifact(session, job):
    folder = flow.directory({"session": session, "job": job})
    return [json.loads((folder / name).read_text(encoding="utf-8")) for name in
            ("browser_surfaces.json", "mesh_summary.json", "mesh_report.json")]


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


def create_app():
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
        config = flow.read_configuration(flow.CASES["axial_stator"])
        stores = [dcc.Store(id=n) for n in ("config", "nozzle-ref", "reference-ref", "volume-ref",
                   "export-ref", "job", "job-completion", "launch-busy", "geometry-data", "geometry-key",
                   "volume-data")]
        return html.Div([
            *stores, dcc.Store(id="session", data=uuid4().hex),
            dcc.Store(id="base-config", data=config), dcc.Interval(id="poll", interval=750),
            html.Header([html.H2("TurboMoC"),
                dcc.Checklist(id="all-settings", options=[{"label": "Show all settings", "value": "all"}],
                              value=[])]),
            html.Div([
                html.Aside([html.Div(id="settings", children=sidebar(config)),
                    html.Small("MoC inputs: SI. Geometry lengths: mm. Mesh sizes: reference conformal plane.")],
                    id="sidebar"),
                html.Main([
                    dcc.Tabs(id="workspace", value="design", children=[
                        dcc.Tab(label="1. Geometric design", value="design"),
                        dcc.Tab(label="2. Two-dimensional meshing", value="setup"),
                        dcc.Tab(label="3. Three-dimensional meshing", value="volume")]),
                    html.Div(id="validation", className="message"),
                    html.Div([
                        html.Div([
                            html.Button("Compute MoC", id="nozzle", n_clicks=0),
                            html.Button("Generate 2D mesh", id="reference", n_clicks=0),
                            html.Button("Generate 3D mesh", id="volume", n_clicks=0),
                            html.Button("Export CGNS", id="export", n_clicks=0),
                            html.Button("Cancel job", id="cancel", n_clicks=0, disabled=True),
                        ], className="actions"),
                        html.Div([html.Span(id="spinner"),
                            html.Span("Ready", id="job-status"),
                            html.Progress(id="job-progress", max=1, value=0)],
                            className="job-strip"),
                    ], className="action-bar"),
                    html.Div(id="job-error", className="message"),
                    # Only the content scrolls. The terminal remains anchored in
                    # the main pane's bottom row, with its summary always visible.
                    html.Div([
                        dcc.Loading(html.Div([
                            html.Section([
                                html.Details([
                                    html.Summary("Blade-to-blade and meridional geometry"),
                                    html.Div([
                                        html.Div([dcc.Graph(id="design-section",
                                            figure=empty("Blade-to-blade geometry"),
                                            className="engineering-plot"),
                                            html.Small(id="design-basis")], className="pane"),
                                        html.Div([dcc.Graph(id="channel", figure=empty("Meridional geometry"),
                                            className="engineering-plot"),
                                            html.Small("Grey region: blade LE-to-TE meridional envelope; "
                                                       "a projection across span, not blade thickness.")],
                                            className="pane"),
                                    ], className="split"),
                                ], open=True, className="plot-group", id="design-2d"),
                                html.Details([html.Summary("Three-dimensional blade shape"),
                                    view_controls("geometry", config), viewport("geometry")],
                                    open=True, className="plot-group", id="geometry-details"),
                            ], id="design-panel"),
                            html.Section([
                                html.Div([
                                    html.Div([dcc.Graph(id="reference-plot", figure=empty("2D reference mesh"),
                                        className="engineering-plot"), html.Small(id="reference-basis")], className="pane"),
                                    html.Div([dcc.Graph(id="meridional-plot", figure=empty("Meridional grid"),
                                        className="engineering-plot"), html.Small(id="meridional-basis")], className="pane"),
                                ], className="split"),
                            ], id="setup-panel"),
                            html.Section([
                                view_controls("mesh", config, mesh_view=True),
                                html.Div(id="volume-status", className="message"),
                                html.Div([
                                    html.Div(viewport("mesh"), className="mesh-pane"),
                                    html.Aside(id="report", children="Generate a 3D mesh to see its report.",
                                               className="report"),
                                ], className="split"),
                            ], id="volume-panel"),
                        ], id="views"), type="circle", delay_show=300),
                        html.Div(id="downloads", className="downloads"),
                    ], className="content-scroll"),
                    html.Details([html.Summary("Terminal · live activity"),
                        html.Pre(id="activity", children="No background job yet.")],
                        open=False, className="terminal", id="terminal"),
                ]),
            ], className="shell"),
        ], id="app")
    app.layout = layout

    @app.callback(Output("base-config", "data"), Output("channel-fields", "children"),
        Input("channel-family", "value"), State("config", "data"), State("base-config", "data"),
        prevent_initial_call=True)
    def channel_family(name, state, base):
        # Switching the dimensional adapter must not reset MoC, blade, domain,
        # or mesh work. Preserve dimensions common to both channel families.
        from copy import deepcopy
        config = deepcopy(state["config"] if state else base)
        previous = config["meridional_channel"]
        channel = flow.read_configuration(flow.CASES[f"{name}_stator"])["meridional_channel"]
        channel.update({k: previous[k] for k in channel if k != "type" and k in previous})
        config["meridional_channel"] = channel
        return config, channel_fields(config)

    @app.callback(Output("config", "data"), Output("validation", "children"),
        Input({"type": "parameter", "path": ALL}, "value"),
        Input("base-config", "data"), State({"type": "parameter", "path": ALL}, "id"))
    def edit(values, base, identifiers):
        try:
            config = validate_configuration(flow.edited_configuration(base, values, identifiers))
            return {"config": config, **flow.keys(config)}, ""
        except (ValueError, TypeError) as exc:
            return None, str(exc)

    @app.callback(Output("sidebar", "className"), Output("design-panel", "style"),
        Output("setup-panel", "style"), Output("volume-panel", "style"),
        Input("workspace", "value"), Input("all-settings", "value"))
    def workspace(name, all_settings):
        return [f"active-{name}" + (" show-all" if all_settings else ""),
            *[{} if n == name else {"display": "none"} for n in ("design", "setup", "volume")]]

    @app.callback(*[Output(f"{pane}-copies", prop) for pane in ("geometry", "mesh")
                    for prop in ("max", "marks", "value")],
        Input("config", "data"), State("geometry-copies", "value"), State("mesh-copies", "value"))
    def blade_counts(state, geometry_count, mesh_count):
        if not state:
            return (no_update,) * 6
        count = state["config"]["meridional_channel"]["num_blades"]
        return tuple(value for current in (geometry_count, mesh_count)
                     for value in (count, {1: "1", count: str(count)}, min(count, max(1, current or 1))))

    @app.callback(Output("geometry-data", "data"), Output("geometry-key", "data"),
        Output("design-section", "figure"), Output("channel", "figure"),
        Output("design-basis", "children"), Input("config", "data"), Input("nozzle-ref", "data"),
        State("geometry-key", "data"))
    def geometry(state, nozzle_ref, previous):
        if not state or not flow.current(nozzle_ref, state["nozzle"]):
            return None, None, empty("Reference section"), empty("Meridional channel"), "Compute current MoC first."
        key = state["geometry"]
        if key == previous:
            return (no_update,) * 5
        try:
            design = flow.get_design(state["config"], nozzle_ref)
            data = preview_geometry(design)
            data["key"] = key
            # Span options should not prevent geometry from being viewed.
            defaults = dict(state["config"]["mesh"]["numerics"], span_layers=8, span_stretch=0)
            channel, _ = plots.meridional_figure(design, defaults, geometry_only=True)
            note = (f"One conformal template · Rref = {design.channel.reference_radius:.4g} mm · "
                    f"reference passage shift = {design.curves['shift_mm']:.4g} mm. "
                    "Blade-to-blade geometry: θ–m′; no mesh cells in this view.")
            return data, key, plots.section_figure(design), channel, note
        except Exception as exc:
            return None, None, empty("Reference section"), empty("Meridional channel"), str(exc)

    @app.callback(Output("reference-plot", "figure"), Output("meridional-plot", "figure"),
        Output("reference-basis", "children"), Output("meridional-basis", "children"),
        Input("config", "data"), Input("geometry-key", "data"), Input("reference-ref", "data"),
        State("nozzle-ref", "data"), State("reference-plot", "figure"))
    def mesh_setup(state, geometry_key, reference, nozzle_ref, previous_figure=None):
        if not state or geometry_key != state["geometry"]:
            return empty("2D reference mesh"), empty("Meridional grid"), "", "Build valid geometry first."
        try:
            design = flow.get_design(state["config"], nozzle_ref)
            if not flow.current(reference, state["reference"]):
                note = "Generate an up-to-date 2D mesh to display both mesh views."
                return empty("2D mesh required"), empty("Meridional grid · 2D mesh required"), note, note
            width = state["config"]["mesh"]["visualization"]["section_edge_width"]
            numerics = state["config"]["mesh"]["numerics"]
            render_key = flow.digest([reference["session"], reference["job"], state["reference"],
                                     numerics["span_layers"], numerics["span_stretch"]])
            # Volume display settings do not affect either 2D plot. Avoid
            # replacing WebGL plots (and their contexts) on those edits.
            previous_meta = (previous_figure or {}).get("layout", {}).get("meta") or {}
            if previous_meta.get("render_key") == render_key:
                if previous_figure["data"][0]["line"]["width"] != width:
                    # Update strokes without resending points or rebuilding the
                    # Plotly WebGL scenes. The blade envelope keeps its outline.
                    section_patch, grid_patch = Patch(), Patch()
                    section_patch["data"][0]["line"]["width"] = width
                    for index in range(1, numerics["span_layers"] + 3):
                        grid_patch["data"][index]["line"]["width"] = width
                    return section_patch, grid_patch, no_update, no_update
                return (no_update,) * 4
            section, _ = reference_artifact(reference["session"], reference["job"])
            grid, details = plots.meridional_figure(design, state["config"]["mesh"]["numerics"],
                                                    section, edge_width=width)
            note = (f"Basis: {details['basis']} × {details['span_layers']} span layers. "
                f"{details['streamwise_stations']} streamwise stations; span spacing "
                f"{details['min_span_spacing_mm']:.3g}–{details['max_span_spacing_mm']:.3g} mm. "
                f"Estimated mapped first height {details['estimated_first_height_mm'][0]:.3g}–"
                f"{details['estimated_first_height_mm'][1]:.3g} mm. "
                "Grid projects the future volume sweep; pseudo-streamlines are geometric, not flow solutions.")
            section_plot = plots.section_mesh_figure(section, design, width)
            section_plot.update_layout(meta={"render_key": render_key})
            return (section_plot,
                    grid, "Current reference mesh, including actual inflation cells.", note)
        except Exception as exc:
            return empty("2D reference mesh"), empty("Meridional grid"), "", str(exc)

    @app.callback(Output("volume-data", "data"), Output("report", "children"),
        Input("volume-ref", "data"))
    def mesh_view(reference):
        if not reference:
            return None, "Generate a 3D mesh to see its report."
        surfaces, summary, report = volume_artifact(reference["session"], reference["job"])
        data = {"key": reference["job"], "surfaces": surfaces,
                "num_blades": summary["geometry"]["num_blades"]}
        return data, report_panel(summary, report)

    # Submission and polling have separate callbacks. A frequent interval must
    # never supersede a slower submission response and lose the worker handle.
    @app.callback(Output("job", "data"), Output("job-error", "children"),
        *[Input(n, "n_clicks") for n in ("nozzle", "reference", "volume", "export", "cancel")],
        State("config", "data"), State("session", "data"), State("job", "data"),
        State("job-completion", "data"),
        *[State(n + "-ref", "data") for n in ("nozzle", "reference", "volume")],
        running=[(Output("launch-busy", "data"), True, False)],
        prevent_initial_call=True)
    def launch(*args):
        state, session, handle, completed, nozzle, reference, volume = args[5:]
        trigger = ctx.triggered_id
        try:
            if trigger == "cancel":
                if handle and completed != handle["job"]:
                    flow.cancel_job(handle)
                return no_update, ""
            if handle and completed != handle["job"]:
                return no_update, "A numerical job is already running."
            if not state:
                raise ValueError("Fix invalid inputs first.")
            handle = flow.start_job(session, trigger, state["config"],
                {"nozzle": nozzle, "reference": reference, "volume": volume})
            return handle, ""
        except Exception as exc:
            return no_update, str(exc)

    @app.callback(Output("nozzle-ref", "data"), Output("reference-ref", "data"),
        Output("volume-ref", "data"), Output("export-ref", "data"),
        Output("job-completion", "data"), Output("job-status", "children"),
        Output("job-progress", "value"), Output("activity", "children"), Output("spinner", "className"),
        Output("workspace", "value"),
        Input("poll", "n_intervals"), Input("job", "data"), State("job-completion", "data"),
        prevent_initial_call=True)
    def poll(interval, handle, completed):
        if not handle or completed == handle["job"]:
            return (no_update,) * 10
        unchanged = [no_update] * 5
        destination = no_update
        try:
            status, log = flow.poll_job(handle)
            elapsed = time.time() - status["started"]
            message = f"{status['message']} · {elapsed:.1f} s"
            if status["state"] != "running":
                unchanged[4] = handle["job"]
                if status["state"] == "complete":
                    result = json.loads((flow.directory(handle) / "result.json").read_text(encoding="utf-8"))
                    index = {"nozzle": 0, "reference": 1, "volume": 2, "export": 3}[handle["action"]]
                    unchanged[index] = result["reference"]
                    # Navigate only after a successful mesh is published.
                    # Failure/cancellation and routine polls leave the user's
                    # selected workspace alone.
                    destination = {"reference": "setup", "volume": "volume"}.get(
                        handle["action"], no_update)
            return (*unchanged, message, status["fraction"], log,
                    "spinner" if status["state"] == "running" else "", destination)
        except (OSError, ValueError) as exc:
            # A transient file read should not discard the job reference or
            # publish a partial artifact. The next interval can retry it.
            return (*unchanged, f"Waiting for worker status: {exc}", None, no_update, "spinner", no_update)

    @app.callback(*[Output(n, "disabled") for n in ("nozzle", "reference", "volume", "export", "cancel")],
        Output("volume-status", "children"),
        *[Output(n, "title") for n in ("reference", "volume", "export")],
        Input("config", "data"), Input("nozzle-ref", "data"),
        Input("reference-ref", "data"), Input("volume-ref", "data"), Input("job", "data"),
        Input("job-completion", "data"), Input("launch-busy", "data"))
    def readiness(state, nozzle, reference, volume, handle, completed, launching):
        busy = bool(launching or (handle and completed != handle["job"]))
        ready_nozzle = bool(state and flow.current(nozzle, state["nozzle"]))
        ready_reference = bool(state and flow.current(reference, state["reference"]))
        ready_volume = bool(state and flow.current(volume, state["volume"]))
        text = "Previous mesh and report shown; generate a mesh for the edited settings." if volume and not ready_volume else ""
        titles = ["" if ready else requirement for ready, requirement in (
            (ready_nozzle, "Compute MoC for the current inputs first."),
            (ready_reference, "Generate an up-to-date 2D mesh first."),
            (ready_volume, "Generate an up-to-date 3D mesh first."))]
        return (busy or not state, busy or not ready_nozzle, busy or not ready_reference,
                busy or not ready_volume, not busy, text, *titles)

    @app.callback(Output("downloads", "children"), Input("nozzle-ref", "data"),
        Input("reference-ref", "data"), Input("volume-ref", "data"), Input("export-ref", "data"),
        Input("job", "data"))
    def downloads(*references):
        links = []
        seen = set()
        for reference in references:
            if not reference or reference["job"] in seen:
                continue
            seen.add(reference["job"])
            folder = flow.directory(reference)
            for path in sorted(folder.iterdir()):
                if path.suffix in (".msh", ".cgns", ".yaml", ".log", ".json") and path.name not in (
                        "browser_surfaces.json", "reference_definition.json", "request.json", "result.json", "status.json"):
                    links.append(html.A(f"{reference.get('action', 'artifact')} · {path.name}",
                        href=f"/artifacts/{reference['session']}/{reference['job']}/{path.name}"))
        return links

    # Display copies and widths are browser operations; no worker is launched.
    app.clientside_callback(
        "function(g,c,s){return dash_clientside.studio.render(g,c,s,[],1,'geometry');}",
        Output("geometry-scene", "children"), Output("geometry-note", "children"),
        Input("geometry-data", "data"), Input("geometry-copies", "value"),
        Input("geometry-surfaces", "value"))
    app.clientside_callback(
        "function(v,c,s,e,w){return dash_clientside.studio.render(v,c,s,e,w,'mesh');}",
        Output("mesh-scene", "children"), Output("mesh-note", "children"),
        Input("volume-data", "data"), Input("mesh-copies", "value"),
        Input("mesh-surfaces", "value"), Input("mesh-edges", "value"),
        Input({"type": "parameter", "path": "mesh.visualization.volume_edge_width"}, "value"))
    return app



