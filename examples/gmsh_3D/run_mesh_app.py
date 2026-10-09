"""Local Dash editor for MoC geometry and meshes; run this file and open port 8051."""

import hashlib
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")  # Browser app: background jobs never open desktop figures.
import barotropy
import diskcache
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import (
    ALL,
    MATCH,
    Dash,
    DiskcacheManager,
    Input,
    Output,
    State,
    ctx,
    dcc,
    html,
    no_update,
)
from flask import abort, send_from_directory

try:
    import dash_vtk  # Register the component assets before serving the layout.
except ImportError as exc:
    raise ImportError(
        'Install browser visualization: pip install ".[mesh-viz]"'
    ) from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import examples.gmsh.run_mesh_generation as nozzle_workflow
import examples.gmsh_3D.geometry_source as geo
import examples.gmsh_3D.meshing_source as msh
import examples.gmsh_3D.run_mesh_generation as workflow
import examples.workflow_logging as log

# =============================================================================
# 1. Local app controls; design and visualization parameters come from the YAML
# =============================================================================

EXAMPLE_DIR = Path(__file__).resolve().parent
APP_OUTPUT = EXAMPLE_DIR / "output" / "dash"
SHARED_NOZZLE_CACHE = EXAMPLE_DIR / "output" / "nozzle_solution.json"
DEFAULT_CASE = "axial_stator"
HOST = "127.0.0.1"
PORT = 8051
DEBUG = False  # Avoid spawning two JAX/Gmsh runtimes with the development reloader.
CASES = {
    name: EXAMPLE_DIR / f"{name}.yaml" for name in ("axial_stator", "radial_stator")
}
# JavaScript represents 1.0 and 1 with the same number type. Use parameter
# semantics rather than the round-tripped YAML value to identify integers.
INTEGER_KEYS = {
    "n", "n_cp", "num_baseline_points", "num_curve_points", "num_points",
    "num_blades", "verbosity", "curvature_elements", "algorithm",
    "recombination_algorithm", "smoothing_steps", "span_layers", "dpi",
}
logger = log.get_logger("gmsh_3D.app")


# =============================================================================
# 2. Configuration editing and session files (no saved blade-coordinate cache)
# =============================================================================


def configuration_key(config, *, mesh=False):
    """Identify the physical design; view changes do not invalidate a mesh."""
    relevant = {
        name: config[name] for name in ("nozzle", "blade", "passage", "machine")
    }
    if mesh:
        relevant["numerics"] = config["mesh"]["numerics"]
        relevant["cgns"] = config["mesh"]["cgns"]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def config_state(config):
    return {
        "config": config,
        "geometry_key": configuration_key(config),
        "mesh_key": configuration_key(config, mesh=True),
    }


def session_directory(session, job=None):
    """Resolve generated UUIDs under the app output directory."""
    identifiers = [session] if job is None else [session, job]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value)
        for value in identifiers
    ):
        raise ValueError("Invalid session/job identifier.")
    return APP_OUTPUT.joinpath(*identifiers)


def load_nozzle(reference):
    """Resolve a browser reference to a validated MoC solution on the server."""
    if not reference:
        raise ValueError("Click Compute MoC before creating geometry or a mesh.")
    path = (
        SHARED_NOZZLE_CACHE
        if reference.get("shared")
        else session_directory(reference["session"], reference["job"])
        / "nozzle_solution.json"
    )
    result = nozzle_workflow.load_or_solve_nozzle(
        reference["parameters"], path, cache_only=True
    )
    if result is None:
        raise ValueError("The MoC solution is missing or outdated. Click Compute MoC.")
    return result


def initial_nozzle(config, session):
    """Seed the UI from a matching cache without ever solving during startup."""
    result = nozzle_workflow.load_or_solve_nozzle(
        config["nozzle"], SHARED_NOZZLE_CACHE, cache_only=True
    )
    if result is None:
        return None
    return {
        "session": session,
        "job": "shared",
        "shared": True,
        "parameters": config["nozzle"],
    }


def flatten(mapping, prefix=""):
    """Yield each YAML leaf and its dotted path."""
    for name, value in mapping.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from flatten(value, path)
        else:
            yield path, value


def edited_configuration(base, values, ids):
    """Apply typed UI values while retaining the YAML's dictionary structure."""
    config = deepcopy(base)
    defaults = dict(flatten(base))
    for value, identifier in zip(values, ids):
        path = identifier["path"]
        if (
            path not in defaults
        ):  # Old controls can briefly remain during a case switch.
            continue
        default = defaults[path]
        if value is not None and path.split(".")[-1] in INTEGER_KEYS:
            if float(value) != int(value):
                raise ValueError(f"{path} requires an integer.")
            value = int(value)
        target = config
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return config


# =============================================================================
# 3. Collapsible YAML controls and the Plotly conformal-plane figure
# =============================================================================


def slider_range(path, value):
    """Give geometric controls useful ranges, retaining the exact number input."""
    key = path.split(".")[-1]
    fixed = {
        "metal_angle_in": (-30, 60, 0.1),
        "metal_angle_out": (0, 85, 0.1),
        "r_trailing": (0.01, 3, 0.01),
        "inlet_opening_ratio": (1, 3, 0.01),
        "num_blades": (2, 100, 1),
        "n_cp": (6, 80, 1),
        "num_baseline_points": (50, 1000, 1),
        "num_curve_points": (50, 1000, 1),
        "num_points": (20, 240, 1),
        "domain_shift": (-0.5, 0.5, 0.001),
        "inlet_chords": (0.1, 4, 0.01),
        "outlet_chords": (0.1, 4, 0.01),
    }
    if key in fixed:
        lower, upper, step = fixed[key]
    elif path.startswith("passage.smooth."):
        lower, upper, step = 0.01, 1.0, 0.01
    elif path.startswith("machine."):
        lower, upper, step = 0.5, max(100, 2 * value), 0.1
    else:
        return None
    return min(lower, value), max(upper, value), step


def parameter_control(path, value):
    label = path.split(".")[-1].replace("_", " ").capitalize()
    bounds = slider_range(path, value) if type(value) in (int, float) else None
    if bounds:
        lower, upper, step = bounds
        control = html.Div(
            [
                dcc.Slider(
                    id={"type": "linked-slider", "path": path},
                    min=lower,
                    max=upper,
                    step=step,
                    value=value,
                    marks=None,
                    updatemode="mouseup",
                    tooltip={"placement": "bottom"},
                ),
                dcc.Input(
                    id={"type": "linked-value", "path": path},
                    type="number",
                    value=value,
                    step=step,
                    debounce=True,
                ),
            ],
            className="slider-input",
        )
    else:
        identifier = {"type": "parameter", "path": path}
        choices = {
            "nozzle.solver": ["conventional", "mln"],
            "passage.domain_shift_mode": ["auto", "manual"],
            "workflow.log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "mesh.numerics.algorithm": [1, 2, 5, 6, 8, 9],
            "mesh.numerics.recombination_algorithm": [0, 1, 2, 3],
            "machine.type": ["axial", "radial"],
        }
        if type(value) is bool:
            control = dcc.Dropdown(
                id=identifier,
                value=value,
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                clearable=False,
            )
        elif path in choices:
            control = dcc.Dropdown(
                id=identifier,
                value=value,
                options=choices[path],
                clearable=False,
                disabled=path == "machine.type",
            )
        else:
            control = dcc.Input(
                id=identifier,
                value=value,
                type="number" if type(value) in (int, float) else "text",
                step=1 if type(value) is int else "any",
                debounce=True,
            )
    return html.Div([html.Label(label, title=path), control], className="parameter")


def configuration_controls(config, prefix=""):
    groups = []
    for name, value in config.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            groups.append(
                html.Details(
                    [
                        html.Summary(name.replace("_", " ").capitalize()),
                        html.Div(
                            configuration_controls(value, path),
                            className="group-content",
                        ),
                    ],
                    open=not prefix and name in ("machine", "blade"),
                    className="parameter-group",
                )
            )
        else:
            groups.append(parameter_control(path, value))
    return groups


def section_figure(data):
    """Plot the fitted blade and the construction/domain curves in theta, -m'."""
    fig = go.Figure()
    if data:
        section = data["section"]
        left, right = np.asarray(section["periodic_left"]), np.asarray(
            section["periodic_right"]
        )
        outer = np.vstack((left, right[::-1], left[:1]))
        fig.add_trace(
            go.Scatter(
                x=outer[:, 0],
                y=outer[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="#f3eee5",
                line={"color": "black", "width": 1},
                name="Flow domain",
            )
        )
        blade = np.asarray(section["blade"])
        fig.add_trace(
            go.Scatter(
                x=blade[:, 0],
                y=blade[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="white",
                line={"color": "black", "width": 2},
                name="Blade",
            )
        )
        for name, color, dash in (
            ("passage_reference", "#a35420", "dash"),
            ("inlet_extension", "#357d83", "solid"),
            ("outlet_extension", "#357d83", "solid"),
        ):
            points = np.asarray(section[name])
            fig.add_trace(
                go.Scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    mode="lines",
                    line={"color": color, "dash": dash, "width": 1.5},
                    name=name.replace("_", " ").capitalize(),
                )
            )
    else:
        fig.add_annotation(
            text="Compute MoC to preview the blade and passage",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    fig.update_layout(
        template="simple_white",
        font={"family": "Times New Roman", "size": 15},
        margin={"l": 65, "r": 30, "t": 30, "b": 65},
        uirevision="conformal",
        xaxis_title="θ (rad, clockwise)",
        yaxis_title="−m′",
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_xaxes(showline=True, mirror=True, zeroline=False)
    fig.update_yaxes(
        showline=True, mirror=True, zeroline=False, scaleanchor="x", scaleratio=1
    )
    return fig


# =============================================================================
# 4. Background jobs: reuse the existing MoC cache and the common mesh workflow
# =============================================================================


def run_job(action, state, reference, session, progress):
    """Compute one immutable input snapshot; each mesh has an isolated directory."""
    config = workflow.validate_configuration(deepcopy(state["config"]))
    job = uuid4().hex
    directory = session_directory(session, job)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "configuration.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    start = perf_counter()
    with log.workflow_logging(
        directory / "workflow.log", config["workflow"]["log_level"]
    ):
        if action == "compute-moc":
            progress(("Load or solve the MoC nozzle…",))
            cache_file = directory / "nozzle_solution.json"
            # Preserve source/version validation when reusing a previous solution.
            candidate = SHARED_NOZZLE_CACHE
            if reference and not reference.get("shared"):
                candidate = (
                    session_directory(reference["session"], reference["job"])
                    / "nozzle_solution.json"
                )
            if candidate.is_file():
                shutil.copyfile(candidate, cache_file)
            nozzle_workflow.load_or_solve_nozzle(config["nozzle"], cache_file)
            result = {
                "session": session,
                "job": job,
                "shared": False,
                "parameters": config["nozzle"],
            }
            return (
                result,
                None,
                f"MoC ready ({perf_counter()-start:.1f} s). Geometry controls are active.",
            )

        if not reference or reference["parameters"] != config["nozzle"]:
            raise ValueError(
                "Nozzle inputs changed or no solution exists. Click Compute MoC first."
            )
        progress(("Generate the mesh and export CGNS…",))
        nozzle = load_nozzle(reference)
        geometry = geo.geometry_from_machine(config["machine"])
        with plt.rc_context():
            barotropy.set_plot_options()
            result = msh.create_mesh(
                nozzle,
                config,
                directory,
                geometry=geometry,
                save_figures=config["workflow"]["save_figures"],
                show_gmsh=False,
            )
        mesh = {
            "session": session,
            "job": job,
            "mesh_key": state["mesh_key"],
            "summary": result,
            "cgns_filename": (
                config["mesh"]["cgns"]["filename"]
                if config["mesh"]["cgns"]["enabled"]
                else None
            ),
        }
        return (
            no_update,
            mesh,
            f"Mesh ready: {sum(result['cells'].values()):,} cells ({perf_counter()-start:.1f} s).",
        )


# =============================================================================
# 5. Dash layout and callbacks (VTK rotation/filtering happens in the browser)
# =============================================================================


def create_app():
    APP_OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(APP_OUTPUT / "callback_cache"))
    app = Dash(
        __name__,
        assets_folder=str(EXAMPLE_DIR / "app_assets"),
        background_callback_manager=DiskcacheManager(cache),
        suppress_callback_exceptions=True,
    )
    app.title = "MoC blade geometry and mesh"

    def layout():
        config = workflow.read_configuration(CASES[DEFAULT_CASE])
        session = uuid4().hex
        reference = initial_nozzle(config, session)
        return html.Div(
            [
                dcc.Store(id="session", data=session),
                dcc.Store(id="base-config", data=config),
                dcc.Store(id="config-state", data=config_state(config)),
                dcc.Store(id="nozzle-reference", data=reference),
                dcc.Store(id="mesh-reference"),
                dcc.Download(id="yaml-download"),
                html.Aside(
                    [
                        html.H1("MoC blade studio"),
                        html.P(
                            "Design geometry, then create and inspect its mesh.",
                            className="muted",
                        ),
                        dcc.Dropdown(
                            id="case",
                            value=DEFAULT_CASE,
                            clearable=False,
                            options=[
                                {"label": "Axial turbine", "value": "axial_stator"},
                                {"label": "Radial turbine", "value": "radial_stator"},
                            ],
                        ),
                        html.P(
                            "Machine lengths: mm · Nozzle inputs: SI", className="muted"
                        ),
                        html.Div(configuration_controls(config), id="controls"),
                        html.P(
                            "Native-window switches apply to the script; this app embeds its views.",
                            className="muted",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Compute MoC",
                                    id="compute-moc",
                                    n_clicks=0,
                                    className="primary",
                                ),
                                html.Button(
                                    "Create mesh", id="create-mesh", n_clicks=0
                                ),
                                html.Button(
                                    "Download YAML",
                                    id="download-yaml",
                                    n_clicks=0,
                                    className="quiet",
                                ),
                            ],
                            className="actions",
                        ),
                    ],
                    className="sidebar",
                ),
                html.Main(
                    [
                        html.Div(
                            [
                                html.H2("Geometry and mesh"),
                                html.Div(
                                    id="job-status",
                                    children=(
                                        "Cached MoC loaded."
                                        if reference
                                        else "Click Compute MoC to begin."
                                    ),
                                ),
                            ],
                            className="topbar",
                        ),
                        html.Div(id="job-progress", className="muted"),
                        html.Div(id="configuration-error", className="error"),
                        html.Div(id="geometry-status", className="status"),
                        dcc.Tabs(
                            id="view-tab",
                            value="conformal",
                            children=[
                                dcc.Tab(label="m′–θ geometry", value="conformal"),
                                dcc.Tab(label="3D geometry", value="geometry"),
                                dcc.Tab(label="3D mesh", value="mesh"),
                            ],
                        ),
                        html.Div(
                            [
                                html.Label("Displayed blades"),
                                dcc.Slider(
                                    id="display-count",
                                    min=1,
                                    max=config["machine"]["num_blades"],
                                    step=1,
                                    value=min(3, config["machine"]["num_blades"]),
                                    marks=None,
                                    tooltip={"always_visible": True},
                                ),
                                html.Span(
                                    "Select surfaces under Mesh › PyVista › Surfaces.",
                                    className="muted",
                                ),
                            ],
                            className="view-controls",
                        ),
                        dcc.Loading(
                            html.Div(
                                [
                                    dcc.Store(id="geometry-data"),
                                    dcc.Store(id="mesh-data"),
                                    html.Div(
                                        dcc.Graph(
                                            id="section-plot",
                                            figure=section_figure(None),
                                            responsive=True,
                                            config={"displaylogo": False},
                                        ),
                                        id="section-panel",
                                    ),
                                    html.Div(
                                        id="vtk-panel",
                                        style={"display": "none", "height": "64vh"},
                                    ),
                                ],
                                className="view-panel",
                            ),
                            type="circle",
                        ),
                        html.Div(id="view-note", className="status"),
                        html.Div(id="mesh-downloads", className="downloads"),
                    ],
                    className="main-panel",
                ),
            ],
            className="app-shell",
        )

    app.layout = layout

    @app.callback(
        Output("controls", "children"),
        Output("base-config", "data"),
        Input("case", "value"),
        prevent_initial_call=True,
    )
    def change_case(case):
        config = workflow.read_configuration(CASES[case])
        return configuration_controls(config), config

    @app.callback(
        Output({"type": "linked-slider", "path": MATCH}, "value"),
        Output({"type": "linked-value", "path": MATCH}, "value"),
        Input({"type": "linked-slider", "path": MATCH}, "value"),
        Input({"type": "linked-value", "path": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def sync_slider(slider, number):
        return (
            (no_update, slider)
            if ctx.triggered_id["type"] == "linked-slider"
            else (number, no_update)
        )

    @app.callback(
        Output("config-state", "data"),
        Output("configuration-error", "children"),
        Input({"type": "parameter", "path": ALL}, "value"),
        Input({"type": "linked-value", "path": ALL}, "value"),
        Input("base-config", "data"),
        State({"type": "parameter", "path": ALL}, "id"),
        State({"type": "linked-value", "path": ALL}, "id"),
    )
    def collect_configuration(values, linked_values, base, ids, linked_ids):
        try:
            config = edited_configuration(
                base, values + linked_values, ids + linked_ids
            )
            return config_state(config), ""
        except (ValueError, TypeError) as exc:
            # An incomplete input must not silently launch a job with older values.
            return {"config": None, "geometry_key": None, "mesh_key": None}, str(exc)

    @app.callback(
        Output("geometry-data", "data"),
        Output("geometry-status", "children"),
        Input("config-state", "data"),
        Input("nozzle-reference", "data"),
        State("geometry-data", "data"),
    )
    def update_geometry(state, reference, previous):
        try:
            if not state or state["config"] is None:
                return None, "Complete the configuration inputs."
            config = state["config"]
            if not reference or reference["parameters"] != config["nozzle"]:
                return None, "Nozzle inputs need a MoC solution. Click Compute MoC."
            if (
                previous
                and previous["geometry_key"] == state["geometry_key"]
                and previous["solution_job"] == reference["job"]
            ):
                return no_update, "Geometry is current."
            workflow.validate_configuration(config)
            start = perf_counter()
            data = geo.preview_geometry(
                load_nozzle(reference), config, span_sections=5, curve_points=140
            )
            data.update(
                geometry_key=state["geometry_key"], solution_job=reference["job"]
            )
            return (
                data,
                f"Geometry updated in {perf_counter()-start:.1f} s · Display surfaces only; no volume mesh.",
            )
        except Exception as exc:
            logger.exception("Geometry preview failed")
            return None, f"Geometry: {exc}"

    @app.callback(Output("section-plot", "figure"), Input("geometry-data", "data"))
    def plot_section(data):
        return section_figure(data)

    @app.callback(
        Output("display-count", "max"),
        Output("display-count", "value"),
        Input("config-state", "data"),
        State("display-count", "value"),
    )
    def clamp_blade_count(state, count):
        maximum = (
            state["config"]["machine"]["num_blades"] if state and state["config"] else 1
        )
        if type(maximum) is not int or maximum < 2:
            maximum = 1
        return maximum, min(max(1, count or 1), maximum)

    @app.callback(
        Output("nozzle-reference", "data"),
        Output("mesh-reference", "data"),
        Output("job-status", "children"),
        Input("compute-moc", "n_clicks"),
        Input("create-mesh", "n_clicks"),
        State("config-state", "data"),
        State("nozzle-reference", "data"),
        State("session", "data"),
        background=True,
        progress=[Output("job-progress", "children")],
        progress_default=("",),
        running=[
            (Output("compute-moc", "disabled"), True, False),
            (Output("create-mesh", "disabled"), True, False),
        ],
        prevent_initial_call=True,
    )
    def compute(progress, moc_clicks, mesh_clicks, state, reference, session):
        try:
            if not state or state["config"] is None:
                raise ValueError("Complete the configuration inputs first.")
            return run_job(ctx.triggered_id, state, reference, session, progress)
        except Exception as exc:
            logger.exception("Background workflow failed")
            return no_update, no_update, f"Could not complete the operation: {exc}"

    @app.callback(
        Output("mesh-data", "data"),
        Output("mesh-downloads", "children"),
        Input("mesh-reference", "data"),
    )
    def load_mesh(reference):
        if not reference:
            return None, []
        directory = session_directory(reference["session"], reference["job"])
        faces = msh.load_mesh_surfaces(directory / "stator_mesh_3d.msh")
        summary = reference["summary"]
        data = {
            "surfaces": {
                name: {"points": points.ravel().tolist(), "polys": polygons.tolist()}
                for name, (points, polygons) in faces.items()
            },
            "num_blades": summary["geometry"]["num_blades"],
            "mesh_key": reference["mesh_key"],
            "cells": sum(summary["cells"].values()),
            "min_sicn": summary["quality"]["min_sicn"],
        }
        files = [
            "stator_mesh_3d.msh",
            "configuration.yaml",
            "mesh_summary.json",
            "workflow.log",
        ]
        if reference["cgns_filename"]:
            files.insert(1, reference["cgns_filename"])
        links = [
            html.A(
                name,
                href=f"/mesh-download/{reference['session']}/{reference['job']}/{name}",
            )
            for name in files
        ]
        return data, links

    app.clientside_callback(
        """function(geometry, mesh, state, tab, count) {
        return window.dash_clientside.mesh_app.renderScene(geometry, mesh, state, tab, count);
    }""",
        Output("vtk-panel", "children"),
        Output("section-panel", "style"),
        Output("vtk-panel", "style"),
        Output("view-note", "children"),
        Input("geometry-data", "data"),
        Input("mesh-data", "data"),
        Input("config-state", "data"),
        Input("view-tab", "value"),
        Input("display-count", "value"),
    )

    @app.callback(
        Output("yaml-download", "data"),
        Input("download-yaml", "n_clicks"),
        State("config-state", "data"),
        State("case", "value"),
        prevent_initial_call=True,
    )
    def download_yaml(clicks, state, case):
        if not state or not state["config"]:
            return no_update
        return dcc.send_string(
            yaml.safe_dump(state["config"], sort_keys=False), f"{case}.yaml"
        )

    @app.server.get("/mesh-download/<session>/<job>/<filename>")
    def download_file(session, job, filename):
        try:
            directory = session_directory(session, job)
            allowed = {
                "stator_mesh_3d.msh",
                "configuration.yaml",
                "mesh_summary.json",
                "workflow.log",
            }
            config = yaml.safe_load((directory / "configuration.yaml").read_text())
            allowed.add(config["mesh"]["cgns"]["filename"])
            if filename not in allowed:
                abort(404)
            return send_from_directory(directory, filename, as_attachment=True)
        except (ValueError, OSError):
            abort(404)

    return app


app = create_app()
server = app.server


def main():
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
