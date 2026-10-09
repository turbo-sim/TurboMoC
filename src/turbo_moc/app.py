"""
turbo_moc.app -- interactive Dash app for MOC nozzle design.

Structured after turbodash's app.py: one Dash app, a compute callback that
calls a single engine function (here, turbo_moc.design_nozzle) and stores its
result in a dcc.Store, and a second callback that fans that result out into
plots. The one deliberate departure from turbodash's pattern: the compute
callback runs in the background (background=True + DiskcacheManager) with
a live progress bar, since a solve here takes 5-30s+ (turbodash's own
algebraic core is sub-second, so it never needed this) -- an explicit
"Compute" button, not live recompute-on-slider-drag, is the intended UX.

Plots are displayed with turbo_moc.plotly (interactive dcc.Graph -- zoom, pan,
hover, and a native download-as-PNG toolbar button, same as turbodash's own
app.py). turbo_moc.plotly's figures are styled (see
``plotting_plotly._apply_journal_style``) to read close to turbo_moc.mpl's
boxed-axes/gridline look rather
than Plotly's own default template, since visual consistency with
examples/plot_design.py's output mattered too -- same colors either way,
only the interactive vs. static trade-off differs. The "Download this
plot" button + format dropdown (PNG/SVG/PDF) is separate from Plotly's own
toolbar button (PNG only): it re-renders the SAME figure with turbo_moc.mpl on
click, so an SVG/PDF download is a real vector file, not a rasterized PNG
relabeled -- display and file-export are genuinely two different renders
of the same data, not one converted into the other.

Run locally::

    conda activate turbo_moc_env
    python -m turbo_moc.app
    (or: turbo_moc-app, the console-script entry point from pyproject.toml)

then open the printed http://127.0.0.1:8050/ URL in a browser.
"""

import base64
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import diskcache
import dash
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import jaxprop as jxp

try:
    # Must be imported at module load (before the Dash server starts), not
    # lazily inside a callback: Dash only registers a component library's
    # JS/CSS assets into the page if the module has been imported by the
    # time the layout is first served. An import deferred to callback time
    # (the pattern used for cadquery, which needs no such registration)
    # left the browser with no idea how to render a dash_vtk.View at all --
    # "Callback error updating sizing-vtk-container.children" with no
    # Python-side traceback, since the failure is purely on the frontend.
    import dash_vtk
    from dash_vtk.utils import to_mesh_state
    DASH_VTK_AVAILABLE = True
except ImportError:
    DASH_VTK_AVAILABLE = False

import turbo_moc
from turbo_moc import design_nozzle, list_supported_fluids, make_progress_reporter, NozzleDesignError
from turbo_moc.core.classes import build_convergent_inlet
from turbo_moc.coupling import derive_rotor_inlet_from_stator
from turbo_moc.geometry import (
    move_control_point_along_normal,
    parametrize_stator_blade,
    parametrize_stator_blade_semi,
    size_rotor_from_pitch,
    size_stator_mode_a,
    size_stator_mode_b,
)
from turbo_moc.rotor import design_rotor_vortex_blade

BLADE_PARAM_FUNCS = {
    "full": parametrize_stator_blade,
    "semi": parametrize_stator_blade_semi,
}

# Tab key -> label, and the two per-tab plot functions: the interactive
# Plotly one (display) and the matplotlib one + filename base (download).
# Adding a new plot tab only means adding one entry to each of these three.
TAB_LABELS = {
    "mesh": "Nozzle contour",
    "axis": "Axis properties", "wall": "Wall properties", "ts": "T-s diagram",
}
PLOT_FUNCS_PLOTLY = {
    "mesh": turbo_moc.plotly.plot_characteristics_mesh,
    "axis": turbo_moc.plotly.plot_axis_properties,
    "wall": turbo_moc.plotly.plot_wall_properties,
    "ts": turbo_moc.plotly.plot_ts_diagram,
}
PLOT_FUNCS_MPL = {
    "mesh": (turbo_moc.mpl.plot_characteristics_mesh, "nozzle_contour_mesh"),
    "axis": (turbo_moc.mpl.plot_axis_properties, "axis_properties"),
    "wall": (turbo_moc.mpl.plot_wall_properties, "wall_properties"),
    "ts": (turbo_moc.mpl.plot_ts_diagram, "ts_diagram"),
}

# --- Background callback manager (see module docstring for why) ---------
cache = diskcache.Cache("./.turbo_moc_app_cache")
background_callback_manager = dash.DiskcacheManager(cache)

app = Dash(__name__, background_callback_manager=background_callback_manager)
app.title = "TurboMoC"
server = app.server  # WSGI app, for gunicorn (matches turbodash's Procfile convention)

FLUIDS = list_supported_fluids()
DEFAULT_FLUID = "Novec649" if "Novec649" in FLUIDS else FLUIDS[0]

DEFAULTS = dict(
    # P0/p_back are in bar (UI + save/load convention) -- converted to Pa
    # only at the design_nozzle() call boundary in run_design. Novec649
    # stator inlet/exit from the meanline design (Parisi, EMPOWER_DTU
    # design_turbine/meanline/Novec649.yaml) RECOMPUTED with
    # degree_reaction=0 (pure impulse) instead of the file's own 0.25 --
    # the 0.25 case leaves the rotor inlet barely SUBSONIC (M_rel=0.999),
    # unusable by this tool's supersonic vortex-flow rotor method; zero
    # reaction pushes the whole enthalpy drop into the stator, giving
    # M_rel=1.14-1.19 (confirmed both in turbodash's own recompute and
    # through this tool's full nozzle+blade+coupling pipeline). P0/Q0 are
    # unchanged by degree_reaction (784351.18 Pa, 0.15); p_back is this
    # recompute's own stator EXIT static pressure (flow_stations[2].p =
    # 130884.61 Pa, alpha=70 deg) -- NOT the turbine/condenser exit
    # pressure. Confirmed converging and reproducing the recompute's own
    # stator-exit Mach (1.726) to 4 significant figures.
    P0=7.8435, T0=397.6290, Q0=0.15, p_back=1.3088,
    y_t=0.01, n=15, rho_d=0.1,
)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _field(label, id, value, **kwargs):
    """Label and input side by side (not label-on-top-of-a-full-width-box
    that's mostly empty) -- the input only needs to be wide enough for a
    handful of digits. label is rendered with MathJax (dcc.Markdown(
    mathjax=True)), so labels are given as LaTeX (e.g. r"$M_{\text{in}}$")
    rather than plain text -- a proper subscripted symbol everywhere
    instead of a spelled-out variable name."""
    return html.Div(
        [
            dcc.Markdown(label, mathjax=True,
                          style={"fontSize": "13px", "fontWeight": "600",
                                  "flex": "1 1 auto", "marginRight": "8px"}),
            dcc.Input(id=id, value=value, type="number",
                       style={"width": "90px", "flex": "0 0 auto"}, **kwargs),
        ],
        style={"display": "flex", "alignItems": "center",
               "justifyContent": "space-between", "marginBottom": "8px"},
    )


def make_table(data, columns):
    """A dash_table.DataTable in turbodash's own convention: monospace,
    right-aligned cells, bold header with a light-gray fill and a
    2px bottom rule -- used for geometry/summary readouts in place of
    inline status text."""
    formatted_columns = [{"name": c, "id": c} for c in columns]
    return dash_table.DataTable(
        data=data,
        columns=formatted_columns,
        style_table={"overflowX": "auto", "marginTop": "8px", "width": "fit-content"},
        style_cell={
            "fontFamily": "monospace",
            "fontSize": "13px",
            "padding": "6px",
            "textAlign": "right",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f0f0f0",
            "borderBottom": "2px solid black",
        },
    )


def _vtk_placeholder(message):
    """Centered gray message inside the Phase 3 VTK container, standing in
    for a dash_vtk.View before a sizing result exists (or if dash-vtk/
    cadquery aren't installed) -- never leave the container visually empty
    with no explanation."""
    return html.Div(message, style={"display": "flex", "alignItems": "center",
                                      "justifyContent": "center", "height": "100%",
                                      "color": "#888", "fontSize": "13px", "padding": "16px",
                                      "textAlign": "center"})


def _dedupe_closed_loop(x, y, tol=1e-6):
    """Drop consecutive near-duplicate points from a closed 2D loop,
    INCLUDING the wraparound last-vs-first pair -- e.g. a stator's
    blade_curve + trailing_edge concatenated this way has an EXACT
    duplicate at the seam (trailing_edge's own endpoints coincide with
    blade_curve's start/end, by construction, to bridge the gap a direct
    closure would leave). A naive closed polygon built straight from that
    (implicitly closing last point back to first) gets a zero-length
    closing edge, which is a degenerate, effectively self-intersecting
    polygon -- confirmed directly: a point-in-polygon self-intersection
    check found exactly one crossing before this dedup, zero after.
    Robust polygon capping (vtkPolygon) handles a non-convex but SIMPLE
    polygon fine; it does not handle a degenerate one, which is what was
    producing the "blanket" mis-triangulated end caps."""
    pts = list(zip(x, y))
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > tol:
            cleaned.append(p)
    if np.hypot(cleaned[-1][0] - cleaned[0][0], cleaned[-1][1] - cleaned[0][1]) <= tol:
        cleaned.pop()
    return [p[0] for p in cleaned], [p[1] for p in cleaned]


def _triangulate_vtk(vtk, polydata):
    """Run polydata through vtkTriangleFilter server-side (VTK's own
    robust, compiled triangulator) before it ever reaches the browser.
    Needed for any mesh built from multi-point vtkPolygon cells (e.g.
    _extrude_polygon_vtk's end caps): dash-vtk's client-side renderer
    (vtk.js) triangulates a raw vtkPolygon cell with a naive fan from its
    first vertex, which breaks badly on non-convex polygons like these
    blade cross-sections -- confirmed visually (cap triangles cutting
    straight across the shape's concave regions, producing a "cone"
    instead of the true outline). Pre-triangulating here sends only plain
    triangles to the client, so there is no client-side ambiguity left to
    get wrong. Harmless/idempotent on meshes that are already all
    triangles (e.g. cadquery's own STL export)."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    return tri.GetOutput()


def _extrude_polygon_vtk(vtk, x, y, depth):
    """A single closed 2D polygon (x, y, already positioned in the global
    X/Y plane -- e.g. one radial-wrapped blade copy) extruded by `depth`
    along Z into a watertight vtkPolyData (bottom cap + side quads + top
    cap). Pure vtk/numpy, no cadquery -- the radial wrap has no inherent
    axial-depth dimension (see turbo_moc.geometry.radial's module
    docstring), so this is a flat, constant-depth preview/export
    extrusion, not a true spanwise-varying solid like the axial wrap's
    _build_annular_blade_solid."""
    x, y = _dedupe_closed_loop(x, y)
    n = len(x)
    points = vtk.vtkPoints()
    for xi, yi in zip(x, y):
        points.InsertNextPoint(xi, yi, 0.0)
    for xi, yi in zip(x, y):
        points.InsertNextPoint(xi, yi, depth)

    polys = vtk.vtkCellArray()
    for i in range(n):
        j = (i + 1) % n
        quad = vtk.vtkPolygon()
        quad.GetPointIds().SetNumberOfIds(4)
        quad.GetPointIds().SetId(0, i)
        quad.GetPointIds().SetId(1, j)
        quad.GetPointIds().SetId(2, j + n)
        quad.GetPointIds().SetId(3, i + n)
        polys.InsertNextCell(quad)

    bottom = vtk.vtkPolygon()
    bottom.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        bottom.GetPointIds().SetId(i, n - 1 - i)
    polys.InsertNextCell(bottom)

    top = vtk.vtkPolygon()
    top.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        top.GetPointIds().SetId(i, i + n)
    polys.InsertNextCell(top)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


controls = html.Div(
    [
        html.Div(
            [
                html.Button("Save design", id="save-nozzle-btn", n_clicks=0,
                             style={"padding": "8px 16px", "fontWeight": "600"}),
                dcc.Upload(
                    id="load-nozzle-upload",
                    accept=".json",
                    children=html.Button(
                        "Load design", style={"padding": "8px 16px", "fontWeight": "600"},
                    ),
                ),
            ],
            style={"display": "flex", "gap": "12px", "marginBottom": "12px"},
        ),
        dcc.Download(id="download-nozzle-json"),

        html.H4("Fluid"),
        dcc.Dropdown(id="fluid_name", options=FLUIDS, value=DEFAULT_FLUID, clearable=False),
        html.Br(),
        dcc.RadioItems(
            id="inlet_mode",
            options=[
                {"label": " Subcooled liquid / single-phase (P0, T0)", "value": "T0"},
                {"label": " Saturated two-phase (P0, Q0)", "value": "Q0"},
            ],
            value="Q0",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field(r"$P_0$ (bar)", "P0", DEFAULTS["P0"]),
        html.Div(_field(r"$T_0$ (K)", "T0", DEFAULTS["T0"]), id="T0_container", style={"display": "none"}),
        html.Div(_field(r"$Q_0$ (-)", "Q0", DEFAULTS["Q0"], min=0, max=1),
                 id="Q0_container"),

        html.H4("Design target"),
        _field(r"$p_{\text{back}}$ (bar)", "p_back", DEFAULTS["p_back"]),

        html.H4("Solver"),
        dcc.RadioItems(
            id="solver",
            options=[
                {"label": " Conventional (rounded arc)", "value": "conventional"},
                {"label": " Minimum-length (sharp corner)", "value": "mln"},
            ],
            value="conventional",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),

        html.H4("Sonic line"),
        dcc.RadioItems(
            id="stage1_mode",
            options=[
                {"label": " Linear", "value": "flat"},
                {"label": " Parabolic (Sauer line)", "value": "sauer"},
            ],
            value="sauer",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),

        html.H4("Geometry"),
        _field(r"$y_t$ (m)", "y_t", DEFAULTS["y_t"]),
        _field(r"$\rho_d$", "rho_d", DEFAULTS["rho_d"]),
        _field(r"$n$", "n", DEFAULTS["n"], step=1, min=5),

        html.H4("Convergent inlet (geometry only)", style={"marginTop": "16px"}),
        dcc.Checklist(
            id="conv_inlet_enable",
            options=[{"label": " Add convergent inlet", "value": "on"}],
            value=[],
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field(r"$L_{\text{conv}}$ (m)", "conv_inlet_length", 0.5 * DEFAULTS["rho_d"]),

        html.Button("Compute", id="compute-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px",
                            "fontWeight": "600"}),
        html.Progress(id="progress-bar", value="0", max="100",
                       style={"width": "100%", "marginTop": "10px"}),
        html.Div(id="progress-label", style={"fontSize": "12px", "color": "#666"}),
        html.Div(id="status-message", style={"marginTop": "8px", "fontSize": "13px"}),

        html.H4("Compare", style={"marginTop": "16px"}),
        html.Div([
            html.Button("Pin as reference", id="pin-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px", "marginRight": "4%"}),
            html.Button("Clear reference", id="clear-ref-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px"}),
        ], style={"display": "flex"}),
        html.Div(id="reference-status", style={"marginTop": "6px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

plots = html.Div(
    [
        html.Div(
            [
                html.Label("Download:", style={"fontSize": "13px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="plot-select",
                    options=[{"label": TAB_LABELS[key], "value": key} for key in TAB_LABELS],
                    value="mesh", clearable=False,
                    style={"width": "180px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                dcc.Dropdown(
                    id="save-format",
                    options=[{"label": fmt.upper(), "value": fmt} for fmt in ("png", "svg", "pdf")],
                    value="png", clearable=False,
                    style={"width": "100px", "display": "inline-block", "verticalAlign": "middle",
                           "marginLeft": "8px"},
                ),
                html.Button("Download this plot", id="download-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-plot"),
                html.Button("Download wall CSV", id="download-wall-csv-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-wall-csv"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        # All five views in one panel (not one-at-a-time tabs) -- a 2-column
        # grid so the whole nozzle design (geometry, mesh, and both property
        # distributions) reads at a glance without clicking through tabs.
        html.Div(
            [
                html.Div(
                    [
                        html.Div(TAB_LABELS[key], style={"fontWeight": "600", "fontSize": "13px",
                                                            "marginBottom": "4px"}),
                        dcc.Graph(id=f"fig-{key}", config={"displaylogo": False},
                                   style={"height": "320px"}),
                    ],
                    style={"border": "1px solid #ddd", "borderRadius": "4px", "padding": "8px"},
                )
                for key in TAB_LABELS
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 2: stator blade parametrization (consumes Phase 1's wall_final)
# --------------------------------------------------------------------------
BLADE_DEFAULTS = dict(
    metal_angle_in=0.0, metal_angle_out=70.0, r_trailing=0.1143,
    inlet_opening_ratio=1.5, n_cp=30,
)

blade_controls = html.Div(
    [
        html.H4("Blade parametrization"),
        dcc.RadioItems(
            id="blade-method",
            options=[
                {"label": " Full (pressure side mirrors the nozzle wall)", "value": "full"},
                {"label": " Semi (pressure side is a single circular arc)", "value": "semi"},
            ],
            value="full",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field(r"$\beta_{\text{metal,in}}$ (deg)", "metal_angle_in", BLADE_DEFAULTS["metal_angle_in"]),
        _field(r"$\beta_{\text{metal,out}}$ (deg)", "metal_angle_out", BLADE_DEFAULTS["metal_angle_out"]),
        _field(r"$r_{\text{TE}}$ (mm)", "r_trailing", BLADE_DEFAULTS["r_trailing"]),
        _field(r"$s / o_1$ (pitch / inlet opening)", "inlet_opening_ratio",
               BLADE_DEFAULTS["inlet_opening_ratio"]),
        _field(r"$n_{cp}$ (B-spline control points)", "n_cp", BLADE_DEFAULTS["n_cp"], step=1, min=6),

        html.Button("Compute blade", id="compute-blade-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px",
                            "fontWeight": "600"}),
        html.Div(id="blade-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

        html.H4("Cascade view", style={"marginTop": "16px"}),
        _field(r"$N_{\text{blades}}$ (to plot)", "n_blades_plot", 2, step=1, min=1),

        html.H4("Edit control points", style={"marginTop": "16px"}),
        dcc.Dropdown(id="cp_index", options=[], value=None, clearable=False,
                      placeholder="Control point index"),
        _field(r"$\Delta n$ (mm, +/-)", "nudge_distance", 1.0),
        html.Div([
            html.Button("Nudge along normal", id="nudge-cp-btn", n_clicks=0,
                         style={"width": "68%", "padding": "6px", "marginRight": "4%"}),
            html.Button("Reset", id="reset-cp-btn", n_clicks=0,
                         style={"width": "28%", "padding": "6px"}),
        ], style={"display": "flex"}),
        html.Div(id="cp-edit-status", style={"marginTop": "6px", "fontSize": "13px"}),

        html.H4("Compare", style={"marginTop": "16px"}),
        html.Div([
            html.Button("Pin as reference", id="pin-blade-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px", "marginRight": "4%"}),
            html.Button("Clear reference", id="clear-blade-ref-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px"}),
        ], style={"display": "flex"}),
        html.Div(id="blade-reference-status", style={"marginTop": "6px", "fontSize": "13px"}),

        html.H4("CAD export (STEP)", style={"marginTop": "16px"}),
        dcc.RadioItems(
            id="step-export-target",
            options=[
                {"label": " Blade solid", "value": "blade"},
                {"label": " Fluid domain (passage)", "value": "passage"},
            ],
            value="blade",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        html.Div(
            id="step-export-blade-fields",
            children=[
                dcc.RadioItems(
                    id="step-kind",
                    options=[
                        {"label": " 2D face", "value": "face"},
                        {"label": " 3D solid (extruded)", "value": "solid"},
                    ],
                    value="solid",
                    labelStyle={"display": "block", "fontSize": "13px"},
                ),
            ],
        ),
        html.Div(
            id="step-export-passage-fields",
            children=[
                _field(r"Inlet distance ($\times$ chord)",
                       "passage_stator_inlet_chords", 1.0, step=0.5, min=0),
                _field(r"Outlet distance ($\times$ chord)",
                       "passage_stator_outlet_chords", 6.0, step=0.5, min=0),
            ],
            style={"display": "none"},
        ),
        _field(r"$L_{\text{extrude}}$ (mm)", "extrude_length", 1.0),
        html.Button("Download STEP", id="download-step-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px"}),
        dcc.Download(id="download-step"),
        html.Div(id="blade-step-status", style={"marginTop": "8px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

blade_plots = html.Div(
    [
        html.Div(
            [
                html.Label("Save format:", style={"fontSize": "13px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="blade-save-format",
                    options=[{"label": fmt.upper(), "value": fmt} for fmt in ("png", "svg", "pdf")],
                    value="png", clearable=False,
                    style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                html.Button("Download this plot", id="blade-download-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-blade-plot"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        html.Div(
            [
                dcc.Graph(id="fig-blade", config={"displaylogo": False},
                           style={"height": "550px", "flex": "2"}),
                html.Div(id="blade-info-table", style={"flex": "1", "paddingTop": "16px"}),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "12px", "alignItems": "flex-start"},
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 3: stator sizing -- couples the Phase 2 2D cascade blade (a
# reference shape) to a real annulus + mass-flow target, producing an
# actual sized stator (r_hub, r_shroud, N_blades, rescaled blade). Pure
# post-hoc rescaling of the Phase 2 result -- never re-invokes design_nozzle
# at a different y_t (see turbo_moc.geometry.sizing module docstring).
# --------------------------------------------------------------------------
SIZING_DEFAULTS = dict(
    # Novec649 meanline design (Parisi, EMPOWER_DTU), recomputed with
    # degree_reaction=0 (see DEFAULTS' own comment) -- stator radius_in,
    # height, blade_count, mass_flow_rate (unaffected by degree_reaction).
    mdot_a=0.52353, n_blades_a=15, r_hub_a=13.391,
    r_hub_b=13.391, H_b=2.9367, n_blades_b=15, mdot_b=0.52353,
)

# r1/r2 are NOT reused from Phase 3's axial sizing defaults -- those are
# post-k-scaled (k~=0.03 for the app's own Novec649 default), ~13-29 mm,
# while the RAW Phase 2 blade fed into the radial map here is still at its
# native size (pitch ~171 mm, chord ~37-114 mm). Feeding that native-size
# blade into a 13-29 mm target annulus compresses it violently and
# non-uniformly (confirmed visually -- distorted, scattered-looking
# blades, not a clean radial sweep). r1/r2 here are instead picked to be
# the same order of magnitude as the blade's own native chord extent.
# n_blades=6 is this (r1, r2)'s own implied_n_blades_radial (not a
# coincidence -- a default N picked without checking that number overlaps
# badly, confirmed visually: 15 blades crammed into a ~6-blade-wide
# annulus at this r1/r2 looks like a pinwheel/sawblade, not a cascade).
RADIAL_DEFAULTS = dict(r1=150.0, r2=300.0, n_blades=6, depth=5.0)

sizing_controls = html.Div(
    [
        html.H4("Stator sizing"),
        dcc.Dropdown(
            id="stator_wrap_type",
            options=[
                {"label": "Axial", "value": "axial"},
                {"label": "Radial", "value": "radial"},
            ],
            value="axial", clearable=False,
            style={"marginBottom": "8px"},
        ),

        html.Div(
            id="stator_axial_fields",
            children=[
                dcc.RadioItems(
                    id="sizing_mode",
                    options=[
                        {"label": " Case A (cycle-level: mass flow target)", "value": "A"},
                        {"label": " Case B (full meanline: all given)", "value": "B"},
                    ],
                    value="A",
                    labelStyle={"display": "block", "fontSize": "13px"},
                ),
                html.Div(
                    id="sizing_mode_a_fields",
                    children=[
                        _field(r"$\dot{m}$ (kg/s)", "sizing_mdot_a", SIZING_DEFAULTS["mdot_a"]),
                        _field(r"$N_{\text{blades}}$", "sizing_n_blades_a", SIZING_DEFAULTS["n_blades_a"], step=1, min=1),
                        _field(r"$r_{\text{hub}}$ (mm)", "sizing_r_hub_a", SIZING_DEFAULTS["r_hub_a"]),
                    ],
                ),
                html.Div(
                    id="sizing_mode_b_fields",
                    children=[
                        _field(r"$r_{\text{hub}}$ (mm)", "sizing_r_hub_b", SIZING_DEFAULTS["r_hub_b"]),
                        _field(r"$H$ (blade height, mm)", "sizing_H_b", SIZING_DEFAULTS["H_b"]),
                        _field(r"$N_{\text{blades}}$", "sizing_n_blades_b", SIZING_DEFAULTS["n_blades_b"], step=1, min=1),
                        _field(r"$\dot{m}$ (kg/s)", "sizing_mdot_b", SIZING_DEFAULTS["mdot_b"]),
                    ],
                    style={"display": "none"},
                ),
                html.Button("Compute sizing", id="compute-sizing-btn", n_clicks=0,
                             style={"width": "100%", "marginTop": "8px", "padding": "8px",
                                    "fontWeight": "600"}),
                html.Div(id="sizing-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

                html.H4("Export", style={"marginTop": "16px"}),
                html.Button("Download scaled blade (JSON)", id="download-sizing-json-btn", n_clicks=0,
                             style={"width": "100%", "padding": "8px"}),
                dcc.Download(id="download-sizing-json"),
                html.Button("Download blade solid (STEP)", id="download-sizing-step-btn", n_clicks=0,
                             style={"width": "100%", "padding": "8px", "marginTop": "8px"}),
                dcc.Download(id="download-sizing-step"),
                html.Div(id="sizing-step-status", style={"marginTop": "8px", "fontSize": "13px"}),
            ],
        ),

        html.Div(
            id="stator_radial_fields",
            children=[
                _field(r"$r_1$ (mm, hub)", "radial_r1", RADIAL_DEFAULTS["r1"]),
                _field(r"$r_2$ (mm, shroud)", "radial_r2", RADIAL_DEFAULTS["r2"]),
                _field(r"$N_{\text{blades}}$", "radial_n_blades", RADIAL_DEFAULTS["n_blades"], step=1, min=1),
                _field(r"Depth (mm, preview/export only)", "radial_depth", RADIAL_DEFAULTS["depth"]),
                html.Button("Compute radial wrap", id="compute-radial-btn", n_clicks=0,
                             style={"width": "100%", "marginTop": "8px", "padding": "8px",
                                    "fontWeight": "600"}),
                html.Div(id="radial-status-message", style={"marginTop": "8px", "fontSize": "13px"}),
            ],
            style={"display": "none"},
        ),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

sizing_plots = html.Div(
    [
        html.Div(
            id="stator_axial_plots",
            children=[
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="sizing-vtk-container",
                        children=_vtk_placeholder("Compute a sizing (Case A or B) to see the 3D solid preview."),
                        style={"height": "520px", "width": "100%", "border": "1px solid #ddd",
                               "borderRadius": "4px"},
                    ),
                ),
                html.Div(id="sizing-info-table", style={"marginTop": "12px"}),
            ],
        ),
        html.Div(
            id="stator_radial_plots",
            children=[
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="radial-vtk-container",
                        children=_vtk_placeholder("Compute a radial wrap to see the 3D preview."),
                        style={"height": "520px", "width": "100%", "border": "1px solid #ddd",
                               "borderRadius": "4px"},
                    ),
                ),
                html.Div(id="radial-info-table", style={"marginTop": "12px"}),
            ],
            style={"display": "none"},
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)


# --------------------------------------------------------------------------
# Phase 4: rotor blade design by the vortex-flow method (standalone --
# doesn't consume Phase 1's output, has its own fluid/inlet state)
# --------------------------------------------------------------------------
ROTOR_DEFAULTS = dict(
    P0_rel=5e5, T0_rel=400.0,
    M_inlet=1.5, M_outlet=1.5, M_lower=1.05, M_upper=2.0,
    beta_inlet=65.0, num_points=60,
)

rotor_controls = html.Div(
    [
        html.H4("Rotor blade (vortex-flow method)"),
        dcc.Dropdown(id="rotor_fluid_name", options=FLUIDS, value=DEFAULT_FLUID, clearable=False),
        html.Br(),

        html.H4("Couple to stator exit", style={"marginTop": "16px"}),
        dcc.Checklist(
            id="couple_to_stator",
            options=[{"label": " Couple to stator exit", "value": "on"}],
            value=[],
            labelStyle={"fontSize": "13px"},
        ),
        _field(r"$\Omega$ (RPM)", "rotor_omega", 20716.71),
        _field(r"$r$ (mm, blade-speed radius)", "rotor_radius_mm", 21.685),
        html.Div(id="rotor-triangle-status", style={"fontSize": "12px"}),

        _field(r"$P_0$ (Pa)", "rotor_P0", ROTOR_DEFAULTS["P0_rel"]),
        _field(r"$T_0$ (K)", "rotor_T0", ROTOR_DEFAULTS["T0_rel"]),

        html.H4("Mach numbers", style={"marginTop": "16px"}),
        _field(r"$M_{\text{in}}$ (uniform flow)", "rotor_M_inlet", ROTOR_DEFAULTS["M_inlet"]),
        _field(r"$M_{\text{out}}$ (uniform flow)", "rotor_M_outlet", ROTOR_DEFAULTS["M_outlet"]),
        _field(r"$M_{\text{lower}}$ (pressure-side arc)", "rotor_M_lower", ROTOR_DEFAULTS["M_lower"]),
        _field(r"$M_{\text{upper}}$ (suction-side arc)", "rotor_M_upper", ROTOR_DEFAULTS["M_upper"]),

        html.H4("Flow angles", style={"marginTop": "16px"}),
        _field(r"$\beta_{\text{in}}$ (deg)", "rotor_beta_inlet", ROTOR_DEFAULTS["beta_inlet"]),

        html.H4("Marching resolution", style={"marginTop": "16px"}),
        _field(r"$n_{\text{points}}$ (per transition arc)", "rotor_num_points",
               ROTOR_DEFAULTS["num_points"], step=1, min=10),

        html.Button("Compute rotor blade", id="compute-rotor-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px",
                            "fontWeight": "600"}),
        html.Div(id="rotor-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

        html.H4("Display", style={"marginTop": "16px"}),
        dcc.Checklist(
            id="rotor-display-options",
            options=[
                {"label": " Show pressure/suction surfaces", "value": "surfaces"},
                {"label": " Show characteristic (Mach) lines", "value": "mach_lines"},
            ],
            value=["surfaces"],
            labelStyle={"display": "block", "fontSize": "13px"},
        ),

        html.H4("Compare", style={"marginTop": "16px"}),
        html.Div([
            html.Button("Pin as reference", id="pin-rotor-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px", "marginRight": "4%"}),
            html.Button("Clear reference", id="clear-rotor-ref-btn", n_clicks=0,
                         style={"width": "48%", "padding": "6px"}),
        ], style={"display": "flex"}),
        html.Div(id="rotor-reference-status", style={"marginTop": "6px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

rotor_plots = html.Div(
    [
        html.Div(
            [
                html.Label("Save format:", style={"fontSize": "13px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="rotor-save-format",
                    options=[{"label": fmt.upper(), "value": fmt} for fmt in ("png", "svg", "pdf")],
                    value="png", clearable=False,
                    style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                html.Button("Download this plot", id="rotor-download-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-rotor-plot"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        html.Div(
            [
                dcc.Graph(id="fig-rotor-blade", config={"displaylogo": False},
                           style={"height": "550px", "flex": "2"}),
                html.Div(id="rotor-info-table", style={"flex": "1", "paddingTop": "16px"}),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "12px", "alignItems": "flex-start"},
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)


# --------------------------------------------------------------------------
# Phase 5: 3D rotor design -- the rotor's twin of Phase 3: couples the
# Phase 4 2D rotor blade (nondimensional-by-r* reference shape) to the SAME
# annulus as the stator (r_hub/r_shroud reused from Phase 3's own sizing
# result -- one through-flow passage in an axial stage), with its own free
# N_blades (generally different from the stator's). Unlike the stator,
# there's no mass-flow equation at this 2D-cascade level for the rotor
# method, so k is derived from pitch consistency alone (see
# turbo_moc.geometry.sizing.size_rotor_from_pitch).
# --------------------------------------------------------------------------
ROTOR_SIZING_DEFAULTS = dict(n_blades=34)  # EMPOWER_DTU Novec649 meanline rotor blade_count
# Same reasoning as RADIAL_DEFAULTS above: r1/r2 picked to match the raw
# Phase 4 rotor blade's own native size, not the axially-scaled Phase 5
# ones. The rotor blade is nondimensional-by-r* (no r_star passed at
# compute time), so `scale` (mm per r*) also needs to be large enough to
# bring it up near that same order of magnitude -- 1.0 mm/r* left it
# under 2 mm, the opposite mismatch (blade far smaller than the target
# annulus this time). n_blades=30 is this (r1, r2, scale)'s own
# implied_n_blades_radial (see RADIAL_DEFAULTS' own comment for why that
# matters).
ROTOR_RADIAL_DEFAULTS = dict(r1=100.0, r2=200.0, n_blades=30, scale=50.0, depth=5.0)

rotor_sizing_controls = html.Div(
    [
        html.H4("3D rotor design"),
        dcc.Dropdown(
            id="rotor_wrap_type",
            options=[
                {"label": "Axial", "value": "axial"},
                {"label": "Radial", "value": "radial"},
            ],
            value="axial", clearable=False,
            style={"marginBottom": "8px"},
        ),

        html.Div(
            id="rotor_axial_fields",
            children=[
                html.Div(id="rotor-sizing-annulus-readout", style={"fontSize": "12px", "marginBottom": "8px"}),
                _field(r"$N_{\text{blades}}$ (rotor)", "rotor_sizing_n_blades",
                       ROTOR_SIZING_DEFAULTS["n_blades"], step=1, min=1),
                html.Button("Compute sizing", id="compute-rotor-sizing-btn", n_clicks=0,
                             style={"width": "100%", "marginTop": "8px", "padding": "8px",
                                    "fontWeight": "600"}),
                html.Div(id="rotor-sizing-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

                html.H4("Export", style={"marginTop": "16px"}),
                html.Button("Download blade solid (STEP)", id="download-rotor-sizing-step-btn", n_clicks=0,
                             style={"width": "100%", "padding": "8px"}),
                dcc.Download(id="download-rotor-sizing-step"),
                html.Div(id="rotor-sizing-step-status", style={"marginTop": "8px", "fontSize": "13px"}),
            ],
        ),

        html.Div(
            id="rotor_radial_fields",
            children=[
                _field(r"$r_1$ (mm, hub)", "rotor_radial_r1", ROTOR_RADIAL_DEFAULTS["r1"]),
                _field(r"$r_2$ (mm, shroud)", "rotor_radial_r2", ROTOR_RADIAL_DEFAULTS["r2"]),
                _field(r"$N_{\text{blades}}$", "rotor_radial_n_blades",
                       ROTOR_RADIAL_DEFAULTS["n_blades"], step=1, min=1),
                _field(r"Scale (mm per r*)", "rotor_radial_scale", ROTOR_RADIAL_DEFAULTS["scale"]),
                _field(r"Depth (mm, preview/export only)", "rotor_radial_depth",
                       ROTOR_RADIAL_DEFAULTS["depth"]),
                html.Button("Compute radial wrap", id="compute-rotor-radial-btn", n_clicks=0,
                             style={"width": "100%", "marginTop": "8px", "padding": "8px",
                                    "fontWeight": "600"}),
                html.Div(id="rotor-radial-status-message", style={"marginTop": "8px", "fontSize": "13px"}),
            ],
            style={"display": "none"},
        ),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

rotor_sizing_plots = html.Div(
    [
        html.Div(
            id="rotor_axial_plots",
            children=[
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="rotor-sizing-vtk-container",
                        children=_vtk_placeholder("Compute a sizing to see the 3D solid preview."),
                        style={"height": "520px", "width": "100%", "border": "1px solid #ddd",
                               "borderRadius": "4px"},
                    ),
                ),
                html.Div(id="rotor-sizing-info-table", style={"marginTop": "12px"}),
            ],
        ),
        html.Div(
            id="rotor_radial_plots",
            children=[
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="rotor-radial-vtk-container",
                        children=_vtk_placeholder("Compute a radial wrap to see the 3D preview."),
                        style={"height": "520px", "width": "100%", "border": "1px solid #ddd",
                               "borderRadius": "4px"},
                    ),
                ),
                html.Div(id="rotor-radial-info-table", style={"marginTop": "12px"}),
            ],
            style={"display": "none"},
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)


# --------------------------------------------------------------------------
# Phase 6: Cascade -- stator (Phase 3) and rotor (Phase 5) rows shown
# together, axially positioned one after another. No new sizing math: just
# combines the two already-sized solids with an axial gap, reusing the
# exact z_offset mechanism export_annular_blade_step/stl already expose. Z
# is each row's own chordwise coordinate, unchanged by the annular wrap
# (see turbo_moc.geometry.annular's module docstring) -- the stator's own
# blade_curve.y span gives its [z_min, z_max], the rotor's own blade.x
# span gives its own, NEGATED (confirmed by this app's earlier annular
# cascade work: the rotor's local chordwise axis runs the opposite way
# from the stator's, so the raw x needs flipping to land downstream
# rather than mirrored/upstream).
# --------------------------------------------------------------------------
cascade_controls = html.Div(
    [
        html.H4("Cascade (stator + rotor)"),
        html.Div(id="cascade-assembly-readout", style={"fontSize": "12px", "marginBottom": "8px"}),
        _field(r"$\Delta z_{\text{gap}}$ (mm, stator TE to rotor LE)", "cascade_axial_gap", 5.0),

        html.H4("2D cascade (mid-span)", style={"marginTop": "16px"}),
        _field(r"$N_{\text{stator}}$ (blades to plot)", "cascade_2d_n_stator", 3, step=1, min=1),
        _field(r"$N_{\text{rotor}}$ (blades to plot)", "cascade_2d_n_rotor", 3, step=1, min=1),

        html.H4("Export", style={"marginTop": "16px"}),
        html.Button("Download stator STEP (assembly position)", id="download-cascade-stator-step-btn",
                     n_clicks=0, style={"width": "100%", "padding": "8px", "marginBottom": "6px"}),
        dcc.Download(id="download-cascade-stator-step"),
        html.Button("Download rotor STEP (assembly position)", id="download-cascade-rotor-step-btn",
                     n_clicks=0, style={"width": "100%", "padding": "8px"}),
        dcc.Download(id="download-cascade-rotor-step"),
        html.Div(id="cascade-step-status", style={"marginTop": "8px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

cascade_plots = html.Div(
    [
        dcc.Loading(
            type="circle",
            children=html.Div(
                id="cascade-vtk-container",
                children=_vtk_placeholder("Run Phase 3 and Phase 5 sizing first to see the assembly."),
                style={"height": "600px", "width": "100%", "border": "1px solid #ddd",
                       "borderRadius": "4px"},
            ),
        ),
        dcc.Graph(id="fig-cascade-2d", config={"displaylogo": False},
                   style={"height": "500px", "marginTop": "16px"}),
    ],
    style={"flex": "1", "padding": "16px"},
)


app.layout = html.Div(
    [
        html.H2("TurboMoC", style={"padding": "16px 16px 0 16px"}),
        dcc.Tabs(
            id="phase-tabs",
            value="phase1",
            children=[
                dcc.Tab(label="Phase 1: Nozzle design", value="phase1",
                         children=html.Div([controls, plots], style={"display": "flex"})),
                dcc.Tab(label="Phase 2: Stator design", value="phase2",
                         children=dcc.Tabs(
                             id="stator-subtabs", value="stator2d",
                             children=[
                                 dcc.Tab(label="2D", value="stator2d",
                                          children=html.Div([blade_controls, blade_plots],
                                                             style={"display": "flex"})),
                                 dcc.Tab(label="3D", value="stator3d",
                                          children=html.Div([sizing_controls, sizing_plots],
                                                             style={"display": "flex"})),
                             ],
                         )),
                dcc.Tab(label="Phase 3: Rotor design", value="phase3",
                         children=dcc.Tabs(
                             id="rotor-subtabs", value="rotor2d",
                             children=[
                                 dcc.Tab(label="2D", value="rotor2d",
                                          children=html.Div([rotor_controls, rotor_plots],
                                                             style={"display": "flex"})),
                                 dcc.Tab(label="3D", value="rotor3d",
                                          children=html.Div([rotor_sizing_controls, rotor_sizing_plots],
                                                             style={"display": "flex"})),
                             ],
                         )),
                dcc.Tab(label="Phase 4: Cascade", value="phase4_cascade",
                         children=html.Div([cascade_controls, cascade_plots], style={"display": "flex"})),
            ],
        ),
        dcc.Store(id="result-store"),
        dcc.Store(id="loaded-nozzle-store"),
        dcc.Store(id="reference-store"),
        dcc.Store(id="blade-store"),
        dcc.Store(id="blade-edited-store"),
        dcc.Store(id="blade-reference-store"),
        dcc.Store(id="stator-sizing-store"),
        dcc.Store(id="stator-radial-store"),
        dcc.Store(id="rotor-store"),
        dcc.Store(id="rotor-reference-store"),
        dcc.Store(id="rotor-sizing-store"),
        dcc.Store(id="rotor-radial-store"),
    ],
    style={"fontFamily": "Helvetica, Arial, sans-serif"},
)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
@app.callback(
    Output("T0_container", "style"),
    Output("Q0_container", "style"),
    Input("inlet_mode", "value"),
)
def toggle_inlet_inputs(mode):
    if mode == "Q0":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


@app.callback(
    Output("result-store", "data"),
    Output("status-message", "children"),
    Input("compute-btn", "n_clicks"),
    State("fluid_name", "value"),
    State("inlet_mode", "value"),
    State("P0", "value"),
    State("T0", "value"),
    State("Q0", "value"),
    State("p_back", "value"),
    State("solver", "value"),
    State("stage1_mode", "value"),
    State("y_t", "value"),
    State("rho_d", "value"),
    State("n", "value"),
    background=True,
    manager=background_callback_manager,
    progress=[Output("progress-bar", "value"), Output("progress-label", "children")],
    prevent_initial_call=True,
)
def run_design(set_progress, n_clicks, fluid_name, inlet_mode, P0, T0, Q0, p_back,
                solver, stage1_mode, y_t, rho_d, n):
    reporter = make_progress_reporter(set_progress)
    set_progress(("0%", "starting..."))
    # rho_t (the Sauer/parabolic sonic-line's throat radius of curvature,
    # only meaningful when stage1_mode="sauer") is no longer an
    # independent field -- derived as rho_d, so the throat's upstream
    # (Sauer-implied) curvature always matches the downstream kernel
    # arc's own radius, keeping the wall's curvature continuous through
    # the throat. NOT y_t + rho_d: that's the y-coordinate of the kernel
    # arc's CENTER (see rotation_around_center's docstring), a position
    # on the axis, not a curvature radius -- a circle's radius of
    # curvature is its own radius rho_d, everywhere on it, regardless of
    # how far its center sits from the axis.
    rho_t = float(rho_d or 0.0)
    try:
        data = design_nozzle(
            fluid_name=fluid_name,
            P0=float(P0) * 1e5,  # bar (UI/save convention) -> Pa
            T0=T0 if inlet_mode == "T0" else None,
            Q0=(Q0 if inlet_mode == "Q0" else float("nan")),
            p_back=float(p_back) * 1e5,  # bar -> Pa
            solver=solver,
            use_true_sauer_line=(stage1_mode == "sauer"),
            y_t=y_t, rho_t=rho_t, rho_d=rho_d, n=int(n),
            progress_callback=reporter,
        )
    except NozzleDesignError as e:
        set_progress(("0%", ""))
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"})

    status_children = [
        html.Span("Converged" if data["converged"] else "Did not fully converge",
                   style={"fontWeight": "600",
                          "color": "#1a7a1a" if data["converged"] else "#b00020"}),
        html.Div(f"exit M = {data['exit_M']:.4f}  (design {data['design_Noz_Mach']:.4f})"),
        html.Div(f"runtime = {data['runtime_s']:.1f} s"),
    ]
    if "q0_nudged_from" in data:
        status_children.append(html.Div(
            f"Note: true Sauer line stalled at Q0={data['q0_nudged_from']:.3g} "
            f"(near-M=1 marching instability) -- auto-nudged to "
            f"Q0={data['meta']['Q0']:.3g} to converge.",
            style={"color": "#b06a00", "marginTop": "4px"},
        ))
    status = html.Div(status_children)
    return data, status


# --------------------------------------------------------------------------
# Phase 1 save/load: the nozzle solve is the most lengthy of the app's
# phases (5-30s+, run in the background above), so a saved file carries
# BOTH the inputs (for the fields) and the full solve result (design_
# nozzle's own "flat, JSON-safe dict" -- no reprocessing needed) so
# loading skips recomputing entirely, not just re-fills the form.
# --------------------------------------------------------------------------
@app.callback(
    Output("download-nozzle-json", "data"),
    Input("save-nozzle-btn", "n_clicks"),
    State("result-store", "data"),
    State("fluid_name", "value"),
    State("inlet_mode", "value"),
    State("P0", "value"),
    State("T0", "value"),
    State("Q0", "value"),
    State("p_back", "value"),
    State("solver", "value"),
    State("stage1_mode", "value"),
    State("y_t", "value"),
    State("rho_d", "value"),
    State("n", "value"),
    prevent_initial_call=True,
)
def save_nozzle_design(n_clicks, result_data, fluid_name, inlet_mode, P0, T0, Q0, p_back,
                        solver, stage1_mode, y_t, rho_d, n):
    if result_data is None:
        return dash.no_update

    cfg = {
        "inputs": {
            "fluid_name": fluid_name, "inlet_mode": inlet_mode, "P0": P0, "T0": T0, "Q0": Q0,
            "p_back": p_back, "solver": solver, "stage1_mode": stage1_mode,
            "y_t": y_t, "rho_d": rho_d, "n": n,
        },
        "outputs": result_data,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"nozzle_design_{timestamp}.json"
    return dict(content=json.dumps(cfg, indent=2), filename=filename, type="application/json")


@app.callback(
    Output("loaded-nozzle-store", "data"),
    Input("load-nozzle-upload", "contents"),
    prevent_initial_call=True,
)
def load_nozzle_file(contents):
    if contents is None:
        return dash.no_update
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")
    return json.loads(decoded)


@app.callback(
    Output("result-store", "data", allow_duplicate=True),
    Output("status-message", "children", allow_duplicate=True),
    Output("fluid_name", "value"),
    Output("inlet_mode", "value"),
    Output("P0", "value"),
    Output("T0", "value"),
    Output("Q0", "value"),
    Output("p_back", "value"),
    Output("solver", "value"),
    Output("stage1_mode", "value"),
    Output("y_t", "value"),
    Output("rho_d", "value"),
    Output("n", "value"),
    Input("loaded-nozzle-store", "data"),
    prevent_initial_call=True,
)
def apply_loaded_nozzle_design(cfg):
    if not cfg:
        return (dash.no_update,) * 12
    inp = cfg.get("inputs", {})
    status = html.Div("Loaded from file (no recompute needed).", style={"color": "#1a7a1a"})
    return (
        cfg.get("outputs"), status,
        inp.get("fluid_name"), inp.get("inlet_mode"), inp.get("P0"), inp.get("T0"), inp.get("Q0"),
        inp.get("p_back"), inp.get("solver"), inp.get("stage1_mode"),
        inp.get("y_t"), inp.get("rho_d"), inp.get("n"),
    )


@app.callback(
    Output("reference-store", "data"),
    Output("reference-status", "children"),
    Input("pin-btn", "n_clicks"),
    Input("clear-ref-btn", "n_clicks"),
    State("result-store", "data"),
    prevent_initial_call=True,
)
def manage_reference(pin_clicks, clear_clicks, current_data):
    if ctx.triggered_id == "clear-ref-btn":
        return None, ""
    if ctx.triggered_id == "pin-btn":
        if not current_data:
            return dash.no_update, html.Div(
                "Nothing to pin yet -- run Compute first.", style={"color": "#b00020"})
        label = f"{current_data['fluid_name']}, {current_data['solver']}, exit M={current_data['exit_M']:.4f}"
        return current_data, html.Div(f"Reference pinned: {label}", style={"color": "#1a7a1a"})
    return dash.no_update, dash.no_update


def _with_convergent_wall(data, enable_vals, L_conv):
    """Copy of `data` with a "convergent_wall" key added if the Phase-1
    convergent-inlet toggle is on -- see turbo_moc.core.classes.
    build_convergent_inlet. Geometry only, computed live/reactively (no
    MOC re-solve), so this is cheap to call from every plot/export
    callback rather than persisting it in result-store."""
    if not data or not enable_vals or "on" not in enable_vals:
        return data
    meta = data.get("meta", {})
    y_t, rho_d = meta.get("y_t"), meta.get("rho_d")
    if y_t is None or rho_d is None or not L_conv:
        return data
    try:
        conv = build_convergent_inlet(float(y_t), float(rho_d), float(L_conv))
    except ValueError:
        return data
    out = dict(data)
    out["convergent_wall"] = conv
    return out


@app.callback(
    [Output(f"fig-{key}", "figure") for key in TAB_LABELS],
    Input("result-store", "data"),
    Input("reference-store", "data"),
    Input("conv_inlet_enable", "value"),
    Input("conv_inlet_length", "value"),
)
def update_plots(data, reference, conv_enable, conv_length):
    if not data:
        empty = go.Figure()
        return [empty] * len(TAB_LABELS)
    ref = reference if reference else None
    data = _with_convergent_wall(data, conv_enable, conv_length)
    return [plot_fn(data, reference=ref) for plot_fn in PLOT_FUNCS_PLOTLY.values()]


@app.callback(
    Output("download-plot", "data"),
    Input("download-btn", "n_clicks"),
    State("plot-select", "value"),
    State("result-store", "data"),
    State("reference-store", "data"),
    State("save-format", "value"),
    State("conv_inlet_enable", "value"),
    State("conv_inlet_length", "value"),
    prevent_initial_call=True,
)
def download_plot(n_clicks, active_tab, data, reference, fmt, conv_enable, conv_length):
    if not data:
        return dash.no_update
    ref = reference if reference else None
    data = _with_convergent_wall(data, conv_enable, conv_length)
    plot_fn, name = PLOT_FUNCS_MPL[active_tab]
    fig, _ax = plot_fn(data, reference=ref)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"{name}.{fmt}")


@app.callback(
    Output("download-wall-csv", "data"),
    Input("download-wall-csv-btn", "n_clicks"),
    State("result-store", "data"),
    State("conv_inlet_enable", "value"),
    State("conv_inlet_length", "value"),
    prevent_initial_call=True,
)
def download_wall_csv(n_clicks, data, conv_enable, conv_length):
    if not data:
        return dash.no_update
    data = _with_convergent_wall(data, conv_enable, conv_length)
    wall = data["wall_final"]
    lines = ["x,y"]
    conv = data.get("convergent_wall")
    if conv and conv.get("x"):
        lines.extend(f"{x},{y}" for x, y in zip(conv["x"], conv["y"]))
    lines.extend(f"{x},{y}" for x, y in zip(wall["x"], wall["y"]))
    return dcc.send_string("\n".join(lines), "nozzle_wall.csv")


@app.callback(
    Output("blade-store", "data"),
    Output("blade-status-message", "children"),
    Output("blade-info-table", "children"),
    Input("compute-blade-btn", "n_clicks"),
    State("result-store", "data"),
    State("blade-method", "value"),
    State("metal_angle_in", "value"),
    State("metal_angle_out", "value"),
    State("r_trailing", "value"),
    State("inlet_opening_ratio", "value"),
    State("n_cp", "value"),
    prevent_initial_call=True,
)
def run_blade(n_clicks, nozzle_data, method, metal_angle_in, metal_angle_out, r_trailing,
              inlet_opening_ratio, n_cp):
    if not nozzle_data:
        return None, html.Div("Run Phase 1 Compute first -- no nozzle wall to build a blade from.",
                                style={"color": "#b00020"}), None

    wall = nozzle_data["wall_final"]
    parametrize_fn = BLADE_PARAM_FUNCS[method or "full"]
    try:
        blade = parametrize_fn(
            wall["x"], wall["y"],
            metal_angle_in=metal_angle_in, metal_angle_out=metal_angle_out,
            r_trailing=r_trailing,
            inlet_opening_ratio=inlet_opening_ratio, n_cp=int(n_cp),
        )
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    status = html.Div([
        html.Span(f"Blade parametrized ({method or 'full'})",
                   style={"fontWeight": "600", "color": "#1a7a1a"}),
    ])
    table_data = [
        {"Quantity": "Pitch", "Value": f"{blade['pitch']:.4f}", "Unit": blade["units"]},
        {"Quantity": "Throat opening", "Value": f"{blade['throat_opening']:.4f}", "Unit": blade["units"]},
        {"Quantity": "Inlet opening", "Value": f"{blade['inlet_opening']:.4f}", "Unit": blade["units"]},
        {"Quantity": "Axial chord (convergent)", "Value": f"{blade['axial_chord_convergent']:.4f}",
         "Unit": blade["units"]},
        {"Quantity": "Axial chord (divergent)", "Value": f"{blade['axial_chord_divergent']:.4f}",
         "Unit": blade["units"]},
        {"Quantity": "Metal angle in", "Value": f"{blade['metal_angle_in']:.2f}", "Unit": "deg"},
        {"Quantity": "Metal angle out", "Value": f"{blade['metal_angle_out']:.2f}", "Unit": "deg"},
        {"Quantity": "TE radius", "Value": f"{blade['r_trailing']:.4f}", "Unit": blade["units"]},
        {"Quantity": "s / o_1", "Value": f"{inlet_opening_ratio:.4f}", "Unit": "-"},
    ]
    return blade, status, make_table(table_data, ["Quantity", "Value", "Unit"])


def _effective_blade(base_blade, edited_blade):
    """The blade actually shown/exported: the edited version if any nudges
    have been applied on top of the last Compute, else the fresh fit."""
    return edited_blade or base_blade


@app.callback(
    Output("sizing_mode_a_fields", "style"),
    Output("sizing_mode_b_fields", "style"),
    Input("sizing_mode", "value"),
)
def _toggle_sizing_mode(mode):
    shown, hidden = {"display": "block"}, {"display": "none"}
    return (shown, hidden) if mode == "A" else (hidden, shown)


@app.callback(
    Output("stator_axial_fields", "style"),
    Output("stator_radial_fields", "style"),
    Output("stator_axial_plots", "style"),
    Output("stator_radial_plots", "style"),
    Input("stator_wrap_type", "value"),
)
def _toggle_stator_wrap_type(wrap_type):
    shown, hidden = {"display": "block"}, {"display": "none"}
    return (shown, hidden, shown, hidden) if wrap_type == "axial" else (hidden, shown, hidden, shown)


@app.callback(
    Output("stator-sizing-store", "data"),
    Output("sizing-status-message", "children"),
    Output("sizing-info-table", "children"),
    Input("compute-sizing-btn", "n_clicks"),
    State("sizing_mode", "value"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("result-store", "data"),
    State("sizing_mdot_a", "value"),
    State("sizing_n_blades_a", "value"),
    State("sizing_r_hub_a", "value"),
    State("sizing_r_hub_b", "value"),
    State("sizing_H_b", "value"),
    State("sizing_n_blades_b", "value"),
    State("sizing_mdot_b", "value"),
    prevent_initial_call=True,
)
def run_stator_sizing(n_clicks, mode, base_blade, edited_blade, nozzle_data,
                       mdot_a, n_blades_a, r_hub_a,
                       r_hub_b, H_b, n_blades_b, mdot_b):
    stator = _effective_blade(base_blade, edited_blade)
    if not stator:
        return None, html.Div("Run Phase 2 first -- no stator blade to size.",
                                style={"color": "#b00020"}), None
    if not nozzle_data:
        return None, html.Div("Run Phase 1 first -- sizing needs the throat sonic state.",
                                style={"color": "#b00020"}), None

    try:
        if mode == "A":
            result = size_stator_mode_a(stator, nozzle_data, mdot_a, int(n_blades_a), r_hub_a)
        else:
            result = size_stator_mode_b(stator, nozzle_data, r_hub_b, H_b, int(n_blades_b), mdot_b)
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    scaled_pitch = result["scaled_blade"]["pitch"]
    if mode == "A":
        table_data = [
            {"Quantity": "k (scale factor)", "Value": f"{result['k']:.2f}", "Unit": "-"},
            {"Quantity": "Pitch (scaled)", "Value": f"{scaled_pitch:.2f}", "Unit": "mm"},
            {"Quantity": "r_hub", "Value": f"{result['r_hub']:.2f}", "Unit": "mm"},
            {"Quantity": "r_shroud", "Value": f"{result['r_shroud']:.2f}", "Unit": "mm"},
            {"Quantity": "H (blade height)", "Value": f"{result['H']:.2f}", "Unit": "mm"},
            {"Quantity": "N_blades", "Value": str(result["n_blades"]), "Unit": "-"},
            {"Quantity": "mdot", "Value": f"{result['mdot']:.2f}", "Unit": "kg/s"},
            {"Quantity": "rho_star", "Value": f"{result['rho_star']:.2f}", "Unit": "kg/m3"},
            {"Quantity": "a_star", "Value": f"{result['a_star']:.2f}", "Unit": "m/s"},
        ]
        status = html.Div("Sizing computed (Case A).", style={"color": "#1a7a1a", "fontWeight": "600"})
    else:
        ok = abs(result["residual"]) <= 0.05
        table_data = [
            {"Quantity": "k_pitch", "Value": f"{result['k_pitch']:.2f}", "Unit": "-"},
            {"Quantity": "k_massflow", "Value": f"{result['k_massflow']:.2f}", "Unit": "-"},
            {"Quantity": "residual", "Value": f"{result['residual'] * 100:.2f}", "Unit": "%"},
            {"Quantity": "Pitch (scaled, from k_pitch)", "Value": f"{scaled_pitch:.2f}", "Unit": "mm"},
            {"Quantity": "r_hub", "Value": f"{result['r_hub']:.2f}", "Unit": "mm"},
            {"Quantity": "r_shroud", "Value": f"{result['r_shroud']:.2f}", "Unit": "mm"},
            {"Quantity": "N_blades", "Value": str(result["n_blades"]), "Unit": "-"},
            {"Quantity": "rho_star", "Value": f"{result['rho_star']:.2f}", "Unit": "kg/m3"},
            {"Quantity": "a_star", "Value": f"{result['a_star']:.2f}", "Unit": "m/s"},
        ]
        color = "#1a7a1a" if ok else "#b00020"
        if ok:
            msg = "Sizing computed (Case B) -- k_pitch/k_massflow agree."
        else:
            msg = ("Sizing computed (Case B) -- LARGE residual: the 2D blade shape "
                    "(metal_angle_out/solidity) may need redesigning, not just rescaling.")
        status = html.Div(msg, style={"color": color, "fontWeight": "600"})

    return result, status, make_table(table_data, ["Quantity", "Value", "Unit"])


@app.callback(
    Output("stator-radial-store", "data"),
    Output("radial-status-message", "children"),
    Output("radial-info-table", "children"),
    Input("compute-radial-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("result-store", "data"),
    State("radial_r1", "value"),
    State("radial_r2", "value"),
    State("radial_n_blades", "value"),
    State("radial_depth", "value"),
    prevent_initial_call=True,
)
def run_radial_stator(n_clicks, base_blade, edited_blade, nozzle_data, r1, r2, n_blades, depth):
    """Diagnostic-only radial sizing (see turbo_moc.geometry.radial's
    module docstring): N_blades is a direct input, not solved -- the
    conformal map's pitch consistency has no free scale lever the way the
    axial Case A's k does (uniform rescale cancels out of the mapped
    pitch once r1/r2 are fixed), so this reports the implied mass flow
    from the given N rather than deriving N from a target mdot."""
    from turbo_moc.geometry import (
        compute_throat_sonic_state, implied_n_blades_radial, mapped_pitch_width,
        mapped_throat_width,
    )

    stator = _effective_blade(base_blade, edited_blade)
    if not stator:
        return None, html.Div("Run Phase 2 first -- no stator blade to wrap.",
                                style={"color": "#b00020"}), None
    if not nozzle_data:
        return None, html.Div("Run Phase 1 first -- the mass-flow diagnostic needs the "
                                "throat sonic state.", style={"color": "#b00020"}), None

    r1, r2, n_blades, depth = float(r1), float(r2), int(n_blades), float(depth)
    try:
        mapped_throat = mapped_throat_width(stator, r1, r2, source="stator")
        mapped_pitch = mapped_pitch_width(stator, r1, r2, source="stator")
        n_implied = implied_n_blades_radial(stator, r1, r2, source="stator")
        rho_star, a_star = compute_throat_sonic_state(nozzle_data)
        mdot_implied = rho_star * a_star * mapped_throat * depth * n_blades * 1e-6
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    result = {
        "r1": r1, "r2": r2, "n_blades": n_blades, "depth": depth,
        "mapped_throat": mapped_throat, "rho_star": rho_star, "a_star": a_star,
        "mdot_implied": mdot_implied,
        "blade": stator,
    }
    table_data = [
        {"Quantity": "2D throat_opening", "Value": f"{stator['throat_opening']:.2f}", "Unit": "mm"},
        {"Quantity": "Mapped throat (radial)", "Value": f"{mapped_throat:.4f}", "Unit": "mm"},
        {"Quantity": "r1 (hub)", "Value": f"{r1:.2f}", "Unit": "mm"},
        {"Quantity": "r2 (shroud)", "Value": f"{r2:.2f}", "Unit": "mm"},
        {"Quantity": "N_blades (entered)", "Value": str(n_blades), "Unit": "-"},
        {"Quantity": "Mapped pitch (at N entered)", "Value": f"{mapped_pitch:.2f}", "Unit": "mm"},
        {"Quantity": "Implied N_blades (no overlap/gaps)", "Value": f"{n_implied:.1f}", "Unit": "-"},
        {"Quantity": "Depth", "Value": f"{depth:.2f}", "Unit": "mm"},
        {"Quantity": "Implied mass flow", "Value": f"{mdot_implied:.4f}", "Unit": "kg/s"},
    ]
    ratio = n_blades / n_implied
    if ratio > 1.15:
        msg = (f"Radial wrap computed -- N_blades ({n_blades}) is {ratio:.1f}x the implied "
               f"count ({n_implied:.1f}): blades will visibly OVERLAP. Lower N_blades.")
        color = "#b00020"
    elif ratio < 0.85:
        msg = (f"Radial wrap computed -- N_blades ({n_blades}) is only {ratio:.1f}x the "
               f"implied count ({n_implied:.1f}): large GAPS between blades. Raise N_blades.")
        color = "#b00020"
    else:
        msg = "Radial wrap computed -- N_blades is close to the implied (no-overlap) count."
        color = "#1a7a1a"
    status = html.Div(msg, style={"color": color, "fontWeight": "600"})
    return result, status, make_table(table_data, ["Quantity", "Value", "Unit"])


@app.callback(
    Output("radial-vtk-container", "children"),
    Input("stator-radial-store", "data"),
)
def _update_radial_vtk_preview(radial_result):
    if not radial_result:
        return _vtk_placeholder("Compute a radial wrap to see the 3D preview.")
    if not DASH_VTK_AVAILABLE:
        return _vtk_placeholder(
            "dash-vtk is not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")
    try:
        import vtk
        from turbo_moc.geometry import wrap_blade_radial
    except ImportError:
        return _vtk_placeholder("vtk is not installed -- 3D solid preview unavailable.")

    try:
        wrapped = wrap_blade_radial(
            radial_result["blade"], radial_result["r1"], radial_result["r2"],
            radial_result["n_blades"])
        append = vtk.vtkAppendPolyData()
        for blade_copy, te_copy in zip(wrapped["blades"], wrapped["trailing_edges"]):
            x = blade_copy["x"] + te_copy["x"]
            y = blade_copy["y"] + te_copy["y"]
            append.AddInputData(_extrude_polygon_vtk(vtk, x, y, radial_result["depth"]))
        append.Update()
        mesh_state = to_mesh_state(_triangulate_vtk(vtk, append.GetOutput()))
    except Exception as e:
        return _vtk_placeholder(f"Error building 3D solid preview: {e}")

    return html.Div(
        [
            html.Div(f"Showing all {radial_result['n_blades']} blades (radial wrap).",
                      style={"position": "absolute", "top": "8px", "left": "8px",
                             "fontSize": "12px", "color": "#666", "zIndex": 1,
                             "backgroundColor": "rgba(255,255,255,0.85)",
                             "padding": "2px 8px", "borderRadius": "4px"}),
            dash_vtk.View(
                background=[0.93, 0.95, 0.97],
                style={"height": "100%", "width": "100%"},
                children=[
                    dash_vtk.GeometryRepresentation(
                        children=[dash_vtk.Mesh(state=mesh_state)],
                        property={
                            "color": [0.55, 0.63, 0.75], "edgeVisibility": False,
                            "interpolation": "Phong",
                            "ambient": 0.12, "diffuse": 0.8, "specular": 0.45, "specularPower": 30,
                        },
                        showCubeAxes=True,
                        cubeAxesStyle={"axisLabels": ["X [mm]", "Y [mm]", "Z (axial) [mm]"]},
                    ),
                ],
            ),
        ],
        style={"position": "relative", "height": "100%", "width": "100%"},
    )


@app.callback(
    Output("download-sizing-json", "data"),
    Input("download-sizing-json-btn", "n_clicks"),
    State("stator-sizing-store", "data"),
    prevent_initial_call=True,
)
def download_sizing_json(n_clicks, sizing_result):
    if not sizing_result:
        return None
    return dcc.send_string(json.dumps(sizing_result, indent=2), "stator_sized_blade.json")


@app.callback(
    Output("sizing-vtk-container", "children"),
    Input("stator-sizing-store", "data"),
)
def _update_sizing_vtk_preview(sizing_result):
    if not sizing_result:
        return _vtk_placeholder("Compute a sizing (Case A or B) to see the 3D solid preview.")

    # True CAD-quality preview: the SAME watertight solid export_annular_
    # blade_step would write to STEP (lofted through every span station,
    # capped at hub/shroud -- see _close_loft_solid), meshed to STL and
    # rendered with VTK's own Phong shading via dash-vtk/vtk.js. This
    # replaces an earlier Plotly Mesh3d preview of just the blade curve
    # swept across span, which had no end caps and read as a hollow shell
    # rather than a solid.
    if not DASH_VTK_AVAILABLE:
        return _vtk_placeholder(
            "dash-vtk is not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")
    try:
        import vtk
        from turbo_moc.geometry import export_annular_blade_stl
    except ImportError:
        return _vtk_placeholder(
            "cadquery / vtk are not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")

    n_blades = int(sizing_result["n_blades"])
    # Show the WHOLE annulus, not just one blade -- cheaply: the cadquery
    # solid (the slow part, a loft + sew + cap) is built ONCE, then the
    # other copies are made by rotating the already-triangulated mesh about
    # the machine (Z) axis with vtkTransformPolyDataFilter and merging with
    # vtkAppendPolyData. That's a few hundred microseconds per copy, not
    # another CAD rebuild, so rendering all 40-70+ real blades costs about
    # the same as rendering one. n_render_cap is only a safety valve
    # against a mistyped pathological blade count, same role as the
    # n_preview cap on the older Plotly annular preview.
    n_render_cap = 200
    n_render = min(n_blades, n_render_cap)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "blade.stl")
            export_annular_blade_stl(
                sizing_result["scaled_blade"], sizing_result["r_hub"], sizing_result["r_shroud"],
                stl_path, n_blades=n_blades, flare="pitch_scale", source="stator",
            )
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()
            base_mesh = reader.GetOutput()

            append = vtk.vtkAppendPolyData()
            for i in range(n_render):
                transform = vtk.vtkTransform()
                transform.RotateZ(i * 360.0 / n_blades)
                tf = vtk.vtkTransformPolyDataFilter()
                tf.SetTransform(transform)
                tf.SetInputData(base_mesh)
                tf.Update()
                append.AddInputData(tf.GetOutput())
            append.Update()
            mesh_state = to_mesh_state(_triangulate_vtk(vtk, append.GetOutput()))
    except Exception as e:
        return _vtk_placeholder(f"Error building 3D solid preview: {e}")

    caption = (f"Showing {n_render} of {n_blades} blades (capped at {n_render_cap})."
               if n_render < n_blades else f"Showing all {n_blades} blades.")

    return html.Div(
        [
            html.Div(caption, style={"position": "absolute", "top": "8px", "left": "8px",
                                       "fontSize": "12px", "color": "#666", "zIndex": 1,
                                       "backgroundColor": "rgba(255,255,255,0.85)",
                                       "padding": "2px 8px", "borderRadius": "4px"}),
            dash_vtk.View(
                # A soft gray-blue viewport tone (not flat white) reads
                # closer to a real CAD viewport (SolidWorks/Fusion-style)
                # and gives the shaded solid some contrast to sit against.
                background=[0.93, 0.95, 0.97],
                style={"height": "100%", "width": "100%"},
                children=[
                    dash_vtk.GeometryRepresentation(
                        children=[dash_vtk.Mesh(state=mesh_state)],
                        property={
                            "color": [0.55, 0.63, 0.75],
                            "edgeVisibility": False,
                            "interpolation": "Phong",
                            # Lower ambient + higher specular than a flat
                            # preview: without real shadow mapping (not
                            # exposed by dash-vtk's high-level API), this is
                            # what reads as "CAD-shaded" rather than
                            # flat-lit -- faces turned away from the light
                            # go noticeably darker, giving the solid depth.
                            "ambient": 0.12, "diffuse": 0.8,
                            "specular": 0.45, "specularPower": 30,
                        },
                        # The reference frame the CAD view is missing: a
                        # bounding-box ruler with tick marks and axis
                        # labels (X / Y / Z, Z being the machine/axial
                        # axis per wrap_blade_annular's convention) around
                        # the solid, the same role a CAD viewer's
                        # coordinate triad plays.
                        showCubeAxes=True,
                        cubeAxesStyle={
                            "axisLabels": ["X [mm]", "Y [mm]", "Z (axial) [mm]"],
                        },
                    ),
                ],
            ),
        ],
        style={"position": "relative", "height": "100%", "width": "100%"},
    )


@app.callback(
    Output("download-sizing-step", "data"),
    Output("sizing-step-status", "children"),
    Input("download-sizing-step-btn", "n_clicks"),
    State("stator-sizing-store", "data"),
    prevent_initial_call=True,
)
def download_sizing_step(n_clicks, sizing_result):
    if not sizing_result:
        return dash.no_update, html.Div("Compute a sizing first.", style={"color": "#b00020"})
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from turbo_moc.geometry import export_annular_blade_step
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_annular_blade_step(
                sizing_result["scaled_blade"], sizing_result["r_hub"], sizing_result["r_shroud"],
                face_path, solid_path, n_blades=int(sizing_result["n_blades"]),
                flare="pitch_scale", source="stator",
            )
            with open(solid_path, "rb") as f:
                content = f.read()
    except ImportError:
        return dash.no_update, html.Div(
            "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

    return dcc.send_bytes(content, "stator_sized_blade.step"), html.Div(
        "Sized stator STEP ready (1 blade).", style={"color": "#1a7a1a"})


@app.callback(
    Output("cp_index", "options"),
    Output("cp_index", "value"),
    Output("blade-edited-store", "data", allow_duplicate=True),
    Output("cp-edit-status", "children", allow_duplicate=True),
    Input("blade-store", "data"),
    prevent_initial_call=True,
)
def reset_cp_editor(blade):
    """A fresh Compute invalidates any in-progress edits (they were nudges
    on top of the PREVIOUS fit) and repopulates the CP index dropdown for
    the new n_cp."""
    if not blade:
        return [], None, None, ""
    n_cp = len(blade["control_points"])
    options = [{"label": str(i), "value": i} for i in range(n_cp)]
    return options, 0, None, ""


@app.callback(
    Output("blade-edited-store", "data", allow_duplicate=True),
    Output("cp-edit-status", "children", allow_duplicate=True),
    Input("nudge-cp-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("cp_index", "value"),
    State("nudge_distance", "value"),
    prevent_initial_call=True,
)
def nudge_cp(n_clicks, base_blade, edited_blade, cp_index, distance):
    current = _effective_blade(base_blade, edited_blade)
    if not current:
        return dash.no_update, html.Div("Compute a blade first.", style={"color": "#b00020"})
    if cp_index is None:
        return dash.no_update, html.Div("Select a control point first.", style={"color": "#b00020"})
    try:
        new_blade = move_control_point_along_normal(current, int(cp_index), float(distance or 0.0))
    except Exception as e:
        return dash.no_update, html.Div(f"Error: {e}", style={"color": "#b00020"})
    status = html.Div(f"Moved CP {cp_index} by {distance} mm along its normal.",
                       style={"color": "#1a7a1a"})
    return new_blade, status


@app.callback(
    Output("blade-edited-store", "data", allow_duplicate=True),
    Output("cp-edit-status", "children", allow_duplicate=True),
    Input("reset-cp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_cp_edits(n_clicks):
    return None, html.Div("Control points reset to the fitted baseline.", style={"color": "#666"})


@app.callback(
    Output("blade-reference-store", "data"),
    Output("blade-reference-status", "children"),
    Input("pin-blade-btn", "n_clicks"),
    Input("clear-blade-ref-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    prevent_initial_call=True,
)
def manage_blade_reference(pin_clicks, clear_clicks, base_blade, edited_blade):
    if ctx.triggered_id == "clear-blade-ref-btn":
        return None, ""
    if ctx.triggered_id == "pin-blade-btn":
        blade = _effective_blade(base_blade, edited_blade)
        if not blade:
            return dash.no_update, html.Div(
                "Nothing to pin yet -- compute a blade first.", style={"color": "#b00020"})
        label = f"pitch={blade['pitch']:.1f} mm, n_cp={blade['n_cp']}"
        return blade, html.Div(f"Blade reference pinned: {label}", style={"color": "#1a7a1a"})
    return dash.no_update, dash.no_update


@app.callback(
    Output("fig-blade", "figure"),
    Input("blade-store", "data"),
    Input("blade-edited-store", "data"),
    Input("blade-reference-store", "data"),
    Input("n_blades_plot", "value"),
    Input("cp_index", "value"),
    Input("step-export-target", "value"),
    Input("passage_stator_inlet_chords", "value"),
    Input("passage_stator_outlet_chords", "value"),
)
def update_blade_plot(base_blade, edited_blade, reference, n_blades, cp_index,
                        step_export_target, inlet_chords, outlet_chords):
    blade = _effective_blade(base_blade, edited_blade)
    if not blade:
        return go.Figure()
    ref = reference if reference else None
    fig = turbo_moc.plotly.plot_blade(blade, n_blades=int(n_blades) if n_blades else 2,
                                  highlight_cp=cp_index, reference=ref)
    if step_export_target == "passage":
        # Preview of the passage this blade's own "Fluid domain (passage)"
        # STEP export would produce -- same helper (and the same inlet/
        # outlet-chord fields) as the actual export, so what's shown here
        # can never disagree with what gets downloaded. mirror=False: this
        # plot is the blade's own natural (unmirrored) orientation, unlike
        # Phase 4's cascade view.
        band_x, band_y = _stator_passage_band_xy(blade, inlet_chords, outlet_chords, mirror=False)
        fig.add_trace(go.Scatter(
            x=band_x + [band_x[0]], y=band_y + [band_y[0]],
            mode="lines", fill="toself",
            line=dict(color="black", width=1.5, dash="dot"),
            fillcolor="#2ecc71", opacity=0.25,
            name="Fluid domain", showlegend=True, hoverinfo="skip",
        ))
    return fig


@app.callback(
    Output("download-blade-plot", "data"),
    Input("blade-download-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("blade-reference-store", "data"),
    State("blade-save-format", "value"),
    State("n_blades_plot", "value"),
    prevent_initial_call=True,
)
def download_blade_plot(n_clicks, base_blade, edited_blade, reference, fmt, n_blades):
    blade = _effective_blade(base_blade, edited_blade)
    if not blade:
        return dash.no_update
    ref = reference if reference else None
    fig, _ax = turbo_moc.mpl.plot_blade(blade, n_blades=int(n_blades) if n_blades else 2, reference=ref)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"stator_blade.{fmt}")


@app.callback(
    Output("step-export-blade-fields", "style"),
    Output("step-export-passage-fields", "style"),
    Input("step-export-target", "value"),
)
def _toggle_step_export_target(target):
    if target == "passage":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


@app.callback(
    Output("download-step", "data"),
    Output("blade-step-status", "children"),
    Input("download-step-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("step-export-target", "value"),
    State("step-kind", "value"),
    State("extrude_length", "value"),
    State("passage_stator_inlet_chords", "value"),
    State("passage_stator_outlet_chords", "value"),
    prevent_initial_call=True,
)
def download_step(n_clicks, base_blade, edited_blade, target, kind, extrude_length,
                    inlet_chords, outlet_chords):
    blade = _effective_blade(base_blade, edited_blade)
    if not blade:
        return dash.no_update, html.Div("Compute a blade first.", style={"color": "#b00020"})

    cad_missing = html.Div(
        "cadquery is not installed -- STEP export unavailable "
        "(see environment.yaml / pyproject.toml's 'cad' extra).",
        style={"color": "#b00020"},
    )

    if target == "passage":
        extrude_length = float(extrude_length or 0.0)
        # 0 (or unset) thickness -> face only, matching export_blade_step's
        # own solid_path=None convention below: a 0-length extrusion isn't
        # a valid CAD solid, and skipping it is exactly what "just give me
        # the surface" means anyway, not an error to route around.
        want_solid = extrude_length > 0.0
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                from turbo_moc.geometry import extract_axial_passage_2d
                face_path = os.path.join(tmpdir, "face.step")
                solid_path = os.path.join(tmpdir, "solid.step") if want_solid else None
                extract_axial_passage_2d(
                    blade, blade["pitch"], face_path, solid_path,
                    extrude_length=extrude_length, source="stator",
                    inlet_chords=float(inlet_chords or 1.0),
                    outlet_chords=float(outlet_chords or 6.0),
                )
                target_path = solid_path if want_solid else face_path
                with open(target_path, "rb") as f:
                    content = f.read()
        except ImportError:
            return dash.no_update, cad_missing
        fname = "stator_passage_solid.step" if want_solid else "stator_passage_face.step"
        return dcc.send_bytes(content, fname), html.Div(
            "Fluid domain STEP ready.", style={"color": "#1a7a1a"})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from turbo_moc.geometry import export_blade_step
            face_path = os.path.join(tmpdir, "blade_face.step")
            solid_path = os.path.join(tmpdir, "blade_solid.step") if kind == "solid" else None
            export_blade_step(blade, face_path, solid_path, extrude_length=extrude_length or 1.0)
            target_path = solid_path if kind == "solid" else face_path
            with open(target_path, "rb") as f:
                content = f.read()
    except ImportError:
        return dash.no_update, cad_missing

    fname = "stator_blade_solid.step" if kind == "solid" else "stator_blade_face.step"
    return dcc.send_bytes(content, fname), html.Div("STEP file ready.", style={"color": "#1a7a1a"})


# --------------------------------------------------------------------------
# Phase 3 callbacks: rotor blade (vortex-flow method)
# --------------------------------------------------------------------------
@app.callback(
    Output("rotor_P0", "value"), Output("rotor_P0", "disabled"),
    Output("rotor_T0", "value"), Output("rotor_T0", "disabled"),
    Output("rotor_M_inlet", "value"), Output("rotor_M_inlet", "disabled"),
    Output("rotor_beta_inlet", "value"), Output("rotor_beta_inlet", "disabled"),
    Output("rotor_fluid_name", "value"), Output("rotor_fluid_name", "disabled"),
    Output("rotor-triangle-status", "children"),
    Input("couple_to_stator", "value"),
    Input("rotor_omega", "value"),
    Input("rotor_radius_mm", "value"),
    Input("blade-store", "data"),
    Input("blade-edited-store", "data"),
    Input("result-store", "data"),
    State("rotor_P0", "value"), State("rotor_T0", "value"),
    State("rotor_M_inlet", "value"), State("rotor_beta_inlet", "value"),
    State("rotor_fluid_name", "value"),
)
def _couple_rotor_inlet_to_stator(couple_vals, omega, r_mm, base_blade, edited_blade, nozzle_data,
                                   p0_cur, t0_cur, m_in_cur, beta_cur, fluid_cur):
    """When "Couple to stator exit" is checked, overwrite (and lock) the
    rotor's P0/T0/M_inlet/beta_inlet/fluid fields with values derived from
    the Phase 1/2 stator's own exit state + the chosen Omega/r, via
    derive_rotor_inlet_from_stator's velocity triangle -- rather than
    leaving them as independently hand-typed numbers that can silently
    drift out of consistency with the upstream design. Unchecked, the
    fields return to ordinary free (enabled) manual entry, same as today."""
    coupled = bool(couple_vals and "on" in couple_vals)
    if not coupled:
        return (p0_cur, False, t0_cur, False, m_in_cur, False, beta_cur, False,
                fluid_cur, False, "")

    stator = _effective_blade(base_blade, edited_blade)
    if not stator:
        return (p0_cur, True, t0_cur, True, m_in_cur, True, beta_cur, True, fluid_cur, True,
                html.Div("Compute a stator blade first (Phase 2).", style={"color": "#b00020"}))
    if not nozzle_data:
        return (p0_cur, True, t0_cur, True, m_in_cur, True, beta_cur, True, fluid_cur, True,
                html.Div("Run Phase 1 first.", style={"color": "#b00020"}))
    if not omega or not r_mm:
        return (p0_cur, True, t0_cur, True, m_in_cur, True, beta_cur, True, fluid_cur, True,
                html.Div("Enter Omega and r.", style={"color": "#b00020"}))

    U = float(omega) * (2.0 * np.pi / 60.0) * (float(r_mm) / 1000.0)
    alpha_deg = float(stator["metal_angle_out"])
    try:
        tri = derive_rotor_inlet_from_stator(nozzle_data, alpha_deg, U)
    except Exception as e:
        return (p0_cur, True, t0_cur, True, m_in_cur, True, beta_cur, True, fluid_cur, True,
                html.Div(f"Error: {e}", style={"color": "#b00020"}))

    # U is a free choice (Omega*r) with no guarantee it's anywhere near the
    # stator exit's own absolute velocity -- if it isn't, W_rel balloons
    # (mostly from the U-sized tangential term), M_rel follows it well past
    # any real supersonic-impulse/reaction design point, and the isentropic
    # reconstruction of P0_rel/T0_rel at that enthalpy can land outside the
    # fluid's valid EOS range entirely (a real crash two steps downstream,
    # in the rotor's OWN FluidManager setup, with a cryptic Brent-solver
    # traceback -- not obviously related to Omega/r at all from there).
    # Surfacing U vs V_abs here, before that happens, is the whole point.
    extreme = tri["M_rel"] > 3.0
    base_info = (f"V_abs={tri['V_abs']:.1f} m/s (M_abs={tri['M_abs']:.2f}), "
                 f"U={U:.1f} m/s -> M_rel={tri['M_rel']:.2f}, beta_rel={tri['beta_rel_deg']:.1f} deg.")
    if extreme:
        msg_color = "#b00020"
        msg = (base_info + " U is much larger than V_abs, which drives M_rel/P0_rel to "
               "physically extreme values (may crash the rotor solve below) -- reduce "
               "Omega or r.")
    elif not tri["is_supersonic"]:
        msg_color = "#b00020"
        msg = (base_info + " Relative inlet is SUBSONIC -- the vortex-flow method requires "
               "supersonic relative inflow; increase U (Omega/r) or check alpha.")
    else:
        msg_color = "#1a7a1a"
        msg = base_info + " Relative inlet is supersonic, OK for this method."

    return (
        tri["P0_rel"], True, tri["T0_rel"], True, tri["M_rel"], True,
        tri["beta_rel_deg"], True, nozzle_data["fluid_name"], True,
        html.Div(msg, style={"color": msg_color, "marginTop": "4px"}),
    )


@app.callback(
    Output("rotor-store", "data"),
    Output("rotor-status-message", "children"),
    Output("rotor-info-table", "children"),
    Input("compute-rotor-btn", "n_clicks"),
    State("rotor_fluid_name", "value"),
    State("rotor_P0", "value"),
    State("rotor_T0", "value"),
    State("rotor_M_inlet", "value"),
    State("rotor_M_outlet", "value"),
    State("rotor_M_lower", "value"),
    State("rotor_M_upper", "value"),
    State("rotor_beta_inlet", "value"),
    State("rotor_num_points", "value"),
    prevent_initial_call=True,
)
def run_rotor(n_clicks, fluid_name, P0, T0, M_inlet, M_outlet, M_lower, M_upper,
              beta_inlet, num_points):
    try:
        data = design_rotor_vortex_blade(
            fluid_name=fluid_name, P0_rel=P0, T0_rel=T0,
            M_inlet=M_inlet, M_outlet=M_outlet, M_lower=M_lower, M_upper=M_upper,
            beta_inlet=beta_inlet,
            backend="HEOS", num_points=int(num_points),
        )
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    status = html.Div([
        html.Span("Blade computed", style={"fontWeight": "600", "color": "#1a7a1a"}),
    ])
    table_data = [
        {"Quantity": "Pitch", "Value": f"{data['pitch']:.4f}", "Unit": data["units"]},
        {"Quantity": "Chord", "Value": f"{data['chord']:.4f}", "Unit": data["units"]},
        {"Quantity": "Solidity", "Value": f"{data['solidity']:.4f}", "Unit": "-"},
        {"Quantity": "Beta inlet", "Value": f"{data['beta_inlet']:.2f}", "Unit": "deg"},
        {"Quantity": "Beta outlet", "Value": f"{data['beta_outlet']:.2f}", "Unit": "deg"},
    ]
    return data, status, make_table(table_data, ["Quantity", "Value", "Unit"])


@app.callback(
    Output("rotor-reference-store", "data"),
    Output("rotor-reference-status", "children"),
    Input("pin-rotor-btn", "n_clicks"),
    Input("clear-rotor-ref-btn", "n_clicks"),
    State("rotor-store", "data"),
    prevent_initial_call=True,
)
def manage_rotor_reference(pin_clicks, clear_clicks, current_data):
    if ctx.triggered_id == "clear-rotor-ref-btn":
        return None, ""
    if ctx.triggered_id == "pin-rotor-btn":
        if not current_data:
            return dash.no_update, html.Div(
                "Nothing to pin yet -- compute a rotor blade first.", style={"color": "#b00020"})
        label = f"{current_data['fluid_name']}, pitch={current_data['pitch']:.3f}"
        return current_data, html.Div(f"Reference pinned: {label}", style={"color": "#1a7a1a"})
    return dash.no_update, dash.no_update


@app.callback(
    Output("fig-rotor-blade", "figure"),
    Input("rotor-store", "data"),
    Input("rotor-reference-store", "data"),
    Input("rotor-display-options", "value"),
)
def update_rotor_plot(data, reference, display_options):
    if not data:
        return go.Figure()
    ref = reference if reference else None
    opts = display_options or []
    return turbo_moc.plotly.plot_rotor_vortex_blade(
        data, show_surfaces=("surfaces" in opts), show_mach_lines=("mach_lines" in opts),
        reference=ref,
    )


@app.callback(
    Output("download-rotor-plot", "data"),
    Input("rotor-download-btn", "n_clicks"),
    State("rotor-store", "data"),
    State("rotor-reference-store", "data"),
    State("rotor-display-options", "value"),
    State("rotor-save-format", "value"),
    prevent_initial_call=True,
)
def download_rotor_plot(n_clicks, data, reference, display_options, fmt):
    if not data:
        return dash.no_update
    ref = reference if reference else None
    opts = display_options or []
    fig, _ax = turbo_moc.mpl.plot_rotor_vortex_blade(
        data, show_surfaces=("surfaces" in opts), show_mach_lines=("mach_lines" in opts),
        reference=ref,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"rotor_vortex_blade.{fmt}")


# --------------------------------------------------------------------------
# Phase 5 callbacks: 3D rotor design -- mirrors Phase 3's own
# run_stator_sizing / _update_sizing_vtk_preview / download_sizing_step
# (see those for the pattern this repeats almost line for line).
# --------------------------------------------------------------------------
@app.callback(
    Output("rotor_axial_fields", "style"),
    Output("rotor_radial_fields", "style"),
    Output("rotor_axial_plots", "style"),
    Output("rotor_radial_plots", "style"),
    Input("rotor_wrap_type", "value"),
)
def _toggle_rotor_wrap_type(wrap_type):
    shown, hidden = {"display": "block"}, {"display": "none"}
    return (shown, hidden, shown, hidden) if wrap_type == "axial" else (hidden, shown, hidden, shown)


@app.callback(
    Output("rotor-sizing-annulus-readout", "children"),
    Input("stator-sizing-store", "data"),
)
def _update_rotor_sizing_annulus_readout(stator_sizing):
    if not stator_sizing:
        return html.Div("Run Phase 3 (stator sizing) first -- the rotor shares its annulus.",
                          style={"color": "#b00020"})
    return html.Div(
        f"Annulus from Phase 3: r_hub={stator_sizing['r_hub']:.2f} mm, "
        f"r_shroud={stator_sizing['r_shroud']:.2f} mm.")


@app.callback(
    Output("rotor-sizing-store", "data"),
    Output("rotor-sizing-status-message", "children"),
    Output("rotor-sizing-info-table", "children"),
    Input("compute-rotor-sizing-btn", "n_clicks"),
    State("rotor-store", "data"),
    State("stator-sizing-store", "data"),
    State("rotor_sizing_n_blades", "value"),
    prevent_initial_call=True,
)
def run_rotor_sizing(n_clicks, rotor_data, stator_sizing, n_blades):
    if not rotor_data:
        return None, html.Div("Run Phase 4 first -- no rotor blade to size.",
                                style={"color": "#b00020"}), None
    if not stator_sizing:
        return None, html.Div("Run Phase 3 (stator sizing) first -- the rotor "
                                "shares its annulus.", style={"color": "#b00020"}), None

    try:
        result = size_rotor_from_pitch(
            rotor_data, stator_sizing["r_hub"], stator_sizing["r_shroud"], int(n_blades))
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    table_data = [
        {"Quantity": "k (scale factor)", "Value": f"{result['k']:.2f}", "Unit": "-"},
        {"Quantity": "Pitch (scaled)", "Value": f"{result['pitch_scaled']:.2f}", "Unit": "mm"},
        {"Quantity": "r_hub", "Value": f"{result['r_hub']:.2f}", "Unit": "mm"},
        {"Quantity": "r_shroud", "Value": f"{result['r_shroud']:.2f}", "Unit": "mm"},
        {"Quantity": "N_blades", "Value": str(result["n_blades"]), "Unit": "-"},
        {"Quantity": "beta_inlet", "Value": f"{rotor_data['beta_inlet']:.2f}", "Unit": "deg"},
        {"Quantity": "beta_outlet", "Value": f"{rotor_data['beta_outlet']:.2f}", "Unit": "deg"},
    ]
    status = html.Div("Sizing computed.", style={"color": "#1a7a1a", "fontWeight": "600"})
    return result, status, make_table(table_data, ["Quantity", "Value", "Unit"])


@app.callback(
    Output("rotor-sizing-vtk-container", "children"),
    Input("rotor-sizing-store", "data"),
)
def _update_rotor_sizing_vtk_preview(sizing_result):
    if not sizing_result:
        return _vtk_placeholder("Compute a sizing to see the 3D solid preview.")

    if not DASH_VTK_AVAILABLE:
        return _vtk_placeholder(
            "dash-vtk is not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")
    try:
        import vtk
        from turbo_moc.geometry import export_annular_blade_stl
    except ImportError:
        return _vtk_placeholder(
            "cadquery / vtk are not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")

    n_blades = int(sizing_result["n_blades"])
    n_render_cap = 200
    n_render = min(n_blades, n_render_cap)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "blade.stl")
            export_annular_blade_stl(
                _mirrored_rotor_blade(sizing_result["scaled_blade"]), sizing_result["r_hub"], sizing_result["r_shroud"],
                stl_path, n_blades=n_blades, flare="pitch_scale", source="rotor",
            )
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()
            base_mesh = reader.GetOutput()

            append = vtk.vtkAppendPolyData()
            for i in range(n_render):
                transform = vtk.vtkTransform()
                transform.RotateZ(i * 360.0 / n_blades)
                tf = vtk.vtkTransformPolyDataFilter()
                tf.SetTransform(transform)
                tf.SetInputData(base_mesh)
                tf.Update()
                append.AddInputData(tf.GetOutput())
            append.Update()
            mesh_state = to_mesh_state(_triangulate_vtk(vtk, append.GetOutput()))
    except Exception as e:
        return _vtk_placeholder(f"Error building 3D solid preview: {e}")

    caption = (f"Showing {n_render} of {n_blades} blades (capped at {n_render_cap})."
               if n_render < n_blades else f"Showing all {n_blades} blades.")

    return html.Div(
        [
            html.Div(caption, style={"position": "absolute", "top": "8px", "left": "8px",
                                       "fontSize": "12px", "color": "#666", "zIndex": 1,
                                       "backgroundColor": "rgba(255,255,255,0.85)",
                                       "padding": "2px 8px", "borderRadius": "4px"}),
            dash_vtk.View(
                background=[0.93, 0.95, 0.97],
                style={"height": "100%", "width": "100%"},
                children=[
                    dash_vtk.GeometryRepresentation(
                        children=[dash_vtk.Mesh(state=mesh_state)],
                        property={
                            "color": [0.55, 0.63, 0.75],
                            "edgeVisibility": False,
                            "interpolation": "Phong",
                            "ambient": 0.12, "diffuse": 0.8,
                            "specular": 0.45, "specularPower": 30,
                        },
                        showCubeAxes=True,
                        cubeAxesStyle={
                            "axisLabels": ["X [mm]", "Y [mm]", "Z (axial) [mm]"],
                        },
                    ),
                ],
            ),
        ],
        style={"position": "relative", "height": "100%", "width": "100%"},
    )


@app.callback(
    Output("rotor-radial-store", "data"),
    Output("rotor-radial-status-message", "children"),
    Output("rotor-radial-info-table", "children"),
    Input("compute-rotor-radial-btn", "n_clicks"),
    State("rotor-store", "data"),
    State("rotor_radial_r1", "value"),
    State("rotor_radial_r2", "value"),
    State("rotor_radial_n_blades", "value"),
    State("rotor_radial_scale", "value"),
    State("rotor_radial_depth", "value"),
    prevent_initial_call=True,
)
def run_radial_rotor(n_clicks, rotor_data, r1, r2, n_blades, scale, depth):
    """Geometric-only radial wrap (see run_radial_stator's docstring for
    why N_blades is a direct input, not solved, for the rotor too)."""
    from turbo_moc.geometry import implied_n_blades_radial, mapped_pitch_width

    if not rotor_data:
        return None, html.Div("Run Phase 4 first -- no rotor blade to wrap.",
                                style={"color": "#b00020"}), None

    r1, r2 = float(r1), float(r2)
    n_blades, scale, depth = int(n_blades), float(scale), float(depth)

    # implied_n_blades_radial/mapped_pitch_width read blade.x/y and pitch
    # directly, with no separate scale factor -- apply `scale` first so
    # the diagnostic matches the geometry that actually gets wrapped/
    # rendered (same scaled copy _update_rotor_radial_vtk_preview builds).
    scaled_rotor = dict(rotor_data)
    scaled_rotor["blade"] = {
        "x": [v * scale for v in rotor_data["blade"]["x"]],
        "y": [v * scale for v in rotor_data["blade"]["y"]],
    }
    scaled_rotor["pitch"] = float(rotor_data["pitch"]) * scale

    try:
        mapped_pitch = mapped_pitch_width(scaled_rotor, r1, r2, source="rotor")
        n_implied = implied_n_blades_radial(scaled_rotor, r1, r2, source="rotor")
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    result = {
        "r1": r1, "r2": r2, "n_blades": n_blades, "scale": scale, "depth": depth,
        "rotor_data": rotor_data,
    }
    table_data = [
        {"Quantity": "r1 (hub)", "Value": f"{r1:.2f}", "Unit": "mm"},
        {"Quantity": "r2 (shroud)", "Value": f"{r2:.2f}", "Unit": "mm"},
        {"Quantity": "N_blades (entered)", "Value": str(n_blades), "Unit": "-"},
        {"Quantity": "Scale", "Value": f"{scale:.4f}", "Unit": "mm / r*"},
        {"Quantity": "Mapped pitch (at N entered)", "Value": f"{mapped_pitch:.2f}", "Unit": "mm"},
        {"Quantity": "Implied N_blades (no overlap/gaps)", "Value": f"{n_implied:.1f}", "Unit": "-"},
        {"Quantity": "Depth", "Value": f"{depth:.2f}", "Unit": "mm"},
    ]
    ratio = n_blades / n_implied
    if ratio > 1.15:
        msg = (f"Radial wrap computed -- N_blades ({n_blades}) is {ratio:.1f}x the implied "
               f"count ({n_implied:.1f}): blades will visibly OVERLAP. Lower N_blades.")
        color = "#b00020"
    elif ratio < 0.85:
        msg = (f"Radial wrap computed -- N_blades ({n_blades}) is only {ratio:.1f}x the "
               f"implied count ({n_implied:.1f}): large GAPS between blades. Raise N_blades.")
        color = "#b00020"
    else:
        msg = "Radial wrap computed -- N_blades is close to the implied (no-overlap) count."
        color = "#1a7a1a"
    status = html.Div(msg, style={"color": color, "fontWeight": "600"})
    return result, status, make_table(table_data, ["Quantity", "Value", "Unit"])


@app.callback(
    Output("rotor-radial-vtk-container", "children"),
    Input("rotor-radial-store", "data"),
)
def _update_rotor_radial_vtk_preview(radial_result):
    if not radial_result:
        return _vtk_placeholder("Compute a radial wrap to see the 3D preview.")
    if not DASH_VTK_AVAILABLE:
        return _vtk_placeholder(
            "dash-vtk is not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")
    try:
        import vtk
        from turbo_moc.geometry import wrap_rotor_blade_radial
    except ImportError:
        return _vtk_placeholder("vtk is not installed -- 3D solid preview unavailable.")

    try:
        wrapped = wrap_rotor_blade_radial(
            radial_result["rotor_data"], radial_result["r1"], radial_result["r2"],
            radial_result["n_blades"], scale=radial_result["scale"])
        append = vtk.vtkAppendPolyData()
        for blade_copy in wrapped["blades"]:
            append.AddInputData(_extrude_polygon_vtk(
                vtk, blade_copy["x"], blade_copy["y"], radial_result["depth"]))
        append.Update()
        mesh_state = to_mesh_state(_triangulate_vtk(vtk, append.GetOutput()))
    except Exception as e:
        return _vtk_placeholder(f"Error building 3D solid preview: {e}")

    return html.Div(
        [
            html.Div(f"Showing all {radial_result['n_blades']} blades (radial wrap).",
                      style={"position": "absolute", "top": "8px", "left": "8px",
                             "fontSize": "12px", "color": "#666", "zIndex": 1,
                             "backgroundColor": "rgba(255,255,255,0.85)",
                             "padding": "2px 8px", "borderRadius": "4px"}),
            dash_vtk.View(
                background=[0.93, 0.95, 0.97],
                style={"height": "100%", "width": "100%"},
                children=[
                    dash_vtk.GeometryRepresentation(
                        children=[dash_vtk.Mesh(state=mesh_state)],
                        property={
                            "color": [0.85, 0.55, 0.45], "edgeVisibility": False,
                            "interpolation": "Phong",
                            "ambient": 0.12, "diffuse": 0.8, "specular": 0.45, "specularPower": 30,
                        },
                        showCubeAxes=True,
                        cubeAxesStyle={"axisLabels": ["X [mm]", "Y [mm]", "Z (axial) [mm]"]},
                    ),
                ],
            ),
        ],
        style={"position": "relative", "height": "100%", "width": "100%"},
    )


@app.callback(
    Output("download-rotor-sizing-step", "data"),
    Output("rotor-sizing-step-status", "children"),
    Input("download-rotor-sizing-step-btn", "n_clicks"),
    State("rotor-sizing-store", "data"),
    prevent_initial_call=True,
)
def download_rotor_sizing_step(n_clicks, sizing_result):
    if not sizing_result:
        return dash.no_update, html.Div("Compute a sizing first.", style={"color": "#b00020"})
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from turbo_moc.geometry import export_annular_blade_step
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_annular_blade_step(
                _mirrored_rotor_blade(sizing_result["scaled_blade"]), sizing_result["r_hub"], sizing_result["r_shroud"],
                face_path, solid_path, n_blades=int(sizing_result["n_blades"]),
                flare="pitch_scale", source="rotor",
            )
            with open(solid_path, "rb") as f:
                content = f.read()
    except ImportError:
        return dash.no_update, html.Div(
            "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

    return dcc.send_bytes(content, "rotor_sized_blade.step"), html.Div(
        "Sized rotor STEP ready (1 blade).", style={"color": "#1a7a1a"})


# --------------------------------------------------------------------------
# Phase 6 callbacks: cascade (stator + rotor shown together)
# --------------------------------------------------------------------------
def _rotor_z_offset(stator_sizing, rotor_sizing, gap):
    """Axial offset [mm] for the rotor row so its INLET sits `gap`
    downstream of the stator's own OUTLET (TE).

    Verified directly (not by eye) from design_rotor_vortex_blade's own
    internal construction: a debug print of _surface()'s x_inlet_far/
    x_outlet_far (added temporarily, then removed) showed the rotor's
    local x INCREASES from inlet (negative x) to outlet (positive x).
    The stator's blade_curve.y runs the opposite way -- suction[0] (the
    actual inlet, start of the camberline-from-inlet-to-throat
    integration) sits at the LARGE-y end, suction[-1]/the TE (outlet) at
    the very-negative-y end -- so this app's shared Z convention is
    "increasing coordinate = upstream" and stator_z_min = min(blade_curve.y)
    is the stator's own OUTLET, the correct anchor point. Negating rotor
    x before using it (rotor_z_raw below) makes the rotor's inlet land at
    rotor_z_raw's MAX, consistent with that same convention -- anchoring
    on max(rotor_z_raw) is therefore correct, and was NOT the bug.

    The actual bug (two min/max flip-flops never touched it): this
    negation only happens here and in the 2D cascade plot -- the 3D
    solid pipeline (wrap_blade_annular/_build_annular_blade_solid) uses
    blade.x RAW, unnegated, then just adds z_offset with no sign logic
    (by design, see export_annular_blade_step's z_offset docstring). Any
    caller that feeds this function's return value into that 3D pipeline
    MUST first negate the rotor blade's x the same way, via
    _mirrored_rotor_blade, or the two halves disagree and the rotor's
    OUTLET lands next to the stator instead of its inlet -- exactly the
    symptom reported."""
    stator_y = stator_sizing["scaled_blade"]["blade_curve"]["y"]
    stator_z_min = min(stator_y)
    rotor_z_raw = [-v for v in rotor_sizing["scaled_blade"]["blade"]["x"]]
    return stator_z_min - float(gap) - max(rotor_z_raw)


def _mirrored_rotor_blade(scaled_blade):
    """Copy of a rotor's scaled_blade with blade.x negated -- needed
    anywhere a z_offset from _rotor_z_offset feeds the 3D annular-wrap
    pipeline (wrap_blade_annular/_build_annular_blade_solid), since that
    pipeline uses blade.x raw/unnegated for Z while _rotor_z_offset (and
    the 2D cascade plot) compute in -x space. See _rotor_z_offset's
    docstring for the full explanation. Only blade.x/y are read by the
    wrap, so copying just "blade" is sufficient."""
    blade = scaled_blade["blade"]
    mirrored = dict(scaled_blade)
    mirrored["blade"] = {**blade, "x": [-v for v in blade["x"]]}
    return mirrored


@app.callback(
    Output("cascade-assembly-readout", "children"),
    Input("stator-sizing-store", "data"),
    Input("rotor-sizing-store", "data"),
)
def _update_cascade_readout(stator_sizing, rotor_sizing):
    missing = []
    if not stator_sizing:
        missing.append("Phase 3 (stator sizing)")
    if not rotor_sizing:
        missing.append("Phase 5 (rotor sizing)")
    if missing:
        return html.Div(f"Run {' and '.join(missing)} first.", style={"color": "#b00020"})
    return html.Div(
        f"Stator: r_hub={stator_sizing['r_hub']:.2f} mm, N={stator_sizing['n_blades']}. "
        f"Rotor: r_hub={rotor_sizing['r_hub']:.2f} mm, N={rotor_sizing['n_blades']}.")


@app.callback(
    Output("fig-cascade-2d", "figure"),
    Input("stator-sizing-store", "data"),
    Input("rotor-sizing-store", "data"),
    Input("cascade_axial_gap", "value"),
    Input("cascade_2d_n_stator", "value"),
    Input("cascade_2d_n_rotor", "value"),
)
def _update_cascade_2d_plot(stator_sizing, rotor_sizing, gap, n_stator, n_rotor):
    """A flat 'blade-to-blade' cascade view at mid-span (r_mid = (r_hub +
    r_shroud)/2, the two rows share one annulus): N_stator/N_rotor copies
    of each row's already-sized 2D blade shape, tiled at the mid-span
    pitch and positioned axially with the same gap as the 3D view -- the
    annular wrap's own "pitch_scale" convention only stretches the
    PITCHWISE coordinate with radius (the chordwise/axial one is
    unchanged across span, see turbo_moc.geometry.annular's docstring),
    so this is genuinely "the same as putting together the 2D shapes" at
    a different pitch, not a 3D slice."""
    fig = go.Figure()
    if not stator_sizing or not rotor_sizing:
        fig.update_layout(annotations=[dict(text="Run Phase 3 and Phase 5 sizing first.",
                                              showarrow=False, font=dict(size=14))])
        return fig

    r_hub = stator_sizing["r_hub"]
    r_mid = 0.5 * (r_hub + stator_sizing["r_shroud"])

    stator_blade = stator_sizing["scaled_blade"]
    pitch_mid_s = stator_blade["pitch"] * (r_mid / r_hub)
    rotor_blade = rotor_sizing["scaled_blade"]
    pitch_mid_r = rotor_blade["pitch"] * (r_mid / rotor_sizing["r_hub"])

    z_offset = _rotor_z_offset(stator_sizing, rotor_sizing, gap or 0.0)

    n_stator = int(n_stator or 1)
    n_rotor = int(n_rotor or 1)

    # Stator: x = pitchwise, y = chordwise/axial (unchanged).
    curve, te = stator_blade["blade_curve"], stator_blade["trailing_edge"]
    for i in range(n_stator):
        dx = i * pitch_mid_s
        showlegend = (i == 0)
        fig.add_trace(go.Scatter(x=[v + dx for v in curve["x"]], y=curve["y"], mode="lines",
                                  line=dict(color="#3d5a80", width=2), name="Stator",
                                  legendgroup="stator", showlegend=showlegend))
        fig.add_trace(go.Scatter(x=[v + dx for v in te["x"]], y=te["y"], mode="lines",
                                  line=dict(color="#3d5a80", width=2),
                                  legendgroup="stator", showlegend=False))

    # Rotor: local axes are chordwise=x, pitchwise=y -- remapped onto the
    # SAME plot axes as the stator (plot_x = pitchwise, plot_y = axial),
    # with the chordwise value negated + shifted by z_offset, matching
    # the 3D view's own axial-orientation convention exactly.
    blade = rotor_blade["blade"]
    for i in range(n_rotor):
        dy = i * pitch_mid_r
        showlegend = (i == 0)
        plot_x = [v + dy for v in blade["y"]]
        plot_y = [-v + z_offset for v in blade["x"]]
        fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="lines",
                                  line=dict(color="#ee6c4d", width=2), name="Rotor",
                                  legendgroup="rotor", showlegend=showlegend))

    fig.update_layout(
        xaxis_title="Pitchwise (mm)", yaxis_title="Axial (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=f"Mid-span cascade (r_mid={r_mid:.1f} mm)",
    )
    return fig


@app.callback(
    Output("cascade-vtk-container", "children"),
    Input("stator-sizing-store", "data"),
    Input("rotor-sizing-store", "data"),
    Input("cascade_axial_gap", "value"),
)
def _update_cascade_vtk_preview(stator_sizing, rotor_sizing, gap):
    if not stator_sizing or not rotor_sizing:
        return _vtk_placeholder("Run Phase 3 and Phase 5 sizing first to see the assembly.")
    if not DASH_VTK_AVAILABLE:
        return _vtk_placeholder(
            "dash-vtk is not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")
    try:
        import vtk
        from turbo_moc.geometry import export_annular_blade_stl
    except ImportError:
        return _vtk_placeholder(
            "cadquery / vtk are not installed -- 3D solid preview unavailable "
            "(pip install \"turbo_moc[cad]\").")

    n_render_cap = 200

    def _row_mesh_state(sizing_result, source, z_offset, tmpdir, name):
        n_blades = int(sizing_result["n_blades"])
        n_render = min(n_blades, n_render_cap)
        stl_path = os.path.join(tmpdir, name)
        export_annular_blade_stl(
            sizing_result["scaled_blade"], sizing_result["r_hub"], sizing_result["r_shroud"],
            stl_path, n_blades=n_blades, flare="pitch_scale", source=source, z_offset=z_offset,
        )
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        base_mesh = reader.GetOutput()

        append = vtk.vtkAppendPolyData()
        for i in range(n_render):
            transform = vtk.vtkTransform()
            transform.RotateZ(i * 360.0 / n_blades)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(transform)
            tf.SetInputData(base_mesh)
            tf.Update()
            append.AddInputData(tf.GetOutput())
        append.Update()
        return to_mesh_state(_triangulate_vtk(vtk, append.GetOutput()))

    try:
        z_offset = _rotor_z_offset(stator_sizing, rotor_sizing, gap or 0.0)
        rotor_sizing_3d = {**rotor_sizing, "scaled_blade": _mirrored_rotor_blade(rotor_sizing["scaled_blade"])}
        with tempfile.TemporaryDirectory() as tmpdir:
            stator_mesh_state = _row_mesh_state(stator_sizing, "stator", 0.0, tmpdir, "stator.stl")
            rotor_mesh_state = _row_mesh_state(rotor_sizing_3d, "rotor", z_offset, tmpdir, "rotor.stl")
    except Exception as e:
        return _vtk_placeholder(f"Error building 3D solid preview: {e}")

    return dash_vtk.View(
        background=[0.93, 0.95, 0.97],
        style={"height": "100%", "width": "100%"},
        children=[
            dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=stator_mesh_state)],
                property={
                    "color": [0.55, 0.63, 0.75], "edgeVisibility": False,
                    "interpolation": "Phong",
                    "ambient": 0.12, "diffuse": 0.8, "specular": 0.45, "specularPower": 30,
                },
                showCubeAxes=True,
                cubeAxesStyle={"axisLabels": ["X [mm]", "Y [mm]", "Z (axial) [mm]"]},
            ),
            dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=rotor_mesh_state)],
                property={
                    "color": [0.85, 0.55, 0.45], "edgeVisibility": False,
                    "interpolation": "Phong",
                    "ambient": 0.12, "diffuse": 0.8, "specular": 0.45, "specularPower": 30,
                },
            ),
        ],
    )


@app.callback(
    Output("download-cascade-stator-step", "data"),
    Output("cascade-step-status", "children", allow_duplicate=True),
    Input("download-cascade-stator-step-btn", "n_clicks"),
    State("stator-sizing-store", "data"),
    prevent_initial_call=True,
)
def download_cascade_stator_step(n_clicks, stator_sizing):
    if not stator_sizing:
        return dash.no_update, html.Div("Run Phase 3 (stator sizing) first.", style={"color": "#b00020"})
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from turbo_moc.geometry import export_annular_blade_step
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_annular_blade_step(
                stator_sizing["scaled_blade"], stator_sizing["r_hub"], stator_sizing["r_shroud"],
                face_path, solid_path, n_blades=int(stator_sizing["n_blades"]),
                flare="pitch_scale", source="stator", z_offset=0.0,
            )
            with open(solid_path, "rb") as f:
                content = f.read()
    except ImportError:
        return dash.no_update, html.Div(
            "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

    return dcc.send_bytes(content, "cascade_stator.step"), html.Div(
        "Stator STEP ready (assembly position).", style={"color": "#1a7a1a"})


@app.callback(
    Output("download-cascade-rotor-step", "data"),
    Output("cascade-step-status", "children", allow_duplicate=True),
    Input("download-cascade-rotor-step-btn", "n_clicks"),
    State("stator-sizing-store", "data"),
    State("rotor-sizing-store", "data"),
    State("cascade_axial_gap", "value"),
    prevent_initial_call=True,
)
def download_cascade_rotor_step(n_clicks, stator_sizing, rotor_sizing, gap):
    if not stator_sizing or not rotor_sizing:
        return dash.no_update, html.Div("Run Phase 3 and Phase 5 sizing first.", style={"color": "#b00020"})
    try:
        z_offset = _rotor_z_offset(stator_sizing, rotor_sizing, gap or 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            from turbo_moc.geometry import export_annular_blade_step
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_annular_blade_step(
                _mirrored_rotor_blade(rotor_sizing["scaled_blade"]), rotor_sizing["r_hub"], rotor_sizing["r_shroud"],
                face_path, solid_path, n_blades=int(rotor_sizing["n_blades"]),
                flare="pitch_scale", source="rotor", z_offset=z_offset,
            )
            with open(solid_path, "rb") as f:
                content = f.read()
    except ImportError:
        return dash.no_update, html.Div(
            "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

    return dcc.send_bytes(content, "cascade_rotor.step"), html.Div(
        "Rotor STEP ready (assembly position).", style={"color": "#1a7a1a"})


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
