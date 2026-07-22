"""
moc.app -- interactive Dash app for MOC nozzle design.

Structured after turbodash's app.py: one Dash app, a compute callback that
calls a single engine function (here, moc.design_nozzle) and stores its
result in a dcc.Store, and a second callback that fans that result out into
plots. The one deliberate departure from turbodash's pattern: the compute
callback runs in the background (background=True + DiskcacheManager) with
a live progress bar, since a solve here takes 5-30s+ (turbodash's own
algebraic core is sub-second, so it never needed this) -- an explicit
"Compute" button, not live recompute-on-slider-drag, is the intended UX.

Plots are displayed with moc.plotly (interactive dcc.Graph -- zoom, pan,
hover, and a native download-as-PNG toolbar button, same as turbodash's own
app.py). moc.plotly's figures are styled (see plotting_plotly._apply_
journal_style) to read close to moc.mpl's boxed-axes/gridline look rather
than Plotly's own default template, since visual consistency with
examples/plot_design.py's output mattered too -- same colors either way,
only the interactive vs. static trade-off differs. The "Download this
plot" button + format dropdown (PNG/SVG/PDF) is separate from Plotly's own
toolbar button (PNG only): it re-renders the SAME figure with moc.mpl on
click, so an SVG/PDF download is a real vector file, not a rasterized PNG
relabeled -- display and file-export are genuinely two different renders
of the same data, not one converted into the other.

Run locally:
    conda activate moc_env
    python -m moc.app
    (or: moc-app, the console-script entry point from pyproject.toml)
then open the printed http://127.0.0.1:8050/ URL in a browser.
"""

import io
import os
import tempfile

import diskcache
import dash
from dash import Dash, html, dcc, Input, Output, State, ctx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import moc
from moc import design_nozzle, list_supported_fluids, make_progress_reporter, NozzleDesignError
from moc.geometry import (
    move_control_point_along_normal,
    parametrize_stator_blade,
    parametrize_stator_blade_semi,
)
from moc.rotor import design_rotor_vortex_blade

BLADE_PARAM_FUNCS = {
    "full": parametrize_stator_blade,
    "semi": parametrize_stator_blade_semi,
}

# Tab key -> label, and the two per-tab plot functions: the interactive
# Plotly one (display) and the matplotlib one + filename base (download).
# Adding a new plot tab only means adding one entry to each of these three.
TAB_LABELS = {
    "contour": "Nozzle contour", "mesh": "Characteristics mesh",
    "axis": "Axis properties", "wall": "Wall properties", "ts": "T-s diagram",
}
PLOT_FUNCS_PLOTLY = {
    "contour": moc.plotly.plot_nozzle_contour,
    "mesh": moc.plotly.plot_characteristics_mesh,
    "axis": moc.plotly.plot_axis_properties,
    "wall": moc.plotly.plot_wall_properties,
    "ts": moc.plotly.plot_ts_diagram,
}
PLOT_FUNCS_MPL = {
    "contour": (moc.mpl.plot_nozzle_contour, "nozzle_contour"),
    "mesh": (moc.mpl.plot_characteristics_mesh, "characteristics_mesh"),
    "axis": (moc.mpl.plot_axis_properties, "axis_properties"),
    "wall": (moc.mpl.plot_wall_properties, "wall_properties"),
    "ts": (moc.mpl.plot_ts_diagram, "ts_diagram"),
}

# --- Background callback manager (see module docstring for why) ---------
cache = diskcache.Cache("./.moc_app_cache")
background_callback_manager = dash.DiskcacheManager(cache)

app = Dash(__name__, background_callback_manager=background_callback_manager)
app.title = "MOC Nozzle Design"
server = app.server  # WSGI app, for gunicorn (matches turbodash's Procfile convention)

FLUIDS = list_supported_fluids()
DEFAULT_FLUID = "Nitrogen" if "Nitrogen" in FLUIDS else FLUIDS[0]

DEFAULTS = dict(
    P0=20e5, T0=113.0, Q0=0.5, p_back=2e5,
    y_t=0.01, n=15, rho_t=0.2, rho_d=0.1,
)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _field(label, id, value, **kwargs):
    return html.Div(
        [
            html.Label(label, style={"fontSize": "13px", "fontWeight": "600", "display": "block"}),
            dcc.Input(id=id, value=value, type="number", style={"width": "100%"}, **kwargs),
        ],
        style={"marginBottom": "10px"},
    )


controls = html.Div(
    [
        html.H4("Fluid & inlet"),
        dcc.Dropdown(id="fluid_name", options=FLUIDS, value=DEFAULT_FLUID, clearable=False),
        html.Br(),
        dcc.RadioItems(
            id="inlet_mode",
            options=[
                {"label": " Subcooled liquid / single-phase (P0, T0)", "value": "T0"},
                {"label": " Saturated two-phase (P0, Q0)", "value": "Q0"},
            ],
            value="T0",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field("P0 (Pa)", "P0", DEFAULTS["P0"]),
        html.Div(_field("T0 (K)", "T0", DEFAULTS["T0"]), id="T0_container"),
        html.Div(_field("Q0 (-)", "Q0", DEFAULTS["Q0"], min=0, max=1),
                 id="Q0_container", style={"display": "none"}),

        html.H4("Design target"),
        _field("Back pressure p_back (Pa)", "p_back", DEFAULTS["p_back"]),

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

        html.H4("Throat (Stage 1) initialization"),
        dcc.RadioItems(
            id="stage1_mode",
            options=[
                {"label": " Flat front (general default -- flashing, "
                          "single-phase, two-phase-dome inlets)", "value": "flat"},
                {"label": " True Sauer sonic line (only valid where the "
                          "isentrope crosses M=1 smoothly, no flashing jump)",
                 "value": "sauer"},
            ],
            value="flat",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),

        html.H4("Geometry"),
        _field("Throat half-height y_t (m)", "y_t", DEFAULTS["y_t"]),
        _field("Throat curvature rho_t", "rho_t", DEFAULTS["rho_t"]),
        _field("Divergent curvature rho_d", "rho_d", DEFAULTS["rho_d"]),
        _field("Front resolution n", "n", DEFAULTS["n"], step=1, min=5),

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
        html.Div(
            "Pin the current design, change some inputs, Compute again -- "
            "both will be overlaid on every plot (reference shown dashed).",
            style={"fontSize": "11px", "color": "#888", "marginTop": "4px"},
        ),
        html.Div(id="reference-status", style={"marginTop": "6px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

plots = html.Div(
    [
        html.Div(
            [
                html.Label("Save format:", style={"fontSize": "13px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="save-format",
                    options=[{"label": fmt.upper(), "value": fmt} for fmt in ("png", "svg", "pdf")],
                    value="png", clearable=False,
                    style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                html.Button("Download this plot", id="download-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-plot"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        dcc.Tabs(
            id="plot-tabs",
            value="contour",
            children=[
                dcc.Tab(label=TAB_LABELS[key], value=key,
                         children=dcc.Graph(id=f"fig-{key}", config={"displaylogo": False}))
                for key in TAB_LABELS
            ],
        ),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 2: stator blade parametrization (consumes Phase 1's wall_final)
# --------------------------------------------------------------------------
BLADE_DEFAULTS = dict(
    metal_angle_in=0.0, metal_angle_out=70.0, r_trailing=0.5,
    leading_edge_x=0.0, leading_edge_y=0.0, inlet_opening_ratio=1.8, n_cp=30,
)

blade_controls = html.Div(
    [
        html.H4("Blade parametrization"),
        html.Div(
            "Builds a closed stator blade profile from Phase 1's nozzle "
            "wall contour (used as the divergent-section suction side). "
            "Run Phase 1 Compute first.",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "10px"},
        ),
        dcc.RadioItems(
            id="blade-method",
            options=[
                {"label": " Full (pressure side mirrors the nozzle wall)", "value": "full"},
                {"label": " Semi (pressure side is a single circular arc)", "value": "semi"},
            ],
            value="full",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        html.Div(
            "Semi is a lighter-weight passage: the pressure side doesn't "
            "reuse the wall at all, just an arc between two points near "
            "the leading/trailing edge -- shorter chord, no fixed-endpoint "
            "constraint on the control-point fit. Default inlet opening "
            "ratio differs by method (1.8 full / 2.5 semi); adjust below "
            "if you switch.",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "10px"},
        ),
        _field("Metal angle in (deg)", "metal_angle_in", BLADE_DEFAULTS["metal_angle_in"]),
        _field("Metal angle out (deg)", "metal_angle_out", BLADE_DEFAULTS["metal_angle_out"]),
        _field("Trailing edge radius (mm)", "r_trailing", BLADE_DEFAULTS["r_trailing"]),
        _field("Leading edge x (mm)", "leading_edge_x", BLADE_DEFAULTS["leading_edge_x"]),
        _field("Leading edge y (mm)", "leading_edge_y", BLADE_DEFAULTS["leading_edge_y"]),
        _field("Inlet opening ratio (pitch / inlet opening)", "inlet_opening_ratio",
               BLADE_DEFAULTS["inlet_opening_ratio"]),
        _field("B-spline control points", "n_cp", BLADE_DEFAULTS["n_cp"], step=1, min=6),

        html.Button("Compute blade", id="compute-blade-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px",
                            "fontWeight": "600"}),
        html.Div(id="blade-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

        html.H4("Cascade view", style={"marginTop": "16px"}),
        _field("Number of blades to plot", "n_blades_plot", 2, step=1, min=1),

        html.H4("Edit control points", style={"marginTop": "16px"}),
        html.Div(
            "True click-and-drag isn't practical in a Plotly/Dash app "
            "without heavy custom JS, so editing here is: pick a control "
            "point (highlighted on the plot), then nudge it by a signed "
            "distance along its local normal. Repeatable -- each nudge "
            "applies on top of the last. Reset clears back to the fitted "
            "baseline (so does Compute blade).",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "8px"},
        ),
        dcc.Dropdown(id="cp_index", options=[], value=None, clearable=False,
                      placeholder="Control point index"),
        _field("Move distance (mm, +/-)", "nudge_distance", 1.0),
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
            id="step-kind",
            options=[
                {"label": " 2D face", "value": "face"},
                {"label": " 3D solid (extruded)", "value": "solid"},
            ],
            value="solid",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field("Extrude length (mm)", "extrude_length", 1.0),
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
        dcc.Graph(id="fig-blade", config={"displaylogo": False}, style={"height": "550px"}),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 3: rotor blade design by the vortex-flow method (standalone --
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
        html.Div(
            "Supersonic impulse/reaction rotor blade by the vortex-flow "
            "method (Goldman & Scullin 1968, NASA TN D-4421) -- see "
            "notes/rotor_vortex_blade.md for the method and what the "
            "pressure/suction/blade curves actually are. Standalone from "
            "Phase 1/2 -- own fluid and inlet state.",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "10px"},
        ),
        dcc.Dropdown(id="rotor_fluid_name", options=FLUIDS, value=DEFAULT_FLUID, clearable=False),
        html.Br(),
        _field("P0 (Pa)", "rotor_P0", ROTOR_DEFAULTS["P0_rel"]),
        _field("T0 (K)", "rotor_T0", ROTOR_DEFAULTS["T0_rel"]),

        html.H4("Mach numbers", style={"marginTop": "16px"}),
        _field("M inlet (uniform flow)", "rotor_M_inlet", ROTOR_DEFAULTS["M_inlet"]),
        _field("M outlet (uniform flow)", "rotor_M_outlet", ROTOR_DEFAULTS["M_outlet"]),
        _field("M lower (pressure-side arc)", "rotor_M_lower", ROTOR_DEFAULTS["M_lower"]),
        _field("M upper (suction-side arc)", "rotor_M_upper", ROTOR_DEFAULTS["M_upper"]),

        html.H4("Flow angles", style={"marginTop": "16px"}),
        _field("beta inlet (deg)", "rotor_beta_inlet", ROTOR_DEFAULTS["beta_inlet"]),
        html.Div(
            "beta outlet is always derived from beta_inlet and "
            "M_inlet/M_outlet via mass-flux continuity (rho*V*cos(beta) "
            "= const) -- not a free input, since an inconsistent value "
            "produces a self-intersecting wall with no warning. To "
            "target a specific exit angle, adjust M_inlet/M_outlet.",
            style={"fontSize": "11px", "color": "#888", "marginTop": "4px"},
        ),

        html.H4("Marching resolution", style={"marginTop": "16px"}),
        _field("Points per transition arc", "rotor_num_points", ROTOR_DEFAULTS["num_points"],
               step=1, min=10),

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
        dcc.Graph(id="fig-rotor-blade", config={"displaylogo": False}, style={"height": "550px"}),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 4: combined stator + rotor cascade view (consumes Phase 2's and
# Phase 3's stores -- doesn't recompute anything, just lays both rows of
# blades out together). Stator is in mm (Phase 2's own units); rotor is in
# r* by default (nondimensional) -- rotor_scale converts r* -> mm so the
# two rows can share one plot, and rotor_x/y_offset position the rotor row
# relative to the stator row (axial gap + any vertical stagger). rotor_shift
# is a fraction of the ROTOR's own pitch -- sliding it sweeps one rotor
# blade past the stator passage, the simplest way to look at the relative
# (unsteady) stator/rotor position without an actual time-accurate solve.
# --------------------------------------------------------------------------
cascade_controls = html.Div(
    [
        html.H4("Stator + rotor cascade"),
        html.Div(
            "Lays out Phase 2's stator row and Phase 3's rotor row "
            "together. Compute both first. Purely geometric overlay -- "
            "no flow coupling -- meant for checking blade counts/pitch "
            "and sweeping the relative rotor position by eye.",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "10px"},
        ),
        _field("Number of stator blades", "n_stator_blades", 4, step=1, min=1),
        _field("Number of rotor blades", "n_rotor_blades", 4, step=1, min=1),

        html.H4("Rotor placement", style={"marginTop": "16px"}),
        html.Div(
            "Stator blades are drawn exactly as in Phase 2 (its own pitch "
            "convention, along x). The rotor's own pitch convention is "
            "along y instead (Phase 3/TN D-4421's GSTAR) -- that's not "
            "changed here. x offset moves the WHOLE rotor row downstream "
            "of the stator (positive = further downstream); y offset "
            "shifts it vertically to line it up with a stator passage.",
            style={"fontSize": "11px", "color": "#888", "marginBottom": "8px"},
        ),
        _field("Rotor scale (mm per r*)", "rotor_scale_mm", 50.0),
        _field("Rotor x offset (mm, downstream gap)", "rotor_x_offset", 15.0),
        _field("Rotor y offset (mm)", "rotor_y_offset", 0.0),

        html.H4("Relative position", style={"marginTop": "16px"}),
        html.Div("Slide the rotor row by a fraction of its own pitch:",
                  style={"fontSize": "12px", "marginBottom": "4px"}),
        dcc.Slider(id="rotor_shift", min=0, max=1, step=0.01, value=0,
                    marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False}),

        html.Div(id="cascade-status-message", style={"marginTop": "8px", "fontSize": "13px"}),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

cascade_plots = html.Div(
    [
        dcc.Graph(id="fig-cascade", config={"displaylogo": False}, style={"height": "650px"}),
    ],
    style={"flex": "1", "padding": "16px"},
)

app.layout = html.Div(
    [
        html.H2("MOC Nozzle & Stator Design", style={"padding": "16px 16px 0 16px"}),
        dcc.Tabs(
            id="phase-tabs",
            value="phase1",
            children=[
                dcc.Tab(label="Phase 1: Nozzle design", value="phase1",
                         children=html.Div([controls, plots], style={"display": "flex"})),
                dcc.Tab(label="Phase 2: Stator blade", value="phase2",
                         children=html.Div([blade_controls, blade_plots], style={"display": "flex"})),
                dcc.Tab(label="Phase 3: Rotor blade (vortex-flow)", value="phase3",
                         children=html.Div([rotor_controls, rotor_plots], style={"display": "flex"})),
                dcc.Tab(label="Phase 4: Stator + rotor cascade", value="phase4",
                         children=html.Div([cascade_controls, cascade_plots], style={"display": "flex"})),
            ],
        ),
        dcc.Store(id="result-store"),
        dcc.Store(id="reference-store"),
        dcc.Store(id="blade-store"),
        dcc.Store(id="blade-edited-store"),
        dcc.Store(id="blade-reference-store"),
        dcc.Store(id="rotor-store"),
        dcc.Store(id="rotor-reference-store"),
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
    State("rho_t", "value"),
    State("rho_d", "value"),
    State("n", "value"),
    background=True,
    manager=background_callback_manager,
    progress=[Output("progress-bar", "value"), Output("progress-label", "children")],
    prevent_initial_call=True,
)
def run_design(set_progress, n_clicks, fluid_name, inlet_mode, P0, T0, Q0, p_back,
                solver, stage1_mode, y_t, rho_t, rho_d, n):
    reporter = make_progress_reporter(set_progress)
    set_progress(("0%", "starting..."))
    try:
        data = design_nozzle(
            fluid_name=fluid_name,
            P0=P0,
            T0=T0 if inlet_mode == "T0" else None,
            Q0=(Q0 if inlet_mode == "Q0" else float("nan")),
            p_back=p_back,
            solver=solver,
            use_true_sauer_line=(stage1_mode == "sauer"),
            y_t=y_t, rho_t=rho_t, rho_d=rho_d, n=int(n),
            progress_callback=reporter,
        )
    except NozzleDesignError as e:
        set_progress(("0%", ""))
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"})

    status = html.Div([
        html.Span("Converged" if data["converged"] else "Did not fully converge",
                   style={"fontWeight": "600",
                          "color": "#1a7a1a" if data["converged"] else "#b00020"}),
        html.Div(f"exit M = {data['exit_M']:.4f}  (design {data['design_Noz_Mach']:.4f})"),
        html.Div(f"runtime = {data['runtime_s']:.1f} s"),
    ])
    return data, status


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


@app.callback(
    [Output(f"fig-{key}", "figure") for key in TAB_LABELS],
    Input("result-store", "data"),
    Input("reference-store", "data"),
)
def update_plots(data, reference):
    if not data:
        empty = go.Figure()
        return [empty] * len(TAB_LABELS)
    ref = reference if reference else None
    return [plot_fn(data, reference=ref) for plot_fn in PLOT_FUNCS_PLOTLY.values()]


@app.callback(
    Output("download-plot", "data"),
    Input("download-btn", "n_clicks"),
    State("plot-tabs", "value"),
    State("result-store", "data"),
    State("reference-store", "data"),
    State("save-format", "value"),
    prevent_initial_call=True,
)
def download_plot(n_clicks, active_tab, data, reference, fmt):
    if not data:
        return dash.no_update
    ref = reference if reference else None
    plot_fn, name = PLOT_FUNCS_MPL[active_tab]
    fig, _ax = plot_fn(data, reference=ref)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"{name}.{fmt}")


@app.callback(
    Output("blade-store", "data"),
    Output("blade-status-message", "children"),
    Input("compute-blade-btn", "n_clicks"),
    State("result-store", "data"),
    State("blade-method", "value"),
    State("metal_angle_in", "value"),
    State("metal_angle_out", "value"),
    State("r_trailing", "value"),
    State("leading_edge_x", "value"),
    State("leading_edge_y", "value"),
    State("inlet_opening_ratio", "value"),
    State("n_cp", "value"),
    prevent_initial_call=True,
)
def run_blade(n_clicks, nozzle_data, method, metal_angle_in, metal_angle_out, r_trailing,
              leading_edge_x, leading_edge_y, inlet_opening_ratio, n_cp):
    if not nozzle_data:
        return None, html.Div("Run Phase 1 Compute first -- no nozzle wall to build a blade from.",
                                style={"color": "#b00020"})

    wall = nozzle_data["wall_final"]
    parametrize_fn = BLADE_PARAM_FUNCS[method or "full"]
    try:
        blade = parametrize_fn(
            wall["x"], wall["y"],
            metal_angle_in=metal_angle_in, metal_angle_out=metal_angle_out,
            r_trailing=r_trailing, leading_edge_x=leading_edge_x, leading_edge_y=leading_edge_y,
            inlet_opening_ratio=inlet_opening_ratio, n_cp=int(n_cp),
        )
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"})

    status = html.Div([
        html.Span(f"Blade parametrized ({method or 'full'})",
                   style={"fontWeight": "600", "color": "#1a7a1a"}),
        html.Div(f"pitch = {blade['pitch']:.2f} mm"),
        html.Div(f"throat opening = {blade['throat_opening']:.2f} mm"),
    ])
    return blade, status


def _effective_blade(base_blade, edited_blade):
    """The blade actually shown/exported: the edited version if any nudges
    have been applied on top of the last Compute, else the fresh fit."""
    return edited_blade or base_blade


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
)
def update_blade_plot(base_blade, edited_blade, reference, n_blades, cp_index):
    blade = _effective_blade(base_blade, edited_blade)
    if not blade:
        return go.Figure()
    ref = reference if reference else None
    return moc.plotly.plot_blade(blade, n_blades=int(n_blades) if n_blades else 2,
                                   highlight_cp=cp_index, reference=ref)


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
    fig, _ax = moc.mpl.plot_blade(blade, n_blades=int(n_blades) if n_blades else 2, reference=ref)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"stator_blade.{fmt}")


@app.callback(
    Output("download-step", "data"),
    Output("blade-step-status", "children"),
    Input("download-step-btn", "n_clicks"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("step-kind", "value"),
    State("extrude_length", "value"),
    prevent_initial_call=True,
)
def download_step(n_clicks, base_blade, edited_blade, kind, extrude_length):
    blade = _effective_blade(base_blade, edited_blade)
    if not blade:
        return dash.no_update, html.Div("Compute a blade first.", style={"color": "#b00020"})
    try:
        from moc.geometry import export_blade_step
    except ImportError:
        return dash.no_update, html.Div(
            "cadquery is not installed -- STEP export unavailable "
            "(see environment.yaml / pyproject.toml's 'cad' extra).",
            style={"color": "#b00020"},
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        face_path = os.path.join(tmpdir, "blade_face.step")
        solid_path = os.path.join(tmpdir, "blade_solid.step") if kind == "solid" else None
        export_blade_step(blade, face_path, solid_path, extrude_length=extrude_length or 1.0)
        target_path = solid_path if kind == "solid" else face_path
        with open(target_path, "rb") as f:
            content = f.read()

    fname = "stator_blade_solid.step" if kind == "solid" else "stator_blade_face.step"
    return dcc.send_bytes(content, fname), html.Div("STEP file ready.", style={"color": "#1a7a1a"})


# --------------------------------------------------------------------------
# Phase 3 callbacks: rotor blade (vortex-flow method)
# --------------------------------------------------------------------------
@app.callback(
    Output("rotor-store", "data"),
    Output("rotor-status-message", "children"),
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
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"})

    status = html.Div([
        html.Span("Blade computed", style={"fontWeight": "600", "color": "#1a7a1a"}),
        html.Div(f"pitch = {data['pitch']:.4f} {data['units']}"),
        html.Div(f"chord = {data['chord']:.4f} {data['units']}"),
        html.Div(f"solidity = {data['solidity']:.4f}"),
        html.Div(f"beta_inlet/outlet = {data['beta_inlet']:.2f} / {data['beta_outlet']:.2f} deg"),
    ])
    return data, status


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
    return moc.plotly.plot_rotor_vortex_blade(
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
    fig, _ax = moc.mpl.plot_rotor_vortex_blade(
        data, show_surfaces=("surfaces" in opts), show_mach_lines=("mach_lines" in opts),
        reference=ref,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"rotor_vortex_blade.{fmt}")


# --------------------------------------------------------------------------
# Phase 4 callback: combined stator + rotor cascade view
# --------------------------------------------------------------------------
@app.callback(
    Output("fig-cascade", "figure"),
    Output("cascade-status-message", "children"),
    Input("blade-store", "data"),
    Input("blade-edited-store", "data"),
    Input("rotor-store", "data"),
    Input("n_stator_blades", "value"),
    Input("n_rotor_blades", "value"),
    Input("rotor_scale_mm", "value"),
    Input("rotor_x_offset", "value"),
    Input("rotor_y_offset", "value"),
    Input("rotor_shift", "value"),
)
def update_cascade_plot(base_blade, edited_blade, rotor_data, n_stator, n_rotor,
                         rotor_scale, rotor_x_offset, rotor_y_offset, rotor_shift):
    stator = _effective_blade(base_blade, edited_blade)
    if not stator or not rotor_data:
        missing = []
        if not stator:
            missing.append("stator blade (Phase 2)")
        if not rotor_data:
            missing.append("rotor blade (Phase 3)")
        return go.Figure(), html.Div(
            f"Compute the {' and '.join(missing)} first.", style={"color": "#b00020"})

    n_stator = int(n_stator or 1)
    n_rotor = int(n_rotor or 1)
    scale = float(rotor_scale or 1.0)
    x_off = float(rotor_x_offset or 0.0)
    y_off = float(rotor_y_offset or 0.0)
    shift = float(rotor_shift or 0.0)

    # Stator: drawn exactly as Phase 2 does (moc.plotly.plot_blade), so its
    # own pitch direction/orientation convention is respected automatically
    # rather than re-guessed here -- same figure this callback then adds
    # the rotor's traces onto, so both share one set of axes (same mm
    # scale by construction, no separate scale-matching needed). plot_blade
    # itself only draws outlines (no fill), so a translucent fill is added
    # here on top, per-copy, in a color distinct from the rotor's.
    fig = moc.plotly.plot_blade(stator, n_blades=n_stator,
                                  show_control_points=False, show_cp_labels=False)
    stator_pitch = stator["pitch"]
    scurve = stator["blade_curve"]
    for i in range(n_stator):
        dx = i * stator_pitch
        fig.add_trace(go.Scatter(
            x=[v + dx for v in scurve["x"]], y=scurve["y"], mode="lines", fill="toself",
            line=dict(color="#2b6cb0", width=0), fillcolor="#2b6cb0", opacity=0.30,
            name="Stator", showlegend=(i == 0), hoverinfo="skip",
        ))

    # Rotor: rotated 90 deg clockwise as a rigid body -- (x,y) -> (y,-x) in
    # its own local (unscaled, r*) frame -- before scaling/offsetting, so
    # the blade's own chord direction (originally along local x) reads
    # along the plot's y after rotation, and what was its pitch direction
    # (local y, TN D-4421's GSTAR) now stacks copies along the plot's x
    # instead. x_off moves the whole row downstream of the stator; y_off/
    # rotor_shift line an individual rotor blade up against a stator
    # passage. Distinct color/fill from the stator's.
    rotor_pitch_mm = rotor_data["pitch"] * scale
    rblade = rotor_data["blade"]
    rx0 = [v * scale + x_off for v in rblade["y"]]
    ry0 = [-v * scale + y_off for v in rblade["x"]]
    for j in range(n_rotor):
        dx = (j + shift) * rotor_pitch_mm
        fig.add_trace(go.Scatter(
            x=[v + dx for v in rx0], y=ry0, mode="lines", fill="toself",
            line=dict(color="black", width=1.5),
            fillcolor="#c0392b", opacity=0.35,
            name="Rotor", showlegend=(j == 0),
        ))

    # plot_blade already pinned numeric xaxis/yaxis ranges to the stator-
    # only extent -- recompute over ALL traces (stator + rotor) now that
    # the rotor row has been added, so it isn't clipped.
    all_x = [x for tr in fig.data for x in tr.x]
    all_y = [y for tr in fig.data for y in tr.y]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    fig.update_layout(
        xaxis=dict(title_text="x (mm)", range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(title_text="y (mm)", range=[y_min - y_pad, y_max + y_pad],
                    scaleanchor="x", scaleratio=1),
        height=650,
    )
    status = html.Div(
        f"stator pitch = {stator['pitch']:.2f} mm, rotor pitch = {rotor_pitch_mm:.2f} mm "
        f"(scale {scale:.2f} mm/r*)",
        style={"color": "#1a7a1a"},
    )
    return fig, status


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
