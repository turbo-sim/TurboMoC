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

import base64
import io
import json
import os
import tempfile
from datetime import datetime

import diskcache
import dash
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import jaxprop as jxp

import moc
from moc import design_nozzle, list_supported_fluids, make_progress_reporter, NozzleDesignError
from moc.geometry import (
    build_meridional_view,
    move_control_point_along_normal,
    parametrize_stator_blade,
    parametrize_stator_blade_semi,
    wrap_blade_radial,
    wrap_rotor_blade_radial,
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
    # P0/p_back are in bar (UI + save/load convention) -- converted to Pa
    # only at the design_nozzle() call boundary in run_design.
    P0=20.0, T0=113.0, Q0=0.5, p_back=2.0,
    y_t=0.01, n=15, rho_t=0.2, rho_d=0.1,
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
            value="T0",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        _field(r"$P_0$ (bar)", "P0", DEFAULTS["P0"]),
        html.Div(_field(r"$T_0$ (K)", "T0", DEFAULTS["T0"]), id="T0_container"),
        html.Div(_field(r"$Q_0$ (-)", "Q0", DEFAULTS["Q0"], min=0, max=1),
                 id="Q0_container", style={"display": "none"}),

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
            value="flat",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),

        html.H4("Geometry"),
        _field(r"$y_t$ (m)", "y_t", DEFAULTS["y_t"]),
        _field(r"$\rho_t$", "rho_t", DEFAULTS["rho_t"]),
        _field(r"$\rho_d$", "rho_d", DEFAULTS["rho_d"]),
        _field(r"$n$", "n", DEFAULTS["n"], step=1, min=5),

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
                    value="contour", clearable=False,
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
    metal_angle_in=0.0, metal_angle_out=70.0, r_trailing=0.5,
    leading_edge_x=0.0, leading_edge_y=0.0, inlet_opening_ratio=1.8, n_cp=30,
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
        _field(r"$x_{\text{LE}}$ (mm)", "leading_edge_x", BLADE_DEFAULTS["leading_edge_x"]),
        _field(r"$y_{\text{LE}}$ (mm)", "leading_edge_y", BLADE_DEFAULTS["leading_edge_y"]),
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
            id="step-kind",
            options=[
                {"label": " 2D face", "value": "face"},
                {"label": " 3D solid (extruded)", "value": "solid"},
            ],
            value="solid",
            labelStyle={"display": "block", "fontSize": "13px"},
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
        dcc.Dropdown(id="rotor_fluid_name", options=FLUIDS, value=DEFAULT_FLUID, clearable=False),
        html.Br(),
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
# --------------------------------------------------------------------------
# Flare (meridional view + flared STEP export) -- an OPTIONAL section
# embedded into both Phase 4 (axial/unrolled) and Phase 5 (radial/
# conformal), not a standalone phase, since it's a property of the stage
# either view is already showing, not a separate design step. Built once
# as _flare_controls(prefix)/_flare_plot(prefix) and registered twice (see
# _register_flare_callbacks below) with distinct ID prefixes ("cascade"
# for Phase 4, "radial" for Phase 5) so both copies coexist in the same
# page. Both the meridional sketch and the flared STEP solid read the SAME
# hub/tip radii fields (see moc/geometry/meridional.py's module docstring
# for why) -- they can never disagree with each other.
# --------------------------------------------------------------------------
def _flare_controls(prefix, scale_input_id=None, gap_input_id=None):
    """scale_input_id/gap_input_id: when given, this prefix's flare panel
    reuses that OTHER (already-on-screen) control instead of rendering its
    own independent rotor-scale/axial-gap field -- geometry the flare view
    shares with the cascade/radial view it's embedded next to must stay
    single-sourced, not editable twice with no guarantee the two numbers
    agree."""
    rotor_fields = [
        _field(r"$r_{\text{hub,in}}$", f"{prefix}_rotor_r_hub_in", 145.0),
        _field(r"$r_{\text{hub,out}}$", f"{prefix}_rotor_r_hub_out", 148.0),
        _field(r"$r_{\text{tip,in}}$", f"{prefix}_rotor_r_tip_in", 188.0),
        _field(r"$r_{\text{tip,out}}$", f"{prefix}_rotor_r_tip_out", 185.0),
    ]
    if scale_input_id is None:
        rotor_fields.append(_field(r"$s_{\text{rotor}}$ (mm per r*)",
                                     f"{prefix}_flare_rotor_scale", 50.0))
    extra_fields = []
    if gap_input_id is None:
        extra_fields.append(_field(r"$\Delta z_{\text{gap}}$ (mm)", f"{prefix}_flare_gap", 5.0))

    return html.Div(
        [
            html.H4("Flare (meridional)", style={"marginTop": "16px"}),
            dcc.Checklist(
                id=f"{prefix}_flare_enable",
                options=[{"label": " Enable", "value": "on"}],
                value=[],
            ),
            html.Div(
                id=f"{prefix}_flare_fields",
                children=[
                    html.H4("Stator radii (mm)", style={"marginTop": "12px"}),
                    _field(r"$r_{\text{hub,in}}$", f"{prefix}_stator_r_hub_in", 140.0),
                    _field(r"$r_{\text{hub,out}}$", f"{prefix}_stator_r_hub_out", 145.0),
                    _field(r"$r_{\text{tip,in}}$", f"{prefix}_stator_r_tip_in", 190.0),
                    _field(r"$r_{\text{tip,out}}$", f"{prefix}_stator_r_tip_out", 188.0),

                    html.H4("Rotor radii (mm)", style={"marginTop": "12px"}),
                    *rotor_fields,
                    *extra_fields,

                    html.H4("Flared STEP export", style={"marginTop": "12px"}),
                    html.Button("Download flared stator STEP", id=f"{prefix}_download_flared_stator_btn",
                                 n_clicks=0, style={"width": "100%", "padding": "6px", "marginBottom": "6px"}),
                    html.Button("Download flared rotor STEP", id=f"{prefix}_download_flared_rotor_btn",
                                 n_clicks=0, style={"width": "100%", "padding": "6px"}),
                    dcc.Download(id=f"{prefix}_download_flared_stator"),
                    dcc.Download(id=f"{prefix}_download_flared_rotor"),
                    html.Div(id=f"{prefix}_flared_step_status", style={"marginTop": "8px", "fontSize": "13px"}),
                ],
                style={"display": "none"},
            ),
        ],
    )


def _flare_plot(prefix, height="380px"):
    return dcc.Graph(id=f"fig-{prefix}-meridional", config={"displaylogo": False},
                       style={"height": height, "flex": "1"})

cascade_controls = html.Div(
    [
        html.H4("Stator + rotor cascade"),
        _field(r"$N_{\text{stator}}$", "n_stator_blades", 4, step=1, min=1),
        _field(r"$N_{\text{rotor}}$", "n_rotor_blades", 4, step=1, min=1),

        html.H4("Rotor placement", style={"marginTop": "16px"}),
        _field(r"$s_{\text{rotor}}$ (mm per r*)", "rotor_scale_mm", 50.0),
        _field(r"$\Delta z_{\text{gap}}$ (mm, stator TE to rotor LE)", "rotor_axial_gap", 5.0),
        dcc.RadioItems(
            id="rotor_downstream_direction",
            options=[
                {"label": " Below (-y)", "value": "below"},
                {"label": " Above (+y)", "value": "above"},
            ],
            value="below",
            labelStyle={"display": "inline-block", "marginRight": "12px", "fontSize": "13px"},
        ),
        _field(r"$\Delta x_{\text{rotor}}$ (mm)", "rotor_lateral_offset", 0.0),

        html.H4("Relative position", style={"marginTop": "16px"}),
        html.Div("Slide the rotor row by a fraction of its own pitch:",
                  style={"fontSize": "12px", "marginBottom": "4px"}),
        dcc.Slider(id="rotor_shift", min=0, max=1, step=0.01, value=0,
                    marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False}),

        _flare_controls("cascade", scale_input_id="rotor_scale_mm", gap_input_id="rotor_axial_gap"),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

cascade_plots = html.Div(
    [
        html.Div(
            [
                dcc.Graph(id="fig-cascade", config={"displaylogo": False},
                           style={"height": "380px", "flex": "2"}),
                _flare_plot("cascade"),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "12px", "alignItems": "flex-start"},
        ),
        html.Div(id="cascade-status-message", style={"width": "50%"}),
    ],
    style={"flex": "1", "padding": "16px"},
)

# --------------------------------------------------------------------------
# Phase 5: conformal (log-spiral) radial wrap of an unrolled axial blade --
# either Phase 2's stator or Phase 3's rotor -- onto an annular passage
# between two radii. See moc/geometry/radial.py's module docstring for the
# mapping itself, and for why stator/rotor need separate wrap functions
# (their own local (x,y) frames put "chordwise" on opposite axes).
# --------------------------------------------------------------------------
radial_controls = html.Div(
    [
        html.H4("Radial (conformal) wrap"),
        dcc.RadioItems(
            id="radial-source",
            options=[
                {"label": " Stator (Phase 2)", "value": "stator"},
                {"label": " Rotor (Phase 3)", "value": "rotor"},
                {"label": " Both (stator + rotor)", "value": "both"},
            ],
            value="stator",
            labelStyle={"display": "block", "fontSize": "13px"},
        ),
        html.Div(id="radial-rotor-scale-container", children=[
            _field(r"$s_{\text{rotor}}$ (mm per r*, if Phase 3 is nondim.)",
                   "radial_rotor_scale", 50.0),
        ], style={"display": "none"}),

        html.Div(id="radial-single-radii-container", children=[
            _field(r"$r_1$ (mm)", "radial_r1", 140.0),
            _field(r"$r_2$ (mm)", "radial_r2", 190.0),
        ]),
        _field(r"$N_{\text{blades}}$", "radial_n_blades", 12, step=1, min=1),
        _field(r"$\theta_0$ (deg)", "radial_theta0", 0.0),

        html.Div(id="radial-both-container", children=[
            html.H4("Coupled stator + rotor radii", style={"marginTop": "16px"}),
            _field(r"$r_{\text{stator,in}}$ (mm)", "radial_r_stator_in", 140.0),
            _field(r"$r_{\text{interface}}$ (mm)", "radial_r_interface", 165.0),
            _field(r"$\Delta r_{\text{gap}}$ (mm)", "radial_gap", 2.0, min=0),
            _field(r"$r_{\text{rotor,out}}$ (mm)", "radial_r_rotor_out", 190.0),
            html.H4("Rotor (2nd row)", style={"marginTop": "16px"}),
            _field(r"$N_{\text{rotor}}$", "radial_n_blades_rotor", 12, step=1, min=1),
            _field(r"$\theta_{0,\text{rotor}}$ (deg)", "radial_theta0_rotor", 0.0),
        ], style={"display": "none"}),

        html.Button("Compute radial wrap", id="compute-radial-btn", n_clicks=0,
                     style={"width": "100%", "marginTop": "8px", "padding": "8px",
                            "fontWeight": "600"}),
        html.Div(id="radial-status-message", style={"marginTop": "8px", "fontSize": "13px"}),

        _flare_controls("radial", scale_input_id="radial_rotor_scale"),
    ],
    style={"width": "300px", "padding": "16px", "borderRight": "1px solid #ddd",
           "overflowY": "auto"},
)

radial_plots = html.Div(
    [
        html.Div(
            [
                html.Label("Save format:", style={"fontSize": "13px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="radial-save-format",
                    options=[{"label": fmt.upper(), "value": fmt} for fmt in ("png", "svg", "pdf")],
                    value="png", clearable=False,
                    style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                html.Button("Download this plot", id="radial-download-btn", n_clicks=0,
                             style={"marginLeft": "12px", "padding": "6px 12px"}),
                dcc.Download(id="download-radial-plot"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        html.Div(
            [
                dcc.Graph(id="fig-radial", config={"displaylogo": False},
                           style={"height": "380px", "flex": "1"}),
                _flare_plot("radial"),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "12px", "alignItems": "flex-start"},
        ),
        html.Div(id="radial-info-table"),
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
                dcc.Tab(label="Phase 5: Radial (conformal) wrap", value="phase5",
                         children=html.Div([radial_controls, radial_plots], style={"display": "flex"})),
            ],
        ),
        dcc.Store(id="result-store"),
        dcc.Store(id="loaded-nozzle-store"),
        dcc.Store(id="reference-store"),
        dcc.Store(id="blade-store"),
        dcc.Store(id="blade-edited-store"),
        dcc.Store(id="blade-reference-store"),
        dcc.Store(id="rotor-store"),
        dcc.Store(id="rotor-reference-store"),
        dcc.Store(id="radial-store"),
        dcc.Store(id="cascade-zrange-store"),
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

    status = html.Div([
        html.Span("Converged" if data["converged"] else "Did not fully converge",
                   style={"fontWeight": "600",
                          "color": "#1a7a1a" if data["converged"] else "#b00020"}),
        html.Div(f"exit M = {data['exit_M']:.4f}  (design {data['design_Noz_Mach']:.4f})"),
        html.Div(f"runtime = {data['runtime_s']:.1f} s"),
    ])
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
    State("rho_t", "value"),
    State("rho_d", "value"),
    State("n", "value"),
    prevent_initial_call=True,
)
def save_nozzle_design(n_clicks, result_data, fluid_name, inlet_mode, P0, T0, Q0, p_back,
                        solver, stage1_mode, y_t, rho_t, rho_d, n):
    if result_data is None:
        return dash.no_update

    cfg = {
        "inputs": {
            "fluid_name": fluid_name, "inlet_mode": inlet_mode, "P0": P0, "T0": T0, "Q0": Q0,
            "p_back": p_back, "solver": solver, "stage1_mode": stage1_mode,
            "y_t": y_t, "rho_t": rho_t, "rho_d": rho_d, "n": n,
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
    Output("rho_t", "value"),
    Output("rho_d", "value"),
    Output("n", "value"),
    Input("loaded-nozzle-store", "data"),
    prevent_initial_call=True,
)
def apply_loaded_nozzle_design(cfg):
    if not cfg:
        return (dash.no_update,) * 13
    inp = cfg.get("inputs", {})
    status = html.Div("Loaded from file (no recompute needed).", style={"color": "#1a7a1a"})
    return (
        cfg.get("outputs"), status,
        inp.get("fluid_name"), inp.get("inlet_mode"), inp.get("P0"), inp.get("T0"), inp.get("Q0"),
        inp.get("p_back"), inp.get("solver"), inp.get("stage1_mode"),
        inp.get("y_t"), inp.get("rho_t"), inp.get("rho_d"), inp.get("n"),
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
    State("plot-select", "value"),
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
def _rotate_traces_display(fig):
    """x -> -x (keeps the CW-180 horizontal/flow orientation already
    confirmed correct -- x still runs the same visual direction as
    before), y unchanged (NOT negated) -- a vertical mirror, not a further
    rotation: a genuine second 180 deg turn ((x,y)->(-x,-y) again) would
    undo the flow-direction fix too and land back on the original tall/
    narrow, pitch-horizontal layout, which isn't what was asked for.
    Flipping y alone swaps which row (stator/rotor) ends up on top
    without touching the horizontal orientation: stator is built at
    higher y than the rotor in this callback's own raw (pre-display-
    transform) coordinates whenever downstream_direction="below" (the
    default -- rotor placed at y_off = stator_y_min - ... - gap, i.e.
    below/more-negative), so leaving y unflipped keeps that same relative
    order (stator above) once displayed."""
    for tr in fig.data:
        x = list(tr.x)
        tr.x = [-xx for xx in x]
    return fig


def _mirror_pitchwise(blade_data):
    """Copy of a Phase-2-style blade dict with its own pitchwise
    coordinate (x, in blade_data's own frame -- chordwise is y, see
    moc/geometry/radial.py's module docstring) negated: flips the
    blade's own lean/incline about its local chordwise axis, without
    moving it relative to the other row or touching its chordwise shape.
    A GLOBAL mirror of the whole assembled figure was tried first, but
    that also flips which row ends up on top (that's controlled by the
    same Y axis this local mirror doesn't touch) -- this is scoped to
    just the one blade's own shape instead."""
    curve = blade_data["blade_curve"]
    te = blade_data["trailing_edge"]
    mirrored = dict(blade_data)
    mirrored["blade_curve"] = {"x": [-v for v in curve["x"]], "y": curve["y"]}
    mirrored["trailing_edge"] = {"x": [-v for v in te["x"]], "y": te["y"]}
    mirrored["control_points"] = [[-p[0], p[1]] for p in blade_data["control_points"]]
    return mirrored


@app.callback(
    Output("fig-cascade", "figure"),
    Output("cascade-status-message", "children"),
    Output("cascade-zrange-store", "data"),
    Input("blade-store", "data"),
    Input("blade-edited-store", "data"),
    Input("rotor-store", "data"),
    Input("n_stator_blades", "value"),
    Input("n_rotor_blades", "value"),
    Input("rotor_scale_mm", "value"),
    Input("rotor_axial_gap", "value"),
    Input("rotor_downstream_direction", "value"),
    Input("rotor_lateral_offset", "value"),
    Input("rotor_shift", "value"),
    Input("result-store", "data"),
    Input("cascade_flare_enable", "value"),
    Input("cascade_stator_r_hub_in", "value"),
    Input("cascade_stator_r_tip_in", "value"),
)
def update_cascade_plot(base_blade, edited_blade, rotor_data, n_stator, n_rotor,
                         rotor_scale, rotor_axial_gap, downstream_direction,
                         rotor_lateral_offset, rotor_shift, nozzle_data,
                         flare_enable, stator_r_hub_in, stator_r_tip_in):
    stator = _effective_blade(base_blade, edited_blade)
    if not stator or not rotor_data:
        missing = []
        if not stator:
            missing.append("stator blade (Phase 2)")
        if not rotor_data:
            missing.append("rotor blade (Phase 3)")
        return go.Figure(), html.Div(
            f"Compute the {' and '.join(missing)} first.", style={"color": "#b00020"}), None

    n_stator = int(n_stator or 1)
    n_rotor = int(n_rotor or 1)
    scale = float(rotor_scale or 1.0)
    gap = float(rotor_axial_gap or 0.0)
    x_off = float(rotor_lateral_offset or 0.0)
    shift = float(rotor_shift or 0.0)

    # Stator: drawn exactly as Phase 2 does (moc.plotly.plot_blade), so its
    # own pitch direction/orientation convention is respected automatically
    # rather than re-guessed here -- same figure this callback then adds
    # the rotor's traces onto, so both share one set of axes (same mm
    # scale by construction, no separate scale-matching needed). plot_blade
    # itself only draws outlines (no fill), so a translucent fill is added
    # here on top, per-copy, in a color distinct from the rotor's.
    stator_display = _mirror_pitchwise(stator)
    fig = moc.plotly.plot_blade(stator_display, n_blades=n_stator,
                                  show_control_points=False, show_cp_labels=False)
    stator_pitch = stator["pitch"]
    scurve = stator_display["blade_curve"]
    for i in range(n_stator):
        dx = i * stator_pitch
        fig.add_trace(go.Scatter(
            x=[v + dx for v in scurve["x"]], y=scurve["y"], mode="lines", fill="toself",
            line=dict(color=moc.plotly.COLOR_RADIAL_STATOR, width=0),
            fillcolor=moc.plotly.COLOR_RADIAL_STATOR, opacity=0.30,
            name="Stator", showlegend=(i == 0), hoverinfo="skip",
        ))

    # Rotor: rotated 90 deg clockwise as a rigid body -- (x,y) -> (y,-x) in
    # its own local (unscaled, r*) frame -- before scaling/offsetting, so
    # the blade's own chord direction (originally along local x) reads
    # along the plot's y after rotation, and what was its pitch direction
    # (local y, TN D-4421's GSTAR) now stacks copies along the plot's x
    # instead. In THIS display, X is the pitch/tangential direction for
    # BOTH rows (stacking copies via each row's own pitch) and Y is the
    # chordwise/flow direction for both -- so the "don't overlap the
    # stator" separation belongs on Y, not X (an earlier version of this
    # put it on X, which just shifted the rotor sideways in the same
    # flow-direction band as the stator -- visually wrong). y_off is
    # DERIVED, not guessed: the stator's own Y extent (its real axial
    # chord, not a fixed number) plus `gap`, placed below (or above, per
    # downstream_direction) it, so the two rows never overlap regardless
    # of the stator's own chord length. x_off is a manual lateral nudge
    # for lining an individual rotor blade up against a stator passage
    # (rotor_shift's pitch-fraction slider does the same thing in relative
    # terms). Distinct color/fill from the stator's.
    stator_y_min = min(scurve["y"])
    stator_y_max = max(scurve["y"])
    rotor_pitch_mm = rotor_data["pitch"] * scale
    rblade = rotor_data["blade"]
    rotor_y_raw = [-v * scale for v in rblade["x"]]
    if downstream_direction == "above":
        y_off = stator_y_max - min(rotor_y_raw) + gap
    else:
        y_off = stator_y_min - max(rotor_y_raw) - gap
    rx0 = [-v * scale + x_off for v in rblade["y"]]
    ry0 = [v + y_off for v in rotor_y_raw]
    for j in range(n_rotor):
        dx = (j + shift) * rotor_pitch_mm
        rx_dx = [v + dx for v in rx0]
        # Two traces, same split as the stator's own (plot_blade's outline
        # + a separate translucent fill): a single trace can't have a
        # fully-opaque line and a translucent fill at once -- Plotly's
        # `opacity` applies to the whole trace, so bundling both here (as
        # an earlier version of this did) made the outline just as faint
        # as the fill.
        fig.add_trace(go.Scatter(
            x=rx_dx, y=ry0, mode="lines",
            line=dict(color="black", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=rx_dx, y=ry0, mode="lines", fill="toself",
            line=dict(color=moc.plotly.COLOR_RADIAL_ROTOR, width=0),
            fillcolor=moc.plotly.COLOR_RADIAL_ROTOR, opacity=0.35,
            name="Rotor", showlegend=(j == 0), hoverinfo="skip",
        ))

    # Rotate for display (see _rotate_traces_display's own docstring), THEN
    # compute the bounding box -- must happen post-rotation so the tight
    # clip below actually matches what's on screen.
    _rotate_traces_display(fig)

    # Shift x and z(=y) so both axes read from 0 (positive), not negative
    # -- the mirror/rotation above keeps them negative in absolute terms
    # (shape preserved, just offset from an arbitrary origin); this only
    # re-zeros the axes, it doesn't change the geometry's shape. Zeroing z
    # too (not just x) is what lets the meridional view below share the
    # same z=0 reference and axis scale (see cascade-zrange-store).
    x_shift = -min(x for tr in fig.data for x in tr.x)
    y_shift = -min(y for tr in fig.data for y in tr.y)
    for tr in fig.data:
        tr.x = [xx + x_shift for xx in tr.x]
        tr.y = [yy + y_shift for yy in tr.y]

    all_x = [x for tr in fig.data for x in tr.x]
    all_y = [y for tr in fig.data for y in tr.y]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    # Clipped close to the LE/TE, not the old 5% padding -- just enough
    # (2%) that the outline isn't cut off by the axes themselves.
    x_pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
    fig.update_layout(
        showlegend=False,
        # No scaleanchor/scaleratio 1:1 lock: the pitchwise (y) span here
        # is much larger than the axial (z) span, so a true 1:1 aspect
        # would letterbox the plot well short of its declared height --
        # shorter than the meridional figure beside it, which fills its
        # full container. Autoscaling both axes independently instead
        # fills the full height, matching that neighbor (see
        # cascade-zrange-store / plot_meridional_view's own z_range).
        xaxis=dict(title_text="y [mm]", range=[x_min - x_pad, x_max + x_pad]),
        # dtick fixed (not autotick) so the gridline spacing matches the
        # meridional plot's z axis exactly, not just the numeric range --
        # autotick can pick different steps for the same range depending
        # on subtle rendered-height differences between the two figures.
        yaxis=dict(title_text="z [mm]", range=[y_min - y_pad, y_max + y_pad], dtick=50),
        height=380,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    chord_stator = _stator_axial_chord(stator)
    chord_rotor = rotor_data["chord"] * scale
    solidity_stator = chord_stator / stator["pitch"]
    solidity_rotor = chord_rotor / rotor_pitch_mm

    table_data = [
        {"Quantity": "Pitch, stator", "Value": f"{stator['pitch']:.2f}", "Unit": "mm"},
        {"Quantity": "Pitch, rotor", "Value": f"{rotor_pitch_mm:.2f}", "Unit": "mm"},
        {"Quantity": "Chord, stator", "Value": f"{chord_stator:.2f}", "Unit": "mm"},
        {"Quantity": "Chord, rotor", "Value": f"{chord_rotor:.2f}", "Unit": "mm"},
        {"Quantity": "Solidity, stator", "Value": f"{solidity_stator:.3f}", "Unit": "-"},
        {"Quantity": "Solidity, rotor", "Value": f"{solidity_rotor:.3f}", "Unit": "-"},
        {"Quantity": "Scale, rotor", "Value": f"{scale:.2f}", "Unit": "mm/r*"},
        {"Quantity": "Offset, rotor (y)", "Value": f"{y_off:.2f}", "Unit": "mm"},
        {"Quantity": "Direction", "Value": downstream_direction, "Unit": "-"},
        {"Quantity": "Gap, axial", "Value": f"{gap:.2f}", "Unit": "mm"},
    ]

    # Choked mass flow rate: only computable once the throat has a real
    # SPAN (blade height), which only exists if flare is enabled here --
    # A_throat = throat_opening (Phase 2's own geometric passage width) x
    # (r_tip_in - r_hub_in) x n_stator (all passages), and rho*/V* come
    # from Phase 1's own solved throat state (wall_final index 0 -- see
    # parametrize_stator_blade's docstring: "throat at index 0"), not
    # assumed to be exactly sonic (M there may be <1 for a flashing/flat-
    # front design) -- V* = M* x a*, with a* from a real jaxprop property
    # call at the throat's own (p, T), same pattern as critical_flow.py.
    flare_on = bool(flare_enable and "on" in flare_enable)
    mdot_row = {"Quantity": "Mass flow rate, choked", "Value": "n/a", "Unit": "kg/s"}
    if flare_on and nozzle_data and stator_r_hub_in is not None and stator_r_tip_in is not None:
        try:
            wall = nozzle_data["wall_final"]
            p_star, T_star, rho_star, M_star = wall["p"][0], wall["T"][0], wall["rho"][0], wall["M"][0]
            fluid = jxp.Fluid(nozzle_data["fluid_name"])
            a_star = fluid.get_state(jxp.PT_INPUTS, p_star, T_star).a
            V_star = M_star * a_star
            throat_height_mm = float(stator_r_tip_in) - float(stator_r_hub_in)
            A_throat_m2 = stator["throat_opening"] * throat_height_mm * n_stator * 1e-6
            mdot_choked = rho_star * V_star * A_throat_m2
            mdot_row["Value"] = f"{mdot_choked:.4f}"
        except Exception:
            pass
    table_data.append(mdot_row)

    status = make_table(table_data, ["Quantity", "Value", "Unit"])
    z_range = [y_min - y_pad, y_max + y_pad]
    return fig, status, z_range


# --------------------------------------------------------------------------
# Phase 5 callbacks: conformal radial wrap of a stator (Phase 2) or rotor
# (Phase 3) blade
# --------------------------------------------------------------------------
@app.callback(
    Output("radial-rotor-scale-container", "style"),
    Output("radial-both-container", "style"),
    Output("radial-single-radii-container", "style"),
    Input("radial-source", "value"),
)
def toggle_radial_extra_fields(source):
    rotor_scale_style = {} if source in ("rotor", "both") else {"display": "none"}
    both_style = {} if source == "both" else {"display": "none"}
    single_radii_style = {"display": "none"} if source == "both" else {}
    return rotor_scale_style, both_style, single_radii_style


@app.callback(
    Output("radial-store", "data"),
    Output("radial-status-message", "children"),
    Output("radial-info-table", "children"),
    Input("compute-radial-btn", "n_clicks"),
    State("radial-source", "value"),
    State("blade-store", "data"),
    State("blade-edited-store", "data"),
    State("rotor-store", "data"),
    State("radial_rotor_scale", "value"),
    State("radial_r1", "value"),
    State("radial_r2", "value"),
    State("radial_n_blades", "value"),
    State("radial_theta0", "value"),
    State("radial_n_blades_rotor", "value"),
    State("radial_theta0_rotor", "value"),
    State("radial_r_stator_in", "value"),
    State("radial_r_interface", "value"),
    State("radial_gap", "value"),
    State("radial_r_rotor_out", "value"),
    prevent_initial_call=True,
)
def run_radial_wrap(n_clicks, source, base_blade, edited_blade, rotor_data, rotor_scale,
                     r1, r2, n_blades, theta0_deg, n_blades_rotor, theta0_rotor_deg,
                     r_stator_in, r_interface, gap, r_rotor_out):
    theta0 = np.radians(float(theta0_deg or 0.0))
    try:
        if source == "both":
            blade = _effective_blade(base_blade, edited_blade)
            missing = []
            if not blade:
                missing.append("stator blade (Phase 2)")
            if not rotor_data:
                missing.append("rotor blade (Phase 3)")
            if missing:
                return None, html.Div(f"Run the {' and '.join(missing)} Compute first.",
                                        style={"color": "#b00020"}), None
            # Coupled radii: ONE shared interface radius (stator exit =
            # rotor inlet reference), not two independently guessed r1/r2
            # pairs -- see the "Coupled stator + rotor radii" note in the
            # controls. `gap` is the row-to-row spacing, inserted as an
            # annular band between the stator's own outer radius and
            # where the rotor's own wrap begins.
            r_stator_in = float(r_stator_in)
            r_interface = float(r_interface)
            r_rotor_in = r_interface + float(gap or 0.0)
            r_rotor_out = float(r_rotor_out)
            stator_radial = wrap_blade_radial(
                blade, r1=r_stator_in, r2=r_interface, n_blades=int(n_blades), theta0=theta0,
            )
            rotor_radial = wrap_rotor_blade_radial(
                rotor_data, r1=r_rotor_in, r2=r_rotor_out, n_blades=int(n_blades_rotor),
                theta0=np.radians(float(theta0_rotor_deg or 0.0)),
                scale=float(rotor_scale or 1.0),
            )
            radial = {
                "mode": "combined", "stator": stator_radial, "rotor": rotor_radial,
                "r1": r_stator_in, "r2": r_rotor_out,
                "r_stator_out": r_interface, "r_rotor_in": r_rotor_in,
                "units": stator_radial["units"],
            }
        elif source == "rotor":
            r1, r2 = float(r1), float(r2)
            if not rotor_data:
                return None, html.Div("Run Phase 3 Compute first -- no rotor blade to wrap.",
                                        style={"color": "#b00020"}), None
            radial = wrap_rotor_blade_radial(
                rotor_data, r1=r1, r2=r2, n_blades=int(n_blades),
                theta0=theta0, scale=float(rotor_scale or 1.0),
            )
            radial["mode"] = "single"
        else:
            r1, r2 = float(r1), float(r2)
            blade = _effective_blade(base_blade, edited_blade)
            if not blade:
                return None, html.Div("Run Phase 2 Compute first -- no stator blade to wrap.",
                                        style={"color": "#b00020"}), None
            radial = wrap_blade_radial(
                blade, r1=r1, r2=r2, n_blades=int(n_blades), theta0=theta0,
            )
            radial["mode"] = "single"
    except Exception as e:
        return None, html.Div(f"Error: {e}", style={"color": "#b00020"}), None

    if radial["mode"] == "combined":
        status = html.Div([
            html.Span("Radial wrap computed (stator + rotor)",
                       style={"fontWeight": "600", "color": "#1a7a1a"}),
        ])
        table_data = [
            {"Quantity": "r_stator_in", "Value": f"{radial['r1']:.2f}", "Unit": radial["units"]},
            {"Quantity": "r_interface", "Value": f"{radial['r_stator_out']:.2f}", "Unit": radial["units"]},
            {"Quantity": "r_rotor_in", "Value": f"{radial['r_rotor_in']:.2f}", "Unit": radial["units"]},
            {"Quantity": "r_rotor_out", "Value": f"{radial['r2']:.2f}", "Unit": radial["units"]},
            {"Quantity": "Number of blades, stator", "Value": str(radial["stator"]["n_blades"]), "Unit": "-"},
            {"Quantity": "Number of blades, rotor", "Value": str(radial["rotor"]["n_blades"]), "Unit": "-"},
        ]
    else:
        status = html.Div([
            html.Span(f"Radial wrap computed ({source})", style={"fontWeight": "600", "color": "#1a7a1a"}),
        ])
        table_data = [
            {"Quantity": "r1", "Value": f"{radial['r1']:.2f}", "Unit": radial["units"]},
            {"Quantity": "r2", "Value": f"{radial['r2']:.2f}", "Unit": radial["units"]},
            {"Quantity": "Number of blades", "Value": str(radial["n_blades"]), "Unit": "-"},
            {"Quantity": "d_theta", "Value": f"{radial['d_theta']:.4f}", "Unit": "rad"},
        ]
    table = make_table(table_data, ["Quantity", "Value", "Unit"])
    return radial, status, table


@app.callback(
    Output("fig-radial", "figure"),
    Input("radial-store", "data"),
)
def update_radial_plot(radial_data):
    if not radial_data:
        return go.Figure()
    return moc.plotly.plot_radial_cascade(radial_data)


@app.callback(
    Output("download-radial-plot", "data"),
    Input("radial-download-btn", "n_clicks"),
    State("radial-store", "data"),
    State("radial-save-format", "value"),
    prevent_initial_call=True,
)
def download_radial_plot(n_clicks, radial_data, fmt):
    if not radial_data:
        return dash.no_update
    fig, _ax = moc.mpl.plot_radial_cascade(radial_data)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), f"radial_cascade.{fmt}")


# --------------------------------------------------------------------------
# Flare callbacks -- registered once per prefix ("cascade" for Phase 4,
# "radial" for Phase 5) via this factory, so the meridional view + flared
# STEP export logic exists exactly once despite appearing in two phases.
# Both the meridional sketch and the exported solid read the SAME hub/tip
# radii fields for a given prefix, so they can never disagree.
# --------------------------------------------------------------------------
def _stator_axial_chord(stator_blade):
    """Stator's own chordwise (flow-direction) extent -- blade_curve's y,
    per parametrize_stator_blade's own rotation convention (see moc/
    geometry/radial.py's module docstring)."""
    ys = stator_blade["blade_curve"]["y"]
    return max(ys) - min(ys)


def _register_flare_callbacks(prefix, scale_input_id=None, gap_input_id=None):
    @app.callback(
        Output(f"{prefix}_flare_fields", "style"),
        Input(f"{prefix}_flare_enable", "value"),
    )
    def _toggle(enable_vals):
        return {} if enable_vals and "on" in enable_vals else {"display": "none"}

    meridional_inputs = [
        Input(f"{prefix}_flare_enable", "value"),
        Input("blade-store", "data"),
        Input("blade-edited-store", "data"),
        Input("rotor-store", "data"),
        Input(f"{prefix}_stator_r_hub_in", "value"),
        Input(f"{prefix}_stator_r_hub_out", "value"),
        Input(f"{prefix}_stator_r_tip_in", "value"),
        Input(f"{prefix}_stator_r_tip_out", "value"),
        Input(f"{prefix}_rotor_r_hub_in", "value"),
        Input(f"{prefix}_rotor_r_hub_out", "value"),
        Input(f"{prefix}_rotor_r_tip_in", "value"),
        Input(f"{prefix}_rotor_r_tip_out", "value"),
        Input(scale_input_id or f"{prefix}_flare_rotor_scale", "value"),
        Input(gap_input_id or f"{prefix}_flare_gap", "value"),
    ]
    # Only the "cascade" (Phase 4) copy sits next to a figure with a
    # comparable z axis (the axial cascade view) -- syncing to its
    # z-range keeps the two visually aligned (see cascade-zrange-store).
    # The "radial" (Phase 5) copy has no such sibling, so it just
    # autoscales to its own data.
    if prefix == "cascade":
        meridional_inputs.append(Input("cascade-zrange-store", "data"))

    @app.callback(Output(f"fig-{prefix}-meridional", "figure"), *meridional_inputs)
    def _update_plot(enable_vals, base_blade, edited_blade, rotor_data,
                      s_hub_in, s_hub_out, s_tip_in, s_tip_out,
                      r_hub_in, r_hub_out, r_tip_in, r_tip_out,
                      rotor_scale, gap, z_range=None):
        if not enable_vals or "on" not in enable_vals:
            return go.Figure()
        stator = _effective_blade(base_blade, edited_blade)
        if not stator or not rotor_data:
            return go.Figure()
        try:
            stator_chord = _stator_axial_chord(stator)
            rotor_chord = rotor_data["chord"] * float(rotor_scale or 1.0)
            meridional = build_meridional_view(
                stator_chord, s_hub_in, s_hub_out, s_tip_in, s_tip_out,
                rotor_chord, r_hub_in, r_hub_out, r_tip_in, r_tip_out,
                gap or 0.0,
            )
        except (TypeError, ValueError):
            return go.Figure()
        return moc.plotly.plot_meridional_view(meridional, z_range=z_range)

    @app.callback(
        Output(f"{prefix}_download_flared_stator", "data"),
        Output(f"{prefix}_flared_step_status", "children", allow_duplicate=True),
        Input(f"{prefix}_download_flared_stator_btn", "n_clicks"),
        State("blade-store", "data"),
        State("blade-edited-store", "data"),
        State(f"{prefix}_stator_r_hub_in", "value"),
        State(f"{prefix}_stator_r_hub_out", "value"),
        State(f"{prefix}_stator_r_tip_in", "value"),
        State(f"{prefix}_stator_r_tip_out", "value"),
        prevent_initial_call=True,
    )
    def _download_stator(n_clicks, base_blade, edited_blade, r_hub_in, r_hub_out, r_tip_in, r_tip_out):
        stator = _effective_blade(base_blade, edited_blade)
        if not stator:
            return dash.no_update, html.Div("Compute a stator blade first (Phase 2).",
                                              style={"color": "#b00020"})
        try:
            from moc.geometry import export_flared_blade_step
        except ImportError:
            return dash.no_update, html.Div(
                "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

        with tempfile.TemporaryDirectory() as tmpdir:
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_flared_blade_step(
                stator, r_hub_in, r_hub_out, r_tip_in, r_tip_out,
                face_path, solid_path, source="stator",
            )
            with open(solid_path, "rb") as f:
                content = f.read()

        return dcc.send_bytes(content, "stator_blade_flared.step"), html.Div(
            "Flared stator STEP ready.", style={"color": "#1a7a1a"})

    @app.callback(
        Output(f"{prefix}_download_flared_rotor", "data"),
        Output(f"{prefix}_flared_step_status", "children", allow_duplicate=True),
        Input(f"{prefix}_download_flared_rotor_btn", "n_clicks"),
        State("rotor-store", "data"),
        State(f"{prefix}_rotor_r_hub_in", "value"),
        State(f"{prefix}_rotor_r_hub_out", "value"),
        State(f"{prefix}_rotor_r_tip_in", "value"),
        State(f"{prefix}_rotor_r_tip_out", "value"),
        State(scale_input_id or f"{prefix}_flare_rotor_scale", "value"),
        prevent_initial_call=True,
    )
    def _download_rotor(n_clicks, rotor_data, r_hub_in, r_hub_out, r_tip_in, r_tip_out, rotor_scale):
        if not rotor_data:
            return dash.no_update, html.Div("Compute a rotor blade first (Phase 3).",
                                              style={"color": "#b00020"})
        try:
            from moc.geometry import export_flared_blade_step
        except ImportError:
            return dash.no_update, html.Div(
                "cadquery is not installed -- STEP export unavailable.", style={"color": "#b00020"})

        # export_flared_blade_step's r_hub/r_tip are in mm -- the rotor
        # blade's own points are only in mm already if it was designed
        # with r_star; otherwise (nondimensional r*) scale them here
        # first, same convention as Phases 4/5's own rotor_scale fields.
        scale = float(rotor_scale or 1.0)
        scaled_blade = {"blade": {
            "x": [v * scale for v in rotor_data["blade"]["x"]],
            "y": [v * scale for v in rotor_data["blade"]["y"]],
        }}

        with tempfile.TemporaryDirectory() as tmpdir:
            face_path = os.path.join(tmpdir, "face.step")
            solid_path = os.path.join(tmpdir, "solid.step")
            export_flared_blade_step(
                scaled_blade, r_hub_in, r_hub_out, r_tip_in, r_tip_out,
                face_path, solid_path, source="rotor",
            )
            with open(solid_path, "rb") as f:
                content = f.read()

        return dcc.send_bytes(content, "rotor_blade_flared.step"), html.Div(
            "Flared rotor STEP ready.", style={"color": "#1a7a1a"})


_register_flare_callbacks("cascade", scale_input_id="rotor_scale_mm", gap_input_id="rotor_axial_gap")
_register_flare_callbacks("radial", scale_input_id="radial_rotor_scale")


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
