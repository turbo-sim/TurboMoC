"""
moc.plotting_plotly -- same functions as moc.plotting_mpl, returning
go.Figure instead of matplotlib axes. Meant for the Dash app (turbodash's
own app.py consumes plotly.graph_objects figures the same way, built from
its plotting_plotly.py) -- a callback calls moc.design_nozzle(...) once,
then these functions to build the figures it returns.

Same inputs as plotting_mpl: the `data` dict from design_nozzle() /
MOCSolver.build_result_data(). No solver/UI objects, no state.

Every function also accepts an optional `reference=<second data dict>` to
overlay a second design for comparison (same color, dashed/lighter) -- see
plotting_mpl's module docstring. Used by moc.app's "pin as reference"
workflow, but standalone too.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jaxprop as jxp

from moc._phase_diagram import get_saturation_dome

COLOR_WALL = "black"
COLOR_PRESSURE = jxp.COLORS_PYTHON[1]
COLOR_MACH = jxp.COLORS_PYTHON[2]
COLOR_C_PLUS = jxp.COLORS_PYTHON[0]
COLOR_C_MINUS = jxp.COLORS_PYTHON[3]
COLOR_CP_LINE = "#440154"  # deep purple (viridis/magma base) -- blade control-point connector
COLOR_CP_FACE = "#f89441"  # plasma orange -- blade control-point marker fill

__all__ = [
    "plot_nozzle_contour", "plot_characteristics_mesh",
    "plot_axis_properties", "plot_wall_properties", "plot_ts_diagram",
    "plot_blade", "plot_rotor_vortex_blade",
]


def _apply_journal_style(fig):
    """Applied at the end of every plot function here, so the interactive
    Plotly figure (used for the app's zoom/pan-able dcc.Graph display)
    actually reproduces jaxprop.set_plot_options()'s rcParams -- the same
    call moc.plotting_mpl makes for the matplotlib/download half of this
    split -- rather than Plotly's own default template. Values below are
    copied straight out of jxp.set_plot_options()'s rcParams dict (serif/
    Times New Roman, inward major+minor ticks, opaque mid-gray grid,
    1.25px axis border) so the two renderers of the same data read as one
    visual language, not just "both roughly boxed with gridlines"."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14),
        legend=dict(bordercolor="black", borderwidth=1, bgcolor="white", font_size=12),
        margin=dict(l=60, r=30, t=30, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    axis_style = dict(
        showline=True, linewidth=1.25, linecolor="black", mirror=True,
        gridcolor="#808080", gridwidth=0.5,
        zeroline=False,
        ticks="inside", tickwidth=1.25, ticklen=6,
        minor=dict(ticks="inside", ticklen=3, tickwidth=0.75, showgrid=False),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _mirror(y):
    return [-v for v in y]


def plot_nozzle_contour(data, mirror=True, show_sauer=True, show_kernel=True,
                         reference=None, label="Current", reference_label="Reference"):
    fig = go.Figure()

    def _draw(d, dash, opacity, lbl):
        wall = d["wall_final"]
        fig.add_trace(go.Scatter(x=wall["x"], y=wall["y"], mode="lines",
                                  line=dict(color=COLOR_WALL, width=2, dash=dash),
                                  opacity=opacity, name=lbl))
        if mirror:
            fig.add_trace(go.Scatter(x=wall["x"], y=_mirror(wall["y"]), mode="lines",
                                      line=dict(color=COLOR_WALL, width=2, dash=dash),
                                      opacity=opacity, showlegend=False))

        if show_sauer and d.get("sauer", {}).get("x"):
            sauer = d["sauer"]
            fig.add_trace(go.Scatter(x=sauer["x"], y=sauer["y"], mode="markers",
                                      marker=dict(color=COLOR_C_PLUS, size=5), opacity=opacity,
                                      showlegend=False))
            if mirror:
                fig.add_trace(go.Scatter(x=sauer["x"], y=_mirror(sauer["y"]), mode="markers",
                                          marker=dict(color=COLOR_C_PLUS, size=5), opacity=opacity,
                                          showlegend=False))

        if show_kernel and d.get("kernel_wall", {}).get("x"):
            kernel = d["kernel_wall"]
            fig.add_trace(go.Scatter(x=kernel["x"], y=kernel["y"], mode="markers",
                                      marker=dict(color=COLOR_WALL, size=5), opacity=opacity,
                                      showlegend=False))
            if mirror:
                fig.add_trace(go.Scatter(x=kernel["x"], y=_mirror(kernel["y"]), mode="markers",
                                          marker=dict(color=COLOR_WALL, size=5), opacity=opacity,
                                          showlegend=False))

    if reference is not None:
        _draw(reference, "dash", 0.5, reference_label)
    _draw(data, "solid", 1.0, label if reference is not None else "Wall")

    if mirror:
        fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))

    fig.update_layout(xaxis_title="x (m)", yaxis_title="y (m)",
                       yaxis=dict(scaleanchor="x", scaleratio=1))
    return _apply_journal_style(fig)


def plot_characteristics_mesh(data, mirror=False, show_wall=True,
                               show_fronts=True, show_sauer=True, show_reflex=True,
                               reference=None, show_reference_mesh=False,
                               label="Current", reference_label="Reference"):
    """The full characteristic grid -- see plotting_mpl.plot_characteristics_mesh
    for why stage*_fronts (not just stage*_c_plus/minus/stage3_mesh) are
    needed to actually look like a grid rather than a one-directional fan,
    and why the reflex-zone connectors are reconstructed from IVP/div_wall
    rather than read from mesh_data.

    reference=<data dict> overlays a second design's WALL only by default
    (dashed); its full characteristic mesh is skipped unless
    show_reference_mesh=True (busy fast otherwise)."""
    fig = go.Figure()

    def _draw_segments(segments, color, lbl, width=1, opacity=0.7, showlegend=True):
        first = True
        for xs, ys in segments:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=width), opacity=opacity,
                name=lbl, showlegend=(first and showlegend), legendgroup=lbl,
            ))
            if mirror:
                fig.add_trace(go.Scatter(
                    x=xs, y=_mirror(ys), mode="lines",
                    line=dict(color=color, width=width), opacity=opacity,
                    showlegend=False, legendgroup=lbl,
                ))
            first = False

    def _draw_mesh(d, opacity_scale=1.0, showlegend=True):
        mesh = d.get("mesh_data", {})
        _draw_segments(mesh.get("stage2_c_plus", []), COLOR_C_PLUS, "Stage 2 (C+)",
                        opacity=0.7 * opacity_scale, showlegend=showlegend)
        _draw_segments(mesh.get("stage2_c_minus", []), COLOR_C_MINUS, "Stage 2 (C-)",
                        opacity=0.7 * opacity_scale, showlegend=showlegend)
        _draw_segments(mesh.get("stage3_mesh", []), COLOR_C_PLUS, "Stage 3",
                        opacity=0.7 * opacity_scale, showlegend=showlegend)
        if show_fronts:
            _draw_segments(mesh.get("stage2_fronts", []), COLOR_C_MINUS, "Stage 2 front",
                            opacity=0.7 * opacity_scale, showlegend=showlegend)
            _draw_segments(mesh.get("stage3_fronts", []), COLOR_C_MINUS, "Stage 3 front",
                            opacity=0.7 * opacity_scale, showlegend=showlegend)

        if show_reflex and d.get("IVP", {}).get("x") and d.get("div_wall", {}).get("x"):
            ivp, div_wall = d["IVP"], d["div_wall"]
            first = True
            for xi, yi, xd, yd in zip(ivp["x"], ivp["y"], div_wall["x"], div_wall["y"]):
                fig.add_trace(go.Scatter(x=[xi, xd], y=[yi, yd], mode="lines",
                                          line=dict(color=COLOR_C_MINUS, width=1),
                                          opacity=0.7 * opacity_scale,
                                          name="Reflex zone", showlegend=(first and showlegend),
                                          legendgroup="reflex"))
                if mirror:
                    fig.add_trace(go.Scatter(x=[xi, xd], y=[-yi, -yd], mode="lines",
                                              line=dict(color=COLOR_C_MINUS, width=1),
                                              opacity=0.7 * opacity_scale,
                                              showlegend=False, legendgroup="reflex"))
                first = False
            fig.add_trace(go.Scatter(x=div_wall["x"], y=div_wall["y"], mode="markers",
                                      marker=dict(color=COLOR_WALL, size=5),
                                      opacity=opacity_scale, showlegend=False))
            if mirror:
                fig.add_trace(go.Scatter(x=div_wall["x"], y=_mirror(div_wall["y"]), mode="markers",
                                          marker=dict(color=COLOR_WALL, size=5),
                                          opacity=opacity_scale, showlegend=False))

        if show_sauer and d.get("sauer", {}).get("x"):
            sauer = d["sauer"]
            fig.add_trace(go.Scatter(x=sauer["x"], y=sauer["y"], mode="lines+markers",
                                      line=dict(color=COLOR_C_PLUS, width=2),
                                      marker=dict(size=4), opacity=opacity_scale,
                                      name="Sonic line", showlegend=showlegend))
            if mirror:
                fig.add_trace(go.Scatter(x=sauer["x"], y=_mirror(sauer["y"]), mode="lines+markers",
                                          line=dict(color=COLOR_C_PLUS, width=2),
                                          marker=dict(size=4), opacity=opacity_scale, showlegend=False))

    def _draw_wall(d, dash, opacity, lbl):
        wall = d["wall_final"]
        fig.add_trace(go.Scatter(x=wall["x"], y=wall["y"], mode="lines",
                                  line=dict(color=COLOR_WALL, width=2, dash=dash),
                                  opacity=opacity, name=lbl))
        if mirror:
            fig.add_trace(go.Scatter(x=wall["x"], y=_mirror(wall["y"]), mode="lines",
                                      line=dict(color=COLOR_WALL, width=2, dash=dash),
                                      opacity=opacity, showlegend=False))

    if reference is not None:
        if show_reference_mesh:
            _draw_mesh(reference, opacity_scale=0.4, showlegend=False)
        _draw_wall(reference, "dash", 0.5, reference_label)

    _draw_mesh(data)
    if show_wall:
        _draw_wall(data, "solid", 1.0, label if reference is not None else "Wall")
        if mirror:
            fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))

    fig.update_layout(xaxis_title="x (m)", yaxis_title="y (m)",
                       yaxis=dict(scaleanchor="x", scaleratio=1))
    return _apply_journal_style(fig)


def _property_vs_x(x, p, M, xlabel, p_label, m_label,
                    ref_x=None, ref_p=None, ref_M=None,
                    label="Current", reference_label="Reference"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    show_legend = ref_x is not None
    if show_legend:
        fig.add_trace(go.Scatter(x=ref_x, y=[v / 1e5 for v in ref_p], mode="lines",
                                  line=dict(color=COLOR_PRESSURE, width=2, dash="dash"),
                                  opacity=0.5, name=f"{reference_label} P", showlegend=False),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=ref_x, y=ref_M, mode="lines",
                                  line=dict(color=COLOR_MACH, width=2, dash="dash"),
                                  opacity=0.5, name=f"{reference_label} M", showlegend=False),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                  line=dict(color="gray", width=2), name=label))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                  line=dict(color="gray", width=2, dash="dash"), name=reference_label))

    fig.add_trace(go.Scatter(x=x, y=[v / 1e5 for v in p], mode="lines",
                              line=dict(color=COLOR_PRESSURE, width=2), name=p_label,
                              showlegend=not show_legend),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=M, mode="lines",
                              line=dict(color=COLOR_MACH, width=2), name=m_label,
                              showlegend=not show_legend),
                  secondary_y=True)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=p_label, color=COLOR_PRESSURE, secondary_y=False)
    fig.update_yaxes(title_text=m_label, color=COLOR_MACH, secondary_y=True)
    return _apply_journal_style(fig)


def plot_axis_properties(data, reference=None, label="Current", reference_label="Reference"):
    """Pressure (left, orange) + Mach (right, green) vs axial distance
    along the centerline -- data['axis']. reference=<data dict> overlays a
    second design's axis properties (dashed)."""
    axis = data["axis"]
    ref = reference["axis"] if reference is not None else None
    return _property_vs_x(axis["x"], axis["p"], axis["M"], "x (m)", "P (bar)", "M (-)",
                           ref_x=ref["x"] if ref else None, ref_p=ref["p"] if ref else None,
                           ref_M=ref["M"] if ref else None, label=label, reference_label=reference_label)


def plot_wall_properties(data, reference=None, label="Current", reference_label="Reference"):
    """Pressure (left, orange) + Mach (right, green) vs axial distance
    along the wall -- data['wall_final']. reference=<data dict> overlays a
    second design's wall properties (dashed)."""
    wall = data["wall_final"]
    ref = reference["wall_final"] if reference is not None else None
    return _property_vs_x(wall["x"], wall["p"], wall["M"], "x (m)", "P (bar)", "M (-)",
                           ref_x=ref["x"] if ref else None, ref_p=ref["p"] if ref else None,
                           ref_M=ref["M"] if ref else None, label=label, reference_label=reference_label)


def plot_ts_diagram(data, show_dome=True, reference=None, label="Current", reference_label="Reference"):
    """T-s diagram -- see plotting_mpl.plot_ts_diagram for the exact
    meaning of the three highlighted points (Total, MoC start, Outlet) and
    why only MoC-start-to-Outlet gets a connecting line."""
    fig = go.Figure()

    if show_dome:
        dome = get_saturation_dome(data["fluid_name"])
        fig.add_trace(go.Scatter(x=dome["liq_s"], y=dome["liq_T"], mode="lines",
                                  line=dict(color="gray", width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=dome["vap_s"], y=dome["vap_T"], mode="lines",
                                  line=dict(color="gray", width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=[dome["crit_s"]], y=[dome["crit_T"]], mode="markers",
                                  marker=dict(color="gray", size=6), showlegend=False))

    def _draw(d, opacity, dash, suffix):
        s_total, T_total = d["s0"], d["T0_stagnation"]
        s_start, T_start = d["axis"]["s"][0], d["axis"]["T"][0]
        s_out, T_out = d["axis"]["s"][-1], d["axis"]["T"][-1]

        fig.add_trace(go.Scatter(x=[s_start, s_out], y=[T_start, T_out], mode="lines",
                                  line=dict(color=COLOR_MACH, width=2, dash=dash),
                                  opacity=opacity, showlegend=False))
        fig.add_trace(go.Scatter(x=[s_total], y=[T_total], mode="markers",
                                  marker=dict(color=COLOR_WALL, size=11, symbol="square"),
                                  opacity=opacity, name=f"Total{suffix}"))
        fig.add_trace(go.Scatter(x=[s_start], y=[T_start], mode="markers",
                                  marker=dict(color=COLOR_C_PLUS, size=11, symbol="circle"),
                                  opacity=opacity, name=f"MoC start{suffix}"))
        fig.add_trace(go.Scatter(x=[s_out], y=[T_out], mode="markers",
                                  marker=dict(color=COLOR_PRESSURE, size=11, symbol="triangle-up"),
                                  opacity=opacity, name=f"Outlet{suffix}"))

    if reference is not None:
        _draw(reference, 0.5, "dash", f" ({reference_label})")
    _draw(data, 1.0, "solid", f" ({label})" if reference is not None else "")

    fig.update_layout(xaxis_title="s (J/kg/K)", yaxis_title="T (K)")
    return _apply_journal_style(fig)


def plot_blade(blade_data, n_blades=2, show_control_points=True, show_cp_labels=True,
                show_trailing_edge=True, highlight_cp=None, reference=None,
                label="Current", reference_label="Reference"):
    """Cascade view of the fitted blade profile -- see plotting_mpl.
    plot_blade for the ported script's mm units, the reference-overlay
    semantics shared with the rest of this module, and what highlight_cp/
    show_cp_labels are for."""
    fig = go.Figure()
    all_x, all_y = [], []

    def _draw(d, dash, opacity, lbl):
        pitch = d["pitch"]
        curve = d["blade_curve"]
        te = d["trailing_edge"]
        cps = d["control_points"]
        for i in range(n_blades):
            dx = i * pitch
            curve_x = [v + dx for v in curve["x"]]
            all_x.extend(curve_x)
            all_y.extend(curve["y"])
            fig.add_trace(go.Scatter(
                x=curve_x, y=curve["y"], mode="lines",
                line=dict(color=COLOR_WALL, width=2, dash=dash), opacity=opacity,
                name=lbl, showlegend=(i == 0),
            ))
            if show_trailing_edge:
                te_x = [v + dx for v in te["x"]]
                all_x.extend(te_x)
                all_y.extend(te["y"])
                fig.add_trace(go.Scatter(
                    x=te_x, y=te["y"], mode="lines",
                    line=dict(color=COLOR_C_PLUS, width=1.5), opacity=opacity,
                    showlegend=False,
                ))
            if show_control_points and i == 0:
                cp_x = [p[0] + dx for p in cps]
                cp_y = [p[1] for p in cps]
                all_x.extend(cp_x)
                all_y.extend(cp_y)
                fig.add_trace(go.Scatter(
                    x=cp_x, y=cp_y, mode="lines+markers",
                    line=dict(color=COLOR_CP_LINE, width=1.2, dash="dash"),
                    marker=dict(size=10, color=COLOR_CP_FACE, line=dict(color="black", width=1.2)),
                    opacity=0.85 * opacity,
                    name="Control points" if lbl != reference_label else None,
                    showlegend=(lbl != reference_label),
                ))

    if reference is not None:
        _draw(reference, "dash", 0.5, reference_label)
    _draw(blade_data, "solid", 1.0, label if reference is not None else "Blade profile")

    if show_control_points and show_cp_labels:
        cps = blade_data["control_points"]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in cps], y=[p[1] for p in cps], mode="text",
            text=[str(i) for i in range(len(cps))],
            textposition="top right",
            textfont=dict(size=11, color=COLOR_CP_LINE, family="DejaVu Sans, Arial, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))

    if show_control_points and highlight_cp is not None:
        cps = blade_data["control_points"]
        if 0 <= highlight_cp < len(cps):
            fig.add_trace(go.Scatter(
                x=[cps[highlight_cp][0]], y=[cps[highlight_cp][1]], mode="markers",
                marker=dict(size=22, color="#ffe600", symbol="star",
                            line=dict(color="#d81b1b", width=2)),
                name=f"Selected CP {highlight_cp}",
            ))

    # Explicit, data-derived ranges on BOTH axes (with padding), plus a
    # fixed figure height -- leaving either axis on autorange while
    # scaleanchor="x"/scaleratio=1 is set means Plotly.js recomputes that
    # axis's range relative to whatever it was left at on the previous
    # Dash re-render (no full reset in between), so it compounds a little
    # every recompute instead of settling -- pinning both numerically, and
    # fixing the container height so it can't join the feedback loop,
    # makes every render fully self-contained.
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    x_pad = 0.05 * x_span
    y_pad = 0.05 * y_span
    fig.update_layout(
        xaxis_title="x (mm)", yaxis_title="y (mm)",
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad], scaleanchor="x", scaleratio=1),
        height=550,
    )
    return _apply_journal_style(fig)


COLOR_BLADE_FILL = "#8c8c8c"


def plot_rotor_vortex_blade(data, show_surfaces=True, show_mach_lines=False,
                              reference=None, label="Current", reference_label="Reference"):
    """Vortex-flow rotor blade (moc.rotor.design_rotor_vortex_blade,
    NASA TN D-4421/Goldman & Scullin 1968) -- see
    notes/rotor_vortex_blade.md for what `pressure`, `suction`, `blade`,
    and the Mach-line segments actually are; NOT the same construction as
    plot_blade (Phase 2's nozzle-derived stator blade) even though both
    return a `pitch`.

    show_surfaces overlays the raw pressure/suction transition-arc +
    circular-arc curves (thin, semi-transparent) on top of the filled
    blade polygon -- useful for sanity-checking the closure against the
    surfaces it was built from. show_mach_lines overlays the recorded
    characteristic segments from the marching recursion (thin, faint).
    """
    fig = go.Figure()

    def _draw(d, dash, opacity, lbl):
        blade = d["blade"]
        fig.add_trace(go.Scatter(
            x=blade["x"], y=blade["y"], mode="lines", fill="toself",
            line=dict(color="black", width=1.5, dash=dash),
            fillcolor=COLOR_BLADE_FILL, opacity=0.55 * opacity if dash == "solid" else 0.25 * opacity,
            name=lbl or "Blade", showlegend=bool(lbl),
        ))
        if show_surfaces:
            for surf_key, color in (("pressure", COLOR_PRESSURE), ("suction", COLOR_MACH)):
                surf = d[surf_key]
                fig.add_trace(go.Scatter(
                    x=surf["x"], y=surf["y"], mode="lines",
                    line=dict(color=color, width=1.5, dash=dash), opacity=0.7 * opacity,
                    name=f"{lbl + ' ' if lbl else ''}{surf_key}", showlegend=True,
                ))
        if show_mach_lines:
            for surf_key in ("pressure", "suction"):
                surf = d[surf_key]
                xs, ys = [], []
                for x0, y0, x1, y1 in surf["mach_lines_inlet"] + surf["mach_lines_outlet"]:
                    xs.extend([x0, x1, None])
                    ys.extend([y0, y1, None])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="#999999", width=0.5), opacity=0.6 * opacity,
                    showlegend=False, hoverinfo="skip",
                ))

    if reference is not None:
        _draw(reference, "dash", 0.6, reference_label)
    _draw(data, "solid", 1.0, label if reference is not None else "")

    units = data.get("units", "r*")
    fig.update_layout(
        xaxis_title=f"x ({units})", yaxis_title=f"y ({units})",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=550,
    )
    return _apply_journal_style(fig)
