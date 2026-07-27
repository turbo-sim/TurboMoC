"""
moc.plotting_mpl -- matplotlib plots built from moc.design_nozzle's result
dict. Static/publication-quality output, matching the color/style conventions
already established across the NICFD_2026 postprocessing scripts (pressure
in orange, Mach in green, jaxprop.COLORS_PYTHON).

Mirrors turbodash's plotting_mpl.py/plotting_plotly.py split: this module
is the matplotlib half; moc.plotting_plotly provides the same functions
returning go.Figure instead, for the Dash app. Both take the SAME `data`
dict (design_nozzle's return value, or MOCSolver.build_result_data()) and
an optional existing axes/figure to draw into -- nothing here calls the
solver or knows about MOCSolver/FluidManager.

Every function also accepts an optional `reference=<second data dict>` to
overlay a second design for comparison (same color, dashed/lighter, to
distinguish it from the current one) -- e.g. pin one design, run another,
compare. Used by moc.app's "pin as reference" workflow, but standalone too.
"""

import matplotlib.pyplot as plt
import jaxprop as jxp

from moc._phase_diagram import get_fluid

# Publication-quality rcParams (fonts, boxed axes with ticks on all sides,
# gridlines, matlab color order) -- this is what plotting_plotly's
# _apply_journal_style is manually reproducing for the interactive figures,
# so both halves of the split read as one consistent visual language.
# Called explicitly here (not just relying on moc.solvers.conventional/mln
# already calling it as an import side effect) so this module's look is
# self-contained regardless of what else has been imported.
jxp.set_plot_options()

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
    "plot_blade", "plot_rotor_blade", "plot_rotor_vortex_blade", "plot_radial_cascade",
    "plot_meridional_view",
]


def _mirror(y):
    return [-v for v in y]


def plot_nozzle_contour(data, ax=None, mirror=True, show_sauer=True, show_kernel=True,
                         reference=None, label="Current", reference_label="Reference"):
    """Wall contour (the actual nozzle shape). mirror=True reflects the
    solved half across the axis for the full profile (this is a 2D
    axisymmetric half-solve throughout moc, same convention as the rest of
    the codebase). reference=<data dict> overlays a second design (dashed,
    lighter) for comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    def _draw(d, ls, alpha, lbl):
        wall = d["wall_final"]
        ax.plot(wall["x"], wall["y"], ls, color=COLOR_WALL, lw=1.5, alpha=alpha, label=lbl)
        if mirror:
            ax.plot(wall["x"], _mirror(wall["y"]), ls, color=COLOR_WALL, lw=1.5, alpha=alpha)

        if show_sauer and d.get("sauer", {}).get("x"):
            sauer = d["sauer"]
            ax.plot(sauer["x"], sauer["y"], "o", color=COLOR_C_PLUS, ms=3, alpha=alpha)
            if mirror:
                ax.plot(sauer["x"], _mirror(sauer["y"]), "o", color=COLOR_C_PLUS, ms=3, alpha=alpha)

        if show_kernel and d.get("kernel_wall", {}).get("x"):
            kernel = d["kernel_wall"]
            ax.plot(kernel["x"], kernel["y"], "o", color=COLOR_WALL, ms=3, alpha=alpha)
            if mirror:
                ax.plot(kernel["x"], _mirror(kernel["y"]), "o", color=COLOR_WALL, ms=3, alpha=alpha)

    if reference is not None:
        _draw(reference, "--", 0.5, reference_label)
    _draw(data, "-", 1.0, label if reference is not None else "Wall")

    if mirror:
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def plot_characteristics_mesh(data, ax=None, mirror=False, show_wall=True,
                               show_fronts=True, show_sauer=True, show_reflex=True,
                               reference=None, show_reference_mesh=False,
                               label="Current", reference_label="Reference"):
    """The full characteristic grid -- both line families, not just one.

    data['mesh_data']'s stage2_c_plus/stage2_c_minus/stage3_mesh are the
    individual characteristic SEGMENTS (radiating fan lines, one
    direction); stage2_fronts/stage3_fronts are the FRONT polylines
    connecting points across the fan at each marching step (the other
    direction) -- together these two families are what makes it a grid
    rather than a one-directional fan. Matches the composite figure
    MOCSolver.plot_results()/MOCSolverMLN.plot_results() draw live during
    a show_plot=True solve (Sauer line, reflex-zone connectors, and all),
    reconstructed here from the saved data dict instead of needing the
    live solver object.

    reference=<data dict> overlays a second design's WALL only by default
    (dashed) -- its full characteristic mesh is skipped unless
    show_reference_mesh=True, since two overlaid meshes get visually busy
    fast.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    def _draw_segments(segments, color, lbl, lw=0.5, alpha=0.7):
        first = True
        for xs, ys in segments:
            ax.plot(xs, ys, "-", color=color, lw=lw, alpha=alpha,
                     label=lbl if first else None)
            if mirror:
                ax.plot(xs, _mirror(ys), "-", color=color, lw=lw, alpha=alpha)
            first = False

    def _draw_mesh(d, alpha_scale=1.0):
        mesh = d.get("mesh_data", {})
        _draw_segments(mesh.get("stage2_c_plus", []), COLOR_C_PLUS, "Stage 2 (C+)", alpha=0.7 * alpha_scale)
        _draw_segments(mesh.get("stage2_c_minus", []), COLOR_C_MINUS, "Stage 2 (C-)", alpha=0.7 * alpha_scale)
        _draw_segments(mesh.get("stage3_mesh", []), COLOR_C_PLUS, "Stage 3", alpha=0.7 * alpha_scale)
        if show_fronts:
            _draw_segments(mesh.get("stage2_fronts", []), COLOR_C_MINUS, "Stage 2 front", alpha=0.7 * alpha_scale)
            _draw_segments(mesh.get("stage3_fronts", []), COLOR_C_MINUS, "Stage 3 front", alpha=0.7 * alpha_scale)

        if show_reflex and d.get("IVP", {}).get("x") and d.get("div_wall", {}).get("x"):
            ivp, div_wall = d["IVP"], d["div_wall"]
            for xi, yi, xd, yd in zip(ivp["x"], ivp["y"], div_wall["x"], div_wall["y"]):
                ax.plot([xi, xd], [yi, yd], "-", color=COLOR_C_MINUS, lw=0.8, alpha=0.7 * alpha_scale)
                if mirror:
                    ax.plot([xi, xd], [-yi, -yd], "-", color=COLOR_C_MINUS, lw=0.8, alpha=0.7 * alpha_scale)
            ax.plot(div_wall["x"], div_wall["y"], "o", color=COLOR_WALL, ms=3, alpha=alpha_scale)
            if mirror:
                ax.plot(div_wall["x"], _mirror(div_wall["y"]), "o", color=COLOR_WALL, ms=3, alpha=alpha_scale)

        if show_sauer and d.get("sauer", {}).get("x"):
            sauer = d["sauer"]
            ax.plot(sauer["x"], sauer["y"], "-o", color=COLOR_C_PLUS, lw=1.5, ms=3,
                     alpha=alpha_scale, label="Sonic line" if alpha_scale == 1.0 else None)
            if mirror:
                ax.plot(sauer["x"], _mirror(sauer["y"]), "-o", color=COLOR_C_PLUS, lw=1.5, ms=3, alpha=alpha_scale)

    def _draw_wall(d, ls, alpha, lbl):
        wall = d["wall_final"]
        ax.plot(wall["x"], wall["y"], ls, color=COLOR_WALL, lw=1.5, alpha=alpha, label=lbl)
        if mirror:
            ax.plot(wall["x"], _mirror(wall["y"]), ls, color=COLOR_WALL, lw=1.5, alpha=alpha)

    if reference is not None:
        if show_reference_mesh:
            _draw_mesh(reference, alpha_scale=0.4)
        _draw_wall(reference, "--", 0.5, reference_label)

    _draw_mesh(data)
    if show_wall:
        _draw_wall(data, "-", 1.0, label if reference is not None else "Wall")
        if mirror:
            ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def _property_vs_x(x, p, M, xlabel, p_label, m_label, ax_p=None,
                    ref_x=None, ref_p=None, ref_M=None,
                    label="Current", reference_label="Reference"):
    if ax_p is None:
        fig, ax_p = plt.subplots(figsize=(6, 4))
    else:
        fig = ax_p.figure
    ax_M = ax_p.twinx()

    show_legend = ref_x is not None
    if show_legend:
        ax_p.plot(ref_x, [v / 1e5 for v in ref_p], color=COLOR_PRESSURE, lw=2, ls="--", alpha=0.5)
        ax_M.plot(ref_x, ref_M, color=COLOR_MACH, lw=2, ls="--", alpha=0.5)
        # Dummy black-ish handles just to label current vs reference in one legend
        ax_p.plot([], [], color="gray", lw=2, ls="-", label=label)
        ax_p.plot([], [], color="gray", lw=2, ls="--", alpha=0.6, label=reference_label)

    ax_p.plot(x, [v / 1e5 for v in p], color=COLOR_PRESSURE, lw=2)
    ax_M.plot(x, M, color=COLOR_MACH, lw=2)

    ax_p.set_xlabel(xlabel)
    ax_p.set_ylabel(p_label, color=COLOR_PRESSURE)
    ax_M.set_ylabel(m_label, color=COLOR_MACH)
    ax_p.tick_params(axis="y", labelcolor=COLOR_PRESSURE)
    ax_M.tick_params(axis="y", labelcolor=COLOR_MACH)
    if show_legend:
        ax_p.legend(loc="best", fontsize=9)
    return fig, (ax_p, ax_M)


def plot_axis_properties(data, ax=None, reference=None, label="Current", reference_label="Reference"):
    """Pressure (left, orange) + Mach (right, green) vs axial distance
    along the centerline -- data['axis']. reference=<data dict> overlays a
    second design's axis properties (dashed)."""
    axis = data["axis"]
    ref = reference["axis"] if reference is not None else None
    return _property_vs_x(axis["x"], axis["p"], axis["M"], "x (m)", "P (bar)", "M (-)", ax_p=ax,
                           ref_x=ref["x"] if ref else None, ref_p=ref["p"] if ref else None,
                           ref_M=ref["M"] if ref else None, label=label, reference_label=reference_label)


def plot_wall_properties(data, ax=None, reference=None, label="Current", reference_label="Reference"):
    """Pressure (left, orange) + Mach (right, green) vs axial distance
    along the wall -- data['wall_final']. reference=<data dict> overlays a
    second design's wall properties (dashed)."""
    wall = data["wall_final"]
    ref = reference["wall_final"] if reference is not None else None
    return _property_vs_x(wall["x"], wall["p"], wall["M"], "x (m)", "P (bar)", "M (-)", ax_p=ax,
                           ref_x=ref["x"] if ref else None, ref_p=ref["p"] if ref else None,
                           ref_M=ref["M"] if ref else None, label=label, reference_label=reference_label)


def plot_ts_diagram(data, ax=None, show_dome=True, show_spinodal=False, reference=None,
                     label="Current", reference_label="Reference"):
    """T-s diagram: the saturation dome (from data['fluid_name']) plus
    three highlighted points of the expansion -- Total (stagnation,
    data['s0']/['T0_stagnation']), MoC start (the first solved axis point,
    ~the sonic/critical state Stage 1 seeds from -- data['axis']['s'][0]/
    ['T'][0]), and Outlet (the last axis point). MoC start and Outlet are
    connected with a straight line (the two endpoints of what the solver
    actually marches, as opposed to Total, which sits upstream of where
    the solve starts at ~zero velocity).

    The dome itself is drawn with jaxprop's own Fluid.plot_phase_diagram
    (the same T-s/phase-diagram plotter used across the rest of this
    codebase's jaxprop-based scripts), not a hand-rolled line plot --
    moc._phase_diagram.get_fluid caches the Fluid per fluid name so the
    saturation line (and, if requested, the spinodal line) is computed
    once and reused, not recomputed on every call. show_spinodal=True adds
    the metastable-region spinodal line (relevant for the subcooled-liquid
    flashing inlets this solver supports) -- off by default since it's a
    real iterative optimization per point, not free like the saturation
    line.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    if show_dome:
        fluid = get_fluid(data["fluid_name"])
        fluid.plot_phase_diagram(
            "s", "T", axes=ax,
            plot_saturation_line=True, plot_critical_point=True,
            plot_spinodal_line=show_spinodal,
        )

    def _draw(d, alpha, ls, suffix):
        s_total, T_total = d["s0"], d["T0_stagnation"]
        s_start, T_start = d["axis"]["s"][0], d["axis"]["T"][0]
        s_out, T_out = d["axis"]["s"][-1], d["axis"]["T"][-1]

        ax.plot([s_start, s_out], [T_start, T_out], ls, color=COLOR_MACH, lw=1.5, alpha=alpha)
        ax.plot(s_total, T_total, "s", color=COLOR_WALL, ms=8, alpha=alpha,
                 label=f"Total{suffix}")
        ax.plot(s_start, T_start, "o", color=COLOR_C_PLUS, ms=8, alpha=alpha,
                 label=f"MoC start{suffix}")
        ax.plot(s_out, T_out, "^", color=COLOR_PRESSURE, ms=8, alpha=alpha,
                 label=f"Outlet{suffix}")

    if reference is not None:
        _draw(reference, 0.5, "--", f" ({reference_label})")
    _draw(data, 1.0, "-", f" ({label})" if reference is not None else "")

    ax.set_xlabel("s (J/kg/K)")
    ax.set_ylabel("T (K)")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def plot_blade(blade_data, ax=None, n_blades=2, show_control_points=True,
                show_cp_labels=True, show_trailing_edge=True, highlight_cp=None,
                reference=None, label="Current", reference_label="Reference"):
    """Cascade view of the fitted blade profile -- moc.geometry.
    parametrize_stator_blade's ``blade_curve`` (closed B-spline contour)
    repeated n_blades times at the design pitch, in mm (this module's
    only plot not in SI, matching the ported blade-parametrization
    script's own working units). reference=<second blade_data dict>
    overlays a second design (dashed, lighter). highlight_cp=<int> marks
    one control point of the CURRENT (not reference) blade with a
    distinct marker -- used by moc.app's interactive control-point editor
    (see moc.geometry.move_control_point_along_normal) to show which
    point a nudge will move. show_cp_labels annotates each control point
    of the CURRENT blade with its index (needed to pick a cp_index for
    that editor) -- reference control points are never labeled, to keep
    an overlaid comparison readable."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    def _draw(d, ls, alpha, lbl):
        pitch = d["pitch"]
        curve = d["blade_curve"]
        te = d["trailing_edge"]
        cps = d["control_points"]
        for i in range(n_blades):
            dx = i * pitch
            ax.plot([v + dx for v in curve["x"]], curve["y"],
                    ls, color=COLOR_WALL, lw=2, alpha=alpha,
                    label=lbl if i == 0 else None)
            if show_trailing_edge:
                ax.plot([v + dx for v in te["x"]], te["y"], ls, color=COLOR_C_PLUS,
                        lw=1.5, alpha=alpha)
            if show_control_points and i == 0:
                cp_x = [p[0] + dx for p in cps]
                cp_y = [p[1] for p in cps]
                ax.plot(cp_x, cp_y, "--", color=COLOR_CP_LINE, lw=1.2, alpha=0.7 * alpha,
                        marker="o", mfc=COLOR_CP_FACE, mec="black", mew=0.8, ms=8, zorder=10)

    if reference is not None:
        _draw(reference, "--", 0.5, reference_label)
    _draw(blade_data, "-", 1.0, label if reference is not None else "Blade profile")

    if show_control_points and show_cp_labels:
        for idx, (cx, cy) in enumerate(blade_data["control_points"]):
            ax.annotate(str(idx), xy=(cx, cy), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, fontweight="bold", color=COLOR_CP_LINE, zorder=12)

    if show_control_points and highlight_cp is not None:
        cps = blade_data["control_points"]
        if 0 <= highlight_cp < len(cps):
            ax.plot(cps[highlight_cp][0], cps[highlight_cp][1], marker="*", mfc="#ffe600",
                     mec="#d81b1b", mew=1.5, ms=22, linestyle="none", zorder=11,
                     label=f"Selected CP {highlight_cp}")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def plot_rotor_blade(data, ax=None, show_junctions=True,
                      reference=None, label="Current", reference_label="Reference"):
    """
    Pressure/suction surfaces from moc.rotor.design_rotor_vortex_blade
    (NASA TN D-4421/Goldman & Scullin 1968 vortex-flow method, validated
    against the report's own worked example -- see vortex_blade.py's
    module docstring), each plotted in ITS OWN local frame (own vortex
    center at the origin) -- the two surfaces are NOT stitched into a
    single closed blade contour here (that needs the leading/trailing-edge
    treatment TN D-4421 covers separately, plus a rounding radius, neither
    implemented yet); `data["pitch"]`/`data["chord"]` give the NATURAL
    (not guessed) blade-to-blade spacing and the lower surface's own
    inlet-to-outlet chord length, derived directly from the geometry.

    show_junctions marks the transition-arc/circular-arc boundary on each
    surface (small circle) -- purely informational; with the corrected
    (real marching-based) construction these should NOT show a visible
    kink. reference=<second data dict> overlays a second design
    (dashed/lighter).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    def _draw(d, ls, alpha, lbl):
        for surf_key, color in (("pressure", COLOR_PRESSURE), ("suction", COLOR_MACH)):
            surf = d[surf_key]
            ax.plot(surf["x"], surf["y"], ls, color=color, lw=2, alpha=alpha,
                     label=f"{lbl} {surf_key}" if lbl else surf_key.capitalize())
            if show_junctions:
                n = len(surf["x"])
                n_pts = d.get("num_points") or n // 3  # fallback if not echoed back
                for idx in (n_pts - 1, n - n_pts):
                    if 0 <= idx < n:
                        ax.plot(surf["x"][idx], surf["y"][idx], "o", mfc="none",
                                mec=color, mew=1.3, ms=7, alpha=alpha, zorder=10)

    if reference is not None:
        _draw(reference, "--", 0.5, reference_label)
    _draw(data, "-", 1.0, label if reference is not None else "")

    units = data.get("units", "r*")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


COLOR_BLADE_FILL = "0.55"


def plot_rotor_vortex_blade(data, ax=None, show_surfaces=True, show_mach_lines=False,
                              reference=None, label="Current", reference_label="Reference"):
    """Filled vortex-flow rotor blade (moc.rotor.design_rotor_vortex_blade's
    "blade" key) -- see notes/rotor_vortex_blade.md for what this closure
    actually is (pressure joined to suction shifted by +pitch, closed by
    tangent segments at beta_inlet/beta_outlet -- NOT pressure+suction of
    the same instance closed directly). show_surfaces overlays the raw
    pressure/suction curves; show_mach_lines overlays the recorded
    characteristic segments from the transition-arc marching.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    def _draw(d, alpha, lbl):
        blade = d["blade"]
        ax.fill(blade["x"], blade["y"], color=COLOR_BLADE_FILL, edgecolor="0.15",
                 lw=1.5, alpha=0.85 * alpha, label=lbl or "Blade")
        if show_surfaces:
            for surf_key, color in (("pressure", COLOR_PRESSURE), ("suction", COLOR_MACH)):
                surf = d[surf_key]
                ax.plot(surf["x"], surf["y"], color=color, lw=1.2, alpha=0.7 * alpha,
                         label=f"{lbl + ' ' if lbl else ''}{surf_key}")
        if show_mach_lines:
            for surf_key in ("pressure", "suction"):
                surf = d[surf_key]
                for x0, y0, x1, y1 in surf["mach_lines_inlet"] + surf["mach_lines_outlet"]:
                    ax.plot([x0, x1], [y0, y1], color="0.6", lw=0.4, alpha=0.6 * alpha)

    if reference is not None:
        _draw(reference, 0.5, reference_label)
    _draw(data, 1.0, label if reference is not None else "")

    units = data.get("units", "r*")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


COLOR_RADIAL_STATOR = "#7e03a8"  # plasma colormap, ~25% stop (purple)
COLOR_RADIAL_ROTOR = "#f89441"  # plasma colormap, ~75% stop (orange)


def _draw_radial_set_mpl(ax, blades, trailing_edges, line_color, fillcolor, name):
    """One source's (stator's or rotor's) blade copies onto an existing
    annular-cascade axes. fillcolor=None draws outline-only (single-source
    mode); a fillcolor draws a translucent fill (combined mode), same
    convention as Phase 4's stator/rotor cascade overlay."""
    for i, (blade, te) in enumerate(zip(blades, trailing_edges)):
        if fillcolor:
            ax.fill(blade["x"], blade["y"], color=fillcolor, edgecolor=line_color,
                     lw=1.2, alpha=0.35, label=name if i == 0 else None)
        else:
            ax.plot(blade["x"], blade["y"], color=line_color, lw=1.2, alpha=0.85,
                     label=name if i == 0 else None)
        if te["x"]:
            ax.plot(te["x"], te["y"], color=COLOR_C_PLUS, lw=1.0, alpha=0.85)


def plot_radial_cascade(radial_data, ax=None, show_radius_circles=True):
    """Full annular cascade view of moc.geometry.wrap_blade_radial(...) /
    wrap_rotor_blade_radial(...) results: every blade copy plus its
    trailing edge, laid out around the annulus, with the r1/r2 bounding
    circles as a visual reference. Also accepts a "combined" dict
    (radial_data["mode"]=="combined", with "stator"/"rotor" sub-dicts) to
    overlay the two rows, same as plotting_plotly's version."""
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    combined = radial_data.get("mode") == "combined"
    if combined:
        stator, rotor = radial_data["stator"], radial_data["rotor"]
        _draw_radial_set_mpl(ax, stator["blades"], stator["trailing_edges"],
                              COLOR_RADIAL_STATOR, COLOR_RADIAL_STATOR, "Stator")
        _draw_radial_set_mpl(ax, rotor["blades"], rotor["trailing_edges"],
                              "black", COLOR_RADIAL_ROTOR, "Rotor")
        n_label = f"stator={stator['n_blades']}, rotor={rotor['n_blades']}"
    else:
        _draw_radial_set_mpl(ax, radial_data["blades"], radial_data["trailing_edges"],
                              COLOR_WALL, None, "Blade")
        n_label = str(radial_data["n_blades"])

    r1, r2 = radial_data["r1"], radial_data["r2"]
    if show_radius_circles:
        t = np.linspace(0, 2 * np.pi, 200)
        circles = [(r1, "r1")]
        if combined:
            circles.append((radial_data["r_stator_out"], "r_interface (stator out)"))
            circles.append((radial_data["r_rotor_in"], "r_rotor_in"))
        circles.append((r2, "r2"))
        for r, lbl in circles:
            ax.plot(r * np.cos(t), r * np.sin(t), ":", color="0.6", lw=1, label=lbl)

    units = radial_data.get("units", "mm")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_aspect("equal")
    ax.set_title(f"Radial cascade ({n_label} blades, r1={r1:.2f}, r2={r2:.2f} {units})")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def plot_meridional_view(meridional_data, ax=None, units="mm"):
    """Meridional (axial-radial, x-r) sketch of a stator + rotor stage
    from moc.geometry.build_meridional_view(...): hub/tip lines for each
    row (straight, per the shared-axial-station flare convention -- see
    that function's docstring), with the passage between them filled."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    def _draw_row(row, color, name):
        x, hub, tip = row["x"], row["hub"], row["tip"]
        ax.fill(x + x[::-1], hub + tip[::-1], color=color, alpha=0.25, label=name)
        ax.plot(x, hub, "-o", color=color, lw=2, ms=4)
        ax.plot(x, tip, "--o", color=color, lw=2, ms=4)

    _draw_row(meridional_data["stator"], COLOR_RADIAL_STATOR, "Stator")
    _draw_row(meridional_data["rotor"], COLOR_RADIAL_ROTOR, "Rotor")

    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"r ({units})")
    ax.set_aspect("equal")
    ax.set_title(f"Meridional view (gap = {meridional_data['gap']:.2f} {units})")
    ax.legend(loc="best", fontsize=9)
    return fig, ax
