"""Plot the package's vortex rotor and three-guide Bezier flow domain.

Run: python examples/moc_design/rotor_flow_domain_construction.py
Edit the controls below; no command-line arguments are needed. The manual
rotor-only case is solved once using the same geometry builders as the app.
Plots show downstream x = -q_y and pitchwise y = q_x, in mm: flow left to right.
No mesh is generated. There is no separate construction implementation here.
"""
from pathlib import Path

# Example controls.
EXAMPLES = Path(__file__).resolve().parents[1]
CASE = EXAMPLES / "apps/cases/rotor_only.yaml"
ROW_ID = "rotor_1"
OUTDIR = EXAMPLES / "output/rotor_flow_domain_construction"
SHOW_FIGURES = True
SAVE_FIGURES = True
SHOW_CONTROL_POINTS = True
FIGURE_SIZE = (10, 4.5)
JOIN_FIGURE_SIZE = (8, 3.5)
ZOOM_CHORD_FRACTION = 0.045
CURVE_SAMPLES = 600

import matplotlib
import numpy as np

from turbo_moc.configuration import project as projects
from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.geometry import passage_2d as planar
from turbo_moc.geometry.rotor_blade import build_rotor_design
from turbo_moc.core.vortex_blade import design_rotor_vortex_blade
from turbo_moc.configuration.rotor import vortex_parameters


SEGMENTS = ("Pressure wall", "TE fillet", "TE stub", "Suction wall", "LE stub", "LE fillet")
COLORS = ("#d97706", "#dc2626", "#a855f7", "#2563eb", "#0891b2", "#16a34a")
REFERENCES = (("inlet_spline", "Inlet extension", "#16a34a"),
              ("blade_reference_spline", "Bezier reference", "#2563eb"),
              ("outlet_spline", "Outlet extension", "#dc2626"))


def display(points):
    """Rotate curves, controls and annotations together, without deformation."""
    points = np.asarray(points)
    return np.stack((-points[..., 1], points[..., 0]), axis=-1)


def plot(ax, points, *args, **kwargs):
    ax.plot(*display(points).T, *args, **kwargs)


def finish(ax, title):
    ax.set(title=title, xlabel="Downstream x [mm]", ylabel="Pitchwise y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)


def draw_domain(ax, curves, pitch, controls=False):
    for name, label, color in REFERENCES:
        points = curves[name](np.linspace(0, 1, CURVE_SAMPLES))
        plot(ax, points, color=color, lw=1.8, label=label)
        for sign in (-1, 1):
            plot(ax, points + [sign * pitch / 2, 0], color=color, lw=1, ls="--")
    for side in ("inlet", "outlet"):
        plot(ax, curves[side], color="black", lw=1.2)
    if controls:
        for name, cp in (("I", curves["inlet_control_points"]),
                         ("B", curves["blade_reference_spline"].c),
                         ("O", curves["outlet_control_points"])):
            plot(ax, cp, "o:", color="0.45", ms=3, lw=1)
            for i, point in enumerate(display(cp)):
                ax.annotate(f"{name}{i}", point, xytext=(4, 4),
                            textcoords="offset points", fontsize=7)


def main():
    if not SHOW_FIGURES:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project = projects.load_project(CASE)
    config = projects.row_configuration(project, ROW_ID)
    print(f"Generating vortex rotor from {CASE}", flush=True)
    result = design_rotor_vortex_blade(**vortex_parameters(solver_parameters(config["rotor_design"], rotor=True)))
    design = build_rotor_design(result, config)
    blade, curves = design.blade, design.curves
    contour = planar.blade_contour(blade)
    middle = curves["blade_reference"]
    guides = curves["blade_reference_guides"]
    chord, pitch = np.ptp(contour[:, 1]), blade["pitch"]

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, layout="constrained")
    ax = axes[0]
    ax.fill(*display(contour).T, color="0.93")
    for spline, label, color in zip(blade["boundary_splines"], SEGMENTS, COLORS):
        points = spline(np.linspace(0, 1, CURVE_SAMPLES))
        plot(ax, points, color=color, lw=2, label=label)
        plot(ax, points[[0, -1]], "o", color=color, ms=3)
    plot(ax, middle, "k--", lw=1, label="Bezier reference")
    plot(ax, guides, "ko", ms=3, label="10/50/90% guides")
    finish(ax, "Blade construction (flow left to right)")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    ax.fill(*display(contour).T, color="0.85")
    draw_domain(ax, curves, pitch, controls=SHOW_CONTROL_POINTS)
    plot(ax, guides, "ko", ms=3, label="Midpoint guides")
    plot(ax, middle[[0, -1]], "o", color="magenta", ms=4, label="Extension joins")
    finish(ax, "Domain: dashed periodics at ± pitch/2")
    ax.legend(fontsize=7, loc="best")

    detail, axes = plt.subplots(1, 2, figsize=JOIN_FIGURE_SIZE, layout="constrained")
    for ax, index, name in ((axes[0], 0, "Inlet"), (axes[1], -1, "Outlet")):
        ax.fill(*display(contour).T, color="0.9")
        draw_domain(ax, curves, pitch)
        anchor = middle[index]
        plot(ax, np.array([anchor]), "o", color="magenta", ms=6)
        x, y = display(anchor)
        half_width = ZOOM_CHORD_FRACTION * chord
        ax.set_xlim(x - half_width, x + half_width)
        ax.set_ylim(y - half_width, y + half_width)
        finish(ax, f"{name} extension/Bezier join")

    clearance = planar.assess_clearance(blade, curves)
    print(f"Chord: {chord:.3f} mm; pitch: {pitch:.3f} mm; reference shift: {curves['shift_mm']:.3f} mm")
    print(f"Minimum clearance: {clearance['minimum_clearance']:.3f} mm; blade contained: {clearance['contained']}")
    if SAVE_FIGURES:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for name, figure in (("rotor_construction", fig), ("rotor_join_diagnostics", detail)):
            for extension in ("png", "svg"):
                figure.savefig(OUTDIR / f"{name}.{extension}", dpi=180)
        print(f"Figures saved to {OUTDIR}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
