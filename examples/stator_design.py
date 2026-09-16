"""Design a stator from a two-phase nozzle; export coordinates and Matplotlib figures.

Run from the repository root: poetry run python examples/stator_design.py
Results are saved in examples/output/stator_design/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import turbo_moc

from turbo_moc.geometry import parametrize_stator_blade

# 1. Cyclopentane stator design conditions from Cioffi et al. (2026),
#    Table 3. This uses the current package geometry, not an exact reproduction
#    of the paper's control-point construction.
data = turbo_moc.design_nozzle(
    fluid_name="Cyclopentane",
    P0=2.513e5,  # Stagnation pressure [Pa].
    Q0=0.025,  # Two-phase inlet: 2.5% vapor by mass.
    p_back=0.95e5,  # Design exit pressure [Pa].
    solver="conventional",
    use_true_sauer_line=True,  # Smooth sonic transition within the mixture.
)

if not data["converged"] or not data["wall_final"]["x"]:
    raise RuntimeError("The nozzle design did not converge.")

wall = data["wall_final"]

# Nozzle coordinates are in meters; the stator geometry is returned in mm.
blade = parametrize_stator_blade(
    wall["x"],
    wall["y"],
    metal_angle_in=0.0,
    metal_angle_out=70.0,
    r_trailing=0.5,
    inlet_opening_ratio=1.5,
)

curve = blade["blade_curve"]
if not np.isfinite(np.column_stack((curve["x"], curve["y"]))).all():
    raise RuntimeError("The stator design produced invalid coordinates.")

# Plotting helpers return (figure, axes); keep the figures for SVG export.
figures = {
    "nozzle_contour": turbo_moc.mpl.plot_nozzle_contour(data)[0],
    "characteristics_mesh": turbo_moc.mpl.plot_characteristics_mesh(data)[0],
    "axis_properties": turbo_moc.mpl.plot_axis_properties(data)[0],
    "wall_properties": turbo_moc.mpl.plot_wall_properties(data)[0],
}
figures["stator_blade"] = turbo_moc.mpl.plot_blade(
    blade,
    n_blades=2,
    show_control_points=False,
)[0]

print(f"Stator pitch: {blade['pitch']:.4f} mm")
# Resolve output relative to this script, regardless of the working directory.
outdir = Path(__file__).resolve().parent / "output" / "stator_design"
outdir.mkdir(parents=True, exist_ok=True)
np.savetxt(
    outdir / "nozzle_wall.csv",
    np.column_stack((wall["x"], wall["y"])),
    delimiter=",",
    header="x_m,y_m",
    comments="",
)
for name in ("blade_curve", "suction", "pressure", "trailing_edge"):
    surface = blade[name]
    np.savetxt(
        outdir / f"{name}.csv",
        np.column_stack((surface["x"], surface["y"])),
        delimiter=",",
        header="x_mm,y_mm",
        comments="",
    )
# Save vector figures that can also be embedded in the documentation.
for name, figure in figures.items():
    figure.savefig(outdir / f"{name}.svg", bbox_inches="tight")
print(f"Coordinates and figures saved to: {outdir}")

# Show figures during interactive runs; smoke tests still exercise all exports.
import os

if os.environ.get("TURBO_MOC_EXAMPLE_SMOKE_TEST") != "1":
    plt.show()
else:
    plt.close("all")
