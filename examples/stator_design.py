"""Design a stator from a two-phase nozzle; export coordinates and Matplotlib figures.

Run from the repository root: poetry run python examples/stator_design.py
Results are saved in examples/output/stator_design/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import moc

from moc.geometry import parametrize_stator_blade

# 1. Expand a saturated mixture from 20 bar to the specified back pressure.
data = moc.design_nozzle(
    fluid_name="nitrogen",
    P0=20e5,  # Stagnation pressure [Pa].
    Q0=0.5,  # Two-phase inlet: vapor mass fraction.
    p_back=2e5,  # Back pressure [Pa].
    solver="conventional",
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
)
curve = blade["blade_curve"]
if not np.isfinite(np.column_stack((curve["x"], curve["y"]))).all():
    raise RuntimeError("The stator design produced invalid coordinates.")

# Plotting helpers return (figure, axes); keep the figures for SVG export.
figures = {
    "nozzle_contour": moc.mpl.plot_nozzle_contour(data)[0],
    "characteristics_mesh": moc.mpl.plot_characteristics_mesh(data)[0],
    "axis_properties": moc.mpl.plot_axis_properties(data)[0],
    "wall_properties": moc.mpl.plot_wall_properties(data)[0],
}
figures["stator_blade"] = moc.mpl.plot_blade(
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

if os.environ.get("MOC_EXAMPLE_SMOKE_TEST") != "1":
    plt.show()
else:
    plt.close("all")
