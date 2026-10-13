"""Design a two-phase rotor; export blade coordinates [m] and Matplotlib figures.

Run from the repository root: poetry run python examples/moc_design/rotor_design.py
Results are saved in examples/output/rotor_design/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import turbo_moc
from turbo_moc.configuration.fluid_inputs import solver_parameters

from turbo_moc.core.vortex_blade import design_rotor_vortex_blade

# 1. Resolve the saturated inlet state in the rotor-relative frame.
pressure = 20e5
quality = 0.5

# 2. Set the inlet/outlet Mach numbers and the two constant-Mach wall arcs.
# The solver derives the outlet angle; r_star sets the physical length scale.
data = design_rotor_vortex_blade(
    **solver_parameters(dict(
        fluid_name="nitrogen",
        inlet_property_pair="PQ_INPUTS",
        inlet_property_1=pressure,  # Relative stagnation pressure [Pa].
        inlet_property_2=quality,  # Relative stagnation vapor mass fraction.
    ), rotor=True),
    M_inlet=1.6,
    M_outlet=1.2,
    M_lower=1.1,
    M_upper=2.0,
    beta_inlet=5.0,
    r_star=0.01,  # Sonic-streamline radius [m]; gives dimensional coordinates.
    num_points=60,
)

blade = data["blade"]
if (
    not blade["x"]
    or not np.isfinite(data["solidity"])
    or not np.isfinite(np.column_stack((blade["x"], blade["y"]))).all()
):
    raise RuntimeError("The rotor design produced invalid blade geometry.")

# Plotting helpers return (figure, axes); keep the figures for SVG export.
figures = {
    "rotor_blade": turbo_moc.mpl.plot_rotor_vortex_blade(
        data,
        show_surfaces=False,
    )[0],
    "rotor_passage": turbo_moc.mpl.plot_rotor_blade(data)[0],
}
print(
    f"Pitch: {data['pitch']:.6f} m  Chord: {data['chord']:.6f} m  "
    f"Solidity: {data['solidity']:.4f}"
)

# Resolve output relative to this script, regardless of the working directory.
outdir = Path(__file__).resolve().parents[1] / "output" / "rotor_design"
outdir.mkdir(parents=True, exist_ok=True)
# Export the assembled blade; passage surfaces have their own local frames.
np.savetxt(
    outdir / "blade.csv",
    np.column_stack((blade["x"], blade["y"])),
    delimiter=",",
    header="x_m,y_m",
    comments="",
)
for name in ("pressure", "suction"):
    surface = data[name]
    np.savetxt(
        outdir / f"{name}_local.csv",
        np.column_stack((surface["x"], surface["y"])),
        delimiter=",",
        header="x_m,y_m",
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
