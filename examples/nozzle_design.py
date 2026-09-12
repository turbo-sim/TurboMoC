"""Design a two-phase nozzle; save wall coordinates [m] and Matplotlib figures.

Run from the repository root: poetry run python examples/nozzle_design.py
Results are saved in examples/output/nozzle_design/.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import moc

smoke_test = os.environ.get("MOC_EXAMPLE_SMOKE_TEST") == "1"

data = moc.design_nozzle(
    fluid_name="nitrogen",
    P0=20e5,           # Stagnation pressure [Pa].
    Q0=0.5,           # Two-phase inlet: vapor mass fraction.
    p_back=2e5,       # Back pressure [Pa].
    solver="conventional",
)

if not data["converged"] or not data["wall_final"]["x"]:
    raise RuntimeError("The nozzle design did not converge.")

wall = data["wall_final"]

figures = {
    "nozzle_contour": moc.mpl.plot_nozzle_contour(data)[0],
    "characteristics_mesh": moc.mpl.plot_characteristics_mesh(data)[0],
    "axis_properties": moc.mpl.plot_axis_properties(data)[0],
    "wall_properties": moc.mpl.plot_wall_properties(data)[0],
}

print(f"Exit Mach number: {data['exit_M']:.4f}")
# Export both walls of the symmetric nozzle, ordered from throat to exit.
if not smoke_test:
    outdir = Path(__file__).resolve().parent / "output" / "nozzle_design"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        outdir / "wall_upper.csv",
        np.column_stack((wall["x"], wall["y"])),
        delimiter=",", header="x_m,y_m", comments="",
    )
    np.savetxt(
        outdir / "wall_lower.csv",
        np.column_stack((wall["x"], -np.asarray(wall["y"]))),
        delimiter=",", header="x_m,y_m", comments="",
    )
    for name, figure in figures.items():
        figure.savefig(outdir / f"{name}.svg", bbox_inches="tight")
    print(f"Coordinates and figures saved to: {outdir}")
    plt.show()
else:
    plt.close("all")
