"""Design a two-phase rotor; export blade coordinates [m] and Matplotlib figures.

Run from the repository root: poetry run python examples/rotor_design.py
Results are saved in examples/output/rotor_design/.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jaxprop as jxp

import moc

smoke_test = os.environ.get("MOC_EXAMPLE_SMOKE_TEST") == "1"

from moc.rotor import design_rotor_vortex_blade

pressure = 20e5
quality = 0.5
temperature = float(jxp.Fluid("nitrogen").get_state(jxp.PQ_INPUTS, pressure, quality).T)

data = design_rotor_vortex_blade(
    fluid_name="nitrogen",
    P0_rel=pressure,  # Relative stagnation pressure [Pa].
    T0_rel=temperature,  # Saturation temperature [K]; required by the rotor API.
    Q0_rel=quality,   # Relative stagnation vapor mass fraction.
    M_inlet=1.6, M_outlet=1.2,
    M_lower=1.1, M_upper=2.0,
    beta_inlet=5.0,
    r_star=0.01,      # Sonic-streamline radius [m]; gives dimensional coordinates.
    num_points=60,
)

blade = data["blade"]
if (not blade["x"] or not np.isfinite(data["solidity"])
        or not np.isfinite(np.column_stack((blade["x"], blade["y"]))).all()):
    raise RuntimeError("The rotor design produced invalid blade geometry.")

figures = {
    "rotor_blade": moc.mpl.plot_rotor_vortex_blade(
        data, show_surfaces=False,
    )[0],
    "rotor_passage": moc.mpl.plot_rotor_blade(data)[0],
}
print(
    f"Pitch: {data['pitch']:.6f} m  Chord: {data['chord']:.6f} m  "
    f"Solidity: {data['solidity']:.4f}"
)

if not smoke_test:
    outdir = Path(__file__).resolve().parent / "output" / "rotor_design"
    outdir.mkdir(parents=True, exist_ok=True)
    # Export the assembled blade; passage surfaces have their own local frames.
    np.savetxt(
        outdir / "blade.csv",
        np.column_stack((blade["x"], blade["y"])),
        delimiter=",", header="x_m,y_m", comments="",
    )
    for name in ("pressure", "suction"):
        surface = data[name]
        np.savetxt(
            outdir / f"{name}_local.csv",
            np.column_stack((surface["x"], surface["y"])),
            delimiter=",", header="x_m,y_m", comments="",
        )
    for name, figure in figures.items():
        figure.savefig(outdir / f"{name}.svg", bbox_inches="tight")
    print(f"Coordinates and figures saved to: {outdir}")
    plt.show()
else:
    plt.close("all")
