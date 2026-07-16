"""
generate_laes_cases.py

Generates both nozzle-design variants -- conventional (rounded-arc kernel)
and MLN (minimum-length, sharp-corner kernel) -- for the LAES subcooled-
liquid-flashing case, at each of several subcooling levels, via
moc.design_nozzle. Replaces the old (pre-packaging) generate_laes_cases.py
scripts that used to live separately in moc_min_length/ and
moc_tool_flashing/ and duplicated most of this driver logic -- both solvers
now share the same case-generation loop, since design_nozzle already
handles solver selection.

For each dT_sub, the design exit Mach number is NOT an arbitrary choice:
it is the actual isentropic exit Mach reached by that subcooled-liquid
state when expanded down to p_out = P0/10, computed automatically by
design_nozzle from p_back (see its docstring).

Run:
    conda activate moc_env
    python examples/generate_laes_cases.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")  # headless: avoid blocking on plt.show() in a loop
import matplotlib.pyplot as plt
import jaxprop as jxp

from moc import design_nozzle, NozzleDesignError

fluid_name = "nitrogen"
pressure = 20e5  # inlet stagnation pressure P0
T_sat = float(jxp.Fluid(fluid_name).get_state(jxp.PQ_INPUTS, pressure, 0.0).T)
p_out = pressure / 10

dT_list = [2.5, 5, 7.5, 10]
solvers = ["conventional", "mln"]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "laes_cases")
os.makedirs(out_dir, exist_ok=True)

results = []

for dT in dT_list:
    T0 = T_sat - dT
    for solver in solvers:
        print(f"\n\n{'=' * 70}\nLAES case dT_sub = {dT} K, solver={solver}"
              f"   (T0={T0:.3f} K)\n{'=' * 70}\n")

        case_dir = os.path.join(out_dir, f"dT_{dT}K", solver)
        os.makedirs(case_dir, exist_ok=True)

        try:
            data = design_nozzle(
                fluid_name=fluid_name, P0=pressure, T0=T0, p_back=p_out,
                solver=solver, y_t=0.01, rho_t=0.2, rho_d=0.1,
            )
        except NozzleDesignError as e:
            print(f"\n***** FAILED case dT={dT}K solver={solver}: {e} *****\n")
            results.append(dict(dT=dT, solver=solver, T0=T0, failed=str(e)))
            continue

        with open(os.path.join(case_dir, "moc_result_data.json"), "w") as f:
            json.dump(data, f, indent=4)

        wall = data["wall_final"]
        results.append(dict(
            dT=dT, solver=solver, T0=T0,
            design_Noz_Mach=data["design_Noz_Mach"], exit_M=data["exit_M"],
            converged=data["converged"], runtime_s=data["runtime_s"],
            x_exit=wall["x"][-1], y_exit=wall["y"][-1],
        ))
        print(f"\nSaved case dT={dT}K solver={solver} -> {case_dir}")

plt.close("all")

print("\n\n===== Summary (LAES flashing cases, both solvers) =====")
for r in results:
    print(r)
