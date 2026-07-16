"""
generate_sauerline_case.py

Generates the dT_sub=2.5K conventional (rounded-arc kernel) flashing nozzle
using the TRUE Sauer (1947) transonic small-perturbation sonic line
(design_nozzle(..., use_true_sauer_line=True)) instead of the flat/
uniform-jump-velocity front used by default (generate_laes_cases.py).

dT=2.5K is the only one of the LAES subcooling cases where the isentrope
crosses M=1 smoothly (no discontinuous flashing jump -- see
jax_solve_critical_state's SMOOTH_SONIC mode in moc.core.classes), which is
the only case where the classical Sauer small-perturbation-around-M=1
assumption actually holds.

Everything downstream of Stage 1 (IVP marching, arc-throat kernel, reflex
zone) is unchanged -- the wall's divergent shape is set by rho_d via
rotation_around_center, independent of the initial front's shape, so this
is a pure Stage-1 swap for direct comparison against the flat-front result
from generate_laes_cases.py.

Run:
    conda activate moc_env
    python examples/generate_sauerline_case.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import jaxprop as jxp

from moc import design_nozzle

fluid_name = "nitrogen"
pressure = 20e5
T_sat = float(jxp.Fluid(fluid_name).get_state(jxp.PQ_INPUTS, pressure, 0.0).T)
p_out = pressure / 10
dT = 2.5
T0 = T_sat - dT

print(f"\n\n{'=' * 70}\nLAES case dT_sub = {dT} K, TRUE Sauer sonic line   "
      f"(T0={T0:.3f} K)\n{'=' * 70}\n")

data = design_nozzle(
    fluid_name=fluid_name, P0=pressure, T0=T0, p_back=p_out,
    solver="conventional", y_t=0.01, rho_t=0.2, rho_d=0.1,
    use_true_sauer_line=True,
)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "conventional_sauerline")
case_dir = os.path.join(out_dir, f"dT_{dT}K")
os.makedirs(case_dir, exist_ok=True)

with open(os.path.join(case_dir, "moc_result_data.json"), "w") as f:
    json.dump(data, f, indent=4)

wall = data["wall_final"]
print(f"\nSaved case dT={dT}K (true Sauer line) -> {case_dir}")
print(f"converged={data['converged']}  exit_M={data['exit_M']:.6f}  "
      f"(design {data['design_Noz_Mach']:.6f})")
print(f"x_exit={wall['x'][-1]:.6f}  y_exit={wall['y'][-1]:.6f}")
