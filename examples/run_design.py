"""
Minimal runnable example for moc.design_nozzle -- run with:

    conda activate moc_env
    python examples/run_design.py

(from the moc/ package root, or anywhere once the package is installed --
`pip install -e .` makes `moc` importable regardless of working directory).
"""
import jaxprop as jxp
from moc import design_nozzle

fluid_name = "nitrogen"
pressure = 20e5
T_sat = float(jxp.Fluid(fluid_name).get_state(jxp.PQ_INPUTS, pressure, 0.0).T)

data = design_nozzle(
    fluid_name=fluid_name,
    P0=pressure,
    T0=T_sat - 10,      # 10 K subcooled-liquid (flashing) inlet
    p_back=pressure / 10,
    solver="conventional",   # or "mln"
)

print(f"\nconverged={data['converged']}  exit_M={data['exit_M']:.4f}  runtime={data['runtime_s']:.1f}s")
wall = data["wall_final"]
print(f"exit wall: x={wall['x'][-1]:.6f}  y={wall['y'][-1]:.6f}  p={wall['p'][-1]:.1f} Pa")
