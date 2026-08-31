"""
Minimal runnable example for moc.design_nozzle -- run with:

    conda activate moc_env
    python examples/run_design.py

(from the moc/ package root, or anywhere once the package is installed --
`pip install -e .` makes `moc` importable regardless of working directory).
"""
import os
import jaxprop as jxp
import moc

fluid_name = "nitrogen"
pressure = 20e5
T_sat = float(jxp.Fluid(fluid_name).get_state(jxp.PQ_INPUTS, pressure, 0.0).T)

data = moc.design_nozzle(
    fluid_name=fluid_name,
    P0=pressure,
    T0=T_sat - 10,      # 10 K subcooled-liquid (flashing) inlet
    p_back=pressure / 10,
    solver="conventional",   # or "mln"
)

# moc.plotly (moc.plotting_plotly) returns go.Figure objects rather than
# drawing to a shared pyplot state -- each one needs its own .show(), and
# outside a notebook that opens a browser tab per figure. Saving to SVG
# (via kaleido) avoids that and leaves a persistent, inspectable result.
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(outdir, exist_ok=True)
moc.plotly.plot_nozzle_contour(data).write_image(os.path.join(outdir, "stator_nozzle_contour.svg"))
moc.plotly.plot_axis_properties(data).write_image(os.path.join(outdir, "stator_axis_properties.svg"))
moc.plotly.plot_wall_properties(data).write_image(os.path.join(outdir, "stator_wall_properties.svg"))
moc.plotly.plot_characteristics_mesh(data).write_image(os.path.join(outdir, "stator_characteristics_mesh.svg"))
print(f"saved figures to {outdir}")

print(f"\nconverged={data['converged']}  exit_M={data['exit_M']:.4f}  runtime={data['runtime_s']:.1f}s")
wall = data["wall_final"]
print(f"exit wall: x={wall['x'][-1]:.6f}  y={wall['y'][-1]:.6f}  p={wall['p'][-1]:.1f} Pa")
