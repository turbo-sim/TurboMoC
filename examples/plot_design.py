"""
plot_design.py -- demonstrates moc.mpl / moc.plotly on a single design.

Run:
    conda activate moc_env
    python examples/plot_design.py

Produces, next to this script, under output/plots/:
  contour_mpl.png, mesh_mpl.png, axis_mpl.png, wall_mpl.png
  contour_plotly.html, mesh_plotly.html, axis_plotly.html, wall_plotly.html
"""

import os
import matplotlib
matplotlib.use("Agg")
import jaxprop as jxp

import moc

fluid_name = "nitrogen"
pressure = 20e5
T_sat = float(jxp.Fluid(fluid_name).get_state(jxp.PQ_INPUTS, pressure, 0.0).T)

data = moc.design_nozzle(
    fluid_name=fluid_name, P0=pressure, T0=T_sat - 10, p_back=pressure / 10,
    solver="conventional",
)
print(f"\nconverged={data['converged']}  exit_M={data['exit_M']:.4f}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "plots")
os.makedirs(out_dir, exist_ok=True)

# --- matplotlib ---
fig, _ = moc.mpl.plot_nozzle_contour(data)
fig.savefig(os.path.join(out_dir, "contour_mpl.png"), dpi=150, bbox_inches="tight")

fig, _ = moc.mpl.plot_characteristics_mesh(data)
fig.savefig(os.path.join(out_dir, "mesh_mpl.png"), dpi=150, bbox_inches="tight")

fig, _ = moc.mpl.plot_axis_properties(data)
fig.savefig(os.path.join(out_dir, "axis_mpl.png"), dpi=150, bbox_inches="tight")

fig, _ = moc.mpl.plot_wall_properties(data)
fig.savefig(os.path.join(out_dir, "wall_mpl.png"), dpi=150, bbox_inches="tight")

# --- plotly ---
moc.plotly.plot_nozzle_contour(data).write_html(os.path.join(out_dir, "contour_plotly.html"))
moc.plotly.plot_characteristics_mesh(data).write_html(os.path.join(out_dir, "mesh_plotly.html"))
moc.plotly.plot_axis_properties(data).write_html(os.path.join(out_dir, "axis_plotly.html"))
moc.plotly.plot_wall_properties(data).write_html(os.path.join(out_dir, "wall_plotly.html"))

print(f"Saved 8 plots -> {out_dir}")
