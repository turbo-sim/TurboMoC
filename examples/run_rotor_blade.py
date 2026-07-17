from pathlib import Path

import matplotlib.pyplot as plt

import moc.plotting_mpl as mpl
from moc.rotor import design_rotor_vortex_blade

data = design_rotor_vortex_blade(
    fluid_name="air",
    P0_rel=5e5, T0_rel=400.0,
    M_inlet=1.5, M_outlet=1.5,
    M_lower=1.05, M_upper=2.0,
    beta_inlet=65.0, beta_outlet=-65.0,
    backend="HEOS",
    num_points=60,
)
print(f"pitch={data['pitch']:.4f}  chord={data['chord']:.4f}  solidity={data['solidity']:.4f}")

fig, ax = mpl.plot_rotor_blade(data)

b = data["blade"]
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.fill(b["x"], b["y"], color="0.55", edgecolor="0.15", lw=1.5)
ax2.set_xlabel(f"x ({data['units']})")
ax2.set_ylabel(f"y ({data['units']})")
ax2.set_aspect("equal")
ax2.set_title("Rotor vortex blade (Goldman & Scullin 1968, TN D-4421)")
plt.show()
out_dir = Path(__file__).parent / "output" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "rotor_vortex_surfaces.png", dpi=140)
fig2.savefig(out_dir / "rotor_vortex_blade.png", dpi=140)
print(f"saved to {out_dir}")
