import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(outdir, exist_ok=True)

import moc.plotting_mpl as mpl
from moc.rotor import design_rotor_vortex_blade
from moc.rotor.vortex_blade import _arc_from_diameter, _line_intersect

data = design_rotor_vortex_blade(
    fluid_name="air",
    P0_rel=5e5, T0_rel=400.0,
    M_inlet=1.6, M_outlet=1.2,
    M_lower=1.1, M_upper=2.0,
    beta_inlet=5,
    backend="HEOS",
    num_points=60,
)

# # Impulse rotor blade example 
# data = design_rotor_vortex_blade(
#     fluid_name="air",
#     P0_rel=5e5, T0_rel=400.0,
#     M_inlet=1.4, M_outlet=1.4,
#     M_lower=1.1, M_upper=2.0,
#     beta_inlet=70,
#     backend="HEOS",
#     num_points=60,
# )

print(f"pitch={data['pitch']:.4f}  chord={data['chord']:.4f}  solidity={data['solidity']:.4f}  "
      f"beta_outlet={data['beta_outlet']:.4f}")

b = data["blade"]
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.fill(b["x"], b["y"], color="0.55", edgecolor="0.15", lw=1.5)
ax2.fill(b["x"], np.array(b["y"]) - data["pitch"], color="0.55", edgecolor="0.15", lw=1.5)
ax2.set_xlabel(f"x ({data['units']})")
ax2.set_ylabel(f"y ({data['units']})")
ax2.set_aspect("equal")
# ax2.set_title("Rotor vortex blade (Goldman & Scullin 1968, TN D-4421)")

lower = data["blade_lower"]
upper = data["blade_upper"]

tan_bi = np.tan(np.radians(data["beta_inlet"]))
tan_bo = np.tan(np.radians(data["beta_outlet"]))
translate = b["translate"]

L0 = np.array([lower["x"][0], lower["y"][0]])
L1 = np.array([lower["x"][-1], lower["y"][-1]])
U0 = np.array([upper["x"][0], upper["y"][0] + translate])
U1 = np.array([upper["x"][-1], upper["y"][-1] + translate])

u_LE, n_LE = np.array([1.0, tan_bi]), np.array([-tan_bi, 1.0])
u_TE, n_TE = np.array([1.0, tan_bo]), np.array([-tan_bo, 1.0])

T_LE = _line_intersect(U0, u_LE, L0, n_LE)
T_TE = _line_intersect(U1, u_TE, L1, n_TE)

interior_ref = np.array([
    0.0, 0.5 * (lower["y"][len(lower["y"]) // 2] + upper["y"][len(upper["y"]) // 2] + translate),
])
le_x, le_y = _arc_from_diameter(L0, T_LE, interior_ref)
te_x, te_y = _arc_from_diameter(L1, T_TE, interior_ref)
print(f"r_LE = {b['le_radius']:.4f}   r_TE (target = pitch/25) = {b['te_radius']:.4f}")

fig5, ax5 = plt.subplots(figsize=(6, 5))
ax5.plot(lower["x"], lower["y"], color="C1", lw=2, label="lower wall")
ax5.plot(upper["x"], [y + translate for y in upper["y"]], color="C2", lw=2, label="upper wall (translated)")
ax5.plot([U0[0], T_LE[0]], [U0[1], T_LE[1]], color="gray", lw=1.5, ls="--", label="straight line (built first)")
ax5.plot([U1[0], T_TE[0]], [U1[1], T_TE[1]], color="gray", lw=1.5, ls="--")

for P, n in [(L0, n_LE), (L1, n_TE)]:
    seg = np.array([P - 0.3 * n, P + 0.3 * n])
    ax5.plot(seg[:, 0], seg[:, 1], color="0.6", lw=1, ls="--")
ax5.plot(le_x, le_y, color="k", lw=2.5, label="LE arc (built from the line's endpoint)")
ax5.plot(te_x, te_y, color="b", lw=2.5, label="TE arc (built from the line's endpoint)")

num_pts = data["num_points"]
n_p = len(data["pressure"]["x"])
n_s = len(data["suction"]["x"])
p_junc_idx = [num_pts - 1, n_p - num_pts]
s_junc_idx = [num_pts - 1, n_s - num_pts]
ax5.plot([data["pressure"]["x"][i] for i in p_junc_idx],
          [data["pressure"]["y"][i] for i in p_junc_idx],
          "o", mfc="none", mec="purple", mew=2, ms=9, zorder=5,
          label="transition-arc / circular-arc junction")
ax5.plot([upper["x"][i] for i in s_junc_idx],
          [upper["y"][i] + translate for i in s_junc_idx],
          "o", mfc="none", mec="purple", mew=2, ms=9, zorder=5)
ax5.plot([L0[0], L1[0]], [L0[1], L1[1]], "*", color="red", ms=16, zorder=6,
          label="circle-arc start (L0/L1)")

ax5.set_xlabel(f"x ({data['units']})")
ax5.set_ylabel(f"y ({data['units']})")
ax5.set_aspect("equal")
ax5.legend(fontsize=7)
# ax5.set_title("Line first (U0->T_LE), then arc as semicircle on L0->T_LE")

fig2.savefig(os.path.join(outdir, "rotor_blade.svg"), bbox_inches="tight")
fig5.savefig(os.path.join(outdir, "rotor_blade_construction.svg"), bbox_inches="tight")
print(f"saved figures to {outdir}")

plt.show()

