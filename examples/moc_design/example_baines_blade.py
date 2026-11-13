"""Baines constant-passage impulse blade: python examples/moc_design/example_baines_blade.py

Construction (Baines 2019, Turbine Design and Analysis, Sec. 4.5.1): pressure surface one circular arc tangent to the
blade angle +-beta at both edges; suction surface the arc concentric with the neighbouring blade's pressure arc, so the
passage width s cos(beta) is constant; straight lines at beta join it to the edges. With sharp edges:
    R_PS = c / (2 sin beta),   R_SS = R_PS - s cos beta,   passage width = s cos beta,   t_max = s (1 - cos beta)
The edges are then closed with circular arcs (the vortex blade closure routine with no ellipse ratio).
Symmetric impulse blading: outlet angle -beta. Lengths in any unit, used consistently. Shows the blade (not saved).
"""
import matplotlib.pyplot as plt
import numpy as np

from turbo_moc.core.impulse_blade import constant_passage_walls
from turbo_moc.core.vortex_blade import _close_blade_le_te

# ---------------------------------------------------------------------------------------------------------------------
# Inputs
BETA = 54.1             # Blade (relative inflow) angle from axial [deg]; outlet angle is -BETA
SPACING = 0.4552        # Blade spacing s of the sharp-edged blade [length]
AXIAL_CHORD = 1.4804    # Axial chord c of the sharp-edged blade [length]
LE_TE_RATIO = 59.088    # Edge size: trailing-edge radius = SPACING / LE_TE_RATIO
LE_ELLIPSE_RATIO = None  # Leading edge: None = circular arc (Baines); a number = ellipse axis ratio
TE_ELLIPSE_RATIO = None  # Trailing edge: None = circular arc (Baines); a number = ellipse axis ratio
NUM_POINTS = 60         # Points on each edge closure
BLADES = 3              # Blades drawn, one spacing apart
# ---------------------------------------------------------------------------------------------------------------------

beta = np.radians(BETA)
lower, upper, construction = constant_passage_walls(SPACING, AXIAL_CHORD, beta, 3 * NUM_POINTS)
x, y, diagnostics = _close_blade_le_te(lower, upper, SPACING, beta, -beta, le_te_ratio=LE_TE_RATIO,
                                       num_points=NUM_POINTS, le_ellipse_ratio=LE_ELLIPSE_RATIO,
                                       te_ellipse_ratio=TE_ELLIPSE_RATIO)
spacing = SPACING + diagnostics["translate"]  # The edge closure shifts the row by `translate`

print(f"beta {BETA} deg, spacing {spacing:.4f}, axial extent {np.ptp(x):.4f}")
for key, value in construction.items():
    print(f"  {key:15s} {value:.4f}")
print(f"  le_radius       {diagnostics['le_radius']:.4f}\n  te_radius       {diagnostics['te_radius']:.4f}")

fig, ax = plt.subplots(figsize=(4, 3.4), constrained_layout=True)
for k in range(-(BLADES // 2), BLADES - BLADES // 2):
    ax.fill(x, np.asarray(y) + k * spacing, color="#0b2f55" if k == 0 else "#8a9aa1", lw=0)
ax.set_aspect("equal")
ax.set_xlabel("x (axial)")
ax.set_ylabel("y (tangential)")
ax.set_title("Constant passage (Baines)", fontsize=10)
plt.show()
