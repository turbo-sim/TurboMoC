"""Constant-passage impulse blade (Baines 2019, Turbine Design and Analysis, Sec. 4.5.1, Fig. 4.13 left).

The baseline against the vortex-flow blade. The pressure surface is one circular arc tangent to the blade
angle at both edges. The suction surface is the arc concentric with the neighbouring blade's pressure arc
(centre shifted one spacing), so the passage has constant width. Straight lines at the blade angle join it
to the edges. With sharp edges, spacing s, axial chord c and blade angle beta fix everything:

    R_PS = c / (2 sin beta),   R_SS = R_PS - s cos beta,   passage width = s cos beta,   t_max = s (1 - cos beta)

The passage width equals the inflow width s cos beta, so the area ratio is one throughout. The solid is
closed with the vortex blade's own edge routine (`_close_blade_le_te`): same wedge leading edge, same
trailing-edge radius pitch/le_te_ratio. The blade is built in the vortex solver's frame (x downstream,
y tangential, pressure wall below) and matched to a vortex result's spacing/chord and axial chord, so both
blades fit the same stage at the same blade count. Symmetric impulse blading: beta_outlet = -beta_inlet.
"""

import numpy as np
from scipy.optimize import fsolve

from turbo_moc.core.vortex_blade import _close_blade_le_te


def _arc(centre_y, radius, half_angle, count):
    """Circular arc above its centre (0, centre_y), from tangent angle +half_angle to -half_angle."""
    phi = np.linspace(-half_angle, half_angle, count)
    return {"x": (radius * np.sin(phi)).tolist(), "y": (centre_y + radius * np.cos(phi)).tolist()}


def constant_passage_walls(spacing, axial_chord, beta, count=200):
    """Open pressure (lower) and suction (upper, the suction arc only) walls of one sharp-edged blade.

    The suction arc spans the tangent points of the edge lines; the closure adds those lines as its stubs.
    """
    radius = 0.5 * axial_chord / np.sin(beta)
    width = spacing * np.cos(beta)
    lower = _arc(-radius * np.cos(beta), radius, beta, count)  # Edges at y = 0
    upper = _arc(spacing - radius * np.cos(beta), radius - width, beta, count)
    return lower, upper, {"R_PS": radius, "R_SS": radius - width, "passage_width": width,
                          "t_max_sharp": spacing - width}


def _blade(spacing, axial_chord, beta, closure):
    lower, upper, construction = constant_passage_walls(spacing, axial_chord, beta, closure["num_points"] * 3)
    x, y, diagnostics = _close_blade_le_te(lower, upper, spacing, beta, -beta, **closure)
    return lower, upper, construction, x, y, diagnostics


EDGE_OPTIONS = ("le_te_ratio", "le_wedge_half_angle", "le_round_tip", "le_ellipse_ratio", "te_ellipse_ratio")


def constant_passage_blade(vortex, **edges):
    """Impulse baseline matched to a vortex rotor result: same blade angle, edge closure (the vortex design's
    own ``EDGE_OPTIONS``), blade spacing (pitch + translate) and axial extent of the closed solid.
    Returns a dict in the vortex result's layout."""
    beta = np.radians(vortex["beta_inlet"])
    closure = {"num_points": vortex["num_points"]} | {key: value for key, value in edges.items() if key in EDGE_OPTIONS}
    target_spacing = vortex["pitch"] + vortex["blade"]["translate"]
    target_extent = float(np.ptp(vortex["blade"]["x"]))

    def mismatch(unknowns):
        _, _, _, x, _, diagnostics = _blade(unknowns[0], unknowns[1], beta, closure)
        return [unknowns[0] + diagnostics["translate"] - target_spacing, np.ptp(x) - target_extent]

    (pitch, axial_chord), _, converged, message = fsolve(mismatch, [target_spacing, target_extent], full_output=True)
    if converged != 1:
        raise RuntimeError(f"Impulse blade matching did not converge: {message}")
    lower, upper, construction, x, y, diagnostics = _blade(pitch, axial_chord, beta, closure)
    return dict(vortex, blade_type="impulse_constant_passage",
                pitch=float(pitch), vortex_pitch=float(pitch), beta_outlet=-float(vortex["beta_inlet"]),
                chord=float(axial_chord), solidity=float(target_extent / target_spacing),
                M_outlet=vortex["M_inlet"], boundary_layer=None,
                pressure={"x": lower["x"], "y": lower["y"]}, suction={"x": upper["x"], "y": upper["y"]},
                blade_lower=lower, blade_upper=upper,
                blade={"x": x.tolist(), "y": y.tolist(), "closure": "circular_arc_le_te_fillet",
                       "le_radius": diagnostics["le_radius"], "te_radius": diagnostics["te_radius"],
                       "translate": diagnostics["translate"], "le_breaks": diagnostics["le_breaks"]},
                construction={key: float(value) for key, value in construction.items()}
                | {"axial_chord_sharp": float(axial_chord), "spacing": float(target_spacing)})
