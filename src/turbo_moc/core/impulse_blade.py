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



def _arc_extent(surface):
    """x range of the constant-radius vortex arc on an uncorrected design surface (own frame, centre at origin)."""
    x, y = np.asarray(surface["x"], float), np.asarray(surface["y"], float)
    on_arc = np.abs(np.hypot(x, y) / surface["R_wall_star"] - 1.0) < 1e-6
    return x[on_arc].min(), x[on_arc].max()


def _tangent_arc(p0, d, b, t_b, count):
    """Circular arc tangent to the line through p0 with direction d and tangent to direction t_b at point b (kept
    fixed). Returns the points from the line tangency T to b (T excluded), and T."""
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    n_left = np.array([-t_b[1], t_b[0]])
    best = None
    for s in (1.0, -1.0):  # Centre on either side of the wall at b
        for sign in (1.0, -1.0):  # Signed distance of the centre to the line equals +-radius
            denom = sign - s * cross(n_left, d)
            if abs(denom) < 1e-12:
                continue
            rho = cross(b - p0, d) / denom
            if rho <= 0:
                continue
            c = b + rho * s * n_left
            t = p0 + np.dot(c - p0, d) * d
            if best is None or (np.dot(t - p0, d) > 0 and np.dot(b - t, d) > 0 and rho < best[0]):
                if np.dot(t - p0, d) > 0 and np.dot(b - t, d) > 0:
                    best = (rho, c, t)
    if best is None:
        raise ValueError("no tangent transition arc between the uniform-flow line and the vortex arc")
    rho, c, t = best
    a0, a1 = np.arctan2(*(t - c)[::-1]), np.arctan2(*(b - c)[::-1])
    a1 = a0 + (a1 - a0 + np.pi) % (2 * np.pi) - np.pi  # Short way round
    angles = np.linspace(a0, a1, count + 1)[1:]
    return np.c_[c[0] + rho * np.cos(angles), c[1] + rho * np.sin(angles)], t


def _arc_transitions(wall, extent, count=40):
    """Vortex wall with each MoC transition replaced by one circular arc tangent to the uniform-flow line and to
    the vortex arc at the vortex arc's own end point (classical pre-MoC construction; the line is shortened)."""
    x, y = np.asarray(wall["x"], float), np.asarray(wall["y"], float)
    on_arc = np.where((x >= extent[0]) & (x <= extent[1]))[0]
    first, last = on_arc[0], on_arc[-1]
    P = np.c_[x, y]
    unit = lambda v: v / np.linalg.norm(v)
    # Inlet: line through P[0] along the first stub, arc ends at the vortex arc start P[first]
    d_in = unit(P[1] - P[0])
    arc_in, t_in = _tangent_arc(P[0], d_in, P[first], unit(P[first + 1] - P[first]), count)
    # Outlet: same construction run backwards from the last point
    d_out = unit(P[-2] - P[-1])
    arc_out, t_out = _tangent_arc(P[-1], d_out, P[last], unit(P[last - 1] - P[last]), count)
    n = 8
    line_in = np.linspace(P[0], t_in, n)
    line_out = np.linspace(t_out, P[-1], n)
    pts = np.vstack([line_in, arc_in[:-1], P[first:last + 1], arc_out[::-1][1:], line_out])
    return ({"x": pts[:, 0].tolist(), "y": pts[:, 1].tolist()},
            {"x_arc_start": float(x[first]), "x_arc_end": float(x[last]), "x_line_end_in": float(t_in[0]), "x_line_start_out": float(t_out[0])})


def _straight_transitions(wall, extent):
    """Vortex wall with each MoC transition (between the constant-radius vortex arc, x in `extent`, and the
    uniform-flow corner) replaced by one straight segment between the same two points."""
    x, y = np.asarray(wall["x"], float), np.asarray(wall["y"], float)
    on_arc = (x >= extent[0]) & (x <= extent[1])
    first, last = np.argmax(on_arc), len(x) - 1 - np.argmax(on_arc[::-1])
    n = max((len(x) - (last - first)) // 2, 4)
    x_in, y_in = np.linspace(x[0], x[first], n, endpoint=False), np.linspace(y[0], y[first], n, endpoint=False)
    x_out, y_out = np.linspace(x[last], x[-1], n + 1)[1:], np.linspace(y[last], y[-1], n + 1)[1:]
    return ({"x": np.concatenate([x_in, x[first:last + 1], x_out]).tolist(),
             "y": np.concatenate([y_in, y[first:last + 1], y_out]).tolist()},
            {"x_arc_start": float(x[first]), "x_arc_end": float(x[last])})


def straight_transition_blade(vortex, transition="straight", **edges):
    """Vortex blade without method-of-characteristics transitions (the no-MoC baseline, paper 4.8): each wall keeps
    the vortex design's constant-radius arc and its uniform-flow end points, and the curved MoC transition between
    them becomes a straight segment. Same arcs, passage ends, spacing, count, chord and edge closure as the vortex
    rotor (its translate is reused); only the transition shape differs, so the flow turns through a kink and the
    waves the MoC transition cancels are left in the passage. Symmetric impulse blading only."""
    from turbo_moc.core.vortex_blade import _close_blade_le_te
    beta = np.radians(vortex["beta_inlet"])
    if abs(vortex["beta_outlet"] + vortex["beta_inlet"]) > 0.05:
        raise ValueError("straight_transition_blade: symmetric impulse blading only (beta_outlet = -beta_inlet).")
    pitch = float(vortex["pitch"])
    build = {"straight": _straight_transitions, "arc": _arc_transitions}[transition]
    lower, arc_lower = build(vortex["blade_lower"], _arc_extent(vortex["pressure"]))
    upper, arc_upper = build(vortex["blade_upper"], _arc_extent(vortex["suction"]))
    closure = {"num_points": vortex["num_points"]} | {key: value for key, value in edges.items() if key in EDGE_OPTIONS}
    x, y, diagnostics = _close_blade_le_te(lower, upper, pitch, beta, -beta, translate=float(vortex["blade"]["translate"]), **closure)
    return dict(vortex, blade_type=f"vortex_{transition}_transition", boundary_layer=None,
                blade_lower=lower, blade_upper=upper,
                blade={"x": x.tolist(), "y": y.tolist(), "closure": "circular_arc_le_te_fillet",
                       "le_radius": diagnostics["le_radius"], "te_radius": diagnostics["te_radius"],
                       "translate": diagnostics["translate"], "le_breaks": diagnostics["le_breaks"]},
                construction={"lower": arc_lower, "upper": arc_upper, "spacing": pitch + float(diagnostics["translate"])})



def thin_arc_walls(spacing, axial_chord, beta, thickness, count=200):
    """Open walls of a classical arc-and-line impulse blade (Capetti / Deich type) thinner than Baines' constant
    passage. Pressure (lower): one arc tangent to +-beta through the sharp edges (as Baines). Suction (upper, +spacing):
    a smaller arc, also tangent to +-beta, spanning x in +-a; the closure adds straight stubs at beta to the edges.
    Its crown sits ``thickness`` (t_max / axial chord, sharp blade) above the pressure crown:
        t = (c/2 - a) (tan beta - tan beta/2)."""
    half = 0.5 * axial_chord
    a = half - thickness * axial_chord / (np.tan(beta) - np.tan(0.5 * beta))
    if not 0 < a < half:
        raise ValueError(f"thin_arc thickness {thickness} out of range for beta {np.degrees(beta):.1f}")
    radius_p, radius_s = half / np.sin(beta), a / np.sin(beta)
    lower = _arc(-radius_p * np.cos(beta), radius_p, beta, count)
    rise = (half - a) * np.tan(beta)  # height of the suction arc's tangent points above the edges
    upper = _arc(rise - radius_s * np.cos(beta), radius_s, beta, count)  # same convention as constant_passage_walls (crown gap = t)
    return lower, upper, {"R_PS": radius_p, "R_SS": radius_s, "suction_arc_half_chord": a, "t_max_sharp": thickness * axial_chord}


def thin_arc_blade(vortex, thickness, **edges):
    """Thin arc-and-line impulse baseline (paper 4.8, 2026-11-11; classical velocity-triangle design in the spirit of
    Bufi's thesis Sec. 3.3.5 / Capetti 1967), matched to a vortex rotor like ``constant_passage_blade``: same blade
    angle, edge closure, spacing (pitch + translate) and axial extent. Thinner than Baines, so the passage widens
    from the entry."""
    beta = np.radians(vortex["beta_inlet"])
    closure = {"num_points": vortex["num_points"]} | {key: value for key, value in edges.items() if key in EDGE_OPTIONS}
    target_spacing = vortex["pitch"] + vortex["blade"]["translate"]
    target_extent = float(np.ptp(vortex["blade"]["x"]))

    def build(pitch, chord):
        lower, upper, construction = thin_arc_walls(pitch, chord, beta, thickness, closure["num_points"] * 3)
        x, y, diagnostics = _close_blade_le_te(lower, upper, pitch, beta, -beta, **closure)
        return lower, upper, construction, x, y, diagnostics

    def mismatch(unknowns):
        *_, x, _, diagnostics = build(*unknowns)
        return [unknowns[0] + diagnostics["translate"] - target_spacing, np.ptp(x) - target_extent]

    (pitch, axial_chord), _, converged, message = fsolve(mismatch, [target_spacing, target_extent], full_output=True)
    if converged != 1:
        raise RuntimeError(f"Thin-arc blade matching did not converge: {message}")
    lower, upper, construction, x, y, diagnostics = build(pitch, axial_chord)
    return dict(vortex, blade_type="impulse_thin_arc",
                pitch=float(pitch), vortex_pitch=float(pitch), beta_outlet=-float(vortex["beta_inlet"]),
                chord=float(axial_chord), solidity=float(target_extent / target_spacing),
                M_outlet=vortex["M_inlet"], boundary_layer=None,
                pressure={"x": lower["x"], "y": lower["y"]}, suction={"x": upper["x"], "y": upper["y"]},
                blade_lower=lower, blade_upper=upper,
                blade={"x": x.tolist(), "y": y.tolist(), "closure": "circular_arc_le_te_fillet",
                       "le_radius": diagnostics["le_radius"], "te_radius": diagnostics["te_radius"],
                       "translate": diagnostics["translate"], "le_breaks": diagnostics["le_breaks"]},
                construction={key: float(value) for key, value in construction.items()}
                | {"thickness": float(thickness), "axial_chord_sharp": float(axial_chord), "spacing": float(target_spacing)})
