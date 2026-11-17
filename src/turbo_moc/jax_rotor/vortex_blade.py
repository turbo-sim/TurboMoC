"""Vortex-flow supersonic rotor blade (Goldman & Scullin, NASA TN D-4421) in JAX.

A line-by-line port of ``turbo_moc.core.vortex_blade.design_rotor_vortex_blade`` (the reference), traceable and
differentiable in the continuous design inputs:

    M_inlet, M_outlet, M_lower, M_upper, beta_inlet, le_te_ratio, le_wedge_half_angle, le_ellipse_ratio,
    te_ellipse_ratio, boundary-layer chord / viscosity / factor, r_star.

The discrete structure of the outline is static (fixed during tracing): ``num_points``, which edge closures are used,
whether the boundary-layer correction is on, and the number of points on each circular arc. The reference sets the
latter from the turning angle, ``max(num_points, int(|alpha_out - alpha_in| in deg) + 2)``; ``structure`` evaluates it
once on concrete inputs and ``design_core`` keeps it fixed, so the geometry is a smooth function of the inputs while
the point count only changes where the reference's own count jumps (whole degrees of turning).

Input checks that raise in the reference (Mach number outside the table, no real outlet angle, wedge radius too large)
are made by ``structure`` on the concrete inputs; inside a trace they cannot raise.

Usage
-----
    iso = JaxIsentrope.from_manager(get_fluid_manager(...))
    blade = design_rotor_vortex_blade(iso, M_inlet=..., ...)          # same keys as the reference (arrays)
    f = make_design_function(iso, base_inputs)                        # f(params dict) -> outputs, for jax.grad / jit
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .isentrope import check_range

CONTINUOUS = ("M_inlet", "M_outlet", "M_lower", "M_upper", "beta_inlet", "le_te_ratio", "le_wedge_half_angle",
              "le_ellipse_ratio", "te_ellipse_ratio", "r_star")
BL_CONTINUOUS = ("chord", "viscosity", "factor")


@dataclass(frozen=True)
class Structure:
    """Static (non-differentiable) layout of the blade outline."""
    num_points: int
    n_circ_lower: int
    n_circ_upper: int
    le_wedge: bool
    le_round_tip: bool
    le_ellipse: bool
    te_ellipse: bool
    boundary_layer: bool
    dimensional: bool


# ----------------------------------------------------------------------------------------------------------------
# Building blocks (same names and algebra as the reference)
# ----------------------------------------------------------------------------------------------------------------

def _solve_beta_outlet(iso, V_inlet, V_outlet, beta_inlet_rad):
    """Mass-flux continuity across the transition: rho V cos(beta) = const (reference: _solve_beta_outlet)."""
    cos_beta_outlet = iso.rho_of_V(V_inlet) * V_inlet * jnp.cos(beta_inlet_rad) / (iso.rho_of_V(V_outlet) * V_outlet)
    return -jnp.arccos(jnp.clip(cos_beta_outlet, -1.0, 1.0)), cos_beta_outlet


def _R_star(iso, nu):
    return iso.a_star / iso.V_of_nu(nu)


def _mach_angle(iso, R_star):
    V = iso.a_star / R_star
    M = V / iso.a_of_V(V)
    return jnp.arcsin(jnp.clip(1.0 / M, -1.0, 1.0))


def _arc_march(iso, nu_far, nu_wall, num_points, upper):
    """TN D-4421 transition-arc marching (reference: _lower_arc_march / _upper_arc_march), as one lax.scan."""
    n_steps = num_points - 1
    delv = ((nu_wall - nu_far) if upper else (nu_far - nu_wall)) / n_steps
    R_wall = _R_star(iso, nu_wall)
    V = nu_far
    phi_start = (-(nu_wall - V) if upper else -(V - nu_wall)) + n_steps * delv

    def step(carry, index):
        phi_prev, x_wall, y_wall = carry
        k = (n_steps + 1) - index
        phi = phi_prev - delv
        nu_k = V + (k - 1) * delv if upper else V - (k - 1) * delv
        radius = _R_star(iso, nu_k)
        tx, ty = radius * jnp.sin(phi), radius * jnp.cos(phi)
        slope_wall = jnp.tan(-phi_prev)
        mu = _mach_angle(iso, radius)
        slope_mach = jnp.tan(-phi + mu) if upper else -jnp.tan(phi + mu)
        temp = y_wall - slope_wall * x_wall
        tempp = ty - slope_mach * tx
        temppp = slope_mach - slope_wall
        x_new = (temp - tempp) / temppp
        y_new = (slope_mach * temp - slope_wall * tempp) / temppp
        return (phi, x_new, y_new), (x_new, y_new, tx, ty)

    zero = jnp.zeros_like(R_wall)
    _, (xs, ys, rx, ry) = jax.lax.scan(step, (phi_start, zero, R_wall), jnp.arange(1, n_steps + 1))
    return jnp.concatenate([zero[None], xs]), jnp.concatenate([R_wall[None], ys]), rx, ry


def _rotate_inlet(x, y, alpha):
    s, c = jnp.sin(alpha), jnp.cos(alpha)
    return y * s + x * c, y * c - x * s


def _rotate_outlet(x, y, alpha):
    s, c = jnp.sin(alpha), jnp.cos(alpha)
    return y * s - x * c, y * c + x * s


def _alphas(nu_inlet, nu_outlet, nu_wall, beta_inlet_rad, beta_outlet_rad, upper):
    if upper:
        return (nu_wall - nu_inlet) - beta_inlet_rad, -((nu_wall - nu_outlet) + beta_outlet_rad)
    return (nu_inlet - nu_wall) - beta_inlet_rad, -((nu_outlet - nu_wall) + beta_outlet_rad)


def _surface(iso, nu_inlet, nu_outlet, nu_wall, beta_inlet_rad, beta_outlet_rad, upper, num_points, n_circ):
    alpha_in, alpha_out = _alphas(nu_inlet, nu_outlet, nu_wall, beta_inlet_rad, beta_outlet_rad, upper)
    x_in_l, y_in_l, rx_in_l, ry_in_l = _arc_march(iso, nu_inlet, nu_wall, num_points, upper)
    x_out_l, y_out_l, rx_out_l, ry_out_l = _arc_march(iso, nu_outlet, nu_wall, num_points, upper)
    x_in, y_in = _rotate_inlet(x_in_l, y_in_l, alpha_in)
    x_out, y_out = _rotate_outlet(x_out_l, y_out_l, alpha_out)
    rx_in, ry_in = _rotate_inlet(rx_in_l, ry_in_l, alpha_in)
    rx_out, ry_out = _rotate_outlet(rx_out_l, ry_out_l, alpha_out)
    R_wall_star = _R_star(iso, nu_wall)
    angle_circ = jnp.linspace(alpha_in, alpha_out, n_circ)
    x_circ, y_circ = R_wall_star * jnp.sin(angle_circ), R_wall_star * jnp.cos(angle_circ)
    x = jnp.concatenate([x_in[::-1], x_circ[1:-1], x_out])
    y = jnp.concatenate([y_in[::-1], y_circ[1:-1], y_out])
    nu = jnp.concatenate([jnp.linspace(nu_wall, nu_inlet, num_points)[::-1], jnp.full(n_circ - 2, nu_wall),
                          jnp.linspace(nu_wall, nu_outlet, num_points)])
    return {"x": x, "y": y, "nu": nu, "x_inlet_far": x_in[-1], "y_inlet_far": y_in[-1],
            "x_outlet_far": x_out[-1], "y_outlet_far": y_out[-1], "R_wall_star": R_wall_star,
            "alpha_inlet": alpha_in, "alpha_outlet": alpha_out,
            # characteristic segments (x0, y0, x1, y1): reference point -> the wall point it locates
            "mach_lines_inlet": jnp.stack([rx_in, ry_in, x_in[1:], y_in[1:]], axis=1),
            "mach_lines_outlet": jnp.stack([rx_out, ry_out, x_out[1:], y_out[1:]], axis=1)}


def _line_intersect(P1, d1, P2, d2):
    # Cramer's rule for P1 + t d1 = P2 + s d2 (reference: np.linalg.solve of the same 2x2 system)
    det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
    b0, b1 = P2[0] - P1[0], P2[1] - P1[1]
    t = (b0 * (-d2[1]) - (-d2[0]) * b1) / det
    return P1 + t * d1


def _arc_from_diameter(L, T, interior_ref, num_points):
    center = 0.5 * (L + T)
    R = jnp.linalg.norm(T - L) / 2.0
    ang1 = jnp.arctan2(L[1] - center[1], L[0] - center[0])
    to_center = center - interior_ref
    radial = L - center
    cross_z = to_center[0] * radial[1] - to_center[1] * radial[0]
    sign = jnp.where(cross_z >= 0, -1.0, 1.0)
    angles = ang1 + sign * jnp.linspace(0, jnp.pi, num_points)
    return center[0] + R * jnp.cos(angles), center[1] + R * jnp.sin(angles)


def _wedge_from_gap(L, T, direction, half_angle_rad, num_points, tip_radius, rounded):
    unit = direction / jnp.linalg.norm(direction)
    half_gap = 0.5 * jnp.linalg.norm(T - L)
    tip = 0.5 * (L + T) - unit * half_gap / jnp.tan(half_angle_rad)
    if not rounded:
        count = max(num_points // 2 + 1, 2)
        to_tip = T + jnp.linspace(0.0, 1.0, count)[:, None] * (tip - T)
        to_wall = tip + jnp.linspace(0.0, 1.0, num_points - count + 1)[1:, None] * (L - tip)
        points = jnp.vstack([to_tip, to_wall])
        return points[:, 0], points[:, 1], [count - 1]
    flank = tip_radius / jnp.tan(half_angle_rad)
    center = tip + unit * tip_radius / jnp.sin(half_angle_rad)
    start = tip + flank * (T - tip) / jnp.linalg.norm(T - tip)
    end = tip + flank * (L - tip) / jnp.linalg.norm(L - tip)
    arc = max(num_points // 2, 4)
    first = (num_points - arc + 2) // 2
    last = num_points - first - arc + 2
    angle_start = jnp.arctan2(*(start - center)[::-1])
    sweep = (jnp.arctan2(*(end - center)[::-1]) - angle_start + jnp.pi) % (2 * jnp.pi) - jnp.pi
    angles = angle_start + jnp.linspace(0.0, sweep, arc)
    points = jnp.vstack([
        T + jnp.linspace(0.0, 1.0, first)[:, None] * (start - T),
        (center + tip_radius * jnp.column_stack([jnp.cos(angles), jnp.sin(angles)]))[1:],
        end + jnp.linspace(0.0, 1.0, last)[1:, None] * (L - end),
    ])
    return points[:, 0], points[:, 1], [first - 1, first + arc - 2]


def _ellipse_from_gap(L, T, outward, ratio, num_points):
    centre = 0.5 * (L + T)
    across = L - centre
    along = ratio * jnp.linalg.norm(across) * outward / jnp.linalg.norm(outward)
    angles = jnp.linspace(0.0, jnp.pi, num_points)
    points = centre + jnp.outer(jnp.cos(angles), across) + jnp.outer(jnp.sin(angles), along)
    return points[:, 0], points[:, 1]


def _translate(lower, upper, pitch, beta_outlet_rad, le_te_ratio):
    """Shift beyond the pitch that gives the trailing-edge arc the radius pitch / le_te_ratio (closed form)."""
    tan_bo = jnp.tan(beta_outlet_rad)
    u = jnp.array([1.0, tan_bo]) / jnp.sqrt(1.0 + tan_bo ** 2)
    n = jnp.array([-u[1], u[0]])
    gap = jnp.array([upper["x"][-1] - lower["x"][-1], upper["y"][-1] - lower["y"][-1]])
    return (2.0 * pitch / le_te_ratio - jnp.dot(gap, n)) / n[1]


def _close_blade_le_te(lower, upper, pitch, beta_inlet_rad, beta_outlet_rad, p, s, translate=None):
    lower_x, lower_y, upper_x, upper_y = lower["x"], lower["y"], upper["x"], upper["y"]
    tan_bi, tan_bo = jnp.tan(beta_inlet_rad), jnp.tan(beta_outlet_rad)
    r_TE_target = pitch / p["le_te_ratio"]
    if translate is None:
        translate = _translate(lower, upper, pitch, beta_outlet_rad, p["le_te_ratio"])
    L0 = jnp.array([lower_x[0], lower_y[0]])
    L1 = jnp.array([lower_x[-1], lower_y[-1]])
    U0 = jnp.array([upper_x[0], upper_y[0] + translate])
    U1 = jnp.array([upper_x[-1], upper_y[-1] + translate])
    u_LE, n_LE = jnp.array([1.0, tan_bi]), jnp.array([-tan_bi, 1.0])
    u_TE, n_TE = jnp.array([1.0, tan_bo]), jnp.array([-tan_bo, 1.0])
    T_LE = _line_intersect(U0, u_LE, L0, n_LE)
    T_TE = _line_intersect(U1, u_TE, L1, n_TE)
    interior_ref = jnp.array([0.0, 0.5 * (lower_y[lower_y.shape[0] // 2] + upper_y[upper_y.shape[0] // 2] + translate)])
    n = s.num_points
    if s.le_ellipse:
        le_x, le_y = _ellipse_from_gap(L0, T_LE, -u_LE, p["le_ellipse_ratio"], n)
    else:
        le_x, le_y = _arc_from_diameter(L0, T_LE, interior_ref, n)
    if s.te_ellipse:
        te_x, te_y = _ellipse_from_gap(L1, T_TE, u_TE, p["te_ellipse_ratio"], n)
    else:
        te_x, te_y = _arc_from_diameter(L1, T_TE, interior_ref, n)
    le_radius = 0.5 * jnp.linalg.norm(T_LE - L0)
    le_breaks = []
    if s.le_wedge:
        le_radius = r_TE_target if s.le_round_tip else jnp.zeros_like(le_radius)
        wedge_x, wedge_y, le_breaks = _wedge_from_gap(L0, T_LE, u_LE, jnp.radians(p["le_wedge_half_angle"]), n,
                                                      le_radius, s.le_round_tip)
        le_x, le_y = wedge_x[::-1], wedge_y[::-1]
    te_radius = 0.5 * jnp.linalg.norm(T_TE - L1)
    closed_x = jnp.concatenate([lower_x, te_x[1:], U1[0][None], upper_x[::-1][1:], T_LE[0][None], le_x[::-1][1:]])
    closed_y = jnp.concatenate([lower_y, te_y[1:], U1[1][None], (upper_y + translate)[::-1][1:], T_LE[1][None],
                                le_y[::-1][1:]])
    return closed_x, closed_y, {"translate": translate, "le_radius": le_radius, "te_radius": te_radius,
                                "le_breaks": le_breaks}


def _displacement_thickness(iso, x, y, nu, length_scale, viscosity, factor):
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.hypot(jnp.diff(x), jnp.diff(y)))]) * length_scale
    V = iso.V_of_nu(nu)
    rho = iso.rho_of_V(V)
    reynolds = jnp.maximum(rho * V * s / viscosity, 1.0)
    return factor * 0.046 * s * reynolds ** -0.2 / length_scale


def _offset_into_blade(wall, other, thickness):
    x, y = wall["x"], wall["y"]
    tx, ty = jnp.gradient(x), jnp.gradient(y)
    norm = jnp.hypot(tx, ty)
    nx, ny = -ty / norm, tx / norm
    middle = x.shape[0] // 2
    towards = jnp.array([jnp.mean(other["x"]) - x[middle], jnp.mean(other["y"]) - y[middle]])
    sign = jnp.where(nx[middle] * towards[0] + ny[middle] * towards[1] > 0, 1.0, -1.0)
    return {"x": x + sign * thickness * nx, "y": y + sign * thickness * ny}


# ----------------------------------------------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------------------------------------------

def _velocities(iso, p):
    V = {k: iso.V_of_M(p[f"M_{k}"]) for k in ("inlet", "outlet", "lower", "upper")}
    nu = {k: iso.nu_of_V(v) for k, v in V.items()}
    beta_inlet_rad = jnp.radians(p["beta_inlet"])
    beta_outlet_rad, cos_beta_outlet = _solve_beta_outlet(iso, V["inlet"], V["outlet"], beta_inlet_rad)
    return V, nu, beta_inlet_rad, beta_outlet_rad, cos_beta_outlet


def structure(iso, num_points=60, le_te_ratio=25.0, le_wedge_half_angle=None, le_round_tip=False,
              le_ellipse_ratio=None, te_ellipse_ratio=None, boundary_layer=None, r_star=None, **flow):
    """Static layout on concrete inputs, plus the reference's input checks (raises like the reference)."""
    check_range(iso, [flow[k] for k in ("M_inlet", "M_outlet", "M_lower", "M_upper")])
    p = {k: jnp.float64(flow[k]) for k in ("M_inlet", "M_outlet", "M_lower", "M_upper", "beta_inlet")}
    V, nu, bi, bo, cos_bo = _velocities(iso, p)
    if abs(float(cos_bo)) > 1.0:
        raise ValueError(f"No real beta_outlet satisfies mass-flux continuity for these inputs "
                         f"(cos(beta_outlet)={float(cos_bo):.4f} outside [-1,1]).")
    counts = []
    for wall, upper in (("lower", False), ("upper", True)):
        a_in, a_out = _alphas(nu["inlet"], nu["outlet"], nu[wall], bi, bo, upper)
        counts.append(max(num_points, int(abs(np.degrees(float(a_out) - float(a_in)))) + 2))
    return Structure(int(num_points), counts[0], counts[1], le_wedge_half_angle is not None, bool(le_round_tip),
                     le_ellipse_ratio is not None, te_ellipse_ratio is not None, boundary_layer is not None,
                     r_star is not None)


def design_core(iso, p, s):
    """Traceable blade design: ``p`` dict of continuous inputs (jnp scalars), ``s`` a static ``Structure``."""
    V, nu, beta_inlet_rad, beta_outlet_rad, _ = _velocities(iso, p)
    pressure = _surface(iso, nu["inlet"], nu["outlet"], nu["lower"], beta_inlet_rad, beta_outlet_rad, False,
                        s.num_points, s.n_circ_lower)
    suction = _surface(iso, nu["inlet"], nu["outlet"], nu["upper"], beta_inlet_rad, beta_outlet_rad, True,
                       s.num_points, s.n_circ_upper)
    tan_bi = jnp.tan(beta_inlet_rad)
    y_last_i = suction["y_inlet_far"] + tan_bi * (pressure["x_inlet_far"] - suction["x_inlet_far"])
    pitch = pressure["y_inlet_far"] - y_last_i
    chord = jnp.hypot(pressure["x_outlet_far"] - pressure["x_inlet_far"],
                      pressure["y_outlet_far"] - pressure["y_inlet_far"])
    scale = p["r_star"] if s.dimensional else jnp.float64(1.0)

    def pack(surf):
        return {"x": surf["x"] * scale, "y": surf["y"] * scale, "R_wall_star": surf["R_wall_star"] * scale,
                "mach_lines_inlet": surf["mach_lines_inlet"] * scale,
                "mach_lines_outlet": surf["mach_lines_outlet"] * scale}

    pressure_packed, suction_packed = pack(pressure), pack(suction)
    tan_bo = jnp.tan(beta_outlet_rad)
    x0i, y0i, x1i = suction["x_inlet_far"], suction["y_inlet_far"], pressure["x_inlet_far"]
    x0o, y0o, x1o = suction["x_outlet_far"], suction["y_outlet_far"], pressure["x_outlet_far"]
    suction_packed["extension_inlet"] = jnp.stack([x0i, y0i, x1i, y0i + tan_bi * (x1i - x0i)]) * scale
    suction_packed["extension_outlet"] = jnp.stack([x0o, y0o, x1o, y0o + tan_bo * (x1o - x0o)]) * scale

    blade_lower = {"x": pressure_packed["x"], "y": pressure_packed["y"]}
    blade_upper = {"x": suction_packed["x"], "y": suction_packed["y"] + pitch * scale}
    displacement, translate = None, None
    if s.boundary_layer:
        translate = _translate(blade_lower, blade_upper, pitch * scale, beta_outlet_rad, p["le_te_ratio"])
        walls_x = jnp.concatenate([blade_lower["x"], blade_upper["x"]])
        length_scale = p["bl_chord"] / (jnp.max(walls_x) - jnp.min(walls_x))
        lower_delta = _displacement_thickness(iso, blade_lower["x"], blade_lower["y"], pressure["nu"], length_scale,
                                              p["bl_viscosity"], p["bl_factor"])
        upper_delta = _displacement_thickness(iso, blade_upper["x"], blade_upper["y"], suction["nu"], length_scale,
                                              p["bl_viscosity"], p["bl_factor"])
        blade_lower, blade_upper = (_offset_into_blade(blade_lower, blade_upper, lower_delta),
                                    _offset_into_blade(blade_upper, blade_lower, upper_delta))
        displacement = {"lower_max_m": jnp.max(lower_delta) * length_scale,
                        "upper_max_m": jnp.max(upper_delta) * length_scale,
                        "lower_m": lower_delta * length_scale, "upper_m": upper_delta * length_scale}

    blade_x, blade_y, diag = _close_blade_le_te(blade_lower, blade_upper, pitch * scale, beta_inlet_rad,
                                                beta_outlet_rad, p, s, translate)
    spacing = pitch * scale + diag["translate"]
    return {
        "pressure": pressure_packed, "suction": suction_packed, "blade_lower": blade_lower, "blade_upper": blade_upper,
        "blade": {"x": blade_x, "y": blade_y, "le_radius": diag["le_radius"], "te_radius": diag["te_radius"],
                  "translate": diag["translate"]},
        "beta_outlet": jnp.degrees(beta_outlet_rad),
        "nu_inlet_deg": jnp.degrees(nu["inlet"]), "nu_outlet_deg": jnp.degrees(nu["outlet"]),
        "nu_lower_deg": jnp.degrees(nu["lower"]), "nu_upper_deg": jnp.degrees(nu["upper"]),
        "pitch": spacing, "vortex_pitch": pitch * scale, "chord": chord * scale, "solidity": chord * scale / spacing,
        "boundary_layer": displacement,
    }


DEFAULTS = {"le_te_ratio": 25.0}  # reference defaults of the optional continuous inputs that are always used


def _params(inputs, s):
    p = {k: jnp.float64(inputs[k] if inputs.get(k) is not None else DEFAULTS.get(k, 0.0)) for k in CONTINUOUS}
    bl = inputs.get("boundary_layer") or {}
    p["bl_chord"] = jnp.float64(bl.get("chord", 0.0))
    p["bl_viscosity"] = jnp.float64(bl.get("viscosity", 1.0))
    p["bl_factor"] = jnp.float64(bl.get("factor", 1.0))
    return p


def make_design_function(iso, inputs):
    """(f, p0, s): f(p) -> design dict, pure in the continuous params p; layout frozen at ``inputs``."""
    s = structure(iso, **inputs)
    p0 = _params(inputs, s)
    return (lambda p: design_core(iso, p, s)), p0, s


def design_rotor_vortex_blade(iso, **inputs):
    """Drop-in counterpart of the reference (minus fluid arguments, which ``iso`` carries); returns arrays."""
    inputs = {k: v for k, v in inputs.items() if k not in ("fluid_name", "P0_rel", "T0_rel", "Q0_rel", "backend")}
    f, p0, s = make_design_function(iso, inputs)
    out = f(p0)
    out["structure"] = s
    if s.le_wedge:
        out["blade"]["le_breaks"] = _wedge_breaks(s)
    return out


def _wedge_breaks(s):
    n = s.num_points
    if not s.le_round_tip:
        return [max(n // 2 + 1, 2) - 1]
    arc = max(n // 2, 4)
    first = (n - arc + 2) // 2
    return [first - 1, first + arc - 2]
