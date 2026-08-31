"""
moc.geometry.blade -- stator blade parametrization built on top of a MOC
nozzle wall contour (the suction side).

Ported from opt_framework/test_2/01_geometry_parametrization/1.1_stator_axial.py,
refactored into moc.design_nozzle()'s "plain args in, flat JSON-safe dict
out" pattern: parametrize_stator_blade(...) takes the nozzle wall contour
(x, y in meters, matching design_nozzle()'s wall_final schema) plus blade
parameters (previously hardcoded module globals in the original script),
and returns a flat dict (all lengths in mm, matching the original script's
working units) -- no solver/UI objects. The geometry helpers below are a
line-for-line port of the original script's module-level functions.

export_blade_step(...) is kept separate/optional since it depends on
cadquery, which the rest of moc does not otherwise require.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from .spline_tool import BSplineTool

__all__ = [
    "parametrize_stator_blade", "parametrize_stator_blade_semi",
    "export_blade_step", "export_flared_blade_step", "move_control_point_along_normal",
]


# --------------------------------------------------------------------------
# Geometry helpers (ported verbatim from 1.1_stator_axial.py)
# --------------------------------------------------------------------------

def _halfcircle_4cp(p_start, p_end, type="leading_edge", slope_end=None, num_plot_points=50):
    x0, y0 = p_start
    x3, y3 = p_end

    if type == "leading_edge":
        radius = abs(x3 - x0) / 2
        x1, y1 = x0, y0 + radius
        y2 = y1
        if slope_end is not None:
            x2 = x3 + slope_end * (y1 - y0)
        else:
            x2 = x3
    else:
        radius = abs(y3 - y0) / 2
        x1, y1 = x0 + radius, y0
        x2, y2 = x1, y3

    control_pts = [(x0, y0, 0), (x1, y1, 0), (x2, y2, 0), (x3, y3, 0)]

    t = np.linspace(0, 1, num_plot_points)
    x_curve = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3 * x3
    y_curve = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3 * y3

    return control_pts, x_curve, y_curve


def _create_trailing_edge_points(last_original, last_mirrored, trailing_edge_radius, num_points=50):
    mx, my, mz = last_mirrored[0], last_mirrored[1], last_mirrored[2]

    center_x = mx
    center_y = my - trailing_edge_radius
    center_z = mz

    angle_start = np.pi / 2
    angle_end = -np.pi / 2

    t = np.linspace(angle_start, angle_end, num_points + 1)
    x = center_x + trailing_edge_radius * np.cos(t)
    y = center_y + trailing_edge_radius * np.sin(t)
    z = np.full_like(x, center_z)

    return x, y, z


def _rotate_points(x, y, angle_deg, clockwise=True):
    theta = np.radians(angle_deg)
    if clockwise:
        theta = -theta
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
    )


def _translate_points(x, y, dx=0, dy=0):
    x = np.asarray(x)
    y = np.asarray(y)
    return x + dx, y + dy


def _create_straight_line(p1, p2, N=50):
    p1, p2 = np.array(p1), np.array(p2)
    t = np.linspace(0, 1, N).reshape(-1, 1)
    pts = p1 + t * (p2 - p1)
    return pts[:, 0], pts[:, 1]


def _get_arc_coordinates(p_start, p_end, num_points=10, clockwise=True):
    x1, y1 = float(p_start[0]), float(p_start[1])
    x2, y2 = float(p_end[0]), float(p_end[1])

    dx = x2 - x1
    dy = y2 - y1
    if dy <= 0:
        raise ValueError("End point y must be > start point y for start to be at the bottom.")

    radius = (dx ** 2 + dy ** 2) / (2 * dy)
    center_x = x1
    center_y = y1 + radius

    start_angle = 1.5 * np.pi
    end_angle_raw = np.arctan2(y2 - center_y, x2 - center_x)

    if clockwise:
        end_angle = end_angle_raw
        while end_angle > start_angle:
            end_angle -= 2 * np.pi
        if (start_angle - end_angle) > 2 * np.pi:
            end_angle += 2 * np.pi
    else:
        end_angle = end_angle_raw
        while end_angle < start_angle:
            end_angle += 2 * np.pi
        if (end_angle - start_angle) > 2 * np.pi:
            end_angle -= 2 * np.pi

    angles = np.linspace(start_angle, end_angle, num_points)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)

    x[0], y[0] = x1, y1
    x[-1], y[-1] = x2, y2

    return x, y


def _get_cubic_spline_points(P0, P3, tangent_start, tangent_end, scale_factor=None, num_points=100):
    P0 = np.array(P0)
    P3 = np.array(P3)
    t_start = np.array(tangent_start, dtype=float)
    t_end = np.array(tangent_end, dtype=float)

    t_start /= np.linalg.norm(t_start)
    t_end /= np.linalg.norm(t_end)

    chord = np.linalg.norm(P3 - P0)

    if scale_factor is None:
        delta = np.abs(P3 - P0)
        scale_start = min(chord / 2, delta[0], delta[1])
        scale_end = min(chord / 10, delta[0], delta[1])
    else:
        scale_start = scale_factor
        scale_end = scale_factor

    P1 = P0 + (t_start * scale_start)
    P2 = P3 - (t_end * scale_end)

    v_start = (P1 - P0) * 3
    v_end = (P3 - P2) * 3

    spline = CubicHermiteSpline(x=[0, 1], y=[P0, P3], dydx=[v_start, v_end])
    t_steps = np.linspace(0, 1, num_points)
    curve_points = spline(t_steps)

    return curve_points[:, 0], curve_points[:, 1]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def parametrize_stator_blade(
    wall_x,
    wall_y,
    metal_angle_in=0.0,
    metal_angle_out=70.0,
    r_trailing=0.5,
    inlet_opening_ratio=1.8,
    n_cp=30,
    num_baseline_points=500,
    num_curve_points=500,
):
    """
    Build a closed stator blade profile from a MOC nozzle wall contour
    (used as the divergent-section suction side), following the approach
    of opt_framework's 1.1_stator_axial.py.

    Parameters
    ----------
    wall_x, wall_y : array_like
        Nozzle wall contour, in METERS, matching design_nozzle()'s
        ``wall_final["x"]``/``["y"]`` (throat at index 0, increasing x
        downstream). Converted to mm internally -- all lengths in the
        returned dict are in mm, matching the original script's units.
    metal_angle_in, metal_angle_out : float
        Camberline metal angles [deg] at the convergent-section inlet and
        at the throat (matching the divergent-section wall angle).
    r_trailing : float
        Trailing-edge circle radius [mm].
    inlet_opening_ratio : float
        pitch / inlet_opening (throat pitch to passage-inlet-opening ratio),
        sets how far upstream the pressure-side Hermite spline reaches.
    n_cp : int
        Number of B-spline control points fitted to the closed contour.
    num_baseline_points, num_curve_points : int
        Resampling / evaluation resolution for the B-spline fit.

    Returns
    -------
    dict
        Flat, JSON-safe dict: ``suction`` / ``pressure`` (raw sides, mm),
        ``blade_curve`` (closed B-spline-fitted contour, mm),
        ``control_points`` (n_cp x 2, mm), ``trailing_edge`` (mm),
        ``pitch``, ``throat_opening``, ``inlet_opening`` (mm), and the
        input parameters echoed back.
    """
    x_vals = np.array(wall_x, dtype=float) * 1e3
    y_vals = np.array(wall_y, dtype=float) * 1e3

    x_mirrored = x_vals
    y_mirrored = -y_vals

    n_prefix = min(130, len(x_mirrored))
    prefix_x = -x_mirrored[:n_prefix]
    prefix_y = y_mirrored[:n_prefix]

    x_mirrored = np.concatenate((prefix_x[::-1], x_mirrored))
    y_mirrored = np.concatenate((prefix_y[::-1], y_mirrored))

    throat_opening = y_vals[0]
    pitch = (2 * max(y_vals) + 2 * r_trailing) / np.cos(np.deg2rad(metal_angle_out))
    axial_chord_divergent = (np.max(x_vals) - np.min(x_vals)) * np.cos(np.deg2rad(metal_angle_out))
    axial_chord_convergent = 0.5 * axial_chord_divergent
    inlet_opening = pitch / inlet_opening_ratio

    # --- Camberline: integrate a linear metal-angle blend from inlet to throat.
    def beta(x):
        return metal_angle_in + (metal_angle_out - metal_angle_in) * x / axial_chord_convergent

    def camberline_slope(x, y):
        return np.tan(np.deg2rad(beta(x)))

    def target_angle_event(x, y):
        return beta(x) - metal_angle_out

    target_angle_event.terminal = True
    target_angle_event.direction = 1

    sol = solve_ivp(
        camberline_slope,
        [0.0, 1000.0],
        [0.0],
        events=target_angle_event,
        rtol=1e-8,
        atol=1e-8,
    )
    x_raw, y_raw = sol.t, sol.y[0]
    x_convergent = np.linspace(x_raw[0], x_raw[-1], len(x_raw))
    y_convergent = np.interp(x_convergent, x_raw, y_raw)

    x_rot, y_rot = _rotate_points(x_convergent, y_convergent, metal_angle_out)
    x_rot = x_rot - x_rot.max()
    y_rot = y_rot + abs(y_rot.min()) + throat_opening

    # --- Suction side: convergent camberline + divergent nozzle wall.
    x_vals, y_vals = _translate_points(x_vals, y_vals, dx=0, dy=(throat_opening - min(y_vals)))
    suction_x = np.concatenate([x_rot, x_vals])
    suction_y = np.concatenate([y_rot, y_vals])
    trailing_x, trailing_y, _ = _create_trailing_edge_points(
        [suction_x[-1], suction_y[-1], 0],
        [suction_x[-1], suction_y[-1] + 2 * r_trailing, 0],
        r_trailing,
    )

    theta = np.deg2rad(90 - metal_angle_out)
    trailing_x, trailing_y = _translate_points(trailing_x, trailing_y, abs(suction_x[0]) + pitch * np.cos(theta))
    trailing_x, trailing_y = _rotate_points(trailing_x, trailing_y, 90 - metal_angle_out)
    suction_x, suction_y = _translate_points(suction_x, suction_y, abs(suction_x[0]))
    x_mirrored, y_mirrored = _translate_points(x_mirrored, y_mirrored, abs(suction_x[-1] - x_mirrored[-1]))

    suction_x, suction_y = _rotate_points(suction_x, suction_y, 90 - metal_angle_out)
    x_mirrored, y_mirrored = _rotate_points(x_mirrored, y_mirrored, 90 - metal_angle_out)
    divergent_straight_x, divergent_straight_y = _create_straight_line(
        [suction_x[-1], suction_y[-1]], [trailing_x[-1], trailing_y[-1]], N=3
    )
    x_mirrored_trans, y_mirrored_trans = _translate_points(x_mirrored, y_mirrored, pitch)

    # --- Pressure side: mirrored wall (translated by pitch) + Hermite spline
    #     bridging the leading-edge region. Tangent is the normal of the
    #     suction side's start direction.
    dxg, dyg = np.gradient(suction_x), np.gradient(suction_y)
    tx, ty = -dyg[0], dxg[0]
    target_tangent = np.array([tx, ty])

    P0 = [suction_x[0] + pitch - inlet_opening, suction_y[0]]
    P3 = [x_mirrored_trans[0], y_mirrored_trans[0]]
    pressure_x, pressure_y = _get_cubic_spline_points(
        P0=P0, P3=P3, tangent_start=[1, -1], tangent_end=target_tangent
    )

    # --- Leading edge fillet joining suction side start to pressure side start.
    dy_pressure, dx_pressure = np.gradient(pressure_y), np.gradient(pressure_x)
    slope_pressure = dy_pressure / dx_pressure
    p_le_start = (suction_x[0], suction_y[0])
    p_le_end = (pressure_x[0], pressure_y[0])
    _, le_x, le_y = _halfcircle_4cp(
        p_le_start, p_le_end, slope_end=slope_pressure[0], type="leading_edge"
    )

    pressure_x = np.concatenate([le_x, pressure_x, x_mirrored_trans])
    pressure_y = np.concatenate([le_y, pressure_y, y_mirrored_trans])
    suction_x = np.concatenate([suction_x, divergent_straight_x])
    suction_y = np.concatenate([suction_y, divergent_straight_y])

    # --- Assemble closed contour, dedupe (stable, order-preserving).
    full_blade_x = np.concatenate([suction_x[::-1], pressure_x])
    full_blade_y = np.concatenate([suction_y[::-1], pressure_y])
    raw_pts = np.vstack((full_blade_x, full_blade_y)).T
    _, unique_indices = np.unique(raw_pts, axis=0, return_index=True)
    cleaned_pts = raw_pts[np.sort(unique_indices)]

    # --- Fit a fixed-endpoint cubic B-spline through the closed contour.
    spline3 = BSplineTool(degree=3)
    baseline_equi = spline3.reparameterize_equidistant(cleaned_pts, num_points=num_baseline_points)
    control_points, knots = spline3.backward_fixed(baseline_equi, n_cp)
    blade_curve = spline3.forward(control_points, knots, num_points=num_curve_points)

    return {
        "suction": {"x": suction_x.tolist(), "y": suction_y.tolist()},
        "pressure": {"x": pressure_x.tolist(), "y": pressure_y.tolist()},
        "blade_curve": {"x": blade_curve[:, 0].tolist(), "y": blade_curve[:, 1].tolist()},
        "control_points": control_points.tolist(),
        "knots": knots.tolist(),
        "trailing_edge": {"x": trailing_x.tolist(), "y": trailing_y.tolist()},
        "pitch": float(pitch),
        "throat_opening": float(throat_opening),
        "inlet_opening": float(inlet_opening),
        "axial_chord_convergent": float(axial_chord_convergent),
        "axial_chord_divergent": float(axial_chord_divergent),
        "metal_angle_in": float(metal_angle_in),
        "metal_angle_out": float(metal_angle_out),
        "r_trailing": float(r_trailing),
        "n_cp": int(n_cp),
        "units": "mm",
    }


def parametrize_stator_blade_semi(
    wall_x,
    wall_y,
    metal_angle_in=0.0,
    metal_angle_out=70.0,
    r_trailing=0.5,
    inlet_opening_ratio=2.5,
    n_cp=30,
    num_baseline_points=500,
    num_curve_points=500,
    trailing_edge_num_points=15,
):
    """
    Build a closed stator blade profile from a MOC nozzle wall contour,
    following opt_framework's stator_parametrization_semi_nozzle.py --
    a lighter-weight alternative to parametrize_stator_blade() that only
    needs the suction-side wall, not a mirrored copy of it:

    - Pressure side: a single circular arc between two points (near the
      leading and trailing edge), not a mirrored-wall-plus-Hermite-spline
      bridge -- so pitch/inlet_opening/the arc's own geometry set the
      passage shape directly, rather than the wall's own divergence.
    - Control points: a plain (non-fixed-endpoint) least-squares B-spline
      fit (BSplineTool.backward), not backward_fixed -- the fitted curve
      is not forced through the exact suction/pressure start/end points.

    Same "plain args in, flat JSON-safe dict out" pattern, same units
    convention (wall_x/wall_y in METERS, everything in the returned dict
    in mm) as parametrize_stator_blade() -- see its docstring for the
    parameter meanings shared between the two. inlet_opening_ratio
    defaults to 2.5 here (vs. 1.8 for the mirrored-wall variant) and
    trailing_edge_num_points to 15 (vs. 50), matching the two original
    scripts' own defaults.
    """
    x_vals = np.array(wall_x, dtype=float) * 1e3
    y_vals = np.array(wall_y, dtype=float) * 1e3

    throat_opening = y_vals[0]
    pitch = (max(y_vals) + 2 * r_trailing) / np.cos(np.deg2rad(metal_angle_out))
    axial_chord_divergent = (np.max(x_vals) - np.min(x_vals)) * np.cos(np.deg2rad(metal_angle_out))
    axial_chord_convergent = 0.5 * axial_chord_divergent
    inlet_opening = pitch / inlet_opening_ratio

    # --- Camberline: integrate a linear metal-angle blend from inlet to throat.
    def beta(x):
        return metal_angle_in + (metal_angle_out - metal_angle_in) * x / axial_chord_convergent

    def camberline_slope(x, y):
        return np.tan(np.deg2rad(beta(x)))

    def target_angle_event(x, y):
        return beta(x) - metal_angle_out

    target_angle_event.terminal = True
    target_angle_event.direction = 1

    sol = solve_ivp(
        camberline_slope,
        [0.0, 1000.0],
        [0.0],
        events=target_angle_event,
        rtol=1e-8,
        atol=1e-8,
    )
    x_raw, y_raw = sol.t, sol.y[0]
    x_convergent = np.linspace(x_raw[0], x_raw[-1], len(x_raw))
    y_convergent = np.interp(x_convergent, x_raw, y_raw)

    x_rot, y_rot = _rotate_points(x_convergent, y_convergent, metal_angle_out)
    x_rot = x_rot - x_rot.max()
    y_rot = y_rot + abs(y_rot.min()) + throat_opening

    # --- Suction side: convergent camberline + divergent nozzle wall.
    x_vals, y_vals = _translate_points(x_vals, y_vals, dx=0, dy=(throat_opening - min(y_vals)))
    suction_x = np.concatenate([x_rot, x_vals])
    suction_y = np.concatenate([y_rot, y_vals])
    trailing_x, trailing_y, _ = _create_trailing_edge_points(
        [suction_x[-1], suction_y[-1], 0],
        [suction_x[-1], suction_y[-1] + 2 * r_trailing, 0],
        r_trailing,
        num_points=trailing_edge_num_points,
    )
    trailing_x, trailing_y = _translate_points(trailing_x, trailing_y, abs(suction_x[0]))
    trailing_x, trailing_y = _rotate_points(trailing_x, trailing_y, 90 - metal_angle_out)
    suction_x, suction_y = _translate_points(suction_x, suction_y, abs(suction_x[0]))

    theta = np.deg2rad(90 - metal_angle_out)

    # --- Pressure side: a single circular arc between a point near the
    #     trailing edge and a point near the leading edge (no mirrored
    #     wall involved).
    p1 = np.array([
        suction_x[-1] - (x_vals[-1] - pitch * np.cos(theta)),
        suction_y[-1] + 2 * r_trailing,
    ])
    p2 = np.array([
        suction_x[0] + (pitch - inlet_opening) * np.cos(theta),
        suction_y[0] + (pitch - inlet_opening) * np.sin(theta),
    ])
    pressure_x, pressure_y = _get_arc_coordinates(p1, p2, len(x_rot))
    pressure_x, pressure_y = _rotate_points(pressure_x, pressure_y, 90 - metal_angle_out)
    suction_x, suction_y = _rotate_points(suction_x, suction_y, 90 - metal_angle_out)

    # --- Straight segment bridging the pressure-side arc to the trailing edge.
    straight_x, straight_y = _create_straight_line(
        [pressure_x[0], pressure_y[0]], [trailing_x[0], trailing_y[0]], N=3
    )
    pressure_x = np.concatenate([pressure_x[::-1], straight_x])
    pressure_y = np.concatenate([pressure_y[::-1], straight_y])

    # --- Leading edge fillet joining suction side start to pressure side start.
    dy_pressure, dx_pressure = np.gradient(pressure_y), np.gradient(pressure_x)
    slope_pressure = dy_pressure / dx_pressure
    p_le_start = (suction_x[0], suction_y[0])
    p_le_end = (pressure_x[0], pressure_y[0])
    _, le_x, le_y = _halfcircle_4cp(
        p_le_start, p_le_end, slope_end=slope_pressure[0], type="leading_edge"
    )
    pressure_x = np.concatenate([le_x, pressure_x])
    pressure_y = np.concatenate([le_y, pressure_y])

    # --- Assemble closed contour, dedupe (stable, order-preserving).
    full_blade_x = np.concatenate([suction_x[::-1], pressure_x])
    full_blade_y = np.concatenate([suction_y[::-1], pressure_y])
    raw_pts = np.vstack((full_blade_x, full_blade_y)).T
    _, unique_indices = np.unique(raw_pts, axis=0, return_index=True)
    cleaned_pts = raw_pts[np.sort(unique_indices)]

    # --- Fit a plain (non-fixed-endpoint) cubic B-spline through the contour.
    spline3 = BSplineTool(degree=3)
    baseline_equi = spline3.reparameterize_equidistant(cleaned_pts, num_points=num_baseline_points)
    control_points, knots = spline3.backward(baseline_equi, n_cp)
    blade_curve = spline3.forward(control_points, knots, num_points=num_curve_points)

    return {
        "suction": {"x": suction_x.tolist(), "y": suction_y.tolist()},
        "pressure": {"x": pressure_x.tolist(), "y": pressure_y.tolist()},
        "blade_curve": {"x": blade_curve[:, 0].tolist(), "y": blade_curve[:, 1].tolist()},
        "control_points": control_points.tolist(),
        "knots": knots.tolist(),
        "trailing_edge": {"x": trailing_x.tolist(), "y": trailing_y.tolist()},
        "pitch": float(pitch),
        "throat_opening": float(throat_opening),
        "inlet_opening": float(inlet_opening),
        "axial_chord_convergent": float(axial_chord_convergent),
        "axial_chord_divergent": float(axial_chord_divergent),
        "metal_angle_in": float(metal_angle_in),
        "metal_angle_out": float(metal_angle_out),
        "r_trailing": float(r_trailing),
        "n_cp": int(n_cp),
        "units": "mm",
        "variant": "semi",
    }


def move_control_point_along_normal(blade_data, cp_index, distance, num_curve_points=None):
    """
    Nudge one B-spline control point of a parametrized blade by `distance`
    (mm, positive/negative) along its local normal -- estimated as the
    90-degree rotation of the discrete tangent through its neighboring
    control points -- then re-evaluate the fitted curve from the updated
    control points.

    True click-and-drag isn't practical in a Plotly/Dash app without heavy
    custom clientside JS (Plotly's `editable` config covers shapes/
    annotations, not scatter markers), so this select-a-point-then-nudge
    workflow is the supported editing mode for moc.app's Phase 2.

    Moving cp_index 0 or -1 (the fixed leading/trailing anchor points from
    backward_fixed) will break their exact match to the raw suction/
    pressure endpoints -- allowed, but usually not what you want.

    Returns a new blade_data dict (does not mutate the input): only
    "control_points" and "blade_curve" change, everything else (pitch,
    trailing_edge, suction/pressure raw sides, knots, ...) is carried over.
    """
    cps = np.array(blade_data["control_points"], dtype=float)
    knots = np.array(blade_data["knots"], dtype=float)
    n = len(cps)
    if not (0 <= cp_index < n):
        raise ValueError(f"cp_index {cp_index} out of range [0, {n - 1}]")

    i_prev = max(cp_index - 1, 0)
    i_next = min(cp_index + 1, n - 1)
    tangent = cps[i_next] - cps[i_prev]
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-12:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / tangent_norm
    normal = np.array([-tangent[1], tangent[0]])

    cps_new = cps.copy()
    cps_new[cp_index] = cps_new[cp_index] + distance * normal

    n_curve = num_curve_points or len(blade_data["blade_curve"]["x"])
    spline3 = BSplineTool(degree=3)
    curve_pts = spline3.forward(cps_new, knots, num_points=n_curve)

    new_data = dict(blade_data)
    new_data["control_points"] = cps_new.tolist()
    new_data["blade_curve"] = {"x": curve_pts[:, 0].tolist(), "y": curve_pts[:, 1].tolist()}
    return new_data


def export_blade_step(blade_data, face_path, solid_path=None, extrude_length=1.0):
    """
    Export the closed blade contour (from parametrize_stator_blade) to
    STEP: a 2D face and, optionally, an extruded 3D solid.

    Requires cadquery (not a core moc dependency -- import kept local so
    the rest of moc.geometry works without it installed).
    """
    import cadquery as cq

    curve = blade_data["blade_curve"]
    te = blade_data["trailing_edge"]

    curve_pts = [cq.Vector(x, y, 0) for x, y in zip(curve["x"], curve["y"])]
    te_pts = [cq.Vector(x, y, 0) for x, y in zip(te["x"], te["y"])]

    blade_wire = (
        cq.Workplane("XY")
        .moveTo(curve_pts[-1].x, curve_pts[-1].y)
        .polyline(te_pts)
        .spline(curve_pts[:-1])
        .close()
    )
    blade_face = cq.Face.makeFromWires(blade_wire.val())
    cq.exporters.export(blade_face, str(face_path))

    if solid_path is not None:
        blade_solid = cq.Workplane("XY").add(blade_face).extrude(extrude_length)
        cq.exporters.export(blade_solid, str(solid_path))


def _closed_wire_2d(cq, curve_xy, te_xy=None):
    """Closed wire from 2D (x,y) point lists -- TE as a sharp polyline plus
    the rest as a smooth spline (curve_xy) when te_xy is given (stator
    convention: parametrize_stator_blade's blade_curve/trailing_edge kept
    separate), or a single smooth closed spline through curve_xy alone
    (rotor convention: design_rotor_vortex_blade's "blade" is already one
    closed, LE/TE-rounded contour)."""
    if te_xy:
        wp = (
            cq.Workplane("XY")
            .moveTo(*curve_xy[-1])
            .polyline(te_xy)
            .spline(curve_xy[:-1])
            .close()
        )
    else:
        wp = cq.Workplane("XY").moveTo(*curve_xy[0]).spline(curve_xy[1:]).close()
    return wp.val()


def export_flared_blade_step(blade_data, r_hub_in, r_hub_out, r_tip_in, r_tip_out,
                              face_path, solid_path=None, source="stator"):
    """
    Export a FLARED 3D blade solid: lofts between a hub-radius copy of the
    2D blade profile (at Z=0) and a tip-radius copy (at Z=blade_height)
    whose pitchwise (tangential) extent is scaled by r_tip_mean/r_hub_mean
    -- pitch = 2*pi*r/Z_blades at constant blade count, so this is the
    physically-correct scaling for "same blade count, wider pitch at
    larger radius", not an arbitrary stretch.

    See moc.geometry.meridional's module docstring for why this needs no
    separate axial (chordwise) shift between hub and tip: flare is
    assumed to share the same axial stations for hub/tip (Anderson et
    al.'s own convention), so it shows up as radius change (the
    meridional view, build_meridional_view) and this pitch scaling here
    -- the two share the SAME hub/tip radii inputs, not independently
    guessed ones.

    Parameters
    ----------
    blade_data : dict
        A stator blade dict (parametrize_stator_blade/_semi's output,
        needs "blade_curve"/"trailing_edge") or a rotor blade dict
        (design_rotor_vortex_blade's output, needs "blade") -- pick which
        via `source`.
    r_hub_in, r_hub_out, r_tip_in, r_tip_out : float
        Hub/tip radii at this row's own inlet/outlet (same length units
        as blade_data, typically mm) -- blade_height and the pitch-scale
        factor are both derived from these, not separate inputs.
    source : str
        "stator" or "rotor" -- which blade_data shape to expect.

    Requires cadquery (not a core moc dependency -- import kept local so
    the rest of moc.geometry works without it installed).
    """
    import cadquery as cq

    if source == "stator":
        curve = blade_data["blade_curve"]
        te = blade_data["trailing_edge"]
        curve_xy = list(zip(curve["x"], curve["y"]))
        te_xy = list(zip(te["x"], te["y"]))
    elif source == "rotor":
        blade = blade_data["blade"]
        curve_xy = list(zip(blade["x"], blade["y"]))
        te_xy = None
    else:
        raise ValueError(f"source must be 'stator' or 'rotor', got {source!r}")

    hub_wire = _closed_wire_2d(cq, curve_xy, te_xy)
    hub_face = cq.Face.makeFromWires(hub_wire)
    cq.exporters.export(hub_face, str(face_path))
    if solid_path is None:
        return

    r_hub_mean = 0.5 * (r_hub_in + r_hub_out)
    r_tip_mean = 0.5 * (r_tip_in + r_tip_out)
    pitch_scale = r_tip_mean / r_hub_mean
    blade_height = r_tip_mean - r_hub_mean

    curve_xy_tip = [(x * pitch_scale, y) for x, y in curve_xy]
    te_xy_tip = [(x * pitch_scale, y) for x, y in te_xy] if te_xy else None
    tip_wire_flat = _closed_wire_2d(cq, curve_xy_tip, te_xy_tip)
    tip_wire = tip_wire_flat.translate((0, 0, blade_height))

    loft_solid = cq.Solid.makeLoft([hub_wire, tip_wire])
    cq.exporters.export(loft_solid, str(solid_path))
