"""
turbo_moc.geometry.blade -- stator blade parametrization built on top of a MOC
nozzle wall contour (the suction side).

Ported from opt_framework/test_2/01_geometry_parametrization/1.1_stator_axial.py,
refactored into turbo_moc.design_nozzle()'s "plain args in, flat JSON-safe dict
out" pattern: parametrize_stator_blade(...) takes the nozzle wall contour
(x, y in meters, matching design_nozzle()'s wall_final schema) plus blade
parameters (previously hardcoded module globals in the original script),
and returns a flat dict (all lengths in mm, matching the original script's
working units) -- no solver/UI objects. The geometry helpers below are a
line-for-line port of the original script's module-level functions.

export_blade_step(...) is kept separate/optional since it depends on
cadquery, which the rest of turbo_moc does not otherwise require.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from .annular import wrap_blade_annular
from .spline_tool import BSplineTool

__all__ = [
    "parametrize_stator_blade", "parametrize_stator_blade_semi",
    "export_blade_step", "export_flared_blade_step", "export_annular_blade_step",
    "export_annular_blade_stl",
    "move_control_point_along_normal",
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
    mirror=False,
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
    mirror : bool
        Flip the blade's concavity (which side is suction/pressure) by
        reflecting the finished shape across the chordwise (y) axis --
        i.e. negating every "x" (pitchwise) coordinate in the returned
        dict, post-construction. NOT the same as negating metal_angle_out:
        that feeds a turning-angle profile AND a separate frame rotation,
        both driven by metal_angle_out, and negating it does not cancel
        cleanly -- it produces a self-crossing, invalid shape (confirmed
        by direct comparison, not assumed). Reflecting the finished 2D
        points is the only method that's actually been verified to give a
        clean mirror image.

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
    # Index of the throat point (wall_x/wall_y's own index 0, the nozzle
    # throat) within suction_x/suction_y -- the convergent-camberline
    # prefix (x_rot/y_rot) comes first, so the throat is the FIRST point
    # of the appended wall segment. Every later transform below (translate/
    # rotate) is a whole-array operation preserving order/length, so this
    # index stays valid all the way to the final returned "suction" arrays.
    idx_throat = len(x_rot)

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

    if mirror:
        suction_x = -suction_x
        pressure_x = -pressure_x
        trailing_x = -trailing_x
        blade_curve = np.column_stack([-blade_curve[:, 0], blade_curve[:, 1]])
        control_points = np.column_stack([-control_points[:, 0], control_points[:, 1]])

    return {
        "suction": {"x": suction_x.tolist(), "y": suction_y.tolist()},
        "pressure": {"x": pressure_x.tolist(), "y": pressure_y.tolist()},
        "blade_curve": {"x": blade_curve[:, 0].tolist(), "y": blade_curve[:, 1].tolist()},
        "control_points": control_points.tolist(),
        "knots": knots.tolist(),
        "trailing_edge": {"x": trailing_x.tolist(), "y": trailing_y.tolist()},
        "pitch": float(pitch),
        "throat_opening": float(throat_opening),
        "throat_point": {"x": float(suction_x[idx_throat]), "y": float(suction_y[idx_throat])},
        "inlet_opening": float(inlet_opening),
        "axial_chord_convergent": float(axial_chord_convergent),
        "axial_chord_divergent": float(axial_chord_divergent),
        "metal_angle_in": float(metal_angle_in),
        "metal_angle_out": float(metal_angle_out),
        "r_trailing": float(r_trailing),
        "n_cp": int(n_cp),
        "mirror": bool(mirror),
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
    mirror=False,
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
    # See parametrize_stator_blade's matching comment: index of the throat
    # point within suction_x/suction_y, valid all the way to the final
    # returned "suction" arrays (only whole-array transforms follow).
    idx_throat = len(x_rot)

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

    if mirror:
        suction_x = -suction_x
        pressure_x = -pressure_x
        trailing_x = -trailing_x
        blade_curve = np.column_stack([-blade_curve[:, 0], blade_curve[:, 1]])
        control_points = np.column_stack([-control_points[:, 0], control_points[:, 1]])

    return {
        "suction": {"x": suction_x.tolist(), "y": suction_y.tolist()},
        "pressure": {"x": pressure_x.tolist(), "y": pressure_y.tolist()},
        "blade_curve": {"x": blade_curve[:, 0].tolist(), "y": blade_curve[:, 1].tolist()},
        "control_points": control_points.tolist(),
        "knots": knots.tolist(),
        "trailing_edge": {"x": trailing_x.tolist(), "y": trailing_y.tolist()},
        "pitch": float(pitch),
        "throat_opening": float(throat_opening),
        "throat_point": {"x": float(suction_x[idx_throat]), "y": float(suction_y[idx_throat])},
        "inlet_opening": float(inlet_opening),
        "axial_chord_convergent": float(axial_chord_convergent),
        "axial_chord_divergent": float(axial_chord_divergent),
        "metal_angle_in": float(metal_angle_in),
        "metal_angle_out": float(metal_angle_out),
        "r_trailing": float(r_trailing),
        "n_cp": int(n_cp),
        "mirror": bool(mirror),
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
    workflow is the supported editing mode for turbo_moc.app's Phase 2.

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

    Requires cadquery (not a core turbo_moc dependency -- import kept local so
    the rest of turbo_moc.geometry works without it installed).
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

    See turbo_moc.geometry.meridional's module docstring for why this needs no
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

    Requires cadquery (not a core turbo_moc dependency -- import kept local so
    the rest of turbo_moc.geometry works without it installed).
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


# OCCT's own working precision is ~1e-7 (mm, this codebase's convention);
# a segment shorter than that is "degenerate" to it even when nonzero, and
# silently corrupts downstream topology (wire mis-ordering, failed lofts)
# instead of raising -- real blade data hits this routinely (e.g. a
# trailing-edge/main-curve junction that's 0.0 by construction, or a
# neighbour a few nanometers off after the 3D wrap's floating-point ops).
# 1e-6 mm (1 nm) comfortably clears that floor while staying far below any
# real geometric feature.
_MIN_EDGE_LENGTH = 1e-6


def _maybe_line_edge(cq, p1, p2, tol=_MIN_EDGE_LENGTH):
    """A straight edge p1->p2, or None if the two points already coincide
    (within tol) -- cq.Edge.makeLine raises on a zero-length line, which a
    literal "closed" input curve (e.g. design_rotor_vortex_blade's "blade",
    whose first and last points are identical by construction) triggers
    every time otherwise."""
    if sum((a - b) ** 2 for a, b in zip(p1, p2)) < tol ** 2:
        return None
    return cq.Edge.makeLine(cq.Vector(*p1), cq.Vector(*p2))


def _dedupe_consecutive(pts, tol=_MIN_EDGE_LENGTH):
    """Drop each point that coincides (within tol) with the one before it --
    cq.Edge.makeSpline's underlying GeomAPI_Interpolate raises
    Standard_ConstructionError on a duplicate/near-zero-length leading
    segment, which real blade data hits routinely (a stator's trailing-edge
    arc and its main curve share an exact junction point by construction).
    Workplane's own .spline() tolerates this silently; raw cq.Edge.makeSpline
    does not, hence this explicit pass before every use of it below."""
    out = [pts[0]]
    for p in pts[1:]:
        if sum((a - b) ** 2 for a, b in zip(p, out[-1])) >= tol ** 2:
            out.append(p)
    return out


def _closed_wire_3d(cq, curve_xyz, te_xyz=None):
    """3D analog of _closed_wire_2d -- same TE-polyline + spline-curve
    split, but built from raw (x, y, z) point lists via cq.Edge/cq.Wire
    directly, since Workplane sketching (moveTo/spline/polyline) is
    plane-locked and these span-station wires lie on a curved (cylindrical)
    surface, not a flat plane.

    Edges are added to the wire builder ONE AT A TIME, in known order, not
    via cq.Wire.assembleEdges's batch add -- that method's underlying
    BRepBuilderAPI_MakeWire.Add(TopTools_ListOfShape) auto-solves edge
    connectivity order-independently, which a near-degenerate segment (see
    _MIN_EDGE_LENGTH) was found to occasionally mis-solve into a
    self-crossing wire (ShapeAnalysis_Wire.CheckOrder() confirmed this on
    real blade data) even though every individual edge/vertex still passed
    validity checks -- feeding edges in already-known sequence order avoids
    that ambiguity entirely."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire

    if te_xyz:
        line_pts = [curve_xyz[-1], *te_xyz]
        line_edges = [
            e for i in range(len(line_pts) - 1)
            if (e := _maybe_line_edge(cq, line_pts[i], line_pts[i + 1])) is not None
        ]
        spline_pts = _dedupe_consecutive([te_xyz[-1], *curve_xyz[:-1]])
        spline_edge = cq.Edge.makeSpline([cq.Vector(*p) for p in spline_pts])
        close_edge = _maybe_line_edge(cq, curve_xyz[-2], curve_xyz[-1])
        edges = [*line_edges, spline_edge, *([close_edge] if close_edge is not None else [])]
    else:
        spline_pts = _dedupe_consecutive(curve_xyz)
        spline_edge = cq.Edge.makeSpline([cq.Vector(*p) for p in spline_pts])
        close_edge = _maybe_line_edge(cq, curve_xyz[-1], curve_xyz[0])
        edges = [spline_edge, *([close_edge] if close_edge is not None else [])]

    builder = BRepBuilderAPI_MakeWire()
    for e in edges:
        builder.Add(e.wrapped)
    return cq.Wire(builder.Wire())


def _cap_wire(cq, wire):
    """A free-form (not necessarily planar) surface patch filling `wire`,
    via BRepOffsetAPI_MakeFilling -- cq.Face.makeFromWires only builds
    planar faces, but the hub/shroud boundary here curves around the
    cylinder axis, so it needs a genuine filling surface instead."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0

    filler = BRepOffsetAPI_MakeFilling()
    for e in wire.Edges():
        filler.Add(e.wrapped, GeomAbs_C0)
    filler.Build()
    return cq.Face(filler.Shape())


def _close_loft_solid(cq, wires):
    """cq.Solid.makeLoft(wires) leaves the hub/shroud ends OPEN here (an
    unfilled hole at each end, not a real solid) -- BRepOffsetAPI_
    ThruSections's automatic end-capping only works for PLANAR boundary
    sections, and both ends of this loft are the same curved-around-the-
    cylinder, non-planar wires as every other span station. This caps
    both ends explicitly (_cap_wire) and sews loft + caps into one
    watertight shell, then builds a genuine closed Solid from it -- so
    CAD software sees a real body, not a surface with holes at the ends."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.TopoDS import TopoDS

    loft_shell = cq.Solid.makeLoft(wires)
    hub_cap = _cap_wire(cq, wires[0])
    tip_cap = _cap_wire(cq, wires[-1])

    sewing = BRepBuilderAPI_Sewing(1e-6)
    for shape in (loft_shell, hub_cap, tip_cap):
        sewing.Add(shape.wrapped)
    sewing.Perform()
    shell = TopoDS.Shell_s(sewing.SewedShape())

    solid_maker = BRepBuilderAPI_MakeSolid(shell)
    return cq.Solid(solid_maker.Solid())


def export_annular_blade_step(blade_data, r_hub, r_shroud, face_path, solid_path=None,
                               n_blades=1, flare="pitch_scale", n_span=13, theta0=0.0,
                               source="stator", z_offset=0.0):
    """
    Export a TRUE annular 3D blade solid: the profile is wrapped
    circumferentially at every span station from r_hub to r_shroud (see
    turbo_moc.geometry.annular's module docstring for the wrap math), then
    lofted through all n_span sections and capped at the hub/shroud ends
    (_close_loft_solid) into one watertight body -- unlike export_flared_
    blade_step's flat 2-wire loft between parallel planes, this actually
    curves around the machine axis. Always exports a SINGLE blade;
    n_blades only affects the pitch-scale flare physics (pitch =
    2*pi*r/n_blades convention) and is not used to multiply the exported
    solid.

    Requires cadquery (not a core turbo_moc dependency -- import kept local
    so the rest of turbo_moc.geometry works without it installed).

    Unlike export_flared_blade_step's hub_face (flat, so a genuine planar
    Face), the hub SECTION here is a non-planar 3D curve -- it curves
    around the cylinder AND varies axially along the profile at the same
    time -- so cq.Face.makeFromWires (planar faces only) can't build a
    face from it. face_path therefore gets the hub WIRE itself (a valid
    STEP curve, still useful as a reference/sketch import), not a face.

    z_offset : float
        Added to every axial (Z) coordinate -- positions this solid in a
        shared global frame against an already-placed neighbor (e.g. a
        rotor STEP file), same purpose as turbo_moc.geometry.
        export_turbogrid_blade's x_offset. No sign/orientation logic is
        applied here (unlike that function's axial_sign) -- a STEP solid
        has no inherent "inlet/outlet" semantics, so z_offset is a plain
        additive shift; figure out which end of THIS blade (its wrapped
        curve/trailing_edge z-range) needs to land where before picking
        a value.
    """
    import cadquery as cq

    wires, solid = _build_annular_blade_solid(
        cq, blade_data, r_hub, r_shroud, n_blades=n_blades,
        flare=flare, n_span=n_span, theta0=theta0, source=source, z_offset=z_offset,
        build_solid=solid_path is not None,
    )
    cq.exporters.export(wires[0], str(face_path))
    if solid_path is None:
        return
    cq.exporters.export(solid, str(solid_path))


def _build_annular_blade_solid(cq, blade_data, r_hub, r_shroud, n_blades=1, flare="pitch_scale",
                                n_span=13, theta0=0.0, source="stator", z_offset=0.0, build_solid=True):
    """Shared by export_annular_blade_step and export_annular_blade_stl --
    builds the per-span wires and (optionally) the watertight lofted+capped
    solid once, so both export formats come from the exact same body
    instead of two independently-built shapes that could drift apart."""
    wrapped = wrap_blade_annular(blade_data, r_hub, r_shroud, n_blades=n_blades,
                                  flare=flare, n_span=n_span, theta0=theta0, source=source)
    curve = wrapped["curve"]
    te = wrapped["trailing_edge"]

    wires = []
    for i in range(wrapped["n_span"]):
        curve_xyz = list(zip(curve["x"][0, i], curve["y"][0, i], curve["z"][0, i] + z_offset))
        te_xyz = (list(zip(te["x"][0, i], te["y"][0, i], te["z"][0, i] + z_offset))
                  if te is not None else None)
        wires.append(_closed_wire_3d(cq, curve_xyz, te_xyz))

    solid = _close_loft_solid(cq, wires) if build_solid else None
    return wires, solid


def export_annular_blade_stl(blade_data, r_hub, r_shroud, stl_path,
                              n_blades=1, flare="pitch_scale", n_span=13, theta0=0.0,
                              source="stator", z_offset=0.0):
    """Export the same watertight annular blade solid as
    export_annular_blade_step, but as an STL mesh -- for CAD-quality (true
    solid, properly shaded) 3D preview in the app via dash-vtk, which reads
    triangulated meshes, not BREP/STEP. Requires cadquery."""
    import cadquery as cq

    _, solid = _build_annular_blade_solid(
        cq, blade_data, r_hub, r_shroud, n_blades=n_blades,
        flare=flare, n_span=n_span, theta0=theta0, source=source, z_offset=z_offset,
    )
    cq.exporters.export(solid, str(stl_path))


def _build_radial_blade_solid(cq, blade_data, r1, r2, depth, source="stator",
                               theta0=0.0, z_offset=0.0, build_solid=True):
    """One radial-wrapped blade (single copy, index 0) as a flat prism of
    thickness `depth` along Z. Unlike the annular wrap, turbo_moc.geometry.
    radial's log-spiral conformal map has no spanwise dimension of its own
    (it maps a 2D cascade profile onto a 2D annulus footprint only), so
    `depth` is a plain extrusion thickness -- the same semantics as the
    app's VTK-only radial preview, built here as a real watertight solid.

    The section is a fine POLYLINE through the wrapped points, not the
    single closed interpolating spline _closed_wire_3d builds: on the
    rotor's contour (long straight LE/TE stubs between densely sampled
    walls) a solid bounded by that one closed spline edge passes
    isValid() but reports a wrong volume under both makeLoft (~0.35x) and
    extrudeLinear (~1.3x), while the polyline prism matches the shoelace
    area * depth to ~10 significant figures. Point spacing is already
    sub-mm, so the faceting is not visible."""
    from .radial import wrap_blade_radial, wrap_rotor_blade_radial

    if source == "stator":
        wrapped = wrap_blade_radial(blade_data, r1, r2, n_blades=1, theta0=theta0)
        x = wrapped["blades"][0]["x"] + wrapped["trailing_edges"][0]["x"]
        y = wrapped["blades"][0]["y"] + wrapped["trailing_edges"][0]["y"]
    elif source == "rotor":
        wrapped = wrap_rotor_blade_radial(blade_data, r1, r2, n_blades=1, theta0=theta0)
        x, y = wrapped["blades"][0]["x"], wrapped["blades"][0]["y"]
    else:
        raise ValueError(f"source must be 'stator' or 'rotor', got {source!r}")

    pts = _dedupe_consecutive(list(zip(x, y, [z_offset] * len(x))))
    if len(pts) > 1 and np.linalg.norm(np.subtract(pts[0], pts[-1])) < _MIN_EDGE_LENGTH:
        pts = pts[:-1]
    bottom = cq.Wire.makePolygon([cq.Vector(*p) for p in pts], close=True)
    top = bottom.translate(cq.Vector(0.0, 0.0, depth))
    solid = (cq.Solid.extrudeLinear(cq.Face.makeFromWires(bottom), cq.Vector(0.0, 0.0, depth))
             if build_solid else None)
    return [bottom, top], solid


def export_radial_blade_step(blade_data, r1, r2, depth, face_path, solid_path=None,
                              source="stator", theta0=0.0, z_offset=0.0):
    """Export a radial-wrapped blade (see _build_radial_blade_solid) as
    STEP -- face_path gets the bottom (z = z_offset) section wire,
    solid_path gets the watertight extruded solid. Requires cadquery (not
    a core dependency -- import kept local)."""
    import cadquery as cq

    wires, solid = _build_radial_blade_solid(
        cq, blade_data, r1, r2, depth, source=source,
        theta0=theta0, z_offset=z_offset, build_solid=solid_path is not None,
    )
    cq.exporters.export(wires[0], str(face_path))
    if solid_path is not None:
        cq.exporters.export(solid, str(solid_path))


def export_radial_blade_stl(blade_data, r1, r2, depth, stl_path,
                             source="stator", theta0=0.0, z_offset=0.0):
    """Export the same watertight radial blade solid as
    export_radial_blade_step, but as an STL mesh for CAD-quality 3D preview
    via dash-vtk. Requires cadquery."""
    import cadquery as cq

    _, solid = _build_radial_blade_solid(cq, blade_data, r1, r2, depth, source=source,
                                          theta0=theta0, z_offset=z_offset)
    cq.exporters.export(solid, str(stl_path))
