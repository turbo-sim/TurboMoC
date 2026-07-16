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

__all__ = ["parametrize_stator_blade", "export_blade_step", "move_control_point_along_normal"]


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
    leading_edge_x=0.0,
    leading_edge_y=0.0,
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
    leading_edge_x, leading_edge_y : float
        Camberline integration start point [mm]. (leading_edge_y is
        unused by the ported algorithm, kept for interface symmetry with
        the original script's globals.)
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
        return metal_angle_in + (metal_angle_out - metal_angle_in) * (
            x - leading_edge_x
        ) / axial_chord_convergent

    def camberline_slope(x, y):
        return np.tan(np.deg2rad(beta(x)))

    def target_angle_event(x, y):
        return beta(x) - metal_angle_out

    target_angle_event.terminal = True
    target_angle_event.direction = 1

    sol = solve_ivp(
        camberline_slope,
        [leading_edge_x, leading_edge_x + 1000.0],
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
