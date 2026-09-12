"""Conformal mapping from an unrolled blade to an annular passage.

The log-spiral mapping wraps either a stator blade produced by
``parametrize_stator_blade`` or a rotor blade produced by
``moc.rotor.design_rotor_vortex_blade`` between two radii. Its reference
point and axial-chord scale are derived from the blade's own point cloud, so
the mapping does not require geometry-specific constants.

The mapping is::

    r = r1 * exp(log(r2/r1) * (chordwise - x1) / c_axial)
    theta = theta0 + log(r2/r1) / c_axial * (pitchwise - y1)

The source axes used for the chordwise and pitchwise coordinates differ:

* A stator's chordwise extent runs mainly along its local y axis and its
  pitchwise extent along x after the final profile rotation.
* A rotor's chordwise direction runs along local x and its pitch runs along y.

Moving along the chord sweeps the radius from ``r1`` to ``r2``; a pitch
offset between adjacent copies becomes the annular blade spacing. The mapping
is angle-preserving by construction.
"""

import numpy as np

__all__ = [
    "apply_conformal_mapping", "rotate_radial_points",
    "wrap_curve_radial", "wrap_blade_radial", "wrap_rotor_blade_radial",
]


def apply_conformal_mapping(chordwise, pitchwise, x1, y1, r1, r2, c_axial, theta0=0.0):
    """Log-spiral conformal map from the unrolled (chordwise, pitchwise)
    frame to Cartesian (X, Y) on an annulus: r=r1 at chordwise=x1, r=r2 at
    chordwise=x1+c_axial, theta=theta0 at pitchwise=y1."""
    chordwise = np.asarray(chordwise, dtype=float)
    pitchwise = np.asarray(pitchwise, dtype=float)
    r = r1 * np.exp(np.log(r2 / r1) * (chordwise - x1) / c_axial)
    theta = theta0 + np.log(r2 / r1) / c_axial * (pitchwise - y1)
    return r * np.cos(theta), r * np.sin(theta)


def rotate_radial_points(x, y, angle_rad):
    """Rigid rotation by angle_rad about the origin -- lays out the
    n_blades copies around the annulus after mapping."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _reference_point_and_scale(chord_vals, pitch_vals):
    """(x1, y1, c_axial) derived from the curve's own point cloud -- the
    point with the largest chordwise coordinate anchors r=r1/theta=theta0,
    and the chordwise span sets c_axial. Needs no per-case tuning, unlike
    the original script's hardcoded version."""
    chord_vals = np.asarray(chord_vals, dtype=float)
    pitch_vals = np.asarray(pitch_vals, dtype=float)
    c_axial = float(np.min(chord_vals) - np.max(chord_vals))
    idx = int(np.argmax(chord_vals))
    y1 = float(pitch_vals[idx])
    x1 = float(chord_vals[idx])
    return x1, y1, c_axial


def wrap_curve_radial(x, y, r1, r2, n_blades, theta0=0.0, chordwise_axis="y"):
    """
    Wrap ONE closed (or open) curve (x, y) onto an annular passage between
    radii r1/r2, laid out as n_blades copies spaced 2*pi/n_blades apart.
    `chordwise_axis` picks which of the curve's own axes runs chordwise
    (the one that maps to radius) -- "y" for a stator-convention curve,
    "x" for a rotor-convention one; see module docstring.

    Returns a list of n_blades {"x", "y"} dicts (the mapped curve, one per
    copy) plus the reference (x1, y1, c_axial, d_theta) used, for reuse
    when wrapping a second curve (e.g. a trailing edge) with the SAME
    reference frame as this one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if chordwise_axis == "y":
        chord_vals, pitch_vals = y, x
    elif chordwise_axis == "x":
        chord_vals, pitch_vals = x, y
    else:
        raise ValueError(f"chordwise_axis must be 'x' or 'y', got {chordwise_axis!r}")

    x1, y1, c_axial = _reference_point_and_scale(chord_vals, pitch_vals)
    x_mapped, y_mapped = apply_conformal_mapping(chord_vals, pitch_vals, x1, y1, r1, r2, c_axial, theta0)

    d_theta = 2.0 * np.pi / n_blades
    copies = []
    for i in range(n_blades):
        xi, yi = rotate_radial_points(x_mapped, y_mapped, i * d_theta)
        copies.append({"x": xi.tolist(), "y": yi.tolist()})
    return copies, dict(x1=x1, y1=y1, c_axial=c_axial, d_theta=d_theta)


def _wrap_with_shared_reference(x, y, x1, y1, c_axial, r1, r2, n_blades, theta0, chordwise_axis):
    """Like wrap_curve_radial, but reusing a reference (x1, y1, c_axial)
    computed from a DIFFERENT curve (e.g. the main blade contour) -- used
    to map a trailing-edge arc consistently with its own blade's wrap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    chord_vals, pitch_vals = (y, x) if chordwise_axis == "y" else (x, y)
    x_mapped, y_mapped = apply_conformal_mapping(chord_vals, pitch_vals, x1, y1, r1, r2, c_axial, theta0)
    d_theta = 2.0 * np.pi / n_blades
    copies = []
    for i in range(n_blades):
        xi, yi = rotate_radial_points(x_mapped, y_mapped, i * d_theta)
        copies.append({"x": xi.tolist(), "y": yi.tolist()})
    return copies


def wrap_blade_radial(blade_data, r1, r2, n_blades, theta0=0.0):
    """
    Wrap a Phase-2-style stator blade (parametrize_stator_blade/_semi's
    output dict) onto an annular passage between radii r1 and r2.

    Parameters
    ----------
    blade_data : dict
        Needs "blade_curve" and "trailing_edge" ({"x", "y"} each, mm).
    r1, r2 : float
        Inner/outer radii of the annular passage (same length units as
        blade_data, mm), reached at the blade's own chordwise extremes.
    n_blades : int
        Number of blade copies laid out around the full annulus.
    theta0 : float
        Angular offset [rad] of the first (index-0) blade copy.

    Returns
    -------
    dict
        "blades": list of n_blades {"x", "y"} closed-contour dicts;
        "trailing_edges": matching list for the trailing-edge arc;
        "r1"/"r2"/"n_blades"/"d_theta" echoed back; "units".
    """
    curve = blade_data["blade_curve"]
    te = blade_data["trailing_edge"]
    blades, ref = wrap_curve_radial(curve["x"], curve["y"], r1, r2, n_blades, theta0, chordwise_axis="y")
    trailing_edges = _wrap_with_shared_reference(
        te["x"], te["y"], ref["x1"], ref["y1"], ref["c_axial"], r1, r2, n_blades, theta0, "y"
    )
    return {
        "blades": blades,
        "trailing_edges": trailing_edges,
        "r1": float(r1), "r2": float(r2),
        "n_blades": int(n_blades), "d_theta": ref["d_theta"],
        "units": blade_data.get("units", "mm"),
    }


def wrap_rotor_blade_radial(rotor_data, r1, r2, n_blades, theta0=0.0, scale=1.0):
    """
    Wrap a Phase-3-style rotor blade (moc.rotor.design_rotor_vortex_blade's
    output dict) onto an annular passage between radii r1 and r2.

    Parameters
    ----------
    rotor_data : dict
        Needs "blade" ({"x", "y"}) -- the closed LE/TE-rounded blade
        contour. Its trailing edge is already part of this closed curve
        (unlike the stator's, which keeps a separate arc), so there is no
        separate trailing-edge overlay here.
    r1, r2 : float
        Inner/outer radii of the annular passage. MUST be in the same
        units as `scale * rotor_data["blade"]` -- rotor_data is often
        nondimensional (r*) unless design_rotor_vortex_blade was called
        with an explicit r_star, so `scale` (e.g. mm per r*, matching
        moc.app's Phase 4 "Rotor scale" convention) converts it first.
    n_blades : int
        Number of blade copies laid out around the full annulus.
    theta0 : float
        Angular offset [rad] of the first (index-0) blade copy.
    scale : float
        Multiplier applied to the raw blade (x, y) before mapping.

    Returns
    -------
    dict
        Same shape as wrap_blade_radial's return, with "trailing_edges"
        as a list of empty {"x": [], "y": []} placeholders (nothing extra
        to draw -- kept for a uniform shape with the stator's result, so
        plot_radial_cascade works unchanged for either source).
    """
    blade = rotor_data["blade"]
    x = [v * scale for v in blade["x"]]
    y = [v * scale for v in blade["y"]]
    blades, ref = wrap_curve_radial(x, y, r1, r2, n_blades, theta0, chordwise_axis="x")
    return {
        "blades": blades,
        "trailing_edges": [{"x": [], "y": []} for _ in range(n_blades)],
        "r1": float(r1), "r2": float(r2),
        "n_blades": int(n_blades), "d_theta": ref["d_theta"],
        "units": "mm" if scale != 1.0 else rotor_data.get("units", "r*"),
    }
