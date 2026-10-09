"""Conformal mapping from an unrolled blade to an annular passage.

The log-spiral mapping wraps either a stator blade produced by
``parametrize_stator_blade`` or a rotor blade produced by
``turbo_moc.rotor.design_rotor_vortex_blade`` between two radii. Its reference
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
    "conformal_length_scale", "mapped_throat_width",
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


def conformal_length_scale(chord, x1, r1, r2, c_axial):
    """Exact local length-magnification factor of the log-spiral conformal
    map at a given chordwise coordinate.

    w = X + iY, as a function of u = chordwise + i*pitchwise, is
    w = C * exp(a*u) with a = log(r2/r1)/c_axial and C a constant complex
    number -- a holomorphic (complex-differentiable) function of u. For any
    holomorphic map, the local length-magnification factor is exactly
    |dw/du| = |a| * |w| = |a| * r (the map's own r-formula, at this
    chordwise coordinate; the absolute value matters here -- this module's
    own _reference_point_and_scale anchors x1 at the chord MAXIMUM, which
    makes c_axial = min(chord)-max(chord), hence a, negative by
    convention, even though the magnification itself is always positive).
    A length measured in the 2D cascade plane at this chordwise location
    is scaled by exactly this factor once mapped onto the annulus -- not
    an approximation, and no numerical search needed.

    The conformal map preserves ANGLES (that's what "conformal" means),
    not lengths -- e.g. the 2D-plane throat_opening is not the actual
    radial-passage throat width; see mapped_throat_width."""
    a = np.log(r2 / r1) / c_axial
    r = r1 * np.exp(a * (chord - x1))
    return float(abs(a) * r)


def _eval_reference(blade_data, r1, r2, source):
    """Shared by mapped_throat_width/mapped_pitch_width/
    implied_n_blades_radial: the (x1, c_axial) the actual wrap derives
    from the main contour (so results stay consistent with the rendered/
    exported geometry), the EVALUATION point's own chordwise coordinate,
    and the radius the map puts it at.

    The evaluation point is the natural "critical section" to check pitch/
    throat consistency at: for a stator, its own throat_point (the
    narrowest passage section). The rotor method has no equivalent
    critical-section concept, so x1 itself (the hub reference point,
    r=r1 by construction -- see _reference_point_and_scale) is used
    instead, a reasonable single representative location in the absence
    of a better one."""
    if source == "stator":
        curve = blade_data["blade_curve"]
        chord_vals, pitch_vals = np.asarray(curve["y"], dtype=float), np.asarray(curve["x"], dtype=float)
        x1, y1, c_axial = _reference_point_and_scale(chord_vals, pitch_vals)
        eval_chord = blade_data["throat_point"]["y"]
    elif source == "rotor":
        blade = blade_data["blade"]
        chord_vals, pitch_vals = np.asarray(blade["x"], dtype=float), np.asarray(blade["y"], dtype=float)
        x1, y1, c_axial = _reference_point_and_scale(chord_vals, pitch_vals)
        eval_chord = x1  # hub reference point (r=r1) -- no throat concept for the rotor method
    else:
        raise ValueError(f"source must be 'stator' or 'rotor', got {source!r}")
    a = np.log(r2 / r1) / c_axial
    r_eval = r1 * np.exp(a * (eval_chord - x1))
    return x1, c_axial, eval_chord, float(r_eval)


def mapped_throat_width(blade_data, r1, r2, source="stator"):
    """The 2D-plane throat_opening, rescaled by the EXACT local conformal
    length factor at the throat's own (chordwise, pitchwise) location --
    see conformal_length_scale. Uses the SAME reference (x1, c_axial) the
    actual wrap (wrap_blade_radial/wrap_rotor_blade_radial) derives from
    the main blade contour, so the result is consistent with the geometry
    that actually gets rendered/exported.

    blade_data : dict
        Stator: needs "blade_curve" ({"x","y"}) and "throat_point"
        ({"x","y"}, see parametrize_stator_blade/_semi) and
        "throat_opening". Rotor: this tool's rotor method has no
        throat_opening/throat_point -- not supported here (its sizing is
        pitch-consistency only to begin with, see
        turbo_moc.geometry.sizing.size_rotor_from_pitch).
    """
    if source != "stator":
        raise ValueError(f"mapped_throat_width only supports source='stator', got {source!r}")
    x1, c_axial, eval_chord, _ = _eval_reference(blade_data, r1, r2, source)
    scale = conformal_length_scale(eval_chord, x1, r1, r2, c_axial)
    return float(blade_data["throat_opening"] * scale)


def mapped_pitch_width(blade_data, r1, r2, source="stator"):
    """The 2D-plane pitch, rescaled by the exact local conformal length
    factor at the evaluation point's chordwise location (the stator's own
    throat, or the rotor's hub reference point -- see _eval_reference) --
    i.e. the actual circumferential footprint ONE mapped blade copy
    occupies there. Unlike the axial wrap, nothing here guarantees this
    matches 2*pi*r/n_blades for whatever n_blades you pick (see this
    module's docstring and implied_n_blades_radial) -- adjacent copies
    will overlap if n_blades is too large for this footprint."""
    x1, c_axial, eval_chord, _ = _eval_reference(blade_data, r1, r2, source)
    scale = conformal_length_scale(eval_chord, x1, r1, r2, c_axial)
    return float(blade_data["pitch"] * scale)


def implied_n_blades_radial(blade_data, r1, r2, source="stator"):
    """The blade count that makes the copy-to-copy spacing
    (2*pi*r_eval/n_blades) match this blade's own mapped pitch at the
    evaluation point (stator: its throat; rotor: its hub reference point,
    see _eval_reference) -- i.e. the n_blades that actually tiles the
    annulus without overlap or gaps, AT THAT RADIUS (pitch still varies
    with span under this map, so this is a single representative value,
    same role as the axial wrap's own pitch-consistency check, not an
    exact guarantee everywhere along the blade). A default N_blades
    chosen far from this will visibly overlap (too large) or leave gaps
    (too small) in the 3D preview."""
    x1, c_axial, eval_chord, r_eval = _eval_reference(blade_data, r1, r2, source)
    mapped_pitch = mapped_pitch_width(blade_data, r1, r2, source=source)
    return float(2.0 * np.pi * r_eval / mapped_pitch)


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
    Wrap a Phase-3-style rotor blade (turbo_moc.rotor.design_rotor_vortex_blade's
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
        turbo_moc.app's Phase 4 "Rotor scale" convention) converts it first.
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
