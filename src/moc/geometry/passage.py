"""
moc.geometry.passage -- single flow-passage (one blade-to-blade channel)
extraction, for CFD meshing with periodic boundary conditions.

Domain picture (per direct user sketch, not the "two blade surfaces as
the domain's own boundary" idea tried earlier in this module's history):
a pitch-wide domain -- two periodic side walls, one pitch apart, each
running VERTICAL from the flat inlet cap down to just short of the
leading edge, then following the actual blade-to-blade centerline down
to just short of the trailing edge, then VERTICAL again down to the flat
outlet cap. Both walls are the exact same bent/curved polyline, one
shifted from the other by exactly one pitch (in x for a stator, in y for
a rotor), so the inlet and outlet cap lengths are BOTH exactly one pitch
regardless of the bends -- a pure pitchwise translation preserves
horizontal distance at every height, bent or curved path or not (and
this is why any future chamfer/fillet on the kinks MUST be applied
identically to both walls -- e.g. filleting the same vertex index on
both, same radius -- rather than picked independently, or the two walls
stop being exact pitch-translates of each other and periodicity breaks).

The middle (non-vertical) section differs by source:

- Stator: a mirrored-wall construction, so the middle section is genuinely
  two STRAIGHT segments -- a line through the leading edge at
  metal_angle_in, a line through the trailing edge at metal_angle_out,
  meeting at their own intersection. The wall passes exactly through the
  blade's own LE/TE corners, and the throat-height centerline runs
  exactly through the mid-pitch point at the trailing edge (the exit
  angle IS the flow direction there).
- Rotor: pressure(c) and suction(c)+pitch+translate are literally the
  two walls design_rotor_vortex_blade's own _close_blade_le_te closes
  into ONE blade's own solid (blade_lower/blade_upper -- "translate" is
  a correction beyond bare +pitch so the TE fillet arc's radius comes
  out right, already stored in blade_data["blade"]["translate"]). Their
  midpoint is therefore the blade's own mid-thickness line, NOT the
  passage. The wall is instead the CENTERLINE that pairs THIS blade's
  own outer wall (suction(c)+pitch+translate) with the NEXT blade's own
  wall (pressure(c)+pitch, not this blade's unshifted pressure) -- the
  blade-midline shifted by one FULL pitch, tracked at every chordwise
  station (not a fixed offset, which is only safe where the local
  blade thickness is less than the offset -- tried and reverted, see
  git history, it sliced through the thick LE bulb). Computed only
  where BOTH surfaces have real data (pressure/suction don't converge
  near the true LE/TE -- they asymptote to PARALLEL lines, pitch apart,
  never meeting -- see moc.rotor.vortex_blade's own module docstring),
  then extended as a STRAIGHT LINE continuing the CENTERLINE's own last
  real trend out to the blade's true LE/TE tip (an earlier version
  blended each surface toward the tip separately instead -- also tried
  and reverted, it bent the wrong way right at the LE since that
  assumed a convergence that doesn't happen). NOTE: the rotor's raw
  chordwise convention has max(c_vals) on the OUTLET side, not the
  inlet (opposite of the stator's) -- see _passage_geometry.

For the stator, both walls sit at +/- half a pitch from the LE/TE
reference line, so the blade's own position runs through the domain's
MIDDLE. For the rotor, wall_ref IS already the true passage centerline,
so one wall is that reference as-is and the other is it shifted by one
full pitch (0/+pitch, no extra centering needed). Either way, sized and
positioned so ONE COMPLETE blade sits fully inside the domain, never
clipped by its own edges, then that one blade is subtracted out as an
internal hole.

extract_radial_passage_2d conformally maps this SAME domain (via
moc.geometry.radial's log-spiral wrap, the same mapping already used to
wrap blades onto an annulus in Phase 5) onto a passage between two radii
-- the straight vertical/cap segments become log-spiral arcs under this
mapping, not straight lines, so they're densely sampled first
(_dense_boundary_points) rather than passed as bare 2-point segments the
way the axial-only wire (extract_axial_passage_2d) can get away with.

Requires cadquery (not a core moc dependency -- import kept local, same
convention as moc.geometry.blade's STEP export functions).
"""

__all__ = ["extract_axial_passage_2d", "extract_radial_passage_2d"]


def _sorted_xy(d):
    """(x, y) pairs from a {"x": [...], "y": [...]} dict, sorted by x --
    np.interp needs its xp argument ascending."""
    pts = sorted(zip(d["x"], d["y"]))
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _passage_geometry(blade_data, pitch, source, inlet_chords, outlet_chords,
                        bend_margin_inlet, bend_margin_outlet):
    """
    Shared core: works out the passage wall's own reference line/curve
    (wall_ref(c), in the blade's local pitchwise/chordwise frame), the
    +/-offset pair that turns it into two periodic walls, the inlet/
    outlet cap chordwise positions, and the blade's own contour (for the
    hole). See module docstring for the construction; this function just
    factors out what both extract_axial_passage_2d and
    extract_radial_passage_2d need identically.

    Returns a dict: corner(offset, c) -> (x, y) in blade_data's own
    units; mid_cs (the middle/curved section's own chordwise stations,
    walked inlet-to-outlet); c_hi/c_lo (cap positions); offset_left/
    offset_right; curve_xy/te_xy (blade's own solid contour, te_xy is
    None for the rotor).
    """
    import math

    import numpy as np

    if source == "stator":
        curve_xy = list(zip(blade_data["blade_curve"]["x"], blade_data["blade_curve"]["y"]))
        te_xy = list(zip(blade_data["trailing_edge"]["x"], blade_data["trailing_edge"]["y"]))
        contour = curve_xy + te_xy
        chord = blade_data["axial_chord_convergent"] + blade_data["axial_chord_divergent"]

        def pw(x, y):
            return x

        def ch(x, y):
            return y

        def to_xy(p, c):
            return (p, c)
    else:
        curve_xy = list(zip(blade_data["blade"]["x"], blade_data["blade"]["y"]))
        te_xy = None
        contour = curve_xy
        chord = blade_data["chord"]

        def pw(x, y):
            return y

        def ch(x, y):
            return x

        def to_xy(p, c):
            return (c, p)

    suction = blade_data["suction"]
    p0 = (suction["x"][0], suction["y"][0])   # leading-edge corner
    p1 = (suction["x"][-1], suction["y"][-1])  # trailing-edge corner

    c_vals = [ch(x, y) for x, y in contour]
    c_span = max(c_vals) - min(c_vals)
    if source == "stator":
        c_hi = max(c_vals) + inlet_chords * abs(chord)
        c_lo = min(c_vals) - outlet_chords * abs(chord)
    else:
        # Rotor's raw chordwise convention has max(c_vals) on the
        # OUTLET side, not the inlet (opposite of the stator's) -- see
        # moc.app's matching note on this swap.
        c_hi = max(c_vals) + outlet_chords * abs(chord)
        c_lo = min(c_vals) - inlet_chords * abs(chord)

    if source == "stator":
        # Line through the LE at metal_angle_in, line through the TE at
        # metal_angle_out -- each line's own `s` (p - slope*c, constant
        # along the line) is evaluated AT that corner, so it passes
        # through the corner exactly (s = p_corner - slope*c_corner =>
        # at c=c_corner, s+slope*c = p_corner). The two lines'
        # intersection is where the middle section switches from the
        # inlet angle to the outlet angle.
        end_slope = (pw(*p1) - pw(*p0)) / (ch(*p1) - ch(*p0))
        sign = 1.0 if end_slope >= 0 else -1.0
        slope_in = sign * abs(math.tan(math.radians(blade_data["metal_angle_in"])))
        slope_out = sign * abs(math.tan(math.radians(blade_data["metal_angle_out"])))
        s_in = pw(*p0) - slope_in * ch(*p0)
        s_out = pw(*p1) - slope_out * ch(*p1)
        c_kink = (s_out - s_in) / (slope_in - slope_out) if slope_in != slope_out else ch(*p0)

        c_bend_in = max(max(c_vals) - bend_margin_inlet * c_span, c_kink)
        c_bend_out = min(min(c_vals) + bend_margin_outlet * c_span, c_kink)

        def wall_ref(c):
            c_eff = min(max(c, c_bend_out), c_bend_in)
            if c_eff >= c_kink:
                return s_in + slope_in * c_eff
            return s_out + slope_out * c_eff

        mid_cs = [c_bend_in]
        if c_bend_in > c_kink > c_bend_out:
            mid_cs.append(c_kink)
        mid_cs.append(c_bend_out)

        offset_left = -0.5 * pitch
        offset_right = 0.5 * pitch
    else:
        # THE PASSAGE CENTERLINE: pairs THIS blade's own outer wall
        # (suction(c) + pitch + translate) with the NEXT blade's own
        # wall (pressure(c) + pitch, not this blade's own unshifted
        # pressure) -- see module docstring.
        pressure = blade_data["pressure"]
        px, py = _sorted_xy(pressure)
        sx, sy = _sorted_xy(suction)
        translate = blade_data["blade"]["translate"]
        suction_shift = pitch + translate

        blade_pts = curve_xy
        le_idx = max(range(len(blade_pts)), key=lambda i: ch(*blade_pts[i]))
        te_idx = min(range(len(blade_pts)), key=lambda i: ch(*blade_pts[i]))
        c_bend_in = ch(*blade_pts[le_idx])   # true leading-edge chordwise position
        c_bend_out = ch(*blade_pts[te_idx])  # true trailing-edge chordwise position

        c_overlap_lo = max(px[0], sx[0])
        c_overlap_hi = min(px[-1], sx[-1])

        cs_real = sorted(c for c in set(px + sx) if c_overlap_lo <= c <= c_overlap_hi)
        y_pressure_real = [float(np.interp(c, px, py)) + pitch for c in cs_real]
        y_suction_real = [float(np.interp(c, sx, sy)) + suction_shift for c in cs_real]
        centerline_real = [0.5 * (p + s) for p, s in zip(y_pressure_real, y_suction_real)]

        slope_hi = (centerline_real[-1] - centerline_real[-2]) / (cs_real[-1] - cs_real[-2])
        slope_lo = (centerline_real[1] - centerline_real[0]) / (cs_real[1] - cs_real[0])
        c_le_val = centerline_real[-1] + slope_hi * (c_bend_in - cs_real[-1])
        c_te_val = centerline_real[0] + slope_lo * (c_bend_out - cs_real[0])

        centerline_cs = [c_bend_out] + cs_real + [c_bend_in]
        centerline_vals = [c_te_val] + centerline_real + [c_le_val]

        def wall_ref(c):
            c_eff = min(max(c, c_bend_out), c_bend_in)
            return float(np.interp(c_eff, centerline_cs, centerline_vals))

        # Walked from the inlet side (c_bend_in) down to the outlet side
        # (c_bend_out) -- matches c_hi/c_lo's own max-is-inlet-for-
        # stator/outlet-for-rotor convention above.
        mid_cs = list(reversed(centerline_cs))

        offset_left = 0.0
        offset_right = pitch

    def corner(offset, c):
        return to_xy(offset + wall_ref(c), c)

    return dict(corner=corner, mid_cs=mid_cs, c_hi=c_hi, c_lo=c_lo,
                 offset_left=offset_left, offset_right=offset_right,
                 curve_xy=curve_xy, te_xy=te_xy)


def extract_axial_passage_2d(blade_data, pitch, face_path, solid_path=None,
                               extrude_length=1.0, source="stator",
                               inlet_chords=1.0, outlet_chords=6.0,
                               bend_margin_inlet=0.15, bend_margin_outlet=0.05):
    """
    One periodic passage domain (a pitch-wide domain, vertical near the
    inlet/outlet and following the blade-to-blade centerline in between,
    with the blade solid subtracted out as an internal hole) in the
    unrolled/axial cascade view, as a STEP face and, optionally, an
    extruded 3D solid. See module docstring for the domain's shape and
    why the stator and rotor get different middle-section constructions.

    Parameters
    ----------
    blade_data : dict
        Stator (parametrize_stator_blade/_semi output -- needs
        "blade_curve"/"trailing_edge"/"suction"/"metal_angle_in"/
        "metal_angle_out"/"axial_chord_convergent"/
        "axial_chord_divergent") or rotor
        (design_rotor_vortex_blade output -- needs
        "blade"/"suction"/"pressure"/"chord") blade dict. Pick which via
        `source`.
    pitch : float
        Blade-to-blade pitch (mm), i.e. the periodic translation. For a
        stator this IS blade_data["pitch"]; for a rotor that's been
        scaled from its own nondimensional r* (see moc.app's
        rotor_scale_mm), pass the SCALED pitch, and pre-scale
        blade_data's own length-valued arrays/fields by the same factor
        (angles need no scaling).
    face_path, solid_path : str
        Output STEP file paths. solid_path is optional -- omit to skip
        the 3D extrusion and only export the 2D face.
    extrude_length : float
        3D solid extrusion depth (mm), only used if solid_path is given.
    source : str
        "stator" or "rotor" -- which raw-surface axis convention to use:
        stator's pitch runs along x (pitchwise), rotor's along y (see
        moc/geometry/radial.py's module docstring on the two blades'
        opposite chordwise/pitchwise axis conventions).
    inlet_chords, outlet_chords : float
        How far the inlet/outlet caps sit upstream of the leading edge /
        downstream of the trailing edge, each as a multiple of the
        blade's own chord (stator: ``axial_chord_convergent +
        axial_chord_divergent``; rotor: "chord") -- e.g. the outlet
        default 6.0
        places the outlet six chords downstream, far enough that the
        outlet boundary condition doesn't interact with the blade's own
        near-field. Decoupled from the stator's own inlet_chords/
        outlet_chords -- pass whichever value applies to `source`. NOTE:
        for the rotor, max(c_vals) (blade's own raw chordwise convention)
        is the OUTLET side, not the inlet (opposite of the stator's), so
        internally outlet_chords is applied there and inlet_chords at
        min(c_vals) -- this function handles that swap; callers just
        pass the physically-correct inlet/outlet values for the source.
    bend_margin_inlet, bend_margin_outlet : float
        Stator only: where the side walls switch between vertical and
        staggered, each as a fraction of the blade's own chordwise
        extent measured inward from the nearer edge (leading edge for
        bend_margin_inlet, trailing edge for bend_margin_outlet).
        Clamped so the vertical sections never cross the LE/TE-angle-
        lines' own intersection point. The rotor's own transition points
        are the natural chordwise overlap of its own suction surface
        with the blade's own true LE/TE (see module docstring) -- these
        two params are unused for it.
    """
    import cadquery as cq

    from .blade import _closed_wire_2d

    g = _passage_geometry(blade_data, pitch, source, inlet_chords, outlet_chords,
                            bend_margin_inlet, bend_margin_outlet)
    corner, mid_cs = g["corner"], g["mid_cs"]
    c_hi, c_lo = g["c_hi"], g["c_lo"]
    offset_left, offset_right = g["offset_left"], g["offset_right"]

    right_path = [c_hi] + mid_cs + [c_lo]
    band_wp = cq.Workplane("XY").moveTo(*corner(offset_left, c_hi))
    for c in right_path:
        band_wp = band_wp.lineTo(*corner(offset_right, c))
    band_wp = band_wp.lineTo(*corner(offset_left, c_lo))
    for c in reversed(mid_cs):
        band_wp = band_wp.lineTo(*corner(offset_left, c))
    band_wp = band_wp.close()  # left wall's final vertical segment, back up to c_hi

    band_face = cq.Face.makeFromWires(band_wp.val())

    blade_wire = _closed_wire_2d(cq, g["curve_xy"], g["te_xy"])
    blade_face = cq.Face.makeFromWires(blade_wire)
    passage = band_face.cut(blade_face)

    cq.exporters.export(passage, str(face_path))

    if solid_path is not None:
        solid = cq.Workplane("XY").add(passage).extrude(extrude_length)
        cq.exporters.export(solid, str(solid_path))


def _dense_boundary_points(g, n_vertical=25, n_cap=15, n_mid=15):
    """The SAME closed boundary extract_axial_passage_2d builds (right
    wall up, inlet cap, ... ), but with EVERY straight segment densely
    resampled instead of just its 2 endpoints -- needed before a
    conformal map, since a straight line in the unrolled frame becomes a
    log-spiral ARC once mapped, not a straight line. This includes the
    inlet/outlet vertical walls and caps (n_vertical/n_cap points each),
    AND each leg of the middle section (mid_cs): for the rotor mid_cs is
    already densely sampled from the blade's own real surface data, but
    for the stator it is only the 2-3 KNOT points of a piecewise-linear
    wall (straight segments at metal_angle_in/metal_angle_out meeting at
    c_kink) -- resampling every mid_cs leg with n_mid points (not just
    mid_cs's own raw points) keeps those straight legs from collapsing
    to a chord once mapped, whichever source built them. Returns a plain
    ordered list of (x, y) points (blade_data's own units), NOT closed
    (the caller should close it).
    """
    corner = g["corner"]
    mid_cs = g["mid_cs"]
    c_hi, c_lo = g["c_hi"], g["c_lo"]
    offset_left, offset_right = g["offset_left"], g["offset_right"]

    def _lerp_c(c0, c1, n):
        return [c0 + (c1 - c0) * k / n for k in range(1, n + 1)]

    def _densify_mid(cs):
        out = []
        for c0, c1 in zip(cs[:-1], cs[1:]):
            out.extend(_lerp_c(c0, c1, n_mid))
        return out

    mid_dense = _densify_mid(mid_cs)

    pts = [corner(offset_left, c_hi)]
    # Inlet cap: offset_left -> offset_right, at c_hi.
    for k in range(1, n_cap + 1):
        off = offset_left + (offset_right - offset_left) * k / n_cap
        pts.append(corner(off, c_hi))
    # Right wall, vertical: c_hi -> mid_cs[0].
    for c in _lerp_c(c_hi, mid_cs[0], n_vertical):
        pts.append(corner(offset_right, c))
    # Right wall, curved/kinked middle section (densified).
    for c in mid_dense:
        pts.append(corner(offset_right, c))
    # Right wall, vertical: mid_cs[-1] -> c_lo.
    for c in _lerp_c(mid_cs[-1], c_lo, n_vertical):
        pts.append(corner(offset_right, c))
    # Outlet cap: offset_right -> offset_left, at c_lo.
    for k in range(1, n_cap + 1):
        off = offset_right + (offset_left - offset_right) * k / n_cap
        pts.append(corner(off, c_lo))
    # Left wall, vertical: c_lo -> mid_cs[-1].
    for c in _lerp_c(c_lo, mid_cs[-1], n_vertical):
        pts.append(corner(offset_left, c))
    # Left wall, curved/kinked middle section, reversed.
    for c in list(reversed(mid_dense)):
        pts.append(corner(offset_left, c))
    # Left wall, vertical: mid_cs[0] -> c_hi.
    for c in _lerp_c(mid_cs[0], c_hi, n_vertical):
        pts.append(corner(offset_left, c))
    return pts


def extract_radial_passage_2d(blade_data, pitch, r1, r2, face_path, solid_path=None,
                                extrude_length=1.0, source="stator",
                                inlet_chords=1.0, outlet_chords=6.0,
                                bend_margin_inlet=0.15, bend_margin_outlet=0.05,
                                n_vertical=25, n_cap=15):
    """
    Conformally maps the SAME periodic passage domain as
    extract_axial_passage_2d (see its docstring/module docstring for the
    construction) onto an annular passage between radii r1/r2, using
    moc.geometry.radial's log-spiral wrap -- the same mapping already
    used to wrap blades onto an annulus for Phase 5. One passage, one
    blade hole (matching n_blades=1 in wrap_curve_radial terms).

    IMPORTANT: r1/r2 land at the BLADE's own leading/trailing edge --
    same convention as wrap_blade_radial/wrap_rotor_blade_radial -- NOT
    at the passage domain's own (extended) inlet/outlet caps. The
    mapping's reference point/scale (see moc.geometry.radial's
    _reference_point_and_scale) is derived from the blade's own contour
    and then shared with the band's outer boundary, so the band's
    inlet/outlet caps (which sit beyond the blade's LE/TE by
    ``inlet_chords``/``outlet_chords``) extrapolate naturally to radii
    outside
    [r1, r2] via the same log-spiral formula, rather than r1/r2
    themselves being squeezed onto the (much wider) extended domain.

    Parameters
    ----------
    blade_data, pitch, source, inlet_chords, outlet_chords,
    bend_margin_inlet, bend_margin_outlet : see extract_axial_passage_2d.
    r1, r2 : float
        Inner/outer radii (same length units as blade_data), reached at
        the blade's own leading/trailing edge (not the extended domain
        caps -- see above).
    face_path, solid_path, extrude_length : see extract_axial_passage_2d.
    n_vertical, n_cap : int
        How densely to resample the domain's straight vertical/cap
        segments before mapping (see _dense_boundary_points) -- these
        become log-spiral arcs, not straight lines, once mapped.
    """
    import cadquery as cq

    from .blade import _closed_wire_2d
    from .radial import _reference_point_and_scale, _wrap_with_shared_reference

    g = _passage_geometry(blade_data, pitch, source, inlet_chords, outlet_chords,
                            bend_margin_inlet, bend_margin_outlet)
    chordwise_axis = "y" if source == "stator" else "x"

    blade_x = [p[0] for p in g["curve_xy"]]
    blade_y = [p[1] for p in g["curve_xy"]]
    if chordwise_axis == "y":
        chord_vals, pitch_vals = blade_y, blade_x
    else:
        chord_vals, pitch_vals = blade_x, blade_y
    x1, y1, c_axial = _reference_point_and_scale(chord_vals, pitch_vals)

    band_pts = _dense_boundary_points(g, n_vertical=n_vertical, n_cap=n_cap)
    band_x = [p[0] for p in band_pts]
    band_y = [p[1] for p in band_pts]
    band_copies = _wrap_with_shared_reference(
        band_x, band_y, x1, y1, c_axial, r1, r2, 1, 0.0, chordwise_axis
    )
    band_mapped = band_copies[0]

    blade_copies = _wrap_with_shared_reference(
        blade_x, blade_y, x1, y1, c_axial, r1, r2, 1, 0.0, chordwise_axis
    )
    blade_mapped = blade_copies[0]

    te_mapped = None
    if g["te_xy"] is not None:
        te_x = [p[0] for p in g["te_xy"]]
        te_y = [p[1] for p in g["te_xy"]]
        te_copies = _wrap_with_shared_reference(
            te_x, te_y, x1, y1, c_axial, r1, r2, 1, 0.0, chordwise_axis
        )
        te_mapped = te_copies[0]

    def _wire_from_points(xs, ys):
        wp = cq.Workplane("XY").moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            wp = wp.lineTo(x, y)
        return wp.close()

    band_wire = _wire_from_points(band_mapped["x"], band_mapped["y"])
    band_face = cq.Face.makeFromWires(band_wire.val())

    blade_curve_xy = list(zip(blade_mapped["x"], blade_mapped["y"]))
    te_xy_mapped = list(zip(te_mapped["x"], te_mapped["y"])) if te_mapped is not None else None
    blade_wire = _closed_wire_2d(cq, blade_curve_xy, te_xy_mapped)
    blade_face = cq.Face.makeFromWires(blade_wire)

    passage = band_face.cut(blade_face)

    cq.exporters.export(passage, str(face_path))

    if solid_path is not None:
        solid = cq.Workplane("XY").add(passage).extrude(extrude_length)
        cq.exporters.export(solid, str(solid_path))
