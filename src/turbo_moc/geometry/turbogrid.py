"""
turbo_moc.geometry.turbogrid -- export a wrapped annular blade (turbo_moc.
geometry.wrap_blade_annular) to ANSYS CFX-BladeGen's .inf + .curve file
triple, the geometry input TurboGrid consumes for turbomachinery blade
meshing.

Format and conventions matched against turbodash's own validated exporter
(turbodash/export_turbogrid.py on turbodash's origin/main, commit ad9fdce --
superseding the earlier rotor-only demos/cyclopentane_case/
export_rotor_turbogrid.py this module first matched at commit 81b3e33)
rather than reverse-engineered from a BladeGen-produced reference file,
since getting an interchange format wrong tends to fail silently (TurboGrid
imports *something*, just not the right shape). Details that are NOT
obvious from the file contents alone, confirmed from that implementation:

* profile.curve's three columns are true Cartesian (X, Y, Z) with the
  blade's own axial/chordwise coordinate as X and (Y, Z) = (r*cos(theta),
  r*sin(theta)) -- i.e. literally turbo_moc.geometry.annular's own (z, x, y)
  triple from wrap_blade_annular, just reordered.
* The header's angle label is "alpha_in"/"alpha_out" for a stator row
  (absolute flow angle convention) but "beta_in"/"beta_out" for a rotor row
  (relative/blade-angle convention) -- not the same label for both.
* Profile span stations are inset 1% of span from r_hub/r_shroud by
  default (hub_shroud_offset_fraction), matching turbodash's own
  convention. NOTE: this does NOT explain a real "Trim operation
  failed... Surfaces ... do not intersect" TurboGrid error hit on a
  single-row stator import -- setting the inset to 0 (spans exactly at
  r_hub/r_shroud) was tried and did NOT fix it, so the actual cause is
  still open; don't assume either setting here is the culprit.
* hub.curve/shroud.curve are just TWO points each (a straight line from
  x_min to x_max at the constant hub/shroud radius), not a dense point
  cloud -- consistent with this module's own "flare has the same axial
  stations, radius doesn't vary along one row's own chord" convention
  (turbo_moc.geometry.meridional's docstring).
* The axial extension of hub.curve/shroud.curve past the blade's own LE/TE
  is ASYMMETRIC by default -- 1x chord upstream, 2x chord downstream (more
  room for wake development on the outlet side), not a single symmetric
  margin.
"""

import shutil

import numpy as np

from .annular import wrap_blade_annular

__all__ = ["export_turbogrid_blade", "read_turbogrid_x_extent", "shift_turbogrid_files"]


# TurboGrid's own curve importer rejects a zero/near-zero-length segment
# outright ("Start and end parametric value are identical"), rather than
# silently tolerating it the way Workplane.spline() did for the STEP export
# -- real blade data hits this every time at the trailing-edge-arc/main-
# curve junction (te[-1] == curve[0] exactly, by construction; see
# turbo_moc.geometry.blade's _MIN_EDGE_LENGTH for the matching STEP-export
# fix). 1e-6 mm is the same floor used there.
_MIN_SEGMENT_LENGTH = 1e-6

# Confirmed against real TurboGrid: a hub/shroud curve extending EXACTLY to
# the blade's own LE/TE (extension fraction 0, e.g. for a "touching" row
# interface) fails with "Start and end parametric value are identical" --
# TurboGrid needs the blade projected strictly inside the curve's domain.
# 2 mm (the margin the user found empirically fixes it) is used as the floor.
_MIN_EXTENSION_MM = 2.0


def _dedupe_consecutive_xyz(x, y, z, tol):
    """Drop each point that coincides (within tol) with the PREVIOUS kept
    point -- catches duplicates anywhere in the sequence, not just a
    leading/trailing pair (unlike a simple first/last check)."""
    keep = [0]
    for i in range(1, len(x)):
        j = keep[-1]
        d = np.hypot(x[i] - x[j], np.hypot(y[i] - y[j], z[i] - z[j]))
        if d > tol:
            keep.append(i)
    keep = np.array(keep)
    return x[keep], y[keep], z[keep]


def _ordered_closed_profile(x, y, z, tol=_MIN_SEGMENT_LENGTH, te_x_hint=None):
    """Reorder a closed-loop (x, y, z) point set so index 0 is the leading
    edge, and the list is left OPEN (first point NOT repeated at the end
    -- TurboGrid's own curve importer apparently closes the loop itself,
    and an explicit repeat of point 0 produced a true zero-length final
    segment for IT to parametrize, not just for us -- "Start and end
    parametric value are identical" is the literal symptom of that double
    closure). Returns (x, y, z, le_index=0, te_index).

    te_x_hint : float or None
        The known trailing-edge region's own mean x (axial/chordwise)
        coordinate, when available (the stator case -- blade_data has a
        separate "trailing_edge" arc, see export_turbogrid_blade). The
        leading edge is then picked as whichever END of the array (its
        own x-min or x-max) is FARTHER from that hint, not just assumed
        to be the x-min end -- blade_curve's own x does not necessarily
        increase from LE to TE (confirmed on real data: the trailing edge
        sits at the LOW end of blade_curve's chordwise coordinate for at
        least some stator designs, the opposite of what a blind argmin(x)
        assumes). None (the rotor case, no separate TE region to check
        against) falls back to argmin(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    x, y, z = _dedupe_consecutive_xyz(x, y, z, tol)

    if np.hypot(x[0] - x[-1], np.hypot(y[0] - y[-1], z[0] - z[-1])) <= tol:
        x, y, z = x[:-1], y[:-1], z[:-1]

    if te_x_hint is None:
        le_index = int(np.argmin(x))
    else:
        le_index = int(np.argmin(x)) if abs(x.min() - te_x_hint) > abs(x.max() - te_x_hint) \
            else int(np.argmax(x))
    x = np.concatenate([x[le_index:], x[:le_index]])
    y = np.concatenate([y[le_index:], y[:le_index]])
    z = np.concatenate([z[le_index:], z[:le_index]])

    # TE is the point CLOSEST to the known trailing-edge region when we
    # have one; otherwise (rotor) fall back to the far end from LE.
    te_index = int(np.argmin(np.abs(x - te_x_hint))) if te_x_hint is not None else int(np.argmax(x))
    return x, y, z, 0, te_index


def _write_profile_curve(path, sections, source):
    # Stator rows report ABSOLUTE flow angles (alpha), rotor rows report
    # RELATIVE/blade angles (beta) -- turbodash's own convention.
    angle_label = "alpha" if source == "stator" else "beta"
    with open(path, "w") as f:
        for i, sec in enumerate(sections):
            f.write(
                f"# Span {i:03d} s={sec['span_fraction']:.8f} "
                f"{angle_label}_in={sec['angle_in_deg']:.6f} "
                f"{angle_label}_out={sec['angle_out_deg']:.6f}\n"
            )
            x, y, z, le_index, te_index = sec["x"], sec["y"], sec["z"], sec["le_index"], sec["te_index"]
            for j, (xi, yi, zi) in enumerate(zip(x, y, z)):
                marker = " le" if j == le_index else (" te" if j == te_index else "")
                f.write(f"{xi:.12e} {yi:.12e} {zi:.12e}{marker}\n")


def _write_endwall_curve(path, radius, x_min, x_max):
    np.savetxt(path, np.array([[x_min, radius, 0.0], [x_max, radius, 0.0]]), fmt="%.12e")


def _write_bladegen_inf(path, n_blades, units, axis, profile_file, hub_file, shroud_file,
                         te_type="CutOffEnd"):
    path.write_text("\n".join([
        "!======  CFX-BladeGen Export  ========",
        f"Axis of Rotation: {axis}",
        f"Number of Blade Sets: {n_blades}",
        "Number of Blades Per Set: 1",
        f"Geometry Units: {units}",
        "Blade 0 LE: EllipseEnd",
        f"Blade 0 TE: {te_type}",
        f"Hub Data File: {hub_file}",
        f"Shroud Data File: {shroud_file}",
        f"Profile Data File: {profile_file}",
        "",
    ]))


def export_turbogrid_blade(blade_data, r_hub, r_shroud, n_blades, output_dir,
                            span_sections=11, hub_shroud_offset_fraction=0.01,
                            inlet_extension_fraction=1.0, outlet_extension_fraction=2.0,
                            x_offset=0.0, flare="pitch_scale", theta0=0.0, source="stator",
                            axis="X", units="MM", te_type="CutOffEnd"):
    """
    Export BladeGen.inf + hub.curve + shroud.curve + profile.curve for
    TurboGrid meshing -- the wrapped-annulus counterpart to turbo_moc.
    geometry.export_annular_blade_step's STEP solid (same wrap math,
    turbo_moc.geometry.wrap_blade_annular, same r_hub/r_shroud/flare
    conventions), but as the plain-text profile-points format TurboGrid
    itself consumes, with no cadquery dependency.

    Parameters
    ----------
    blade_data : dict
        Stator (parametrize_stator_blade/_semi's output) or rotor
        (design_rotor_vortex_blade's output, already scaled to mm) --
        pick which via `source`.
    r_hub, r_shroud : float
        Hub/shroud radii (mm).
    n_blades : int
        Blade count -- written as "Number of Blade Sets" and used for the
        pitch-scale flare physics (see wrap_blade_annular).
    output_dir : Path
        Directory the four files are written into (created if missing).
    span_sections : int
        Number of profile.curve span stations (>= 2).
    hub_shroud_offset_fraction : float
        Inset the first/last profile station this fraction of span in
        from r_hub/r_shroud -- TurboGrid wants near-wall profiles, not
        sections sitting exactly on the hub/shroud wall.
    inlet_extension_fraction, outlet_extension_fraction : float
        How far hub.curve/shroud.curve extend upstream/downstream of the
        blade's own LE/TE, as a fraction of its axial chord (asymmetric by
        default -- more room downstream for wake development). Set the
        side facing an adjacent row to 0 -- extensions belong at the
        machine's true inlet/outlet, not padding an internal row interface
        (see shift_turbogrid_files's docstring).
    x_offset : float
        Added to every axial (X) coordinate after the fact -- positions
        this row in a shared global frame against an already-exported
        neighbor (e.g. a turbodash rotor), equivalent to exporting at 0
        then calling shift_turbogrid_files, just in one step.
    flare, theta0 : see wrap_blade_annular.
    source : "stator" or "rotor".
    axis : Axis of Rotation written to BladeGen.inf ("X", "Y", or "Z").
    units : Geometry Units written to BladeGen.inf.
    te_type : Blade 0 TE treatment written to BladeGen.inf.

    Returns
    -------
    dict
        Paths written: {"inf", "profile", "hub", "shroud"}.
    """
    if span_sections < 2:
        raise ValueError(f"span_sections must be >= 2, got {span_sections}")
    if not 0.0 <= hub_shroud_offset_fraction < 0.5:
        raise ValueError("hub_shroud_offset_fraction must be in [0, 0.5)")

    output_dir.mkdir(parents=True, exist_ok=True)

    r_lo = r_hub + hub_shroud_offset_fraction * (r_shroud - r_hub)
    r_hi = r_shroud - hub_shroud_offset_fraction * (r_shroud - r_hub)
    span_fractions = np.linspace(hub_shroud_offset_fraction, 1.0 - hub_shroud_offset_fraction,
                                  span_sections)

    angle_in = float(blade_data.get("metal_angle_in", float("nan")))
    angle_out = float(blade_data.get("metal_angle_out", float("nan")))

    wrapped = wrap_blade_annular(blade_data, r_lo, r_hi, n_blades=1, flare=flare,
                                  n_span=span_sections, theta0=theta0, source=source)
    curve = wrapped["curve"]
    te = wrapped["trailing_edge"]

    # The blade's own raw chordwise coordinate does NOT necessarily run LE
    # -> TE in the +axial direction -- confirmed on real data, it runs the
    # opposite way for at least some stator designs. Decided ONCE (span 0
    # is representative -- this blade has no spanwise twist) and applied
    # to every span, so LE always ends up on the low-x (upstream-facing)
    # side and TE on the high-x (downstream-facing) side, regardless of
    # which way the raw design happens to run. This is purely about which
    # END of the BLADE faces which way -- it does not change the overall
    # "+x is downstream" axis convention the adjoining row (e.g. a
    # turbodash rotor) also uses.
    axial_sign = 1.0
    if source == "stator":
        te_z0 = float(np.mean(te["z"][0, 0]))
        cz0 = curve["z"][0, 0]
        le_z0 = float(cz0[int(np.argmax(np.abs(cz0 - te_z0)))])
        if le_z0 > te_z0:
            axial_sign = -1.0

    sections = []
    for i, span_fraction in enumerate(span_fractions):
        te_x_hint = None
        if source == "stator":
            cz, cx, cy = curve["z"][0, i], curve["x"][0, i], curve["y"][0, i]
            tz, tx, ty = te["z"][0, i], te["x"][0, i], te["y"][0, i]
            z_full = axial_sign * np.concatenate([tz, cz])
            x_full = np.concatenate([tx, cx])
            y_full = np.concatenate([ty, cy])
            te_x_hint = axial_sign * float(np.mean(tz))
        else:
            z_full, x_full, y_full = curve["z"][0, i], curve["x"][0, i], curve["y"][0, i]
        x, y, z, le_index, te_index = _ordered_closed_profile(z_full, x_full, y_full, te_x_hint=te_x_hint)
        sections.append(dict(span_fraction=float(span_fraction), x=x, y=y, z=z,
                              le_index=le_index, te_index=te_index,
                              angle_in_deg=angle_in, angle_out_deg=angle_out))

    # sec["x"] is the axial/chordwise coordinate (post-reorder, still the
    # same raw values the wrap carries through unchanged from blade_data --
    # see wrap_blade_annular's module docstring) -- its own span gives the
    # blade's true axial chord directly, no need to re-derive it from the
    # unwrapped 2D data.
    z_min_all = min(float(np.min(sec["x"])) for sec in sections)
    z_max_all = max(float(np.max(sec["x"])) for sec in sections)
    axial_chord = z_max_all - z_min_all
    # Which END (x_min or x_max) is actually the LE side is NOT assumed --
    # blade_curve's own chordwise coordinate does not necessarily increase
    # from LE to TE (confirmed on real data: it runs the opposite way for
    # at least some stator designs). sections[0]'s own le_index=0 point vs
    # its te_index point settles it for real, every time.
    le_x = float(sections[0]["x"][0])
    te_x = float(sections[0]["x"][sections[0]["te_index"]])
    le_at_min = le_x <= te_x
    # TurboGrid needs the blade's own LE/TE to sit strictly INSIDE the
    # hub/shroud curve's domain, not exactly on its endpoint -- an
    # extension of exactly 0 (e.g. for a "touching" row interface) leaves
    # zero margin there, which TurboGrid can't build a valid parametrization
    # from ("Start and end parametric value are identical", confirmed
    # against real TurboGrid). MIN_EXTENSION_MM floors it to a small but
    # nonzero margin regardless of the requested fraction.
    min_ext, max_ext = (inlet_extension_fraction, outlet_extension_fraction) if le_at_min \
        else (outlet_extension_fraction, inlet_extension_fraction)
    x_min = z_min_all - max(min_ext * axial_chord, _MIN_EXTENSION_MM) + x_offset
    x_max = z_max_all + max(max_ext * axial_chord, _MIN_EXTENSION_MM) + x_offset
    if x_offset:
        for sec in sections:
            sec["x"] = sec["x"] + x_offset

    profile_path = output_dir / "profile.curve"
    hub_path = output_dir / "hub.curve"
    shroud_path = output_dir / "shroud.curve"
    inf_path = output_dir / "BladeGen.inf"

    _write_profile_curve(profile_path, sections, source)
    _write_endwall_curve(hub_path, r_hub, x_min, x_max)
    _write_endwall_curve(shroud_path, r_shroud, x_min, x_max)
    _write_bladegen_inf(inf_path, n_blades, units, axis, "profile.curve", "hub.curve",
                         "shroud.curve", te_type=te_type)

    return {"inf": inf_path, "profile": profile_path, "hub": hub_path, "shroud": shroud_path}


def read_turbogrid_x_extent(hub_curve_path):
    """(x_min, x_max) of a hub.curve/shroud.curve file -- format-agnostic
    (works on ANY BladeGen-format export, not just this module's own, since
    the X column convention is shared; see this module's docstring). Use
    this on a row's hub.curve where the facing side's extension fraction
    was exported as 0 to get that row's TRUE blade/domain edge there, for
    computing the axial offset to hand to shift_turbogrid_files."""
    x = np.loadtxt(hub_curve_path, usecols=0)
    return float(x.min()), float(x.max())


def shift_turbogrid_files(src_dir, dst_dir, x_offset):
    """Copy a BladeGen export (profile.curve + hub.curve + shroud.curve +
    BladeGen.inf -- from THIS module or any other tool writing the same
    format, e.g. turbodash's) into dst_dir with every X (axial) coordinate
    shifted by x_offset. BladeGen.inf has no coordinates, so it's just
    copied unchanged.

    This is the tool for lining up two independently-exported blade rows
    (each with its own local axial origin) so they share one global axial
    frame with no overlap and no gap at their interface -- same principle
    turbodash's own assembly exporter uses (one additive offset per row),
    just applied here as a standalone post-export step since the two rows
    come from different, unconnected design tools. Pick one row to anchor
    (x_offset=0, leave it as exported) and shift the OTHER by:

        x_offset = (anchor_x_max_at_interface + gap) - this_row_x_min_at_interface

    where both x_max/x_min are read (read_turbogrid_x_extent) from each
    row's hub.curve AFTER re-exporting with that row's facing-side
    extension fraction set to 0 -- otherwise you're offsetting by each
    row's own inlet/outlet padding instead of its true blade/domain edge,
    and the rows will overlap or leave an unwanted gap at the interface.

    Radius (hub/shroud) continuity at the interface is NOT handled here --
    shifting coordinates only repositions a row axially. If the two rows'
    hub/shroud radii differ at the interface, that's a design mismatch
    (fix it in whichever tool designed that row), not a file-shift fix.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hub.curve", "shroud.curve"):
        points = np.loadtxt(src_dir / name)
        points[:, 0] += x_offset
        np.savetxt(dst_dir / name, points, fmt="%.12e")

    with open(src_dir / "profile.curve") as f_in, open(dst_dir / "profile.curve", "w") as f_out:
        for line in f_in:
            if line.startswith("#") or not line.strip():
                f_out.write(line)
                continue
            parts = line.split()
            marker = f" {parts[3]}" if len(parts) > 3 else ""
            x, y, z = float(parts[0]) + x_offset, float(parts[1]), float(parts[2])
            f_out.write(f"{x:.12e} {y:.12e} {z:.12e}{marker}\n")

    shutil.copy(src_dir / "BladeGen.inf", dst_dir / "BladeGen.inf")
    return {"inf": dst_dir / "BladeGen.inf", "profile": dst_dir / "profile.curve",
            "hub": dst_dir / "hub.curve", "shroud": dst_dir / "shroud.curve"}
