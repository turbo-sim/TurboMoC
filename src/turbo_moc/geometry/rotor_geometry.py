"""Adapt the unchanged vortex solver to the common conformal blade geometry.

The solver's x is downstream and y is tangential. The shared numerical chart
uses q = (y, -x), with clockwise tangential angle and downstream-negative q_y.
Only a similarity transform is applied to the blade. Annular pitch comes from
the blade count; a linear cascade preserves the solver's natural pitch/chord.
"""

import numpy as np
from scipy.interpolate import make_interp_spline

from . import geometry_2d as planar
from .geometry_3d import Design, require_containment
from .flow_interfaces import apply_interfaces
from .linear_cascade import LinearCascadeDefinition, LinearCascadeMapping
from .meridional_channel import MeridionalChannel, geometry_from_machine, validate_machine


def _spline(points, tangents=None):
    """Interpolate a solver boundary segment without smoothing away its shape."""
    points = np.asarray(points, dtype=float)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, distance > 1e-12]]
    if len(points) < 2:
        return None
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    degree = min(3, len(points) - 1)
    # Analytic flow directions give tangent joins at both edge fillets. Tiny
    # tangent discontinuities from unconstrained fitting can invert Gmsh's
    # first inflation cells even when the displayed polygon looks smooth.
    boundary = (None if degree != 3 or tangents is None else
                ([(1, arclength[-1] * tangents[0])], [(1, arclength[-1] * tangents[1])]))
    return make_interp_spline(arclength / arclength[-1], points, k=degree, bc_type=boundary)


def _boundary_splines(result, points):
    """Retain the pressure/TE/stub/suction/stub/LE joins from the solver.

    Fitting the entire closed contour as one spline can ring at the joins.
    Separate interpolants preserve the solver's explicitly constructed edges
    and allow Gmsh to share exactly the same endpoint at each join.
    """
    lower, upper = len(result["blade_lower"]["x"]), len(result["blade_upper"]["x"])
    arc = int(result["num_points"])
    lengths = (lower, arc, 2, upper, 2, arc)
    if sum(lengths) - 5 != len(points):
        raise ValueError("Unexpected vortex blade boundary layout; cannot transfer its CAD safely.")
    incoming = planar.metal_direction(result["beta_inlet"])
    outgoing = planar.metal_direction(result["beta_outlet"])
    directions = ((incoming, outgoing), (outgoing, -outgoing),
                  (-outgoing, -outgoing), (-outgoing, -incoming),
                  (-incoming, -incoming), (-incoming, incoming))
    splines, start = [], 0
    for count, tangents in zip(lengths, directions):
        spline = _spline(points[start:start + count], tangents)
        if spline is not None:
            splines.append(spline)
        start += count - 1
    return splines


def _blade_reference(contour, count):
    """Midpoint of the solid profile's pitchwise extent at each chord station.

    Intersecting the complete polygon handles rounded edges and avoids treating
    the solver's open passage walls as the two sides of one solid blade.
    """
    top, bottom = contour[:, 1].max(), contour[:, 1].min()
    y = np.linspace(top, bottom, count)
    a, b = contour[:-1], contour[1:]
    dy = b[:, 1] - a[:, 1]
    transverse = np.abs(dy) > 1e-12
    a, b, dy = a[transverse], b[transverse], dy[transverse]
    fraction = (y[:, None] - a[None, :, 1]) / dy[None, :]
    valid = (fraction >= -1e-10) & (fraction <= 1 + 1e-10)
    crossings = a[None, :, 0] + fraction * (b - a)[None, :, 0]
    left = np.min(np.where(valid, crossings, np.inf), axis=1)
    right = np.max(np.where(valid, crossings, -np.inf), axis=1)
    if not np.isfinite([left, right]).all():
        raise ValueError("Could not construct a continuous vortex blade passage reference.")
    return np.column_stack(((left + right) / 2, y))


def _flow_domain(blade, settings):
    """Build smooth periodics around the rotor, with the shared extension controls."""
    options = planar.validate_flow_domain_settings({
        key: settings[key] for key in planar.DEFAULTS if key in settings
    })
    count = options["num_points"]
    contour = planar.blade_contour(blade)
    middle = _blade_reference(contour, count)
    chord = np.ptp(contour[:, 1])
    beta_in, beta_out = blade["metal_angle_in"], blade["metal_angle_out"]
    middle_spline = make_interp_spline(
        np.linspace(0, 1, count), middle, k=3,
        bc_type=([(1, [chord * np.tan(np.deg2rad(beta_in)), -chord])],
                 [(1, [chord * np.tan(np.deg2rad(beta_out)), -chord])]),
    )
    upstream = settings["inlet_chords"] * chord
    downstream = settings["outlet_chords"] * chord
    if not np.isfinite([upstream, downstream]).all() or min(upstream, downstream) <= 0:
        raise ValueError("Flow-domain inlet_chords and outlet_chords must be positive.")
    inlet_point = middle[0] + [-.5 * upstream * np.tan(np.deg2rad(beta_in)), upstream]
    outlet_point = middle[-1] + [.5 * downstream * np.tan(np.deg2rad(beta_out)), -downstream]
    inlet, inlet_cp, inlet_spline = planar.cubic_blend(
        inlet_point, middle[0], [0., -1.], planar.metal_direction(beta_in),
        upstream * options["inlet_axial_tangent"], upstream * options["inlet_blade_tangent"], count)
    outlet, outlet_cp, outlet_spline = planar.cubic_blend(
        middle[-1], outlet_point, planar.metal_direction(beta_out), [0., -1.],
        downstream * options["outlet_blade_tangent"], downstream * options["outlet_axial_tangent"], count)
    reference = np.vstack((inlet, middle[1:], outlet[1:]))
    domain = planar._domain(reference, blade["pitch"]) | {
        "blade_reference": middle, "blade_reference_spline": middle_spline,
        "inlet_extension": inlet, "outlet_extension": outlet,
        "inlet_control_points": inlet_cp, "outlet_control_points": outlet_cp,
        "inlet_spline": inlet_spline, "outlet_spline": outlet_spline,
        "reference_spline_names": ["inlet_spline", "blade_reference_spline", "outlet_spline"],
        "reference_sample_counts": [count, count, count],
    }
    mode, shift = settings.get("domain_shift_mode", "auto"), settings.get("domain_shift", 0.)
    planar.validate_domain_shift(mode, shift)
    return (planar.auto_center_flow_domain(blade, domain) if mode == "auto"
            else planar.apply_domain_shift(blade, domain, shift))


def build_rotor_design(rotor_result, config):
    """Return a canonical Design from one completed vortex-flow blade solution.

    Solver lengths may be metres or sonic-radius units: the common chord
    normalization removes that scale. No thermodynamic or vortex calculations
    are repeated during geometry, span mapping, or mesh generation.
    """
    dimensions = config["meridional_channel"]
    validate_machine(dimensions)
    native = planar.xy(rotor_result["blade"])
    if len(native) < 6 or not np.isfinite(native).all() or not np.allclose(native[0], native[-1]):
        raise ValueError("A finite, closed vortex rotor blade is required.")
    q = np.column_stack((native[:, 1], -native[:, 0]))
    chord = np.ptp(q[:, 1])
    if chord <= 0 or rotor_result["pitch"] <= 0:
        raise ValueError("The vortex rotor must have positive chord and pitch.")
    origin = q[np.argmax(q[:, 1])]
    if dimensions["type"] == "linear":
        scale = dimensions["meridional_chord"] / chord
        mapping = LinearCascadeMapping(LinearCascadeDefinition(
            dimensions["meridional_chord"], dimensions["blade_height"],
            float(rotor_result["pitch"] * scale)))
    else:
        mapping = MeridionalChannel(geometry_from_machine(dimensions))
        scale = mapping.reference_chord / chord
    q = (q - origin) * scale
    splines = _boundary_splines(rotor_result, q)
    # Display/clearance use the same spline geometry transferred to Gmsh.
    sampled = np.vstack([spline(np.linspace(0, 1, max(8, len(spline.c) * 3)))[:-1]
                         for spline in splines] + [splines[0](np.array([0.]))])
    blade = {
        "provider": "vortex_rotor", "pitch": mapping.pitch,
        "natural_pitch": float(rotor_result["pitch"] * scale),
        "boundary_splines": splines,
        "blade_curve": {"x": sampled[:, 0].tolist(), "y": sampled[:, 1].tolist()},
        "trailing_edge": {"x": [], "y": []},
        "metal_angle_in": float(rotor_result["beta_inlet"]),
        "metal_angle_out": float(rotor_result["beta_outlet"]),
    }
    curves = _flow_domain(blade, config["flow_domain"])
    curves.update(blade_curve=planar.xy(blade["blade_curve"]), trailing_edge=np.empty((0, 2)))
    assembly = config.get("assembly", {})
    curves = apply_interfaces(mapping, blade, curves, assembly, config["flow_domain"], .5)
    require_containment(blade, curves)
    design = Design(mapping, blade, curves, assembly, config["flow_domain"])
    design.section(0.)
    design.section(1.)
    return design
