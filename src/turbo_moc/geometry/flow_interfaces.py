"""Physical multirow interfaces expressed in each row's conformal chart.

The blade and meridional mapping are left unchanged. Only the inlet/outlet
extensions are rebuilt, so their swept caps coincide with a specified axial
plane or radial cylinder at every span station, including tapered blades.
"""

import numpy as np
from scipy.optimize import brentq

from turbo_moc.geometry import passage_2d as planar


def interface_prime(mapping, interface, placement, eta):
    """Convert a global interface coordinate to local dimensionless m'."""
    family = interface["geometry_type"]
    if mapping.is_linear or family not in ("axial", "radial"):
        raise ValueError("Shared interfaces require an axial or radial meridional channel.")
    axis = 1 if family == "axial" else 0  # meridional points contain (r, z)
    coordinate = float(interface["coordinate_mm"])
    if family == "axial":
        coordinate -= float(placement.get("streamwise_offset_mm", 0.))
    elif coordinate <= 0:
        raise ValueError("A radial interface must have a positive radius.")

    ends = mapping.meridional_points(np.array([0., 1.]), eta)[:, axis]
    if ends[1] == ends[0]:
        raise ValueError("The requested interface cannot intersect this meridional channel.")
    if mapping._straight:
        # This is the inverse of a straight meridional curve, independent of
        # whether that curve is axial, radial-inflow or radial-outflow.
        parameter = (coordinate - ends[0]) / (ends[1] - ends[0])
    else:
        def residual(u):
            return mapping.meridional_points(u, eta)[axis] - coordinate
        low, high = -1., 2.
        for _ in range(12):
            if residual(low) * residual(high) <= 0:
                break
            low, high = 2 * low, 2 * high
        else:
            raise ValueError("The physical interface does not intersect the meridional channel.")
        parameter = brentq(residual, low, high, xtol=1e-13)
    return float(mapping.prime_from_parameter(parameter, eta))


def apply_interfaces(mapping, blade, curves, assembly, settings, eta):
    """Reconstruct smooth extensions against the prescribed physical caps.

    The adjoining blade-reference endpoint and tangent are retained. Handle
    proportions follow the existing flow-domain settings; their dimensional
    lengths use the newly derived extension. Automatic pitchwise placement is
    deliberately not repeated here.
    """
    interfaces = assembly.get("interfaces", {})
    if not interfaces:
        return curves
    result = dict(curves)
    options = planar.DEFAULTS | {key: settings[key] for key in planar.DEFAULTS if key in settings}
    count = options["num_points"]
    middle = (curves["blade_reference"] if "blade_reference" in curves else
              np.vstack((curves["convergent"], curves["divergent"][1:])))
    contour = planar.blade_contour(blade)
    for side, interface in interfaces.items():
        if side not in ("inlet", "outlet"):
            raise ValueError(f"Unknown shared-interface boundary: {side}.")
        q_y = -mapping.reference_radius * interface_prime(
            mapping, interface, assembly.get("placement", {}), float(eta))
        inlet = side == "inlet"
        anchor = middle[0 if inlet else -1]
        extension = q_y - anchor[1] if inlet else anchor[1] - q_y
        blade_limit = contour[:, 1].max() if inlet else contour[:, 1].min()
        clearance = q_y - blade_limit if inlet else blade_limit - q_y
        if min(extension, clearance) <= 1e-10:
            raise ValueError(f"Shared {side} interface intersects the blade at span eta={eta:g}.")
        angle = blade["metal_angle_in" if inlet else "metal_angle_out"]
        direction = planar.metal_direction(angle)
        cap = anchor + [(-.5 if inlet else .5) * extension * np.tan(np.deg2rad(angle)),
                        extension if inlet else -extension]
        start, end = (cap, anchor) if inlet else (anchor, cap)
        tangents = ([0., -1.], direction) if inlet else (direction, [0., -1.])
        handles = (("inlet_axial_tangent", "inlet_blade_tangent") if inlet else
                   ("outlet_blade_tangent", "outlet_axial_tangent"))
        samples, controls, spline = planar.cubic_blend(
            start, end, *tangents, *(extension * options[key] for key in handles), count)
        result[f"{side}_extension"] = samples
        result[f"{side}_control_points"] = controls
        result[f"{side}_spline"] = spline
    reference = np.vstack((result["inlet_extension"], middle[1:], result["outlet_extension"][1:]))
    result.update(planar._domain(reference, blade["pitch"]))
    return result
