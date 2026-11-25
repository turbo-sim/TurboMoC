"""Geometry-only rotor sections: no vortex solve, no supersonic inlet required.

Two blade types are built from geometric inputs alone and returned in the rotor-result layout used by
``geometry.rotor_blade.build_rotor_design`` (solver frame: x downstream, y tangential, pressure side below):

``reaction``
    Subsonic reaction section from TurboDash's blade parametrization. The camberline angle is TurboDash's
    ``turning_angle`` (half-turning location, turning concentration, ``interpolation="angle"``); the camber is the
    integral of its slope. Each side is offset by half the full thickness of TurboDash's modified NACA law
    (``thickness_models.full_thickness`` with ``family="naca"``) along the camberline normal. The endpoint camber
    metrics ds/dxi = c/cos(beta) are passed to the law so that the leading-edge radius and trailing-edge wedge are
    physical. The trailing edge is a finite cut-off normal to the camberline.

``constant_passage``
    Baines constant-passage impulse blade (examples/moc_design/example_baines_blade.py) from the blade angle and
    the pitch-to-chord ratio alone, closed with the vortex blade's own edge routine.

Conventions, applied once here: TurboDash angles (inlet/outlet metal angle, counterclockwise chart azimuth) map to
the TurboMoC solver frame with y = -theta, so beta_solver = -beta_TurboDash; all thickness inputs are fractions of
the meridional (axial) chord, which is one in the returned section. TurboDash is an optional dependency: its two
leaf modules are imported lazily, only when a reaction blade is built.
"""
from math import pi

import numpy as np

REACTION_DEFAULTS = {
    "inlet_metal_angle_deg": 0.0,       # TurboDash convention
    "outlet_metal_angle_deg": 60.0,     # TurboDash convention
    "turning_location": 0.52,           # Half-turning location, fraction of chord
    "turning_concentration": 1.2,
    "maximum_thickness": 0.12,          # Full thickness / meridional chord
    "maximum_thickness_location": 0.30,
    "leading_edge_radius": 0.01,        # / meridional chord
    "trailing_edge_thickness": 0.015,   # / meridional chord
    "trailing_edge_wedge_angle_deg": 8.0,
    "pitch_to_chord": 0.8727,           # Used by linear cascades; annular rows take the pitch from the blade count
    "num_points": 241,                  # Camberline stations per side
}
CONSTANT_PASSAGE_DEFAULTS = {
    "blade_angle_deg": 60.0,            # Inlet angle from axial; outlet -blade_angle (symmetric impulse)
    "pitch_to_chord": 0.6,              # Final spacing / axial extent of the closed blade (linear cascades)
}
GEOMETRIC_BLADE_TYPES = {"reaction": REACTION_DEFAULTS, "constant_passage": CONSTANT_PASSAGE_DEFAULTS}


def _turbodash():
    try:
        from turbodash.geometry.blade_parametrization.geometry import turning_angle
        from turbodash.geometry.blade_parametrization.thickness_models import full_thickness
    except ImportError as exc:  # Optional dependency
        raise ImportError("The reaction blade uses TurboDash's blade parametrization "
                          "(turbodash.geometry.blade_parametrization.thickness_models, family 'naca'). "
                          "Install turbodash-core >= 0.9 or put its checkout on the path.") from exc
    return turning_angle, full_thickness


def validate_reaction(p):
    for key in REACTION_DEFAULTS:
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"rotor_design.reaction.{key} must be a finite number.")
    if not 0 < p["turning_location"] < 1:
        raise ValueError("rotor_design.reaction.turning_location must lie strictly between 0 and 1.")
    if p["turning_concentration"] <= 0:
        raise ValueError("rotor_design.reaction.turning_concentration must be positive.")
    if not 0 < p["maximum_thickness_location"] < 1:
        raise ValueError("rotor_design.reaction.maximum_thickness_location must lie strictly between 0 and 1.")
    for key in ("maximum_thickness", "leading_edge_radius", "pitch_to_chord"):
        if p[key] <= 0:
            raise ValueError(f"rotor_design.reaction.{key} must be positive.")
    if p["trailing_edge_thickness"] < 0 or p["trailing_edge_thickness"] >= p["maximum_thickness"]:
        raise ValueError("rotor_design.reaction.trailing_edge_thickness must be in [0, maximum_thickness).")
    if not 0 <= p["trailing_edge_wedge_angle_deg"] < 90:
        raise ValueError("rotor_design.reaction.trailing_edge_wedge_angle_deg must be in [0, 90).")
    if max(abs(p["inlet_metal_angle_deg"]), abs(p["outlet_metal_angle_deg"])) >= 85:
        raise ValueError("rotor_design.reaction metal angles must be within +-85 degrees.")
    if type(p["num_points"]) is not int or p["num_points"] < 41:
        raise ValueError("rotor_design.reaction.num_points must be an integer >= 41.")


def validate_constant_passage(p):
    for key in CONSTANT_PASSAGE_DEFAULTS:
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"rotor_design.constant_passage.{key} must be a finite number.")
    if not 5 <= p["blade_angle_deg"] < 85:
        raise ValueError("rotor_design.constant_passage.blade_angle_deg must be in [5, 85).")
    if p["pitch_to_chord"] <= 0:
        raise ValueError("rotor_design.constant_passage.pitch_to_chord must be positive.")


def reaction_section(p):
    """Camberline, sides and full thickness of one reaction section in TurboDash's chart (chord 1).

    Returns a dict with xi, beta [rad], camber (n, 2), plus/minus sides (n, 2) and full thickness t(xi).
    """
    turning_angle, full_thickness = _turbodash()
    camber_parameters = {"turning_location": p["turning_location"],
                         "turning_concentration": p["turning_concentration"],
                         "inlet_metal_angle_deg": p["inlet_metal_angle_deg"],
                         "outlet_metal_angle_deg": p["outlet_metal_angle_deg"],
                         "interpolation": "angle"}
    count = int(p["num_points"])
    xi = .5 * (1 - np.cos(np.linspace(0., pi, count)))  # Clustered at both edges (square-root nose)
    xi = np.unique(np.r_[xi, p["turning_location"], p["maximum_thickness_location"]])
    # Integrate the camber slope on a dense grid, as TurboDash's build_section does.
    dense = np.unique(np.r_[np.linspace(0., 1., max(4001, 8 * count + 1)), xi])
    slope = np.tan(turning_angle(dense, camber_parameters))
    integral = np.r_[0., np.cumsum(.5 * (slope[1:] + slope[:-1]) * np.diff(dense))]
    beta = turning_angle(xi, camber_parameters)
    camber = np.column_stack((xi, np.interp(xi, dense, integral)))
    thickness = {"family": "naca", "leading_edge_radius_m": float(p["leading_edge_radius"]),
                 "maximum_thickness_m": float(p["maximum_thickness"]),
                 "maximum_thickness_location": float(p["maximum_thickness_location"]),
                 "trailing_edge_thickness_m": float(p["trailing_edge_thickness"]),
                 "trailing_edge_wedge_angle_deg": float(p["trailing_edge_wedge_angle_deg"]),
                 "trailing_edge_shape": "cut_off"}
    # Endpoint camber metrics ds/dxi keep the leading radius and trailing wedge physical (unit chord).
    metrics = {"leading_metric_m": 1. / np.cos(beta[0]), "trailing_metric_m": 1. / np.cos(beta[-1])}
    full = np.asarray(full_thickness(xi, thickness, 1.0, **metrics), dtype=float)
    if np.any(full[1:] <= 0):
        raise ValueError("The NACA thickness law gives a non-positive thickness inside the blade; "
                         "check the trailing-edge thickness and wedge angle.")
    normal = np.column_stack((-np.sin(beta), np.cos(beta)))
    return {"xi": xi, "beta": beta, "camber": camber, "thickness": full,
            "plus": camber + .5 * full[:, None] * normal, "minus": camber - .5 * full[:, None] * normal,
            "camber_parameters": camber_parameters, "thickness_parameters": thickness}


def reaction_blade(p):
    """Reaction rotor result in the solver frame (x = chord fraction, y = -theta), unit meridional chord."""
    validate_reaction(p)
    s = reaction_section(p)
    flip = np.array([1., -1.])
    # TurboDash puts the pressure side on "plus" for positive turning; after y -> -y it is the lower side.
    turning = s["beta"][-1] - s["beta"][0]
    pressure, suction = (s["plus"], s["minus"]) if turning >= 0 else (s["minus"], s["plus"])
    pressure, suction, camber = pressure * flip, suction * flip, s["camber"] * flip
    # Closed contour: pressure TE -> LE -> suction TE (one smooth curve round the nose), then the TE cut.
    contour = np.vstack((pressure[::-1], suction[1:], pressure[-1:]))
    beta_in, beta_out = -float(p["inlet_metal_angle_deg"]), -float(p["outlet_metal_angle_deg"])
    return {
        "blade_type": "reaction", "section_layout": "le_wrap_te_cut",
        "beta_inlet": beta_in, "beta_outlet": beta_out, "pitch": float(p["pitch_to_chord"]),
        "axial_chord": 1.0, "leading_edge_point": camber[0].tolist(), "num_points": int(len(s["xi"])),
        "blade": {"x": contour[:, 0].tolist(), "y": contour[:, 1].tolist(), "closure": "naca_round_le_cut_te",
                  "le_radius": float(p["leading_edge_radius"]), "te_thickness": float(p["trailing_edge_thickness"])},
        "blade_lower": {"x": pressure[:, 0].tolist(), "y": pressure[:, 1].tolist()},
        "blade_upper": {"x": suction[:, 0].tolist(), "y": suction[:, 1].tolist()},
        "camberline": {"x": camber[:, 0].tolist(), "y": camber[:, 1].tolist(),
                       "beta_deg": (-np.degrees(s["beta"])).tolist(), "xi": s["xi"].tolist()},
        "thickness": {"xi": s["xi"].tolist(), "full": s["thickness"].tolist()},
        "parameters": dict(p), "turbodash": {"camberline": s["camber_parameters"],
                                             "thickness": s["thickness_parameters"]},
    }


def constant_passage_geometric(p, *, le_te_ratio, num_points, **edges):
    """Baines constant-passage blade from blade angle and pitch/chord only (no vortex result)."""
    from scipy.optimize import brentq
    from turbo_moc.core.impulse_blade import EDGE_OPTIONS, _blade
    validate_constant_passage(p)
    beta = np.radians(p["blade_angle_deg"])
    closure = {"num_points": int(num_points), "le_te_ratio": le_te_ratio} | {
        key: value for key, value in edges.items() if key in EDGE_OPTIONS and value is not None}
    target = float(p["pitch_to_chord"])

    def mismatch(spacing):  # Final spacing / axial extent of the closed solid at unit sharp axial chord
        _, _, _, x, _, diagnostics = _blade(spacing, 1.0, beta, closure)
        return (spacing + diagnostics["translate"]) / np.ptp(x) - target

    spacing = brentq(mismatch, .05 * target, 5. * target, xtol=1e-12)
    lower, upper, construction, x, y, diagnostics = _blade(spacing, 1.0, beta, closure)
    pitch = spacing + diagnostics["translate"]
    return {
        "blade_type": "constant_passage", "beta_inlet": float(p["blade_angle_deg"]),
        "beta_outlet": -float(p["blade_angle_deg"]), "pitch": float(pitch), "num_points": int(num_points),
        "pressure": lower, "suction": upper, "blade_lower": lower, "blade_upper": upper,
        "blade": {"x": x.tolist(), "y": y.tolist(), "closure": "circular_arc_le_te_fillet",
                  "le_radius": diagnostics["le_radius"], "te_radius": diagnostics["te_radius"],
                  "translate": diagnostics["translate"], "le_breaks": diagnostics["le_breaks"]},
        "construction": {key: float(value) for key, value in construction.items()} | {"spacing": float(pitch)},
        "parameters": dict(p),
    }


def row_pitch_to_chord(config):
    """Pitch / meridional chord at the reference radius for annular rows, else None (linear cascades)."""
    dimensions = config["meridional_channel"]
    if dimensions["type"] == "linear":
        return None
    from turbo_moc.geometry.meridional_channel import MeridionalChannel, geometry_from_machine
    mapping = MeridionalChannel(geometry_from_machine(dimensions))
    return float(mapping.pitch / mapping.reference_chord)


def geometric_rotor(config):
    """Build the selected geometry-only blade for one rotor row configuration."""
    design = config["rotor_design"]
    blade_type = design.get("blade_type", "vortex")
    params = dict(GEOMETRIC_BLADE_TYPES[blade_type]) | dict(design.get(blade_type) or {})
    ratio = row_pitch_to_chord(config)
    if ratio is not None:  # Annular rows: the blade count fixes the pitch
        params["pitch_to_chord"] = ratio
    if blade_type == "reaction":
        return reaction_blade(params)
    edges = {key: design.get(key) for key in ("le_wedge_half_angle", "le_round_tip", "le_ellipse_ratio",
                                              "te_ellipse_ratio")}
    return constant_passage_geometric(params, le_te_ratio=design["le_te_ratio"],
                                      num_points=design["num_points"], **edges)
