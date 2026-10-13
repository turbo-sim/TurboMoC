"""Inputs for the vortex rotor provider and its relative-frame inlet state.

Numerical rotor construction remains in ``rotor.vortex_blade``. This module
owns only configuration, including migration of the earlier planning rows.
"""
from copy import deepcopy

import numpy as np

from turbo_moc.configuration.row import SMOOTH_FIELDS, validate_mesh_configuration
from turbo_moc.configuration.fluid_inputs import INLET_FIELDS, validate_inlet_inputs


ROTOR_DEFAULTS = {
    "fluid_name": "Novec649", "P0_rel": 5e5, "T0_rel": 400.0,
    "M_inlet": 1.5, "M_outlet": 1.5, "M_lower": 1.05, "M_upper": 2.0,
    "beta_inlet": 65.0, "num_points": 60, "le_te_ratio": 25.0,
}
SPEED_TYPES = {"actual": "Actual speed [rpm]", "blade_velocity_ratio": "Blade velocity ratio U/c₀ [-]",
               "specific": "Specific speed [-]"}
COUPLING_DEFAULTS = {"mode": "manual", "rotational_speed": {"type": "actual", "value": 20716.71}}
FLOW_DOMAIN_DEFAULTS = {
    "inlet_chords": 1.5, "outlet_chords": 2.0,
    "domain_shift_mode": "auto", "domain_shift": 0.0,
    "inlet_axial_tangent": 0.35, "inlet_blade_tangent": 0.35,
    "outlet_blade_tangent": 0.15, "outlet_axial_tangent": 0.55,
    "num_points": 120,
}


def default_axial_interspace_mm(meridional_channel):
    """Initial clearance: 10% of the rotor's midspan axial chord.

    Linear chord variation makes the midspan chord the mean of the hub and
    tip values. This is an initial length, not a constraint: saved gaps remain
    unchanged when the user subsequently edits the rotor dimensions.
    """
    midspan_chord = 0.5 * (meridional_channel["hub_chord"] + meridional_channel["tip_chord"])
    return 0.1 * midspan_chord


def normalize_rotor_geometry(geometry):
    """Upgrade a placeholder without replacing saved dimensions or mesh inputs."""
    result = deepcopy(geometry)
    # "conventional" was a selector on planning-only rows, never a rotor solver.
    # Only migrate it when no implemented rotor inputs were present.
    planning = "rotor_design" not in result
    if planning and result.get("parametrization_type", "conventional") == "conventional":
        result["parametrization_type"] = "vortex"
    result.setdefault("parametrization_type", "vortex")
    for name, defaults in (("rotor_design", ROTOR_DEFAULTS),
                           ("inlet_coupling", COUPLING_DEFAULTS),
                           ("flow_domain", FLOW_DOMAIN_DEFAULTS)):
        values = result.setdefault(name, {})
        if isinstance(values, dict):
            if name == "inlet_coupling" and "rpm" in values:
                if "rotational_speed" in values:
                    raise ValueError("Specify rotational_speed or legacy rpm, not both.")
                values["rotational_speed"] = {"type": "actual", "value": values.pop("rpm")}
            if name == "flow_domain":
                smooth = values.pop("smooth", {})
                if not isinstance(smooth, dict):
                    raise ValueError("flow_domain.smooth must be a mapping.")
                for key, value in smooth.items():
                    values.setdefault(key, value)
            for key, value in defaults.items():
                if name == "rotor_design" and key in ("P0_rel", "T0_rel") and any(
                        field in values for field in INLET_FIELDS):
                    continue
                values.setdefault(key, deepcopy(value))
    meridional_channel = result.get("meridional_channel", {})
    if isinstance(meridional_channel, dict) and meridional_channel.get("type") == "axial":
        if "axial_interspace_mm" not in result:
            result["axial_interspace_mm"] = default_axial_interspace_mm(meridional_channel)
    return result


def _finite(value, name, *, minimum=None, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}.")


def inlet_radius_mm(config):
    """Representative blade-speed radius: midspan at the rotor leading edge.

    Linear cascades have no physical radius and can only use a manual inlet.
    The axial gap changes row placement, never this radius.
    """
    meridional_channel = config["meridional_channel"]
    if meridional_channel["type"] == "linear":
        return None
    if meridional_channel["type"] == "radial":
        return float(meridional_channel["inlet_radius"])
    return float(meridional_channel["hub_radius"] + meridional_channel["blade_height"] / 2)


def validate_rotor_configuration(config):
    """Validate a row without solving thermodynamics or requiring an upstream job.

    A saved coupled rotor may temporarily have no active stator (for example
    after switching design mode). Dependency readiness is checked at execution.
    """
    from turbo_moc.geometry.meridional_channel import validate_machine
    from turbo_moc.geometry.passage_2d import validate_domain_shift, validate_flow_domain_settings

    required = {"parametrization_type", "meridional_channel", "rotor_design",
                "inlet_coupling", "flow_domain", "mesh", "workflow"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Rotor configuration requires {sorted(required)}.")
    if config["parametrization_type"] != "vortex":
        raise ValueError("Rotor parametrization_type must be vortex.")
    validate_machine(config["meridional_channel"])
    design = config["rotor_design"]
    paired_fields = (set(ROTOR_DEFAULTS) - {"P0_rel", "T0_rel"}) | set(INLET_FIELDS)
    if not isinstance(design, dict) or set(design) not in (set(ROTOR_DEFAULTS), paired_fields):
        raise ValueError(f"rotor_design requires {tuple(ROTOR_DEFAULTS)}.")
    if not isinstance(design["fluid_name"], str) or not design["fluid_name"].strip():
        raise ValueError("rotor_design.fluid_name must be a non-empty fluid name.")
    validate_inlet_inputs(design)
    for name in ("P0_rel", "T0_rel", "le_te_ratio"):
        if name not in design:
            continue
        _finite(design[name], f"rotor_design.{name}", minimum=0, strict=True)
    for name in ("M_inlet", "M_outlet", "M_lower", "M_upper"):
        _finite(design[name], f"rotor_design.{name}", minimum=1, strict=True)
    _finite(design["beta_inlet"], "rotor_design.beta_inlet")
    if abs(design["beta_inlet"]) >= 90:
        raise ValueError("rotor_design.beta_inlet must be strictly between -90 and 90 degrees.")
    if type(design["num_points"]) is not int or design["num_points"] < 10:
        raise ValueError("rotor_design.num_points must be an integer >= 10.")
    coupling = config["inlet_coupling"]
    if not isinstance(coupling, dict) or set(coupling) != set(COUPLING_DEFAULTS):
        raise ValueError("inlet_coupling requires mode and rotational_speed.")
    if coupling["mode"] not in ("manual", "stator_exit"):
        raise ValueError("inlet_coupling.mode must be manual or stator_exit.")
    speed = coupling["rotational_speed"]
    if not isinstance(speed, dict) or set(speed) != {"type", "value"}:
        raise ValueError("inlet_coupling.rotational_speed requires type and value.")
    if speed["type"] not in SPEED_TYPES:
        raise ValueError(f"rotational_speed.type must be one of {tuple(SPEED_TYPES)}.")
    # Signed speed represents reversing the rotor's rotation direction.
    _finite(speed["value"], "rotational_speed.value")
    domain = config["flow_domain"]
    if not isinstance(domain, dict) or set(domain) != set(FLOW_DOMAIN_DEFAULTS):
        raise ValueError(f"flow_domain requires {tuple(FLOW_DOMAIN_DEFAULTS)}.")
    validate_flow_domain_settings({name: domain[name] for name in SMOOTH_FIELDS})
    validate_domain_shift(domain["domain_shift_mode"], domain["domain_shift"])
    for name in ("inlet_chords", "outlet_chords"):
        _finite(domain[name], f"flow_domain.{name}", minimum=0, strict=True)
    validate_mesh_configuration(config["mesh"])
    return config
