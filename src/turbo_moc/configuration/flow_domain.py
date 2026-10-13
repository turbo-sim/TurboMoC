"""Nested flow-domain inputs and adapters for the existing geometry routines."""
from copy import deepcopy


FIELDS = {"extension_type": "extension_type", "extension_length": "chords",
          "axial_tangent": "axial_tangent", "blade_tangent": "blade_tangent"}
DEFAULTS = {
    "domain_shift_mode": "auto", "domain_shift_value": 0.0, "number_of_points": 120,
    "inlet": {"extension_type": "bspline", "extension_length": 1.5,
              "axial_tangent": 0.35, "blade_tangent": 0.35},
    "outlet": {"extension_type": "straight", "extension_length": 2.0,
               "axial_tangent": 0.55, "blade_tangent": 0.15},
}


def normalize(domain, *, defaults=None):
    """Upgrade flat/smooth inputs without changing values or the caller's data."""
    if not isinstance(domain, dict):
        raise ValueError("flow_domain must be a mapping.")
    source = deepcopy(domain)
    smooth = source.pop("smooth", {})
    if not isinstance(smooth, dict):
        raise ValueError("flow_domain.smooth must be a mapping.")
    for name, value in smooth.items():
        source.setdefault(name, value)
    for old, new in (("domain_shift", "domain_shift_value"), ("num_points", "number_of_points")):
        if old in source:
            if new in source:
                raise ValueError(f"Use flow_domain.{new} or legacy {old}, not both.")
            source[new] = source.pop(old)
    for side in ("inlet", "outlet"):
        values = source.setdefault(side, {})
        if not isinstance(values, dict):
            raise ValueError(f"flow_domain.{side} must be a mapping.")
        for new, suffix in FIELDS.items():
            old = f"{side}_{suffix}"
            if old in source:
                if new in values:
                    raise ValueError(f"Use flow_domain.{side}.{new} or legacy {old}, not both.")
                values[new] = source.pop(old)
        for name, value in (defaults or {}).get(side, {}).items():
            values.setdefault(name, deepcopy(value))
        values.setdefault("extension_type", "bspline" if side == "inlet" else "straight")
        unknown = values.keys() - FIELDS.keys()
        if unknown:
            raise ValueError(f"Unknown flow_domain.{side} options: {sorted(unknown)}.")
        source[side] = {name: values[name] for name in FIELDS if name in values}
    for name, value in (defaults or {}).items():
        source.setdefault(name, deepcopy(value))
    unknown = source.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown flow_domain options: {sorted(unknown)}.")
    return {name: source[name] for name in DEFAULTS if name in source}


def to_flat(domain):
    """Provide the unchanged numerical arguments for new or saved old inputs."""
    domain = normalize(domain)
    result = {old: domain[new] for new, old in (
        ("domain_shift_mode", "domain_shift_mode"), ("domain_shift_value", "domain_shift"),
        ("number_of_points", "num_points")) if new in domain}
    for side in ("inlet", "outlet"):
        result.update({f"{side}_{suffix}": domain[side][name]
                       for name, suffix in FIELDS.items() if name in domain[side]})
    return result


def to_passage(domain):
    """Adapt to the historical passage/smooth contract used by planar geometry."""
    result = to_flat(domain)
    result["smooth"] = {name: result.pop(name) for name in list(result)
                        if name not in {"domain_shift_mode", "domain_shift", "inlet_chords", "outlet_chords"}}
    return result
