"""Editable fluid input pairs and adapters to the solvers' stagnation states."""
from copy import deepcopy
import re

import jaxprop as jxp
import numpy as np


INLET_FIELDS = ("inlet_property_pair", "inlet_property_1", "inlet_property_2")
PROPERTIES = {
    "P": ("Pressure", "Pa"), "T": ("Temperature", "K"),
    "Q": ("Vapor quality", "kg/kg"),
    "Dmass": ("Mass density", "kg/m³"), "Dmolar": ("Molar density", "mol/m³"),
    "Hmass": ("Specific enthalpy", "J/kg"), "Hmolar": ("Molar enthalpy", "J/mol"),
    "Smass": ("Specific entropy", "J/(kg·K)"), "Smolar": ("Molar entropy", "J/(mol·K)"),
    "Umass": ("Specific internal energy", "J/kg"), "Umolar": ("Molar internal energy", "J/mol"),
}
_PROPERTY_PATTERN = "(" + "|".join(PROPERTIES) + ")"


def pair_properties(pair):
    """Return the two properties in CoolProp's required input order."""
    if not isinstance(pair, str) or pair not in jxp.INPUT_PAIRS:
        raise ValueError("Select a supported inlet property pair.")
    match = re.fullmatch(_PROPERTY_PATTERN + _PROPERTY_PATTERN + "_INPUTS", pair)
    if match is None:
        raise ValueError(f"Unknown properties in inlet pair {pair!r}.")
    return match.groups()


def property_label(pair, index):
    name, unit = PROPERTIES[pair_properties(pair)[index - 1]]
    return f"Inlet property {index}: {name.lower()} [{unit}]"


def inlet_values(parameters, *, rotor=False):
    """Display historical PQ/PT inputs without changing the saved case."""
    if any(name in parameters for name in INLET_FIELDS):
        return {name: parameters.get(name) for name in INLET_FIELDS}
    pressure, temperature, quality = ("P0_rel", "T0_rel", "Q0_rel") if rotor else ("P0", "T0", "Q0")
    q0 = parameters.get(quality)
    saturated = q0 is not None and np.isfinite(q0)
    return dict(zip(INLET_FIELDS, (
        "PQ_INPUTS" if saturated else "PT_INPUTS", parameters[pressure],
        q0 if saturated else parameters.get(temperature))))


def validate_inlet_inputs(parameters):
    """Validate explicit pair inputs without performing thermodynamic work."""
    if not any(name in parameters for name in INLET_FIELDS):
        return
    if not all(name in parameters for name in INLET_FIELDS):
        raise ValueError("Specify the inlet property pair and both inlet properties.")
    properties = pair_properties(parameters[INLET_FIELDS[0]])
    for field, prop in zip(INLET_FIELDS[1:], properties):
        value = parameters[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        if prop == "Q" and not 0 <= value <= 1:
            raise ValueError(f"{field}: vapor quality must be in [0, 1].")
        if prop in ("P", "T", "Dmass", "Dmolar") and value <= 0:
            raise ValueError(f"{field}: {PROPERTIES[prop][0].lower()} must be positive.")


def solver_parameters(parameters, *, rotor=False):
    """Resolve any selected pair to PT or PQ, preserving two-phase quality."""
    result = deepcopy(parameters)
    if not any(name in result for name in INLET_FIELDS):
        return result
    validate_inlet_inputs(result)
    pair, first, second = (result.pop(name) for name in INLET_FIELDS)
    fluid = jxp.Fluid(result["fluid_name"], backend=result.get("backend", "HEOS"))
    try:
        state = fluid.get_state(jxp.INPUT_PAIRS[pair], first, second)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve the {result['fluid_name']} inlet with {pair}: {exc}") from exc
    pressure, temperature, quality = ("P0_rel", "T0_rel", "Q0_rel") if rotor else ("P0", "T0", "Q0")
    p0, t0 = float(state.p), float(state.T)
    q0 = float(state.Q) if state.is_two_phase else float("nan")
    if not np.isfinite(p0) or not np.isfinite(t0) or p0 <= 0 or t0 <= 0:
        raise ValueError("The inlet state requires finite, positive pressure and temperature.")
    result.update({pressure: p0, temperature: t0, quality: q0})
    return result
