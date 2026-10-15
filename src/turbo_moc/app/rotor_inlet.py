"""Resolve a row's inlet from selected, current upstream results.

The project retains manual inputs when coupling is enabled. Derived values are
an execution snapshot, not edits to those manual inputs. The legacy coupling
model transfers the stator exit static state without modelling flow through the
interspace; the gap still sets the physical row placement independently.
"""
from copy import deepcopy
from functools import lru_cache
import json

from turbo_moc.configuration import project as projects
from turbo_moc.configuration.rotor import inlet_radius_mm
from turbo_moc.configuration.fluid_inputs import INLET_FIELDS
from turbo_moc.core.coupling import derive_rotor_inlet_from_stator
from turbo_moc.app.rotor_speed import resolve_speed, stator_reference


@lru_cache(maxsize=32)
def _triangle(serialized_nozzle, angle, speed):
    # Cache only this small derived display calculation, never a MoC solve.
    return derive_rotor_inlet_from_stator(json.loads(serialized_nozzle), angle, speed)


def resolve_configuration(project, row_id, registry):
    """Return the effective numerical inputs and their upstream provenance."""
    from turbo_moc.app import workflow as flow
    config = projects.row_configuration(project, row_id)
    row = projects.get_row(project, row_id)
    if row["role"] != "rotor" or config["inlet_coupling"]["mode"] == "manual":
        return config
    upstream = projects.upstream_stator(project, row_id)
    if upstream is None:
        raise ValueError("Coupled rotor inlet requires an active upstream stator. Select manual inlet or Stator + Rotor.")
    upstream_config = projects.row_configuration(project, upstream["id"])
    reference = (registry or {}).get(upstream["id"], {}).get("nozzle")
    try:
        nozzle = flow.load_nozzle(reference, upstream_config)
    except ValueError as exc:
        raise ValueError("Compute the upstream stator MoC for its current inputs before coupling the rotor inlet.") from exc
    if reference.get("row_id", "stator_1") != upstream["id"]:
        raise ValueError("The selected stator result belongs to another blade row.")
    if not nozzle.get("converged"):
        raise ValueError("The upstream stator MoC must converge before coupling the rotor inlet.")
    radius = inlet_radius_mm(config)
    if radius is None:
        raise ValueError("Stator-to-rotor coupling requires an axial or radial machine.")
    from math import isfinite
    angle = upstream_config["blade_geometry"]["metal_angle_out"]
    specification = config["inlet_coupling"]["rotational_speed"]
    reference_state = {}
    if specification["type"] != "actual":
        try:
            reference_state = stator_reference(upstream_config, nozzle, angle)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Cannot compute the rotor speed reference: {exc}") from exc
    resolved_speed = resolve_speed(specification, radius, **{
        key: reference_state[key] for key in ("enthalpy_drop", "exit_density", "mass_flow")
        if key in reference_state})
    speed = resolved_speed["U"]
    # Only the exit state is consumed by the existing velocity-triangle model.
    # Avoid serializing the entire characteristic net on every sidebar edit.
    exit_state = {"fluid_name": project["machine"]["fluid_name"], "wall_final": {
        key: [nozzle["wall_final"][key][-1]] for key in ("p", "T", "rho", "M")}}
    try:
        triangle = deepcopy(_triangle(json.dumps(exit_state, sort_keys=True), angle, speed))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot compute the coupled rotor inlet: {exc}") from exc
    if (not all(isfinite(triangle[name]) for name in ("P0_rel", "T0_rel", "M_rel", "beta_rel_deg"))
            or min(triangle[name] for name in ("P0_rel", "T0_rel", "M_rel")) <= 0):
        raise ValueError("The coupled rotor inlet requires finite, positive pressure, temperature and Mach number.")
    inputs = config["rotor_design"]
    for name in INLET_FIELDS:
        inputs.pop(name, None)
    inputs.update(P0_rel=triangle["P0_rel"], T0_rel=triangle["T0_rel"],
                  M_inlet=triangle["M_rel"], beta_inlet=triangle["beta_rel_deg"])
    inputs.pop("Q0_rel", None)
    if isfinite(triangle.get("Q0_rel", float("nan"))):
        inputs["Q0_rel"] = triangle["Q0_rel"]
    config["rotor_source"] = {"row_id": upstream["id"], "reference": {
        key: reference[key] for key in ("session", "job", "key")},
        "alpha_deg": angle, "inlet_radius_mm": radius, "triangle": triangle,
        "rotational_speed": {**resolved_speed, **reference_state}}
    return config


def inlet_warnings(config):
    """Report unsupported vortex inlets without hiding the calculated state."""
    design = config.get("rotor_design")
    if design is None:
        return []
    warnings = []
    if design["M_inlet"] <= 1:
        warnings.append("The vortex rotor requires a supersonic relative inlet (M > 1). Adjust speed or inlet conditions.")
    if design["beta_inlet"] < 0:
        warnings.append("Negative relative inlet angle (beta_inlet < 0): the rotor blade concavity is reversed. Reduce speed or adjust inlet conditions.")
    return warnings


def require_supersonic_inlet(config):
    """Block unsupported Mach numbers and angles before starting the solver."""
    warnings = inlet_warnings(config)
    if warnings:
        raise ValueError(" ".join(warnings))
