"""Resolve a row's inlet from selected, current upstream results.

The project retains manual inputs when coupling is enabled. Derived values are
an execution snapshot, not edits to those manual inputs. The legacy coupling
model transfers the stator exit static state without modelling flow through the
interspace; the gap still sets the physical row placement independently.
"""
from copy import deepcopy
from functools import lru_cache
import json

from ... import configuration_project as projects
from ...configuration_rotor import inlet_radius_mm
from ...coupling import derive_rotor_inlet_from_stator


@lru_cache(maxsize=32)
def _triangle(serialized_nozzle, angle, speed):
    # Cache only this small derived display calculation, never a MoC solve.
    return derive_rotor_inlet_from_stator(json.loads(serialized_nozzle), angle, speed)


def resolve_configuration(project, row_id, registry):
    """Return the effective numerical inputs and their upstream provenance."""
    from . import workflow as flow
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
    from math import isfinite, pi
    speed = config["inlet_coupling"]["rpm"] * 2 * pi / 60 * radius / 1000
    angle = upstream_config["blade_geometry"]["metal_angle_out"]
    # Only the exit state is consumed by the existing velocity-triangle model.
    # Avoid serializing the entire characteristic net on every sidebar edit.
    exit_state = {"fluid_name": nozzle["fluid_name"], "wall_final": {
        key: [nozzle["wall_final"][key][-1]] for key in ("p", "T", "rho", "M")}}
    try:
        triangle = deepcopy(_triangle(json.dumps(exit_state, sort_keys=True), angle, speed))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot compute the coupled rotor inlet: {exc}") from exc
    if (not all(isfinite(triangle[name]) for name in ("P0_rel", "T0_rel", "M_rel", "beta_rel_deg"))
            or min(triangle[name] for name in ("P0_rel", "T0_rel", "M_rel")) <= 0):
        raise ValueError("The coupled rotor inlet requires finite, positive pressure, temperature and Mach number.")
    inputs = config["rotor_design"]
    inputs.update(fluid_name=nozzle["fluid_name"], P0_rel=triangle["P0_rel"],
                  T0_rel=triangle["T0_rel"], M_inlet=triangle["M_rel"],
                  beta_inlet=triangle["beta_rel_deg"])
    config["rotor_source"] = {"row_id": upstream["id"], "reference": {
        key: reference[key] for key in ("session", "job", "key")},
        "alpha_deg": angle, "inlet_radius_mm": radius, "triangle": triangle}
    return config


def require_supersonic_inlet(config):
    """Block unsuitable coupled conditions while leaving the readout visible."""
    if "rotor_design" in config and config["rotor_design"]["M_inlet"] <= 1:
        raise ValueError("The vortex rotor requires a supersonic relative inlet (M > 1). Adjust speed or inlet conditions.")
