"""Planar stage design: the MoC stator and the Goldman vortex rotor as linear cascades, nothing annular.

The stator is the MoC nozzle blade and the rotor the TN D-4421 vortex-flow blade, each scaled to its
chord with its own pitch/chord (a similarity transform only). The rotor inflow is the stator's MoC
exit state seen at the blade speed U. The only stage constraint is a whole number of rotor passages
per stator pitch, met by the rotor chord.
"""

from copy import deepcopy

import numpy as np

ROWS = ("stator_1", "rotor_1")


def _row(project, role):
    return deepcopy(next(row for row in project["rows"] if row["role"] == role))


def _linear(row, chord, overrides):
    from .design import merge
    row = merge(row, deepcopy(overrides))
    row["geometry"].pop("axial_interspace_mm", None)
    row["geometry"]["meridional_channel"] = {"type": "linear", "meridional_chord": float(chord),
                                             "blade_height": float(row["geometry"]["meridional_channel"]["blade_height"])}
    return row


def _resolved(project, row_id):
    from turbo_moc.app import workflow as flow
    from turbo_moc.configuration import project as model
    project = model.validate_project(model.normalize_project(project))
    return flow.resolved_configuration(project, row_id, {})


def _project(source, mode, row):
    return {"machine": {"design_mode": mode, "geometry_type": "linear", "fluid_name": source["machine"]["fluid_name"]},
            "workflow": deepcopy(source.get("workflow", {})), "rows": [row]}


def _gap_side(row, side, half_gap, chord):
    """Extend the passage by half the row gap beyond the blade edge, straight, so that the rows meet midway."""
    from .design import merge
    return merge(row, {"geometry": {"flow_domain": {side: {"extension_type": "straight",
                                                          "extension_length": half_gap / chord}}}})


def stator(source, settings, half_gap):
    """MoC nozzle and its stator blade as a linear cascade at the requested chord."""
    from turbo_moc.geometry import blade_row
    from turbo_moc.workflows.nozzle import solve_nozzle
    row = _gap_side(_linear(_row(source, "stator"), settings["chord"], settings.get("row", {})),
                    "outlet", half_gap, settings["chord"])
    config = _resolved(_project(source, "stator", row), row["id"])
    nozzle = solve_nozzle(config["method_of_characteristics"])
    design = blade_row.build_design(nozzle, config)
    return {"config": config, "nozzle": nozzle, "design": design}


def exit_state(fluid_name, nozzle, alpha_deg):
    """Static state, velocity and flow components at the MoC stator exit (uniform, isentropic)."""
    import jaxprop as jxp
    wall = nozzle["wall_final"]
    fluid = jxp.Fluid(fluid_name)
    state = fluid.get_state(jxp.DmassT_INPUTS, float(wall["rho"][-1]), float(wall["T"][-1]))
    speed = float(wall["M"][-1]) * float(state.a)
    alpha = np.radians(alpha_deg)
    return {"p": float(state.p), "rho": float(state.rho), "h": float(state.h), "s": float(state.s),
            "a": float(state.a), "M": float(wall["M"][-1]), "c": speed, "alpha": float(alpha_deg),
            "c_axial": speed * np.cos(alpha), "c_tangential": speed * np.sin(alpha)}


def relative_inflow(fluid_name, exit_, blade_speed):
    """The stator exit seen from a rotor moving at U along the swirl: relative Mach, angle and total state."""
    import jaxprop as jxp
    fluid = jxp.Fluid(fluid_name)
    w_tangential = exit_["c_tangential"] - blade_speed
    speed = float(np.hypot(exit_["c_axial"], w_tangential))
    total = fluid.get_state(jxp.HmassSmass_INPUTS, exit_["h"] + 0.5 * speed ** 2, exit_["s"])
    quality = float(total.Q) if total.is_two_phase else float("nan")
    return {"W": speed, "M": speed / exit_["a"], "beta": float(np.degrees(np.arctan2(w_tangential, exit_["c_axial"]))),
            "P0_rel": float(total.p), "T0_rel": float(total.T), "Q0_rel": quality, "p": exit_["p"],
            "rho": exit_["rho"], "axial_mach": exit_["c_axial"] / exit_["a"]}


def vortex_rotor(fluid_name, inflow, settings):
    """TN D-4421 vortex blade at the relative inflow; impulse: the exit Mach equals the inlet Mach."""
    from turbo_moc.core.vortex_blade import design_rotor_vortex_blade
    quality = inflow["Q0_rel"]
    return design_rotor_vortex_blade(
        fluid_name, inflow["P0_rel"], inflow["T0_rel"], inflow["M"], settings.get("M_outlet", inflow["M"]),
        settings["M_lower"], settings["M_upper"], inflow["beta"],
        Q0_rel=quality if np.isfinite(quality) else float("nan"), le_te_ratio=settings["le_te_ratio"],
        le_wedge_half_angle=settings.get("le_wedge_half_angle"))


def rotor(source, settings, inflow, stator_pitch, half_gap):
    """Rotor blade at the chord that fits ``passages`` pitches into one stator pitch.

    ``spacing_factor`` sets the blades that much farther apart than the vortex design's own spacing,
    leaving the blade itself unchanged: a starting margin on the passage area.
    """
    from turbo_moc.geometry.rotor_blade import build_rotor_design
    blade = vortex_rotor(source["machine"]["fluid_name"], inflow, settings)
    blade = dict(blade, pitch=blade["pitch"] * settings.get("spacing_factor", 1.0))
    pitch_chord = blade["pitch"] / np.ptp(np.asarray(blade["blade"]["x"]))
    if stator_pitch is None:
        passages, chord = 1, float(settings["chord"])
    else:
        passages = settings.get("passages") or max(1, int(round(stator_pitch / (pitch_chord * settings["chord"]))))
        chord = stator_pitch / (passages * pitch_chord)
    row = _gap_side(_linear(_row(source, "rotor"), chord, settings.get("row", {})), "inlet", half_gap, chord)
    row["geometry"]["inlet_coupling"] = {"mode": "manual", "rotational_speed": {"type": "actual", "value": 0.0}}
    config = _resolved(_project(source, "rotor", row), row["id"])
    design = build_rotor_design(blade, config)
    return {"config": config, "blade": blade, "design": design, "chord": chord, "passages": passages,
            "pitch_chord": pitch_chord}


def section(design, numerics):
    """The planar passage mesh of one row, in the arrays ``planar_mesh`` assembles."""
    import gmsh
    from turbo_moc.meshing.mesh_3d import generate_section
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    gmsh.initialize(["planar_section", "-v", "2"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        mesh = generate_section(design.blade, design.curves, numerics)
    finally:
        gmsh.finalize()
    xy = np.asarray(mesh["xy"], dtype=float)
    cells = [np.asarray(cell, dtype=np.int64) for block in mesh["cells"].values() for cell in block]
    master, slave = np.asarray(mesh["master"], dtype=np.int64), np.asarray(mesh["slave"], dtype=np.int64)
    return {"xy": xy, "cells": cells, "boundaries": {name: np.asarray(edges, dtype=np.int64)
                                                     for name, edges in mesh["boundaries"].items()},
            "master": master, "slave": slave, "pitch": (xy[slave] - xy[master]).mean(axis=0),
            "summary": mesh["summary"]}


def design_stage(source, workflow):
    """Both rows, the velocity triangle and the planar sections of a two-row stage at a fixed cycle."""
    fluid = source["machine"]["fluid_name"]
    rows = workflow["rows"]
    half_gap = 0.5 * float(workflow["axial_gap_mm"])
    built = stator(source, rows["stator_1"], half_gap)
    stator_design = built["design"]
    alpha = built["config"]["blade_geometry"]["metal_angle_out"]
    exit_ = exit_state(fluid, built["nozzle"], alpha)
    inflow = relative_inflow(fluid, exit_, float(workflow["blade_speed"]))
    rotor_built = rotor(source, rows["rotor_1"], inflow, float(stator_design.blade["pitch"]), half_gap)
    numerics = {row: built_row["config"]["mesh"]["numerics"]
                for row, built_row in (("stator_1", built), ("rotor_1", rotor_built))}
    return {"stator": built, "rotor": rotor_built, "stator_exit": exit_, "relative_inflow": inflow,
            "sections": {row: section(item["design"], numerics[row])
                         for row, item in (("stator_1", built), ("rotor_1", rotor_built))}}


def prescribed_inflow(fluid_name, nozzle, settings, outlet_pressure):
    """Relative inflow from a chosen Mach number and angle, its static state on the cycle inlet isentrope.

    The static pressure is ``settings["p"]`` or, by default, the cycle outlet pressure (an impulse rotor).
    """
    import jaxprop as jxp
    fluid = jxp.Fluid(fluid_name)
    wall = nozzle["wall_final"]
    entropy = float(fluid.get_state(jxp.DmassT_INPUTS, float(wall["rho"][0]), float(wall["T"][0])).s)
    static = fluid.get_state(jxp.PSmass_INPUTS, float(settings.get("p", outlet_pressure)), entropy)
    speed = float(settings["M"]) * float(static.a)
    total = fluid.get_state(jxp.HmassSmass_INPUTS, float(static.h) + 0.5 * speed ** 2, entropy)
    beta = float(settings["beta"])
    return {"W": speed, "M": float(settings["M"]), "beta": beta, "P0_rel": float(total.p), "T0_rel": float(total.T),
            "Q0_rel": float(total.Q) if total.is_two_phase else float("nan"), "p": float(static.p),
            "rho": float(static.rho), "axial_mach": float(settings["M"]) * np.cos(np.radians(beta))}


def design_rotor_only(source, workflow):
    """The rotor alone at a prescribed relative inflow (``workflow["relative_inflow"]``: M, beta [deg], optional p [Pa]).

    The MoC nozzle is still solved for the barotropic model along the cycle isentrope; no stator is built.
    """
    from turbo_moc.workflows.nozzle import solve_nozzle
    fluid = source["machine"]["fluid_name"]
    stator_row = _linear(_row(source, "stator"), workflow["rows"]["stator_1"]["chord"],
                         workflow["rows"]["stator_1"].get("row", {}))
    config = _resolved(_project(source, "stator", stator_row), stator_row["id"])
    nozzle = solve_nozzle(config["method_of_characteristics"])
    inflow = prescribed_inflow(fluid, nozzle, workflow["relative_inflow"], float(workflow["outlet_pressure"]))
    built = rotor(source, workflow["rows"]["rotor_1"], inflow, None, 0.5 * float(workflow["axial_gap_mm"]))
    return {"stator": {"config": config, "nozzle": nozzle}, "rotor": built, "relative_inflow": inflow,
            "stator_exit": None, "sections": {"rotor_1": section(built["design"], built["config"]["mesh"]["numerics"])}}
