"""Planar stage and rotor-alone cases: the MoC stator and the Goldman vortex rotor as 2D cascades in Fluent.

On an axial machine the rotating-frame forces on a cylinder (centrifugal and Coriolis) are radial,
so the blade-to-blade plane with the rotor zone translating at U is the stage at that radius, and
the rotor alone is a stationary cascade fed with its relative inflow. Radial equilibrium and
spanwise stream-tube changes are left out. Fluent's 2D flows are per metre of depth.

``python -m turbo_moc.fluent.stage.planar <workflow.yaml> [stage|rotor|stage-extract|rotor-extract]`` designs the rows
(``planar_design``), assembles the planar mesh, writes the Fluent case and, when the workflow asks
for it, solves it and extracts the fields; the ``-extract`` modes only re-extract a solved case.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import case, planar_design, planar_mesh
from .setup import solve

STATOR, ROTOR = case.STATOR, case.ROTOR
VELOCITIES = ("x-velocity", "y-velocity")
FIELDS = ("absolute-pressure", "density", "x-velocity", "y-velocity", "relative-x-velocity", "relative-y-velocity")


def swirl_direction(stator):
    """+1 when the stator exit flow runs towards +x in the planar frame (the flow runs towards -y)."""
    points = stator["xy"][np.unique(stator["boundaries"]["periodic_left"])]
    points = points[np.argsort(points[:, 1])][: max(3, len(points) // 10)]
    return -float(np.sign(np.polyfit(points[:, 1], points[:, 0], 1)[0]))


def inflow_direction(rotor):
    """+1 when the rotor's inlet extension runs towards +x as the flow runs towards -y."""
    points = rotor["xy"][np.unique(rotor["boundaries"]["periodic_left"])]
    points = points[np.argsort(points[:, 1])][-max(3, len(points) // 10):]
    return -float(np.sign(np.polyfit(points[:, 1], points[:, 0], 1)[0]))


def build_mesh(output_dir, design, rotor_only=False):
    sections, passages = design["sections"], design["rotor"]["passages"]
    if "stator_1" not in sections:
        rotor = sections["rotor_1"]
        return {"mesh_file": planar_mesh.write_cgns(output_dir / "mesh" / "rotor_2d.cgns", {"rotor_1": rotor}),
                "swirl_direction": inflow_direction(rotor), "rotor_passages": 1,
                "stator_pitch_m": float(rotor["pitch"][0]) / 1e3, "rotor_pitch_m": float(rotor["pitch"][0]) / 1e3,
                "cells": {"rotor_1": len(rotor["cells"])}}
    stator, rotor = planar_mesh.assemble(sections["stator_1"], sections["rotor_1"], passages)
    zones = {"rotor_1": sections["rotor_1"]} if rotor_only else {"stator_1": stator, "rotor_1": rotor}
    name = "rotor_2d.cgns" if rotor_only else "stage_2d.cgns"
    return {"mesh_file": planar_mesh.write_cgns(output_dir / "mesh" / name, zones),
            "swirl_direction": swirl_direction(stator), "rotor_passages": 1 if rotor_only else passages,
            "stator_pitch_m": float(stator["pitch"][0]) / 1e3,
            "rotor_pitch_m": float(sections["rotor_1"]["pitch"][0]) / 1e3,
            "cells": {row: len(section["cells"]) for row, section in zones.items()}}


def import_mesh(settings, mesh_file, rows):
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    types = case.boundary_types(settings.setup.boundary_conditions)
    return {f"{row}_{name}": case.zone(types, row.split("_fluid")[0], name)
            for row in rows for name in planar_mesh.BOUNDARIES}


def flow_model(settings, options):
    """Inviscid when the workflow says ``viscous: false``; the turbulence residuals then drop out."""
    if options.get("viscous", True):
        return options
    settings.setup.models.viscous.model = "inviscid"
    return dict(options, residual_equations={key: value for key, value in options["residual_equations"].items()
                                             if key not in ("k", "omega")})


def translational_periodicity(settings, zones, rows, pitch_m):
    """Each row's sides paired one sector apart along +x (2D needs the explicit translation vector)."""
    for row in rows:
        settings.mesh.modify_zones.make_periodic(zone_name=zones[f"{row}_periodic_left"],
                                                 shadow_zone_name=zones[f"{row}_periodic_right"],
                                                 rotate_periodic=False, create=True, auto_translation=False,
                                                 direction=[pitch_m, 0.0])
    periodic = settings.setup.boundary_conditions.periodic
    for name in periodic.get_object_names():
        periodic[name].periodic.rotationally_periodic = "Translational"


def turbulence(boundary, options, backflow=False):
    boundary.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    prefix = "backflow_" if backflow else ""
    setattr(boundary.turbulence, f"{prefix}turbulent_intensity", options["turbulence_intensity"])
    setattr(boundary.turbulence, f"{prefix}turbulent_viscosity_ratio", options["turbulence_viscosity_ratio"])


def outlet_boundary(settings, zones, pressure, options):
    outlet = settings.setup.boundary_conditions.pressure_outlet[zones[f"{ROTOR}_outlet"]]
    outlet.momentum.gauge_pressure.value = pressure
    if options.get("viscous", True):
        turbulence(outlet, options, backflow=True)


def configure_stage(solver, mesh, state, expressions, cycle, options):
    """Stator at rest, rotor zone translating at U along the stator swirl, a non-conformal interface between them."""
    settings = solver.settings
    rows = (STATOR, ROTOR)
    zones = import_mesh(settings, mesh["mesh_file"], rows)
    names = case.configure_material(settings, expressions, options["material_name"])
    modify = settings.mesh.modify_zones
    modify.zone_type(zone_names=[zones[f"{STATOR}_inlet"]], new_type="pressure-inlet")
    modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
    modify.zone_type(zone_names=[zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"]], new_type="interface")
    translational_periodicity(settings, zones, rows, mesh["stator_pitch_m"])
    options = flow_model(settings, options)
    case.configure_numerics(settings, options)
    speed = mesh["swirl_direction"] * cycle["blade_speed"]
    frame = settings.setup.cell_zone_conditions.fluid[names[ROTOR]].reference_frame
    frame.frame_motion = True
    frame.reference_frame_velocity = [speed, 0.0]
    settings.setup.mesh_interfaces.create_manually(name="stator_rotor", zone_list_1=[zones[f"{STATOR}_outlet"]],
                                                   zone_list_2=[zones[f"{ROTOR}_inlet"]])
    inlet = settings.setup.boundary_conditions.pressure_inlet[zones[f"{STATOR}_inlet"]]
    inlet.momentum.gauge_total_pressure.value = state["P0"]
    inlet.momentum.supersonic_or_initial_gauge_pressure.value = state["P0"]
    inlet.momentum.direction_specification_method = "Normal to Boundary"
    if options.get("viscous", True):
        turbulence(inlet, options)
    outlet_boundary(settings, zones, cycle["outlet_pressure"], options)
    settings.mesh.check()
    return {"cell_zones": names, "boundaries": zones, "blade_speed_m_s": speed}


def mass_flow_inflow(inflow, mesh):
    """The design relative inflow of one rotor passage, per metre of depth, and its planar direction."""
    beta = np.radians(inflow["beta"])
    return {"p": inflow["p"], "W": inflow["W"],
            "direction": [mesh["swirl_direction"] * np.sin(beta), -np.cos(beta)],
            "mass_flow": inflow["rho"] * inflow["W"] * np.cos(beta) * mesh["rotor_pitch_m"]}


def configure_rotor(solver, mesh, state, expressions, cycle, options):
    """The rotor alone, at rest, fed through a mass-flow inlet with the stator's design flow and direction.

    A choked stator imposes the mass flow; a pressure inlet would instead lose total pressure as soon as
    the inflow turns subsonic (Fluent relates p0 and p incompressibly for a pressure-defined density).
    """
    settings = solver.settings
    rows = (ROTOR,)
    zones = import_mesh(settings, mesh["mesh_file"], rows)
    names = case.configure_material(settings, expressions, options["material_name"])
    modify = settings.mesh.modify_zones
    modify.zone_type(zone_names=[zones[f"{ROTOR}_inlet"]], new_type="mass-flow-inlet")
    modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
    translational_periodicity(settings, zones, rows, mesh["rotor_pitch_m"])
    options = flow_model(settings, options)
    case.configure_numerics(settings, options)
    inflow = mass_flow_inflow(cycle["relative_inflow"], mesh)
    inlet = settings.setup.boundary_conditions.mass_flow_inlet[zones[f"{ROTOR}_inlet"]]
    inlet.momentum.mass_flow_specification = "Mass Flow Rate"
    inlet.momentum.mass_flow_rate.value = inflow["mass_flow"]
    inlet.momentum.supersonic_gauge_pressure.value = inflow["p"]
    inlet.momentum.direction_specification = "Direction Vector"
    for component, value in zip(inlet.momentum.flow_direction, inflow["direction"]):
        component.value = value
    if options.get("viscous", True):
        turbulence(inlet, options)
    outlet_boundary(settings, zones, cycle["outlet_pressure"], options)
    settings.mesh.check()
    return {"cell_zones": names, "boundaries": zones, "inflow": inflow}


def solve_rotor(solver, inflow, options):
    """Start from the uniform design inflow, then the staged strategy."""
    from ..common import set_order
    settings = solver.settings
    initialization = settings.solution.initialization
    initialization.initialization_type = "standard"
    initialization.defaults["pressure"] = inflow["p"]
    for component, value in zip(VELOCITIES, inflow["direction"]):
        initialization.defaults[component] = value * inflow["W"]
    initialization.standard_initialize()
    for stage in options["solution_strategy"]:
        set_order(settings, first_order=stage["order"] == "first")
        settings.solution.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor = (
            stage["time_scale_factor"])
        print(f"    {stage['iterations']} iterations, {stage['order']} order", flush=True)
        settings.solution.run_calculation.iterate(iter_count=stage["iterations"])


def extract(solver, configured, path):
    """Node and face values on the interior, flow and blade surfaces, the mass flows and the tangential blade forces.

    Face normals are not requested: Fluent 24.2 crashes when they are asked for on a 2D interface zone.
    """
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest
    settings = solver.settings
    data = solver.fields.field_data
    types = case.boundary_types(settings.setup.boundary_conditions)
    surfaces = [name for name, kind in types.items()
                if ("interior" in name and kind == "interior") or name in configured["boundaries"].values()]
    arrays = {}
    for surface in surfaces:
        if "_periodic_" in surface:
            continue
        mesh = data.get_field_data(SurfaceFieldDataRequest(
            surfaces=[surface], data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesCentroid]))[surface]
        arrays[f"{surface}|vertices"] = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
        arrays[f"{surface}|centroids"] = np.asarray(mesh.face_centroids, dtype=float).reshape(-1, 3)
        for location, node_value in (("node", True), ("face", False)):
            for field in FIELDS:
                arrays[f"{surface}|{location}|{field}"] = np.asarray(data.get_field_data(ScalarFieldDataRequest(
                    field_name=field, surfaces=[surface], node_value=node_value))[surface], dtype=float)
    definitions = settings.solution.report_definitions
    reports = {}
    for row, key in (("stator_1", f"{STATOR}_blade"), ("rotor_1", f"{ROTOR}_blade")):
        if key in configured["boundaries"]:
            name = f"planar_force_{row}"
            definitions.force.create(name=name)
            definitions.force[name].zones = [configured["boundaries"][key]]
            definitions.force[name].force_vector = [1.0, 0.0]
            reports[name] = f"force|{row}"
    for key in (f"{STATOR}_inlet", f"{ROTOR}_inlet", f"{ROTOR}_outlet"):
        zone = configured["boundaries"].get(key)
        if zone and types.get(zone) != "interface":
            name = f"planar_mass_{key}"
            definitions.surface.create(name=name)
            definitions.surface[name].report_type = "surface-massflowrate"
            definitions.surface[name].surface_names = [zone]
            reports[name] = f"mass_flow|{key}"
    values = definitions.compute(report_defs=list(reports))
    values = {key: value[0] for item in values for key, value in item.items()}
    for name, label in reports.items():
        arrays[label] = np.array(values[name], dtype=float)
    np.savez_compressed(path, **arrays)
    return path


def summary(design, mesh, cycle):
    rotor = design["rotor"]
    return {"stator_exit": design["stator_exit"], "relative_inflow": design["relative_inflow"],
            "rotor": {"chord_mm": rotor["chord"], "passages": rotor["passages"], "pitch_chord": rotor["pitch_chord"],
                      "beta_inlet": rotor["blade"]["beta_inlet"], "beta_outlet": rotor["blade"]["beta_outlet"]},
            "stator": {"pitch_mm": float(design["stator"]["design"].blade["pitch"])} if "design" in design["stator"] else None,
            "cycle": {key: value for key, value in cycle.items() if key != "relative_inflow"},
            "mesh": {key: str(value) if isinstance(value, Path) else value for key, value in mesh.items()}}


def extract_case(config_file, name):
    """Re-read a solved planar case and extract its fields again, without redesigning or solving."""
    from ..common import load_dependencies
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    fluent_dir = (config_file.parent / workflow["output_dir"]).resolve() / "fluent"
    configured = json.loads((fluent_dir / f"{name}_summary.json").read_text(encoding="utf-8"))
    _, pyfluent = load_dependencies(False)
    options = workflow["fluent"]
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=2,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode="no_gui", start_timeout=options["start_timeout"], cwd=str(fluent_dir))
    try:
        solver.settings.file.read_case_data(file_name=str(fluent_dir / f"{name}.cas.h5"))
        from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
        for zone, kind in case.boundary_types(solver.settings.setup.boundary_conditions).items():
            if "non-overlapping" in zone:
                points = solver.fields.field_data.get_field_data(SurfaceFieldDataRequest(
                    surfaces=[zone], data_types=[SurfaceDataType.Vertices]))[zone].vertices
                points = np.asarray(points, dtype=float).reshape(-1, 3)
                extent = np.ptp(points[:, 0]) * 1e3 if len(points) else 0.0
                print(f"    {zone}: {len(points)} vertices, x extent {extent:.4f} mm", flush=True)
        print(f"    Extracted: {extract(solver, configured, fluent_dir / f'{name}_fields.npz')}", flush=True)
    finally:
        solver.exit()


def main(config_file, mode="stage"):
    if mode.endswith("-extract"):
        return extract_case(config_file, "rotor_2d" if mode.startswith("rotor") else "stage_2d")
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    source = yaml.safe_load((config_file.parent / workflow["project_file"]).read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rotor_only = mode == "rotor"
    name = "rotor_2d" if rotor_only else "stage_2d"

    print("1/5 Goldman vortex rotor (and MoC stator) and their planar passages", flush=True)
    prescribed = rotor_only and "relative_inflow" in workflow
    design = (planar_design.design_rotor_only if prescribed else planar_design.design_stage)(source, workflow)
    mesh = build_mesh(output_dir, design, rotor_only)
    cycle = {"blade_speed": float(workflow["blade_speed"]), "outlet_pressure": float(workflow["outlet_pressure"]),
             "relative_inflow": design["relative_inflow"]}
    inflow = design["relative_inflow"]
    print(f"    rotor inflow M_rel {inflow['M']:.3f} at {inflow['beta']:.2f} deg (axial Mach {inflow['axial_mach']:.3f}); "
          f"rotor chord {design['rotor']['chord']:.2f} mm, {design['rotor']['passages']} passages per stator pitch; "
          f"cells {mesh['cells']}", flush=True)

    from ..barotropic_model import generate_barotropic_model
    from ..common import load_dependencies
    barotropy, pyfluent = load_dependencies(False)
    print("2/5 Barotropic model along the stator inlet isentrope", flush=True)
    parameters = dict(design["stator"]["config"]["method_of_characteristics"])
    if "backend" in workflow["barotropic"]:
        parameters["backend"] = workflow["barotropic"]["backend"]
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, design["stator"]["nozzle"],
                                                   workflow["barotropic"], output_dir / "barotropy")

    options = workflow["fluent"]
    print(f"3/5 Launch Fluent 2D ({options['ui_mode']})", flush=True)
    fluent_dir = output_dir / "fluent"
    fluent_dir.mkdir(exist_ok=True)
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=2,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode=options["ui_mode"], start_timeout=options["start_timeout"],
                                    cwd=str(fluent_dir), cleanup_on_exit=False)
    print(f"4/5 Configure the planar {'rotor' if rotor_only else 'stage'}", flush=True)
    configured = (configure_rotor if rotor_only else configure_stage)(solver, mesh, state, expressions, cycle, options)
    case_file = fluent_dir / f"{name}.cas.h5"
    solver.settings.file.batch_options.confirm_overwrite = False
    solver.settings.file.write_case(file_name=str(case_file))
    (fluent_dir / f"{name}_summary.json").write_text(json.dumps(
        {**configured, **summary(design, mesh, cycle), "inlet_state": state, "case_file": str(case_file)},
        indent=2, default=float), encoding="utf-8")
    print(f"5/5 Case saved: {case_file}", flush=True)
    if options.get("solve", False):
        if rotor_only:
            solve_rotor(solver, configured["inflow"], options)
        else:
            solve(solver, options, VELOCITIES)
        solver.settings.file.write_case_data(file_name=str(case_file))
        extract(solver, configured, fluent_dir / f"{name}_fields.npz")
        print(f"    Solved, saved and extracted: {case_file}", flush=True)
        if options["ui_mode"] == "no_gui":
            solver.exit()
    return case_file


if __name__ == "__main__":
    main(*sys.argv[1:])
