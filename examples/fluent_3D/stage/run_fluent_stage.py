"""Design an axial stator + rotor stage with the app pipeline and open a configured 3D Fluent case.

The rotor inlet is coupled to the stator exit at a prescribed speed. The app's
own jobs compute the blades and the 3D meshes, and its project export writes the
combined stage CGNS. The Fluent case (barotropic material, rotating rotor zone,
mixing plane, rotational periodicity) is saved and left open, unsolved.

Run from the repository root:
    python examples/fluent_3D/stage/run_fluent_stage.py
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_stage.yaml"


def run_jobs(project, actions, session, registry):
    from turbo_moc.app import calculations
    from turbo_moc.app import workflow as flow
    for action in actions:
        handle = calculations.start(session, project, "all", action, registry)
        while handle:
            while True:
                status, log = flow.poll_job(handle)
                if status["state"] != "running":
                    break
                time.sleep(2)
            print(f"    {action} {handle['row_id']}: {status['state']} - {status['message'][-90:]}")
            if status["state"] != "complete":
                raise RuntimeError(log[-3000:])
            result = json.loads((flow.directory(handle) / "result.json").read_text(encoding="utf-8"))
            registry.setdefault(handle["row_id"], {})[action] = result["reference"]
            handle = calculations.advance(handle, registry, log)


def operating_point(project, registry):
    from turbo_moc.app import design_statistics as statistics
    from turbo_moc.app import workflow as flow
    rows = {row["id"]: row for row in project["rows"]}
    stator_config = flow.resolved_configuration(project, "stator_1", registry)
    rotor_config = flow.resolved_configuration(project, "rotor_1", registry)
    nozzle = flow.load_nozzle(registry["stator_1"]["nozzle"], stator_config)
    rotor_result = flow.load_nozzle(registry["rotor_1"]["nozzle"], rotor_config)
    stator = statistics.stator_statistics(project, rows["stator_1"], stator_config, nozzle)
    rotor = statistics.rotor_statistics(project, rows["rotor_1"], rotor_config, rotor_result, stator)
    exit_station = next(item for item in rotor["stations"] if item["station"].startswith("Rotor exit"))
    source = rotor_config["rotor_source"]
    design = {"stator": {"stations": stator["stations"], "performance": stator["performance"],
                         "mass_flow": stator["mass_flow"]},
              "rotor": {"stations": rotor["stations"], "performance": rotor["performance"]},
              "num_blades": {row: int(flow.resolved_configuration(project, row, registry)["meridional_channel"]
                                      ["num_blades"]) for row in ("stator_1", "rotor_1")}}
    return {"stator_config": stator_config, "nozzle": nozzle,
            "rotor_exit_pressure": float(exit_station["p"]), "rpm": float(source["rotational_speed"]["rpm"]),
            "relative_inlet_mach": float(source["triangle"]["M_rel"]),
            "relative_inlet_angle": float(source["triangle"]["beta_rel_deg"]),
            "stator_exit_angle": float(source["alpha_deg"]), "stage": rotor["stage"], "design": design}


def registry_from_jobs(jobs_dir):
    """Rebuild the app registry of the latest session from its saved job results."""
    sessions = [path for path in Path(jobs_dir).iterdir() if any(path.glob("*/result.json"))]
    session = max(sessions, key=lambda path: max(item.stat().st_mtime for item in path.glob("*/result.json")))
    registry = {}
    for result_file in sorted(session.glob("*/result.json"), key=lambda path: path.stat().st_mtime):
        reference = json.loads(result_file.read_text(encoding="utf-8"))["reference"]
        registry.setdefault(reference["row_id"], {})[reference["action"]] = reference
    return registry


def write_operating_point(output_dir, project, registry):
    point = operating_point(project, registry)
    point["swirl_sign"] = swirl_sign(project, registry)
    summary = {key: value for key, value in point.items() if key not in ("stator_config", "nozzle")}
    (output_dir / "operating_point.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return point, summary


def swirl_sign(project, registry):
    """+1 when the stator exit swirl turns counterclockwise about +z in the exported frame."""
    from turbo_moc.app import workflow as flow
    config = flow.resolved_configuration(project, "stator_1", registry)
    design = flow.get_design(config, registry["stator_1"]["nozzle"])
    curve = np.asarray(design.curves["periodic_left"])
    mapping = design.meridional_channel
    xyz = mapping.map_points(curve[-2:], 0.5)
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    return 1.0 if np.unwrap(theta)[-1] > np.unwrap(theta)[0] else -1.0


def prepare(workflow, output_dir):
    from turbo_moc.app import project_export
    from turbo_moc.app import workflow as flow
    from turbo_moc.configuration import project as model

    flow.OUTPUT = output_dir / "jobs"
    flow.OUTPUT.mkdir(parents=True, exist_ok=True)
    project = model.set_design_mode(model.default_project(workflow["geometry_type"]), "stator_rotor")
    rotor = next(row for row in project["rows"] if row["role"] == "rotor")
    rotor["geometry"]["inlet_coupling"] = {"mode": "stator_exit",
                                           "rotational_speed": {"type": "actual", "value": float(workflow["rpm"])}}
    project = model.validate_project(model.normalize_project(project))
    session, registry = uuid4().hex, {}
    print("1/5 Blade designs and 3D meshes (app jobs)")
    run_jobs(project, ("nozzle", "reference", "volume"), session, registry)
    data, _ = project_export.build_archive(project, registry)
    mesh_dir = output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(mesh_dir)
    (output_dir / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return project, registry


STATOR, ROTOR = "stator_1_fluid", "rotor_1_fluid"


def zone(types, row, name):
    return next(key for key in types if key.endswith(f"{row}_{name}"))


def create_mixing_plane(solver, name, zone1, zone2):
    """General turbo interface with the mixing-plane pitch change (choice 2 of turbo-create)."""
    settings = solver.settings
    settings.setup.turbo_models.enabled = True
    solver.execute_tui(f"/define/turbo-model/turbo-create {name} {zone1} () {zone2} () 2")
    if name not in settings.setup.mesh_interfaces.interface.get_object_names():
        raise RuntimeError(f"Fluent did not create the mixing-plane interface {name!r}.")


def configure_stage(solver, mesh_file, state, expressions, point, options):
    from turbo_moc.fluent.common import boundary_types, configure_numerics, MACH_EXPRESSION
    settings = solver.settings
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    general = settings.setup.general
    general.solver.type = "pressure-based"
    general.solver.time = "steady"
    general.operating_conditions.operating_pressure = 0.0
    general.operating_conditions.gravity.enable = False
    models = settings.setup.models
    models.energy.enabled = False
    models.viscous.model = "k-omega"
    models.viscous.k_omega_model = "sst"
    for name, definition in expressions.items():
        settings.setup.named_expressions[name] = {"definition": definition}
    settings.setup.named_expressions[MACH_EXPRESSION] = {"definition": "VelocityMagnitude / barotropic_speed_of_sound"}
    material = settings.setup.materials.fluid.create(name=options["material_name"])
    material.density = {"option": "expression", "expression": "barotropic_density"}
    material.viscosity = {"option": "expression", "expression": "barotropic_viscosity"}
    fluids = settings.setup.cell_zone_conditions.fluid
    names = {row: next(name for name in fluids.get_object_names() if name.startswith(row))
             for row in (STATOR, ROTOR)}
    for name in names.values():
        fluids[name].general.material = options["material_name"]

    types = boundary_types(settings.setup.boundary_conditions)
    zones = {f"{row}_{name}": zone(types, row.split("_fluid")[0], name)
             for row in (STATOR, ROTOR) for name in ("inlet", "outlet", "blade", "hub", "shroud",
                                                    "periodic_left", "periodic_right")}
    modify = settings.mesh.modify_zones
    modify.zone_type(zone_names=[zones[f"{STATOR}_inlet"]], new_type="pressure-inlet")
    modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
    modify.zone_type(zone_names=[zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"]], new_type="interface")
    for row in (STATOR, ROTOR):
        modify.make_periodic(zone_name=zones[f"{row}_periodic_left"], shadow_zone_name=zones[f"{row}_periodic_right"],
                             rotate_periodic=True, create=True, auto_translation=True)
    configure_numerics(settings, options)

    omega = point["swirl_sign"] * point["rpm"] * 2 * np.pi / 60
    rotor_frame = fluids[names[ROTOR]].reference_frame
    rotor_frame.frame_motion = True
    rotor_frame.mrf_omega.value = omega
    create_mixing_plane(solver, "stator_rotor_mixing_plane", zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"])

    bc = settings.setup.boundary_conditions
    inlet = bc.pressure_inlet[zones[f"{STATOR}_inlet"]]
    inlet.momentum.gauge_total_pressure.value = state["P0"]
    inlet.momentum.supersonic_or_initial_gauge_pressure.value = state["P0"]
    inlet.momentum.direction_specification_method = "Normal to Boundary"
    inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    inlet.turbulence.turbulent_intensity = options["turbulence_intensity"]
    inlet.turbulence.turbulent_viscosity_ratio = options["turbulence_viscosity_ratio"]
    outlet = bc.pressure_outlet[zones[f"{ROTOR}_outlet"]]
    outlet.momentum.gauge_pressure.value = point["rotor_exit_pressure"]
    outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    outlet.turbulence.backflow_turbulent_intensity = options["turbulence_intensity"]
    outlet.turbulence.backflow_turbulent_viscosity_ratio = options["turbulence_viscosity_ratio"]
    shroud = bc.wall[zones[f"{ROTOR}_shroud"]].momentum
    shroud.wall_motion = "Moving Wall"
    shroud.relative = False
    shroud.rotating = False
    shroud.speed.value = 0.0

    settings.mesh.check()
    settings.solution.initialization.initialization_type = "hybrid"
    settings.solution.run_calculation.iter_count = options["solution_strategy"][0]["iterations"]
    return {"cell_zones": names, "boundaries": zones, "rotor_omega_rad_s": omega}


def main(config_file=CONFIG_FILE):
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project, registry = prepare(workflow, output_dir)
    point, summary = write_operating_point(output_dir, project, registry)
    print("    operating point:", json.dumps({key: value for key, value in summary.items() if key != "design"},
                                              default=float)[:400])

    from turbo_moc.fluent.barotropic_model import generate_barotropic_model
    from turbo_moc.fluent.common import load_dependencies
    barotropy, pyfluent = load_dependencies(False)
    print("2/5 Barotropic model along the stator inlet isentrope")
    parameters = dict(point["stator_config"]["method_of_characteristics"])
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, point["nozzle"],
                                                   workflow["barotropic"], output_dir / "barotropy")
    print("3/5 Launch Fluent (GUI) and import the stage mesh")
    options = workflow["fluent"]
    fluent_dir = output_dir / "fluent"
    fluent_dir.mkdir(exist_ok=True)
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=3,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode=options["ui_mode"], start_timeout=options["start_timeout"],
                                    cwd=str(fluent_dir), cleanup_on_exit=False)
    print("4/5 Configure materials, rotating rotor zone, mixing plane, periodicity and boundaries")
    setup = configure_stage(solver, output_dir / "mesh" / "stage.cgns", state, expressions, point, options)
    case_file = fluent_dir / options["case_filename"]
    solver.settings.file.batch_options.confirm_overwrite = False
    solver.settings.file.write_case(file_name=str(case_file))
    (fluent_dir / "setup_summary.json").write_text(json.dumps({**setup, "inlet_state": state, **summary,
                                                             "case_file": str(case_file)}, indent=2, default=float),
                                                 encoding="utf-8")
    print(f"5/5 Case saved: {case_file}")
    print("    Fluent is open: check the setup, initialize (hybrid) and run the calculation.")


if __name__ == "__main__":
    main(*sys.argv[1:])
