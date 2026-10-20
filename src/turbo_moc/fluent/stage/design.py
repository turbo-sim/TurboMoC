"""App pipeline for the stage: blade designs, 3D meshes, stage CGNS and the MoC design point."""

import io
import json
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

ROWS = ("stator_1", "rotor_1")


def run_jobs(project, actions, session, registry):
    """Run the app jobs row by row; returns the blade sizing reported by each row's design job."""
    from turbo_moc.app import calculations
    from turbo_moc.app import workflow as flow
    sizing = {}
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
            if "blade_sizing" in result:
                sizing[handle["row_id"]] = result["blade_sizing"]
            handle = calculations.advance(handle, registry, log)
    return sizing


def merge(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value
    return target


def stage_project(workflow):
    from turbo_moc.configuration import project as model
    if "project_file" in workflow:
        project = yaml.safe_load(Path(workflow["project_file"]).read_text(encoding="utf-8"))
    else:
        project = model.set_design_mode(model.default_project(workflow["geometry_type"]), "stator_rotor")
    rotor = next(row for row in project["rows"] if row["role"] == "rotor")
    rotor["geometry"]["inlet_coupling"] = {"mode": "stator_exit",
                                           "rotational_speed": {"type": "actual", "value": float(workflow["rpm"])}}
    rotor["geometry"]["axial_interspace_mm"] = float(workflow["axial_interspace_mm"])
    if "fluid_name" in workflow:
        project["machine"]["fluid_name"] = workflow["fluid_name"]
    for row in project["rows"]:
        merge(row, workflow.get("rows", {}).get(row["id"], {}))
    return model.validate_project(model.normalize_project(project))


def with_suggested_blade_counts(project, sizing):
    """Use each row's natural blade count, so the passages keep the pitch/chord of their own design."""
    from turbo_moc.configuration import project as model
    for row in project["rows"]:
        if row["id"] in sizing:
            row["geometry"]["meridional_channel"]["num_blades"] = int(sizing[row["id"]]["suggested_num_blades"])
    return model.validate_project(model.normalize_project(project))


def prepare(workflow, output_dir):
    """Run the app jobs and export the combined stage mesh to ``output_dir/mesh``."""
    from turbo_moc.app import project_export
    from turbo_moc.app import workflow as flow

    flow.OUTPUT = output_dir / "jobs"
    flow.OUTPUT.mkdir(parents=True, exist_ok=True)
    project = stage_project(workflow)
    registry = {}
    if workflow.get("blade_counts") == "suggested":
        sizing = run_jobs(project, ("nozzle",), uuid4().hex, registry)
        project = with_suggested_blade_counts(project, sizing)
        print("    blade counts:", {row["id"]: row["geometry"]["meridional_channel"]["num_blades"]
                                    for row in project["rows"]})
        registry = {}
    run_jobs(project, ("nozzle", "reference", "volume"), uuid4().hex, registry)
    data, _ = project_export.build_archive(project, registry)
    mesh_dir = output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(mesh_dir)
    (output_dir / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return project, registry


def registry_from_jobs(jobs_dir):
    """Rebuild the app registry of the latest session from its saved job results."""
    sessions = [path for path in Path(jobs_dir).iterdir() if any(path.glob("*/result.json"))]
    session = max(sessions, key=lambda path: max(item.stat().st_mtime for item in path.glob("*/result.json")))
    registry = {}
    for result_file in sorted(session.glob("*/result.json"), key=lambda path: path.stat().st_mtime):
        reference = json.loads(result_file.read_text(encoding="utf-8"))["reference"]
        registry.setdefault(reference["row_id"], {})[reference["action"]] = reference
    return registry


def operating_point(project, registry):
    from turbo_moc.app import design_statistics as statistics
    from turbo_moc.app import workflow as flow
    rows = {row["id"]: row for row in project["rows"]}
    configs = {row: flow.resolved_configuration(project, row, registry) for row in ROWS}
    nozzle = flow.load_nozzle(registry["stator_1"]["nozzle"], configs["stator_1"])
    rotor_result = flow.load_nozzle(registry["rotor_1"]["nozzle"], configs["rotor_1"])
    stator = statistics.stator_statistics(project, rows["stator_1"], configs["stator_1"], nozzle)
    rotor = statistics.rotor_statistics(project, rows["rotor_1"], configs["rotor_1"], rotor_result, stator)
    exit_station = next(item for item in rotor["stations"] if item["station"].startswith("Rotor exit"))
    source = configs["rotor_1"]["rotor_source"]
    design = {"stator": {"stations": stator["stations"], "performance": stator["performance"],
                         "mass_flow": stator["mass_flow"]},
              "rotor": {"stations": rotor["stations"], "performance": rotor["performance"]},
              "num_blades": {row: int(configs[row]["meridional_channel"]["num_blades"]) for row in ROWS}}
    return {"stator_config": configs["stator_1"], "nozzle": nozzle,
            "rotor_exit_pressure": float(exit_station["p"]), "rpm": float(source["rotational_speed"]["rpm"]),
            "relative_inlet_mach": float(source["triangle"]["M_rel"]),
            "relative_inlet_angle": float(source["triangle"]["beta_rel_deg"]),
            "stator_exit_angle": float(source["alpha_deg"]), "stage": rotor["stage"], "design": design}


def swirl_sign(project, registry):
    """+1 when the stator exit swirl turns counterclockwise about +z in the exported frame."""
    from turbo_moc.app import workflow as flow
    config = flow.resolved_configuration(project, "stator_1", registry)
    design = flow.get_design(config, registry["stator_1"]["nozzle"])
    curve = np.asarray(design.curves["periodic_left"])
    xyz = design.meridional_channel.map_points(curve[-2:], 0.5)
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    return 1.0 if np.unwrap(theta)[-1] > np.unwrap(theta)[0] else -1.0


def write_operating_point(output_dir, project, registry):
    """Write the MoC design point, the stage statistics and the swirl sign to ``operating_point.json``."""
    point = operating_point(project, registry)
    point["swirl_sign"] = swirl_sign(project, registry)
    summary = {key: value for key, value in point.items() if key not in ("stator_config", "nozzle")}
    (output_dir / "operating_point.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return point, summary
