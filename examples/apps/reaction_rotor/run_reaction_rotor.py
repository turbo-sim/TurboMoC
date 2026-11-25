"""Subsonic reaction rotor (TurboDash camberline + modified NACA thickness) through the normal TurboMoC app jobs.

    python examples/apps/reaction_rotor/run_reaction_rotor.py [baseline|shifted|all]

Axial rotor: 15 mm meridional chord, 50 mm mean radius, 10 mm span, 24 blades. TurboDash angles 0 -> 60 deg,
max full thickness 0.12 c at 0.30 c, LE radius 0.01 c, TE thickness 0.015 c, 8 deg TE wedge, finite cut-off.
Cases: baseline (half-turning location 0.52, concentration 1.2) and shifted (0.65, 2.0).

For each case: write the project YAML (reproducible case file), reload it, run the rotor blade job, the 2D reference
mesh job and the 3D volume job exactly as the studio does (flow.start_job -> worker subprocess). The volume job
reloads the persisted reference section. Results (job handles, mesh summaries) go to output/<case>/summary.json.
"""
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import yaml

from turbo_moc.app import workflow as flow
from turbo_moc.configuration import project as projects

HERE = Path(__file__).resolve().parent
CASES = {"baseline": (0.52, 1.2), "shifted": (0.65, 2.0)}
REACTION = {"inlet_metal_angle_deg": 0.0, "outlet_metal_angle_deg": 60.0, "maximum_thickness": 0.12,
            "maximum_thickness_location": 0.30, "leading_edge_radius": 0.01, "trailing_edge_thickness": 0.015,
            "trailing_edge_wedge_angle_deg": 8.0, "pitch_to_chord": 0.8727, "num_points": 241}


def build_project(location, concentration):
    project = projects.set_design_mode(projects.default_project("axial"), "rotor")
    row = projects.get_row(project, "rotor_1")
    row["geometry"]["meridional_channel"].update(hub_radius=45.0, blade_height=10.0, hub_chord=15.0,
                                                 tip_chord=15.0, num_blades=24)
    row["geometry"]["inlet_coupling"]["mode"] = "manual"
    design = row["geometry"]["rotor_design"]
    design["blade_type"] = "reaction"
    design["reaction"] = dict(REACTION, turning_location=location, turning_concentration=concentration)
    numerics = row["mesh"]["numerics"]
    numerics.update(blade_size=0.15, domain_size=0.6, span_layers=20, span_stretch=1.5)
    numerics["inflation_layers"].update(first_height=0.01, growth_ratio=1.2, thickness=0.3)
    row["mesh"]["cgns"]["filename"] = "reaction_rotor_3d.cgns"
    return projects.validate_project(projects.normalize_project(project))


def wait(handle):
    while True:
        state, log = flow.poll_job(handle)
        if state["state"] != "running":
            if state["state"] != "complete":
                raise RuntimeError(f"{handle['action']} failed: {state['message']}\n{log[-3000:]}")
            return flow.read_json(flow.directory(handle) / "result.json")
        time.sleep(2)


def run_case(name):
    location, concentration = CASES[name]
    out = HERE / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    flow.OUTPUT = out / "jobs"
    case_file = HERE / f"reaction_rotor_{name}.yaml"
    case_file.write_text(yaml.safe_dump(build_project(location, concentration), sort_keys=False), encoding="utf-8")
    project = projects.load_project(case_file)  # Saved and reloaded, as from the studio
    config = flow.resolved_configuration(project, "rotor_1", {})
    session, registry, timings = uuid4().hex, {}, {}
    for action, refs in (("nozzle", lambda: {}),
                         ("reference", lambda: {"nozzle": registry["nozzle"]}),
                         ("volume", lambda: {"nozzle": registry["nozzle"], "reference": registry["reference"]})):
        start = time.time()
        handle = flow.start_job(session, action, config, refs(), project=project, row_id="rotor_1",
                                row_role="rotor", registry={})
        result = wait(handle)
        timings[action] = time.time() - start
        registry[action] = {key: handle[key] for key in ("session", "job", "key", "row_id")}
        registry[action + "_result"] = result
        print(f"[{name}] {action} done in {timings[action]:.0f} s", flush=True)
    keys = flow.keys(config, registry["nozzle"])
    edited = json.loads(json.dumps(config))
    edited["rotor_design"]["reaction"]["maximum_thickness"] += 0.01  # A geometry edit
    stale = flow.keys(edited, registry["nozzle"])
    summary = {"case": name, "turning_location": location, "turning_concentration": concentration,
               "jobs": {k: registry[k] for k in ("nozzle", "reference", "volume")}, "timings_s": timings,
               "volume_result": registry["volume_result"], "reference_result": registry["reference_result"],
               "blade_result": registry["nozzle_result"],
               "volume_current": flow.current(registry["volume"], keys["volume"]),
               "volume_current_after_geometry_edit": flow.current(registry["volume"], stale["volume"]),
               "blade_current_after_geometry_edit": flow.current(registry["nozzle"], flow.keys(edited)["nozzle"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("volume_current", "volume_current_after_geometry_edit",
                                             "blade_current_after_geometry_edit")}))
    return summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for case in (CASES if which == "all" else [which]):
        run_case(case)
