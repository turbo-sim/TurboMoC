"""Stage invalidation, worker provenance and asynchronous result publication."""
from copy import deepcopy
from pathlib import Path
import json
import time

import pytest

from turbo_moc.apps.app import workflow as flow
from turbo_moc import configuration_project as projects
from app_helpers import call, row_state

CASE_DIR = Path(__file__).resolve().parents[1] / "examples/apps/cases"
CASES = {n: CASE_DIR / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}


def test_stage_dependencies_exclude_span_and_view_but_include_cgns_output():
    config = flow.read_configuration(CASES["axial_stator"])
    original = flow.keys(config)
    for path, value in ((("mesh", "pyvista", "show_mesh_edges"), False),
                        (("mesh", "visualization", "section_edge_width"), 0.9),
                        (("mesh", "visualization", "volume_edge_width"), 2.0)):
        modified = deepcopy(config)
        target = modified
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        assert flow.keys(modified) == original
    modified = deepcopy(config)
    modified["mesh"]["cgns"]["filename"] = "other.cgns"
    assert flow.keys(modified)["reference"] != original["reference"]
    assert flow.keys(modified)["geometry"] == original["geometry"]
    config["mesh"]["numerics"]["span_layers"] += 1
    changed = flow.keys(config)
    assert changed["reference"] == original["reference"]
    assert changed["volume"] != original["volume"]
    config["mesh"]["numerics"]["blade_size"] *= 2
    assert flow.keys(config)["reference"] != changed["reference"]
    config["meridional_channel"]["num_blades"] += 1
    assert flow.keys(config)["geometry"] != original["geometry"]


def test_new_moc_job_invalidates_downstream_artifacts_with_unchanged_inputs(small_design):
    _, config = small_design
    first = {"session": "a" * 32, "job": "b" * 32}
    second = {"session": "a" * 32, "job": "c" * 32}
    before, after = flow.keys(config, first), flow.keys(config, second)
    assert before["nozzle"] == after["nozzle"]
    assert all(before[n] != after[n] for n in ("geometry", "reference", "volume"))


def test_each_moc_worker_solves_and_hands_off_only_its_own_result(
        small_design, tmp_path, monkeypatch):
    from turbo_moc.apps.app import worker
    nozzle, config = small_design
    calls = []
    def solve(parameters, *, progress_callback):
        calls.append(parameters)
        progress_callback(0.5, "3 of 4")
        return nozzle
    monkeypatch.setattr(worker, "solve_nozzle", solve)
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    for job in ("b" * 32, "c" * 32):
        handle = {"session": "a" * 32, "job": job, "action": "nozzle",
                  "key": flow.keys(config)["nozzle"], "started": time.time()}
        folder = flow.directory(handle)
        folder.mkdir(parents=True)
        flow.write_json(folder / "request.json", {
            "handle": handle, "config": config, "references": {},
            "source_revision": flow.SOURCE_REVISION})
        assert worker.run(folder / "request.json") == 0
        assert flow.load_nozzle(handle, config) == nozzle
        assert json.loads((folder / "status.json").read_text())["state"] == "complete"
        assert not (folder / "nozzle_solution.json").exists()
    assert calls == [config["method_of_characteristics"]] * 2
    config["method_of_characteristics"]["n"] += 1
    with pytest.raises(ValueError, match="current inputs"):
        flow.load_nozzle(handle, config)


def test_job_service_requires_current_2d_before_launching_volume(small_design, monkeypatch):
    _, config = small_design
    monkeypatch.setattr(flow, "load_nozzle", lambda *args: {})
    with pytest.raises(ValueError, match="up-to-date 2D"):
        flow.start_job("a" * 32, "volume", config, {"nozzle": {}, "reference": None})
    old = {"key": "previous-section-settings"}
    with pytest.raises(ValueError, match="up-to-date 2D"):
        flow.start_job("a" * 32, "volume", config, {"nozzle": {}, "reference": old})


def test_submission_and_polling_keep_separate_outputs(small_design, monkeypatch):
    """A periodic poll response cannot replace/discard a submitted worker handle."""
    from turbo_moc.apps.app import create_app
    _, config = small_design
    app = create_app()
    handle = {"session": "a" * 32, "job": "b" * 32, "action": "nozzle",
              "row_id": "stator_1", "row_role": "stator",
              "started": time.time(), "key": flow.keys(config)["nozzle"]}
    monkeypatch.setattr(flow, "start_job", lambda *args, **kwargs: handle)
    launch_key = next(key for key in app.callback_map if key.startswith("..job.data"))
    metadata = app.callback_map[launch_key]
    assert not any(item["id"] == "poll" for item in metadata["inputs"])
    state = row_state(config)
    values = {("config", "data"): state, ("session", "data"): "a" * 32,
        ("row-references", "data"): {}, ("base-project", "data"): state["project"],
        ("calculation-scope", "value"): ["stator_1"], ("nozzle", "n_clicks"): 1}
    result = call(app, "..job.data", "nozzle.n_clicks", values)
    assert {key: value for key, value in result["job"]["data"].items() if key != "queue"} == handle
    assert result["job"]["data"]["queue"]["rows"] == ["stator_1"]
    monkeypatch.setattr(flow, "poll_job", lambda handle: (
        {"state": "running", "message": "Native Gmsh operation", "fraction": None,
         "started": handle["started"]}, "live output"))
    values[("job", "data")] = handle
    values[("poll", "n_intervals")] = 1
    result = call(app, "..row-references.data", "poll.n_intervals", values)
    assert "job" not in result
    assert result["spinner"]["className"] == "spinner running"
    assert result["activity"]["children"] == "live output"


def test_machine_family_switch_preserves_row_design_and_mesh_inputs():
    from turbo_moc.apps.app import create_app
    config = flow.read_configuration(CASES["axial_stator"])
    config["method_of_characteristics"]["n"] = 18
    config["flow_domain"]["outlet_chords"] = 2.7
    config["mesh"]["numerics"]["blade_size"] = 0.37
    app = create_app()
    state = row_state(config)
    key = next(k for k in app.callback_map if k.startswith("..base-project.data"))
    callback = app.callback_map[key]["callback"].__wrapped__
    result = callback("stator", "radial", state, state["project"], "stator_1")
    changed_project = result[0]
    changed = projects.row_configuration(changed_project, "stator_1")
    assert changed["meridional_channel"]["type"] == "radial"
    for name in ("method_of_characteristics", "blade_geometry", "flow_domain", "mesh"):
        assert changed[name] == config[name]
    # Obsolete axial-only controls may still arrive during the browser update.
    edited = flow.edited_project(changed_project, [20], [
        {"type": "parameter", "row": "stator_1", "path": "meridional_channel.hub_chord"}])
    assert "hub_chord" not in projects.row_configuration(edited, "stator_1")["meridional_channel"]
    assert edited == changed_project


@pytest.mark.parametrize("action,destination", [("reference","mesh"), ("volume","mesh")])
def test_only_successful_meshing_navigates_to_result(action, destination, tmp_path, monkeypatch):
    from turbo_moc.apps.app import create_app
    from dash import no_update
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    app = create_app()
    project = projects.default_project()
    key = next(k for k in app.callback_map if k.startswith("..row-references.data"))
    callback = app.callback_map[key]["callback"].__wrapped__
    handle = {"session": "a"*32, "job": "b"*32, "row_id": "stator_1", "row_role": "stator",
              "action": action, "started": time.time()}
    folder = flow.directory(handle)
    folder.mkdir(parents=True)
    flow.write_json(folder / "result.json", {"reference": handle})
    status = {"state": "running", "message": "Working", "fraction": None, "started": handle["started"]}
    monkeypatch.setattr(flow, "poll_job", lambda *args: (status, "live log"))
    for outcome in ("running", "failed", "cancelled"):
        status["state"] = outcome
        result = callback(1, handle, None, {}, project)
        assert result[6] is no_update and result[7] is no_update
    status.update(state="complete", fraction=1)
    completed = callback(1, handle, None, {}, project)
    assert completed[6] == destination and completed[7] == "stator_1"
    assert completed[0]["stator_1"][action] == handle
    repeated = callback(2, handle, handle["job"], completed[0], project)
    assert repeated[6] is no_update and repeated[7] is no_update


def test_completed_job_is_merged_into_its_row_even_when_that_row_is_inactive(tmp_path, monkeypatch):
    from turbo_moc.apps.app import create_app
    from dash import no_update
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    project = projects.set_design_mode(projects.default_project(), "rotor")
    existing = {"rotor_1": {"planning": {"notes": "kept"}}, "stator_1": {"nozzle": {"job": "previous"}}}
    handle = {"session": "a"*32, "job": "b"*32, "row_id": "stator_1", "row_role": "stator",
              "action": "reference", "started": time.time()}
    folder = flow.directory(handle)
    folder.mkdir(parents=True)
    flow.write_json(folder / "result.json", {"reference": handle})
    status = {"state": "complete", "message": "Ready", "fraction": 1, "started": handle["started"]}
    monkeypatch.setattr(flow, "poll_job", lambda *args: (status, "row log"))
    app = create_app()
    key = next(k for k in app.callback_map if k.startswith("..row-references.data"))
    callback = app.callback_map[key]["callback"].__wrapped__
    result = callback(1, handle, None, existing, project)
    assert result[0]["stator_1"]["reference"] == handle
    assert result[0]["stator_1"]["nozzle"] == existing["stator_1"]["nozzle"]
    assert result[0]["rotor_1"] == existing["rotor_1"]
    assert result[6] is no_update and result[7] is no_update
    assert "reference" not in existing["stator_1"], "Publishing a result must not mutate browser input state."
