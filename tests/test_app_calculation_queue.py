"""Ordered row jobs, dependency handoff, cancellation and actual Dash callbacks."""
from copy import deepcopy
import time
from uuid import uuid4

import pytest

from turbo_moc.configuration import project as projects
from turbo_moc.app import calculations, create_app, rotor_inlet, workflow as flow
from app_helpers import call


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    launched = []

    def start(session, action, config, references, **kwargs):
        rotor_inlet.require_supersonic_inlet(config)
        handle = {"session": session, "job": uuid4().hex, "row_id": kwargs["row_id"],
                  "row_role": kwargs["row_role"], "action": action, "started": time.time(),
                  "key": flow.keys(config, references.get("nozzle"))[action]}
        folder = flow.directory(handle)
        folder.mkdir(parents=True)
        flow.write_json(folder / "status.json", {"state": "running", "message": "Working", "fraction": None})
        flow.write_json(folder / "result.json", {"reference": handle})
        flow.write_json(folder / "moc_result.json", {"converged": True, "fluid_name": "Air",
            "wall_final": {"p": [100000.], "T": [280.], "rho": [1.2], "M": [2.]}})
        launched.append((handle, deepcopy(config)))
        return handle

    monkeypatch.setattr(flow, "start_job", start)
    monkeypatch.setattr(flow, "poll_job", lambda handle: (
        flow.read_json(flow.directory(handle) / "status.json"), f"{handle['row_id']} live output\n"))
    monkeypatch.setattr(flow, "cancel_job", lambda handle: flow.write_json(
        flow.directory(handle) / "status.json", {"state": "cancelled", "message": "Cancelled", "fraction": None}))
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {
        "P0_rel": 150000., "T0_rel": 330., "M_rel": 1.7, "beta_rel_deg": 45., "U": 110.})
    return project, launched


def complete(handle):
    flow.write_json(flow.directory(handle) / "status.json", {
        "state": "complete", "message": "Ready", "fraction": 1})
    return flow.read_json(flow.directory(handle) / "result.json")["reference"]


def current_profiles(project):
    return {row["id"]: {"nozzle": {"key": flow.keys(flow.resolved_configuration(
        project, row["id"], {}))["nozzle"], "session": "a"*32, "job": uuid4().hex}}
        for row in projects.active_rows(project)}


def test_scope_is_separate_from_design_mode_and_uses_row_order(queue):
    project, _ = queue
    original = deepcopy(project)
    assert [row["id"] for row in calculations.scoped_rows(project, "all")] == ["stator_1", "rotor_1"]
    assert [row["id"] for row in calculations.scoped_rows(project, "rotor_1")] == ["rotor_1"]
    assert [item["label"] for item in calculations.scope_options(project)] == ["Stator", "Rotor"]
    assert [row["id"] for row in calculations.scoped_rows(project, ["rotor_1", "stator_1"])] == ["stator_1", "rotor_1"]
    with pytest.raises(ValueError, match="Check at least one"):
        calculations.scoped_rows(project, [])
    with pytest.raises(ValueError, match="active row"):
        calculations.scoped_rows(project, ["missing"])
    assert project == original


def test_both_blade_designs_handoff_new_stator_before_coupled_rotor(queue):
    project, launched = queue
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    with pytest.raises(ValueError, match="upstream stator"):
        calculations.start("a"*32, project, "rotor_1", "nozzle", {})
    first = calculations.start("a"*32, project, "all", "nozzle", {})
    assert first["row_id"] == "stator_1"
    assert "queue" not in launched[0][0]
    saved = deepcopy(first["queue"]["project"])
    project["rows"][1]["geometry"]["rotor_design"]["M_inlet"] = 3.0
    registry = {"stator_1": {"nozzle": complete(first)}}
    second = calculations.advance(first, registry, "Stator finished")
    assert second["row_id"] == "rotor_1"
    assert second["queue"]["project"] == saved
    assert launched[1][1]["rotor_design"]["M_inlet"] == 1.7
    assert launched[1][1]["rotor_source"]["reference"]["job"] == first["job"]
    assert calculations.advance(first, registry, "Duplicate poll") == second
    assert len(launched) == 2, "Overlapping polling must not launch duplicate row jobs."
    registry["rotor_1"] = {"nozzle": complete(second)}
    assert calculations.advance(second, registry, "Rotor finished") is None


@pytest.mark.parametrize("action", ["reference", "volume"])
def test_mesh_queue_checks_every_rows_dependencies_before_launch(queue, action):
    project, launched = queue
    registry = current_profiles(project)
    if action == "reference":
        registry["rotor_1"] = {}
    else:
        for row in projects.active_rows(project):
            config = flow.resolved_configuration(project, row["id"], registry)
            registry[row["id"]]["reference"] = {"key": flow.keys(config, registry[row["id"]]["nozzle"])["reference"]}
        registry["rotor_1"].pop("reference")
    with pytest.raises(ValueError, match="rotor_1"):
        calculations.start("a"*32, project, "all", action, registry)
    assert launched == []


def test_current_meshes_are_skipped_but_explicit_design_recomputes(queue):
    project, launched = queue
    registry = current_profiles(project)
    for row in projects.active_rows(project):
        config = flow.resolved_configuration(project, row["id"], registry)
        registry[row["id"]]["reference"] = {"key": flow.keys(config, registry[row["id"]]["nozzle"])["reference"]}
    assert calculations.start("a"*32, project, "all", "reference", registry) is None
    registry["rotor_1"].pop("reference")
    handle = calculations.start("a"*32, project, "all", "reference", registry)
    assert handle["row_id"] == "rotor_1" and handle["queue"]["index"] == 1
    recompute = calculations.start("a"*32, project, "stator_1", "nozzle", registry)
    assert recompute["row_id"] == "stator_1" and len(launched) == 2


def test_cancellation_preserves_finished_result_and_stops_remaining_rows(queue):
    project, launched = queue
    handle = calculations.start("a"*32, project, "all", "nozzle", {})
    reference = complete(handle)
    calculations.cancel(handle)
    assert flow.poll_job(handle)[0]["state"] == "complete"
    assert calculations.advance(handle, {"stator_1": {"nozzle": reference}}, "done") is None
    assert len(launched) == 1


def test_dash_queue_publishes_each_row_navigates_and_keeps_live_logs(queue):
    project, launched = queue
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    app = create_app()
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    values = {("config", "data"): {"project": project, "valid": True, "supported": False, "row_id": "rotor_1"},
        ("session", "data"): "a"*32, ("calculation-scope", "value"): ["stator_1", "rotor_1"], ("row-references", "data"): {},
        ("base-project", "data"): project, ("nozzle", "n_clicks"): 1}
    submitted = call(app, "..job.data", "nozzle.n_clicks", values)
    first = submitted["job"]["data"]
    assert first["row_id"] == "stator_1"
    values[("job", "data")] = first
    complete(first)
    published = call(app, "..row-references.data", "poll.n_intervals", values)
    assert published["workspace"]["value"] == "design"
    assert published["selected-row"]["value"] == "stator_1"
    assert published["spinner"]["className"] == "spinner running"
    second = published["job"]["data"]
    assert second["row_id"] == "rotor_1" and "row 2 of 2" in published["job-status"]["children"]
    values[("row-references", "data")] = published["row-references"]["data"]
    values[("job-completion", "data")] = first["job"]
    values[("job", "data")] = second
    complete(second)
    finished = call(app, "..row-references.data", "poll.n_intervals", values)
    assert set(finished["row-references"]["data"]) == {"stator_1", "rotor_1"}
    assert finished["selected-row"]["value"] == "rotor_1"
    assert finished["spinner"]["className"] == ""
    assert finished["job-progress"]["value"] == 1
    assert "stator_1 live output" in finished["activity"]["children"]
    assert "rotor_1 live output" in finished["activity"]["children"]


def test_failed_first_row_stops_queue_and_preserves_existing_results(queue):
    project, launched = queue
    handle = calculations.start("a"*32, project, "all", "nozzle", {})
    flow.write_json(flow.directory(handle) / "status.json", {"state": "failed", "message": "Solver failed", "fraction": None})
    app = create_app()
    registry = {"other": {"notes": "retained"}}
    result = call(app, "..row-references.data", "poll.n_intervals", {
        ("job", "data"): handle, ("row-references", "data"): registry, ("base-project", "data"): project})
    assert "row-references" not in result and "job" not in result
    assert result["spinner"]["className"] == ""
    assert "failed" in result["job-status"]["children"]
    assert len(launched) == 1


def test_unsuitable_new_coupled_inlet_stops_queue_and_keeps_stator(queue, monkeypatch):
    project, launched = queue
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    handle = calculations.start("a"*32, project, "all", "nozzle", {})
    reference = complete(handle)
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {
        "P0_rel": 150000., "T0_rel": 330., "M_rel": .8, "beta_rel_deg": 45., "U": 110.})
    result = call(create_app(), "..row-references.data", "poll.n_intervals", {
        ("job", "data"): handle, ("row-references", "data"): {}, ("base-project", "data"): project})
    assert result["row-references"]["data"] == {"stator_1": {"nozzle": reference}}
    assert "Queue stopped" in result["job-status"]["children"]
    assert "supersonic" in result["activity"]["children"]
    assert "job" not in result and result["spinner"]["className"] == ""
    assert len(launched) == 1
