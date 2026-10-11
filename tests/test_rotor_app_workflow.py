"""Rotor inlet provenance and row-scoped background jobs, without a CFD run."""
from copy import deepcopy
import json
import math

import pytest

from turbo_moc import configuration_project as projects
from turbo_moc.apps.app import workflow as flow, rotor_inlet


@pytest.fixture
def coupled_project(tmp_path, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "rotor_1")["geometry"]["inlet_coupling"].update(mode="stator_exit", rpm=4000)
    config = projects.row_configuration(project, "stator_1")
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_1",
                 "key": flow.keys(config)["nozzle"]}
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    folder = flow.directory(reference)
    folder.mkdir(parents=True)
    flow.write_json(folder / "moc_result.json", {"fluid_name": "Air", "converged": True,
        "wall_final": {"p": [100000.], "T": [280.], "rho": [1.2], "M": [2.]}})
    calls = []

    def triangle(serialized, angle, speed):
        calls.append((json.loads(serialized), angle, speed))
        return {"P0_rel": 150000. + speed, "T0_rel": 330., "M_rel": 1.7,
                "beta_rel_deg": angle - 20., "U": speed, "is_supersonic": True}
    monkeypatch.setattr(rotor_inlet, "_triangle", triangle)
    return project, {"stator_1": {"nozzle": reference}}, calls


def test_coupling_uses_upstream_result_and_physical_inlet_radius_preserving_manual_values(coupled_project):
    project, registry, calls = coupled_project
    original = deepcopy(project)
    config = flow.resolved_configuration(project, "rotor_1", registry)
    assert project == original
    assert config["rotor_design"]["fluid_name"] == "Air"
    assert config["rotor_design"]["M_inlet"] == 1.7
    assert config["rotor_design"]["beta_inlet"] == 45.
    assert calls[-1][2] == pytest.approx(4000 * 2 * math.pi / 60 * .105)
    assert config["rotor_source"]["row_id"] == "stator_1"
    assert "axial_interspace_mm" not in config
    # A gap changes the domain and export placement, but not the inlet physics.
    project["rows"][1]["geometry"]["axial_interspace_mm"] = 8
    changed = flow.keys(flow.resolved_configuration(project, "rotor_1", registry))
    assert flow.keys(config)["nozzle"] == changed["nozzle"]
    assert flow.keys(config)["geometry"] != changed["geometry"]
    assert flow.keys(config)["volume"] != changed["volume"]


def test_upstream_edits_and_recomputation_invalidate_coupled_rotor(coupled_project):
    project, registry, _ = coupled_project
    initial = flow.keys(flow.resolved_configuration(project, "rotor_1", registry))
    project["rows"][0]["geometry"]["blade_geometry"]["metal_angle_out"] -= 2
    changed = flow.keys(flow.resolved_configuration(project, "rotor_1", registry))
    assert changed["nozzle"] != initial["nozzle"]
    assert changed["volume"] != initial["volume"]
    reference = registry["stator_1"]["nozzle"]
    old_folder = flow.directory(reference)
    reference["job"] = "c" * 32
    new_folder = flow.directory(reference)
    new_folder.mkdir(parents=True)
    (new_folder / "moc_result.json").write_bytes((old_folder / "moc_result.json").read_bytes())
    assert flow.keys(flow.resolved_configuration(project, "rotor_1", registry))["nozzle"] != changed["nozzle"]
    project["rows"][0]["geometry"]["method_of_characteristics"]["P0"] += 100
    with pytest.raises(ValueError, match="upstream stator MoC"):
        flow.resolved_configuration(project, "rotor_1", registry)


def test_coupled_rotor_remains_visible_after_browser_number_roundtrip(coupled_project):
    from turbo_moc.apps.app.application import ready_rows

    project, registry, _ = coupled_project
    config = flow.resolved_configuration(project, "rotor_1", registry)
    # JSON.stringify in the browser writes an integral JS Number as 65, even
    # if Python originally sent 65.0. This also affects nested inlet provenance.
    def browser_number(value):
        number = float(value)
        return int(number) if number.is_integer() else number

    submitted = json.loads(json.dumps(config), parse_float=browser_number)
    assert isinstance(config["rotor_design"]["beta_inlet"], float)
    assert isinstance(submitted["rotor_design"]["beta_inlet"], int)
    assert isinstance(submitted["rotor_source"]["inlet_radius_mm"], int)
    reference = {"session": "a" * 32, "job": "c" * 32, "row_id": "rotor_1",
                 "key": flow.keys(submitted)["nozzle"]}
    registry["rotor_1"] = {"nozzle": reference}

    assert flow.digest(True) != flow.digest(1)
    assert flow.keys(submitted, reference) == flow.keys(config, reference)
    assert [row[0]["id"] for row in ready_rows({"project": project}, registry)] == [
        "stator_1", "rotor_1"]

    submitted["rotor_design"]["beta_inlet"] += 1e-8
    assert not flow.current(reference, flow.keys(submitted)["nozzle"])


def test_manual_inlet_is_independent_and_coupling_requires_active_stator(coupled_project):
    project, registry, _ = coupled_project
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "manual"
    initial = flow.resolved_configuration(project, "rotor_1", registry)
    project = projects.set_design_mode(project, "rotor")
    standalone = flow.resolved_configuration(project, "rotor_1", {})
    assert standalone == {key: value for key, value in initial.items() if key != "assembly"}
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    with pytest.raises(ValueError, match="active upstream stator"):
        flow.resolved_configuration(project, "rotor_1", registry)


def test_rotor_job_rechecks_derived_inputs_before_launch(coupled_project, monkeypatch):
    project, registry, _ = coupled_project
    config = flow.resolved_configuration(project, "rotor_1", registry)
    class Process:
        pid = 123
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: Process())
    handle = flow.start_job("d" * 32, "nozzle", config, {}, project=project,
        row_id="rotor_1", row_role="rotor", registry=registry)
    saved = flow.read_json(flow.directory(handle) / "request.json")
    assert saved["config"]["rotor_source"] == config["rotor_source"]
    assert saved["project"] == project
    flow.PROCESSES.pop((handle["session"], handle["job"]))
    config["rotor_design"]["P0_rel"] += 1
    with pytest.raises(ValueError, match="does not match"):
        flow.start_job("d" * 32, "nozzle", config, {}, project=project,
            row_id="rotor_1", row_role="rotor", registry=registry)


def test_coupled_subsonic_state_remains_visible_but_cannot_run(coupled_project, monkeypatch):
    project, registry, _ = coupled_project
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {
        "P0_rel": 100000., "T0_rel": 300., "M_rel": .8, "beta_rel_deg": 25.})
    config = flow.resolved_configuration(project, "rotor_1", registry)
    assert config["rotor_design"]["M_inlet"] == .8
    with pytest.raises(ValueError, match="supersonic relative inlet"):
        flow.start_job("d" * 32, "nozzle", config, {}, project=project,
            row_id="rotor_1", row_role="rotor", registry=registry)


def test_property_failure_is_a_readable_inlet_error(coupled_project, monkeypatch):
    project, registry, _ = coupled_project
    def fail(*args):
        raise RuntimeError("state outside property range")
    monkeypatch.setattr(rotor_inlet, "_triangle", fail)
    with pytest.raises(ValueError, match="Cannot compute the coupled rotor inlet.*property range"):
        flow.resolved_configuration(project, "rotor_1", registry)
