"""Rotor inlet provenance and row-scoped background jobs, without a CFD run."""
from copy import deepcopy
import json
import math

import pytest

from turbo_moc.configuration import project as projects
from turbo_moc.app import workflow as flow, rotor_inlet


@pytest.fixture
def coupled_project(tmp_path, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    project["machine"]["fluid_name"] = "Air"
    projects.get_row(project, "rotor_1")["geometry"]["inlet_coupling"].update(
        mode="stator_exit", rotational_speed={"type": "actual", "value": 4000})
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
    project["rows"][0]["geometry"]["method_of_characteristics"]["inlet_property_1"] += 100
    with pytest.raises(ValueError, match="upstream stator MoC"):
        flow.resolved_configuration(project, "rotor_1", registry)


def test_machine_fluid_change_invalidates_upstream_and_coupled_rotor(coupled_project):
    project, registry, _ = coupled_project
    stator_before = flow.keys(projects.row_configuration(project, "stator_1"))
    rotor_before = flow.keys(projects.row_configuration(project, "rotor_1"))
    project["machine"]["fluid_name"] = "Nitrogen"
    assert flow.keys(projects.row_configuration(project, "stator_1"))["nozzle"] != stator_before["nozzle"]
    assert flow.keys(projects.row_configuration(project, "rotor_1"))["nozzle"] != rotor_before["nozzle"]
    with pytest.raises(ValueError, match="upstream stator MoC"):
        flow.resolved_configuration(project, "rotor_1", registry)


def test_coupled_rotor_remains_visible_after_browser_number_roundtrip(coupled_project):
    from turbo_moc.app.application import ready_rows

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


def test_coupling_preserves_manual_pair_and_uses_upstream_state(coupled_project):
    from turbo_moc.configuration.fluid_inputs import INLET_FIELDS
    project, registry, _ = coupled_project
    design = project["rows"][1]["geometry"]["rotor_design"]
    design.update(inlet_property_pair="PT_INPUTS", inlet_property_1=5e5, inlet_property_2=400.)
    original = deepcopy(design)
    projects.validate_project(project)
    coupled = flow.resolved_configuration(project, "rotor_1", registry)["rotor_design"]
    assert not set(INLET_FIELDS) & coupled.keys()
    assert coupled["T0_rel"] == 330.
    assert design == original
    project["rows"][1]["geometry"]["inlet_coupling"]["mode"] = "manual"
    assert flow.resolved_configuration(project, "rotor_1", registry)["rotor_design"] == {
        **original, "fluid_name": project["machine"]["fluid_name"]}


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


@pytest.mark.parametrize("kind,value", [("blade_velocity_ratio", .4), ("specific", .7)])
def test_dimensionless_speed_drives_the_coupled_velocity_triangle(coupled_project, monkeypatch, kind, value):
    from turbo_moc.app.rotor_speed import resolve_speed
    project, registry, calls = coupled_project
    coupling = project["rows"][1]["geometry"]["inlet_coupling"]
    coupling["rotational_speed"] = {"type": kind, "value": value}
    reference = {"enthalpy_drop": 50000., "exit_density": 2., "mass_flow": 5.,
                 "c0": math.sqrt(100000.), "reference_pressure": 95000.}
    monkeypatch.setattr(rotor_inlet, "stator_reference", lambda *args: reference)
    config = flow.resolved_configuration(project, "rotor_1", registry)
    expected = resolve_speed(coupling["rotational_speed"], config["rotor_source"]["inlet_radius_mm"],
        enthalpy_drop=50000., exit_density=2., mass_flow=5.)
    assert calls[-1][2] == pytest.approx(expected["U"])
    assert config["rotor_source"]["rotational_speed"]["rpm"] == pytest.approx(expected["rpm"])
    assert config["rotor_source"]["rotational_speed"]["reference_pressure"] == 95000.


@pytest.mark.parametrize("mach,beta,expected", [
    (1.0, 54., "supersonic relative inlet"),
    (1.39, -65., "Negative relative inlet angle"),
    (.73, -35., "Negative relative inlet angle"),
])
def test_unsupported_coupled_inlet_cannot_start_solver(coupled_project, monkeypatch, mach, beta, expected):
    project, registry, _ = coupled_project
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {
        "P0_rel": 100000., "T0_rel": 300., "M_rel": mach, "beta_rel_deg": beta})
    config = flow.resolved_configuration(project, "rotor_1", registry)
    assert config["rotor_design"]["beta_inlet"] == beta
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Solver must not start"))
    with pytest.raises(ValueError, match=expected):
        flow.start_job("d" * 32, "nozzle", config, {}, project=project,
            row_id="rotor_1", row_role="rotor", registry=registry)
