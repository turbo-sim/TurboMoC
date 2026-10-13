from pathlib import Path
"""Machine/row identity, project migration and worker-scoping regressions."""
from copy import deepcopy
import json
import time

import pytest

from turbo_moc.configuration.row import read_configuration, to_shared
from turbo_moc.configuration import project as projects
from turbo_moc.app import workflow as flow


def test_legacy_cases_and_project_cases_share_the_same_row_contract():
    current = read_configuration(CASES["axial_stator"])
    wrapped = projects.normalize_project(to_shared(current))
    projects.validate_project(wrapped)
    assert wrapped["machine"] == {"design_mode": "stator", "geometry_type": "axial", "fluid_name": "Cyclopentane"}
    assert projects.row_configuration(wrapped, "stator_1") == current
    loaded = projects.load_project(CASES["axial_stator"])
    assert projects.row_configuration(loaded, "stator_1") == current


def test_mode_switch_retains_inactive_row_settings_and_order():
    initial = projects.default_project()
    paired = projects.set_design_mode(initial, "stator_rotor")
    rotor = projects.get_row(paired, "rotor_1")
    rotor["geometry"]["parametrization_type"] = "vortex"
    rotor["mesh"]["numerics"]["span_layers"] = 17
    rotor["placement"]["angular_phase_deg"] = 12.5
    stator = projects.get_row(paired, "stator_1")
    stator["geometry"]["method_of_characteristics"]["n"] = 22
    rotor_only = projects.set_design_mode(paired, "rotor")
    assert [row["id"] for row in projects.active_rows(rotor_only)] == ["rotor_1"]
    restored = projects.set_design_mode(rotor_only, "stator_rotor")
    assert restored == paired
    assert initial["machine"]["design_mode"] == "stator" and len(initial["rows"]) == 1
    projects.validate_project(restored)


def test_rotor_only_planning_project_can_gain_an_upstream_stator():
    project = projects.load_project(CASE_DIR / "rotor_only.yaml")
    existing = deepcopy(project["rows"][0])
    paired = projects.set_design_mode(project, "stator_rotor")
    assert [row["role"] for row in projects.active_rows(paired)] == ["stator", "rotor"]
    assert projects.get_row(paired, existing["id"]) == existing
    projects.validate_project(paired)


def test_channel_family_switch_preserves_independent_row_design_and_mesh_settings():
    paired = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(paired, "stator_1")["mesh"]["numerics"]["blade_size"] = 0.37
    projects.get_row(paired, "rotor_1")["mesh"]["numerics"]["blade_size"] = 0.71
    radial = projects.set_machine_type(paired, "radial")
    assert radial["machine"]["geometry_type"] == "radial"
    for before, after in zip(paired["rows"], radial["rows"]):
        assert after["geometry"]["meridional_channel"]["type"] == "radial"
        assert after["mesh"] == before["mesh"]
        for name in set(before["geometry"]) - {"meridional_channel", "axial_interspace_mm"}:
            assert after["geometry"][name] == before["geometry"][name]
        assert "axial_interspace_mm" not in after["geometry"]
    assert paired["machine"]["geometry_type"] == "axial"
    projects.validate_project(radial)


def test_linear_family_has_no_radius_or_blade_count_and_cannot_form_a_pair():
    linear = projects.set_machine_type(projects.default_project(), "linear")
    meridional_channel = projects.get_row(linear, "stator_1")["geometry"]["meridional_channel"]
    assert set(meridional_channel) == {"type", "meridional_chord", "blade_height"}
    projects.validate_project(linear)
    with pytest.raises(ValueError, match="axial or radial"):
        projects.set_design_mode(linear, "stator_rotor")
    paired = projects.set_design_mode(projects.default_project(), "stator_rotor")
    with pytest.raises(ValueError, match="linear cascade"):
        projects.set_machine_type(paired, "linear")
    invalid = deepcopy(paired)
    invalid["machine"]["geometry_type"] = "linear"
    with pytest.raises(ValueError, match="linear cascade"):
        projects.validate_project(invalid)


@pytest.mark.parametrize("family", ["axial", "radial", "linear"])
def test_rotor_is_a_supported_numerical_configuration(family):
    rotor = projects.set_design_mode(projects.default_project(family), "rotor")
    projects.validate_project(rotor)
    assert [row["role"] for row in projects.active_rows(rotor)] == ["rotor"]
    config = projects.row_configuration(rotor, "rotor_1")
    assert config["parametrization_type"] == "vortex"
    assert config["rotor_design"]["M_inlet"] > 1


def test_rotor_planning_mesh_controls_use_the_same_validation_as_stator_controls():
    rotor = projects.set_design_mode(projects.default_project(), "rotor")
    projects.get_row(rotor, "rotor_1")["mesh"]["visualization"]["volume_edge_width"] = -1
    with pytest.raises(ValueError, match="volume_edge_width must be positive"):
        projects.validate_project(rotor)


def test_row_identifiers_are_unique_and_survive_reordering():
    project = projects.default_project()
    second = deepcopy(project["rows"][0])
    second["id"] = "stator_2"
    project["rows"].insert(0, second)
    edited = flow.edited_project(project, [0.73, 25], [
        {"type": "parameter", "row": "stator_1", "path": "mesh.numerics.blade_size"},
        {"type": "parameter", "row": "stator_2", "path": "method_of_characteristics.n"}])
    assert projects.get_row(edited, "stator_1")["mesh"]["numerics"]["blade_size"] == 0.73
    assert projects.get_row(edited, "stator_2")["geometry"]["method_of_characteristics"]["n"] == 25
    assert projects.get_row(edited, "stator_2")["mesh"] == projects.get_row(project, "stator_2")["mesh"]
    projects.validate_project(edited)
    edited["rows"][0]["id"] = "stator_1"
    with pytest.raises(ValueError, match="Duplicate"):
        projects.validate_project(edited)


def test_display_placement_is_excluded_from_mesh_identity():
    project = projects.default_project()
    config = projects.row_configuration(project, "stator_1")
    moved = projects.update_row_placement(project, "stator_1", streamwise_offset_mm=30, angular_phase_deg=8)
    projects.validate_project(moved)
    assert flow.keys(projects.row_configuration(moved, "stator_1")) == flow.keys(config)
    assert projects.get_row(project, "stator_1")["placement"]["streamwise_offset_mm"] == 0
    radial = projects.set_machine_type(moved, "radial")
    assert projects.get_row(radial, "stator_1")["placement"]["streamwise_offset_mm"] == 0
    projects.get_row(radial, "stator_1")["placement"]["streamwise_offset_mm"] = 3
    with pytest.raises(ValueError, match="only for axial"):
        projects.validate_project(radial)


def test_worker_scope_rejects_mismatched_role_and_foreign_row_artifacts_before_launch(monkeypatch):
    config = read_configuration(CASES["axial_stator"])
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match="role does not match"):
        flow.start_job("a" * 32, "nozzle", config, {}, row_id="rotor_1", row_role="rotor")
    with pytest.raises(ValueError, match="another row"):
        flow.start_job("a" * 32, "nozzle", config, {"nozzle": {"row_id": "stator_2"}}, row_id="stator_1")
    with pytest.raises(ValueError, match="another row"):
        flow.start_job("a" * 32, "nozzle", config, {"nozzle": {"job": "historical"}}, row_id="stator_2")
    monkeypatch.setattr(flow, "load_nozzle", lambda *args: {})
    with pytest.raises(ValueError, match="up-to-date 2D"):
        flow.start_job("a" * 32, "export_2d", config, {"nozzle": {}, "reference": None})


def test_job_snapshot_retains_row_and_project_identity(tmp_path, monkeypatch):
    project = projects.default_project()
    config = projects.row_configuration(project, "stator_1")
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    class Process:
        pid = 123
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: Process())
    handle = flow.start_job("a" * 32, "nozzle", config, {}, project=project, row_id="stator_1")
    folder = flow.directory(handle)
    request = json.loads((folder / "request.json").read_text())
    assert request["handle"]["row_id"] == "stator_1"
    assert request["handle"]["row_role"] == "stator"
    assert request["project"] == project
    assert request["config"] == config
    assert (folder / "project.yaml").is_file()
    flow.PROCESSES.pop((handle["session"], handle["job"]))


def test_job_snapshot_rejects_mismatched_row_configuration(monkeypatch):
    project = projects.default_project()
    config = projects.row_configuration(project, "stator_1")
    config["mesh"]["numerics"]["blade_size"] = 0.99
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match="does not match"):
        flow.start_job("a" * 32, "nozzle", config, {}, project=project)


def test_worker_2d_export_reports_row_and_reads_only_selected_reference(tmp_path, monkeypatch):
    from turbo_moc.app import worker
    config = read_configuration(CASES["axial_stator"])
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_2"}
    request = {"config": config, "references": {"reference": reference},
               "handle": {"row_id": "stator_2", "row_role": "stator", "action": "export_2d", "started": time.time()}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    calls = []
    def export_reference(folder, cfg, output, *, progress):
        calls.append((folder, cfg, output))
        progress("Write two-dimensional grid", 0.5)
        return {"file": "section.cgns"}
    monkeypatch.setattr(worker.mesh, "export_reference_mesh", export_reference)
    assert worker.run(path) == 0
    assert calls[0][0] == flow.directory(reference)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["row_id"] == "stator_2" and status["message"].startswith("[stator_2]")
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["export"]["file"] == "section.cgns"


def test_axial_interspace_uses_the_farthest_tapered_trailing_edge():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    stator = projects.get_row(project, "stator_1")
    stator["geometry"]["meridional_channel"].update(hub_chord=20, tip_chord=24)
    stator["placement"]["streamwise_offset_mm"] = 5
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"]["axial_interspace_mm"] = 3.5
    rotor["placement"].update(streamwise_offset_mm=99, angular_phase_deg=12)
    projects.validate_project(project)
    resolved = projects.assembly_placement(project, "rotor_1")
    assert resolved == {"streamwise_offset_mm": 32.5, "angular_phase_deg": 12}
    clearance = [resolved["streamwise_offset_mm"] - 5 - chord for chord in (20, 24)]
    assert min(clearance) == 3.5
    assert projects.assembly_interspaces(project) == [{
        "upstream_id": "stator_1", "downstream_id": "rotor_1", "geometry_type": "axial",
        "gap_mm": 3.5, "prescribed": True, "upstream_trailing_edge_mm": 29.0,
        "downstream_leading_edge_mm": 32.5, "flow_direction": "axial"}]
    assert rotor["placement"]["streamwise_offset_mm"] == 99, "Assembly placement must not mutate saved row data."


@pytest.mark.parametrize("gap", [-1, float("nan"), float("inf"), -float("inf"), True, "1.0", None])
def test_axial_interspace_requires_a_finite_non_negative_length(gap):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "rotor_1")["geometry"]["axial_interspace_mm"] = gap
    with pytest.raises(ValueError, match="axial_interspace_mm must be finite and non-negative"):
        projects.validate_project(project)


def test_axial_interspace_migrates_and_invalidates_domains_but_retains_solver():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "rotor_1")["geometry"].pop("axial_interspace_mm")
    migrated = projects.normalize_project(project)
    assert projects.get_row(migrated, "rotor_1")["geometry"]["axial_interspace_mm"] == 2
    assert "axial_interspace_mm" not in projects.get_row(project, "rotor_1")["geometry"]
    before = flow.keys(flow.resolved_configuration(migrated, "stator_1", {}))
    edited = flow.edited_project(migrated, [8.5], [
        {"type": "parameter", "row": "rotor_1", "path": "axial_interspace_mm"}])
    projects.validate_project(edited)
    assert projects.get_row(edited, "rotor_1")["geometry"]["axial_interspace_mm"] == 8.5
    assert projects.assembly_placement(edited, "rotor_1")["streamwise_offset_mm"] == 28.5
    after = flow.keys(flow.resolved_configuration(edited, "stator_1", {}))
    assert after["nozzle"] == before["nozzle"]
    assert after["geometry"] != before["geometry"]
    assert after["volume"] != before["volume"]
    radial = projects.set_machine_type(edited, "radial")
    assert "axial_interspace_mm" not in projects.get_row(radial, "rotor_1")["geometry"]
    axial = projects.set_machine_type(radial, "axial")
    assert projects.get_row(axial, "rotor_1")["geometry"]["axial_interspace_mm"] == 2


def test_single_row_placement_is_not_affected_by_an_inactive_interspace():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "stator_1")["placement"]["streamwise_offset_mm"] = 7
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"]["axial_interspace_mm"] = 8
    rotor["placement"]["streamwise_offset_mm"] = 41
    stator_only = projects.set_design_mode(project, "stator")
    assert projects.assembly_placement(stator_only, "stator_1")["streamwise_offset_mm"] == 7
    assert projects.assembly_interspaces(stator_only) == []
    rotor_only = projects.set_design_mode(project, "rotor")
    assert projects.assembly_placement(rotor_only, "rotor_1")["streamwise_offset_mm"] == 41
    assert projects.assembly_interspaces(rotor_only) == []
    paired_again = projects.set_design_mode(rotor_only, "stator_rotor")
    assert projects.assembly_placement(paired_again, "rotor_1")["streamwise_offset_mm"] == 35


@pytest.mark.parametrize("inlet,outlet,next_inlet,next_outlet,gap,direction", [
    (100, 80, 75, 60, 5, "inflow"),
    (100, 80, 85, 65, -5, "inflow"),
    (80, 100, 105, 120, 5, "outflow"),
    (80, 100, 95, 115, -5, "outflow"),
])
def test_radial_interspace_is_implied_by_radii_and_reports_overlap(
        inlet, outlet, next_inlet, next_outlet, gap, direction):
    project = projects.set_design_mode(projects.default_project("radial"), "stator_rotor")
    projects.get_row(project, "stator_1")["geometry"]["meridional_channel"].update(
        inlet_radius=inlet, outlet_radius=outlet)
    projects.get_row(project, "rotor_1")["geometry"]["meridional_channel"].update(
        inlet_radius=next_inlet, outlet_radius=next_outlet)
    projects.validate_project(project)  # Negative implied gaps remain valid planning configurations.
    record = projects.assembly_interspaces(project)[0]
    assert record["gap_mm"] == gap and record["flow_direction"] == direction
    assert record["prescribed"] is False
    assert record["upstream_trailing_edge_mm"] == outlet
    assert record["downstream_leading_edge_mm"] == next_inlet
    assert projects.assembly_placement(project, "rotor_1")["streamwise_offset_mm"] == 0
    assert "axial_interspace_mm" not in projects.get_row(project, "rotor_1")["geometry"]


def test_ordered_assembly_spacing_uses_stable_ids_and_each_upstream_row_extent():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    first_stator, first_rotor = project["rows"]
    first_stator["id"], first_rotor["id"] = "fixed_A", "moving_A"
    first_stator["placement"]["streamwise_offset_mm"] = 5
    first_rotor["geometry"]["axial_interspace_mm"] = 3
    second_stator, second_rotor = deepcopy(first_stator), deepcopy(first_rotor)
    second_stator["id"], second_rotor["id"] = "fixed_B", "moving_B"
    second_stator["placement"]["streamwise_offset_mm"] = 70
    second_stator["geometry"]["meridional_channel"].update(hub_chord=25, tip_chord=30)
    second_rotor["geometry"]["axial_interspace_mm"] = 4
    project["rows"].extend([second_stator, second_rotor])
    projects.validate_project(project)
    assert projects.assembly_placement(project, "moving_A")["streamwise_offset_mm"] == 28
    assert projects.assembly_placement(project, "moving_B")["streamwise_offset_mm"] == 104
    records = projects.assembly_interspaces(project)
    assert [(r["upstream_id"], r["downstream_id"]) for r in records] == [
        ("fixed_A", "moving_A"), ("moving_A", "fixed_B"), ("fixed_B", "moving_B")]
    assert [record["gap_mm"] for record in records] == [3, 22, 4]

CASE_DIR = Path(__file__).resolve().parents[1] / "examples/apps/cases"
CASES = {n: CASE_DIR / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}
FLUENT_CONFIG = Path(__file__).resolve().parents[1] / "examples/fluent_2D/workflow_fluent_stator_2D.yaml"


@pytest.mark.parametrize("action", ["reference", "volume"])
def test_app_mesh_jobs_generate_cgns_and_remove_native_mesh(action, tmp_path, monkeypatch):
    from turbo_moc.app import worker
    config = read_configuration(CASES["axial_stator"])
    config["mesh"]["cgns"]["enabled"] = False  # App output remains mandatory.
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_1"}
    request = {"config": config, "references": {"nozzle": reference, "reference": reference},
               "handle": {"row_id": "stator_1", "row_role": "stator", "action": action,
                          "started": time.time()}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    monkeypatch.setattr(flow, "get_design", lambda *args: object())
    monkeypatch.setattr(flow, "directory", lambda *args: tmp_path / "reference-source")
    monkeypatch.setattr(worker.mesh, "load_section", lambda *args: ({}, object()))

    def generate(*args, progress):
        (tmp_path / "temporary.msh").write_text("internal topology")
        progress("Mesh ready", 1)
        return {"cells": 1}

    def convert(source, cfg, output, *, progress):
        assert source == tmp_path  # Convert this newly generated mesh.
        assert cfg["mesh"]["cgns"]["enabled"] is True
        assert (source / "temporary.msh").exists()
        progress("CGNS ready", 1)
        filename = output / "mesh.cgns"
        filename.write_text("converted output")
        return {"file": str(filename), "native_mesh_file": str(source / "temporary.msh")}

    monkeypatch.setattr(worker.mesh, "reference_mesh", generate)
    monkeypatch.setattr(worker.mesh, "volume_mesh", generate)
    monkeypatch.setattr(worker.mesh, "export_reference_mesh", convert)
    monkeypatch.setattr(worker.mesh, "export_mesh", convert)
    assert worker.run(path) == 0
    assert (tmp_path / "mesh.cgns").exists()
    assert not list(tmp_path.glob("*.msh"))
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"
    assert json.loads((tmp_path / "result.json").read_text())["export"]["file"].endswith(".cgns")
