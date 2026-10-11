"""Project ZIP contents, freshness checks and the browser download endpoint."""
import base64
from copy import deepcopy
from io import BytesIO
import json
from uuid import uuid4
from zipfile import ZipFile

import pytest
import yaml

pytest.importorskip("dash_vtk")

from turbo_moc import configuration_project as projects
from turbo_moc.apps.app import create_app, project_export, workflow as flow
from turbo_moc.io.cgns_adf import read_adf_bytes
from test_cgns_stage import row_file


@pytest.fixture
def completed_project(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)

    def create(family="axial", paired=False):
        project = projects.default_project(family)
        if paired:
            project = projects.set_design_mode(project, "stator_rotor")
        registry = {}
        session = uuid4().hex
        for row in projects.active_rows(project):
            references = registry.setdefault(row["id"], {})
            for index, stage in enumerate(("nozzle", "reference", "volume")):
                handle = {"session": session, "job": uuid4().hex, "action": stage,
                          "row_id": row["id"], "row_role": row["role"], "started": 1700000000 + index}
                effective = flow.resolved_configuration(project, row["id"], registry)
                handle["key"] = flow.keys(effective, references.get("nozzle"))[stage]
                folder = flow.directory(handle)
                folder.mkdir(parents=True)
                request = {"handle": handle, "project": project, "config": effective,
                           "references": deepcopy(references),
                           "source_revision": flow.SOURCE_REVISION}
                flow.write_json(folder / "request.json", request)
                flow.write_json(folder / "status.json", {"state": "complete"})
                (folder / "activity.log").write_text(f"{row['id']} {stage} activity\n", encoding="utf-8")
                for filename in ("moc_result.json", "result.json", "browser_surfaces.json", "reference_definition.json"):
                    (folder / filename).write_text("{}")
                (folder / "internal.npz").write_bytes(b"not for export")
                references[stage] = handle
                if stage == "reference":
                    summary = {"section_nodes": 12, "section_cells": 8, "elapsed_s": 1.0}
                    flow.write_json(folder / "reference_summary.json", summary)
                    name = row["mesh"]["cgns"]["filename"].removesuffix(".cgns") + "_2d.cgns"
                    row_file(folder / name, 2)
                if stage == "volume":
                    summary = {"volume_nodes": 36, "cells": {"hex": 16},
                               "quality": {"min_sicn": 0.8}, "elapsed_s": 2.0}
                    flow.write_json(folder / "mesh_summary.json", summary)
                    flow.write_json(folder / "mesh_report.json", {
                        "metrics": {"skewness": {"min": 0.0, "mean": 0.1, "max": 0.2}},
                        "definitions": {"skewness": "test definition"}, "worst_cells": {"skewness": []}})
                    row_file(folder / row["mesh"]["cgns"]["filename"])
        return project, registry

    return create


@pytest.mark.parametrize("family,paired", [("axial", False), ("radial", False), ("linear", False), ("axial", True)])
def test_zip_contains_only_selected_files_and_consolidates_reports(completed_project, family, paired):
    project, registry = completed_project(family, paired)
    data, filename = project_export.build_archive(project, registry)
    assert filename.endswith(".zip")
    with ZipFile(BytesIO(data)) as archive:
        expected = {"project.yaml", "README.txt", "mesh_report.json", "activity.log", "stage.cgns"}
        if family == "linear":
            expected.add("stage_2d.cgns")
        for row in projects.active_rows(project):
            name = row["mesh"]["cgns"]["filename"]
            expected.add(f"meshes/{row['id']}/{name}")
            if family == "linear":
                expected.add(f"meshes/{row['id']}/{name.removesuffix('.cgns')}_2d.cgns")
            assert archive.read(f"meshes/{row['id']}/{name}") == (
                flow.directory(registry[row['id']]["volume"]) / name).read_bytes()
        assert set(archive.namelist()) == expected
        assert yaml.safe_load(archive.read("project.yaml")) == project
        report = json.loads(archive.read("mesh_report.json"))
        base = read_adf_bytes(archive.read("stage.cgns")).get(label="CGNSBase_t")
        assert {zone.name for zone in base.find_all("Zone_t")} == {f"{row_id}_fluid" for row_id in registry}
        for row_id in registry:
            volume = report["rows"][row_id]["meshes"]["3d"]
            assert volume["summary"]["cells"] == {"hex": 16}
            assert volume["quality"]["metrics"]["skewness"]["max"] == 0.2
            assert volume["quality"]["definitions"]["skewness"] == "test definition"
        logs = archive.read("activity.log").decode()
        for row_id in registry:
            for stage in ("nozzle", "reference", "volume"):
                assert logs.count(f"{row_id} {stage} activity") == 1
        readme = archive.read("README.txt").decode()
        assert "Minimum SICN: 0.8" in readme and "periodic_left" in readme
        if paired:
            assert len(report["interfaces"]) == 1
            assert {side["boundary"] for side in report["interfaces"][0]["exported_boundaries"]} == {"inlet", "outlet"}
            assert all(row["meshes"]["3d"]["coordinate_basis"] == "assembly_cartesian"
                       for row in report["rows"].values())
            assert "Machine placement applied" in readme


def test_only_linear_2d_can_be_downloaded_without_a_volume(completed_project):
    for family in ("axial", "radial", "linear"):
        project, registry = completed_project(family)
        registry["stator_1"].pop("volume")
        if family != "linear":
            with pytest.raises(ValueError, match="Generate a current 3D mesh"):
                project_export.build_archive(project, registry)
        else:
            data, _ = project_export.build_archive(project, registry)
            with ZipFile(BytesIO(data)) as archive:
                assert len([name for name in archive.namelist() if name.endswith(".cgns")]) == 2
                assert read_adf_bytes(archive.read("stage.cgns")).get(label="CGNSBase_t").data[0] == 2
                assert "3D: not generated" in archive.read("README.txt").decode()


def test_span_edits_keep_linear_2d_but_exclude_outdated_3d(completed_project):
    project, registry = completed_project("linear")
    project["rows"][0]["mesh"]["numerics"]["span_layers"] += 1
    data, _ = project_export.build_archive(project, registry)
    with ZipFile(BytesIO(data)) as archive:
        assert all(name == "stage.cgns" or name.endswith("_2d.cgns")
                   for name in archive.namelist() if name.endswith(".cgns"))
        assert "out of date" in archive.read("README.txt").decode()
        assert yaml.safe_load(archive.read("project.yaml")) == project


@pytest.mark.parametrize("change", ["geometry", "cgns", "nozzle", "reference", "failed", "missing", "foreign", "source"])
def test_outdated_or_incomplete_meshes_are_not_exported(completed_project, change):
    project, registry = completed_project()
    row = project["rows"][0]
    references = registry[row["id"]]
    folder = flow.directory(references["volume"])
    if change == "geometry":
        row["geometry"]["blade_geometry"]["metal_angle_out"] += 1
    elif change == "cgns":
        row["mesh"]["cgns"]["length_scale"] *= 2
    elif change in ("nozzle", "reference"):
        references[change] = {**references[change], "job": uuid4().hex}
    elif change == "failed":
        flow.write_json(folder / "status.json", {"state": "failed"})
    elif change == "missing":
        (folder / row["mesh"]["cgns"]["filename"]).unlink()
    else:
        request = flow.read_json(folder / "request.json")
        if change == "foreign":
            request["handle"]["row_id"] = "stator_2"
        else:
            request["source_revision"] = "earlier-source"
        flow.write_json(folder / "request.json", request)
    with pytest.raises(ValueError, match="Generate a current"):
        project_export.build_archive(project, registry)


def test_display_edits_and_inactive_rows_do_not_export_unwanted_artifacts(completed_project):
    project, registry = completed_project("axial")
    # Retaining an inactive row does not change a standalone row's mesh.
    project = projects.set_design_mode(project, "stator_rotor")
    project["rows"][0]["placement"]["angular_phase_deg"] = 12
    project["rows"][0]["mesh"]["visualization"]["volume_edge_width"] = 0.8
    project["machine"]["design_mode"] = "stator"
    data, _ = project_export.build_archive(project, registry)
    with ZipFile(BytesIO(data)) as archive:
        assert not any("rotor_1" in name for name in archive.namelist())
        assert "rotor_1" not in archive.read("activity.log").decode()
        assert yaml.safe_load(archive.read("project.yaml")) == project


@pytest.mark.parametrize("change", ["gap", "rotor_phase", "stator_position", "rotor_chord"])
def test_assembly_edits_reject_meshes_with_old_interfaces_or_placement(completed_project, change):
    project, registry = completed_project("axial", paired=True)
    stator, rotor = project["rows"]
    if change == "gap":
        rotor["geometry"]["axial_interspace_mm"] += 1
    elif change == "rotor_phase":
        rotor["placement"]["angular_phase_deg"] += 10
    elif change == "stator_position":
        stator["placement"]["streamwise_offset_mm"] += 5
    else:
        rotor["geometry"]["meridional_channel"]["hub_chord"] += 2
    meshes, omitted = project_export.collect_meshes(project, registry)
    assert any("out of date" in reason for reason in omitted)
    assert not any(mesh["row"]["id"] == rotor["id"] for mesh in meshes)
    if change in ("gap", "stator_position"):
        assert not meshes, "Both rows depend on the shared physical interface."


def test_browser_download_returns_a_zip_and_gates_active_jobs(completed_project):
    from app_helpers import call

    project, registry = completed_project()
    app = create_app()
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    state = {"project": project, "valid": True}
    values = {("config", "data"): state, ("row-references", "data"): registry,
              ("download-project", "n_clicks"): 1}
    ready = call(app, "..download-project.disabled", "config.data", values)
    assert ready["download-project"]["disabled"] is False
    result = call(app, "..project-download.data", "download-project.n_clicks", values)
    download = result["project-download"]["data"]
    assert download["type"] == "application/zip"
    with ZipFile(BytesIO(base64.b64decode(download["content"]))) as archive:
        assert "project.yaml" in archive.namelist()
    values[("job", "data")] = registry["stator_1"]["volume"]
    ready = call(app, "..download-project.disabled", "job.data", values)
    assert ready["download-project"]["disabled"] is True
    result = call(app, "..project-download.data", "download-project.n_clicks", values)
    assert "project-download" not in result
    assert "active job" in result["project-download-message"]["children"]


def test_partial_stage_is_blocked_even_with_a_complete_stator(completed_project):
    project, registry = completed_project("axial", paired=True)
    registry["rotor_1"].pop("volume")
    with pytest.raises(ValueError, match="Every active row"):
        project_export.build_archive(project, registry)


def test_inconsistent_coordinate_scales_block_stage_export(completed_project):
    project, registry = completed_project("axial", paired=True)
    # Keep both saved jobs current, but give the rotor a different output scale.
    row = project["rows"][1]
    row["mesh"]["cgns"]["length_scale"] *= 2
    reference = registry[row["id"]]["volume"]
    request = flow.read_json(flow.directory(reference) / "request.json")
    request["project"] = project
    request["config"] = flow.resolved_configuration(project, row["id"], registry)
    reference["key"] = flow.keys(request["config"], registry[row["id"]]["nozzle"])["volume"]
    flow.write_json(flow.directory(reference) / "request.json", request)
    with pytest.raises(ValueError, match="same CGNS length_scale"):
        project_export.build_archive(project, registry)
