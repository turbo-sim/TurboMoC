"""Independent report selections and correct pooling of passage statistics."""
from pathlib import Path

import numpy as np
import pytest

from turbo_moc.app import application, mesh_reporting as reporting
from turbo_moc.configuration import project as projects
from app_helpers import call, components

CASES = Path(__file__).resolve().parents[1] / "examples/apps/cases"


def entry(row_id, values, status="Current"):
    arrays = {"skewness": np.array(values), "orthogonality": 1 - np.array(values),
              "aspect_ratio": 1 + 100 * np.array(values)}
    return {"id": row_id, "label": row_id, "status": status, "arrays": arrays,
        "summary": {"cells": {"hexahedra": len(values)}, "volume_nodes": 8 * len(values),
            "quality": {"min_sicn": .1}}, "report": {"definitions": {name: name for name in arrays}}}


def test_total_pools_samples_and_reports_missing_and_outdated_rows():
    entries = [entry("stator_1", [0, .1, .2]), entry("rotor_1", [.9], "Outdated")]
    pooled = reporting.aggregate(entries)
    assert pooled["skewness"]["mean"] == pytest.approx(.3)
    assert pooled["skewness"]["p95"] == pytest.approx(np.quantile([0, .1, .2, .9], .95))
    assert pooled["orthogonality"]["min"] == pytest.approx(.1)
    assert pooled["aspect_ratio"]["max"] == 91
    assert all(sum(metric["counts"]) == 4 for metric in pooled.values())
    rows = [{"id": e["id"], "label": e["label"]} for e in entries]
    text = str(reporting.total_panel(rows, entries, []))
    assert "2 of 2 row meshes available; 1 current" in text
    assert "outdated mesh included" in text
    assert "Partial / outdated" in text
    assert "Cells with skewness > 0.8" in text
    partial = str(reporting.total_panel(rows, entries[:1], [("Rotor", "not generated")]))
    assert "1 of 2 row meshes available" in partial and "not generated" in partial


def test_report_staleness_detects_recomputed_predecessor(monkeypatch):
    refs = {stage: {"key": stage, "session": "a" * 32, "job": stage}
            for stage in ("nozzle", "reference", "volume")}
    monkeypatch.setattr(reporting.flow, "resolved_configuration", lambda *args: {})
    monkeypatch.setattr(reporting.flow, "keys", lambda *args: {stage: stage for stage in refs})
    monkeypatch.setattr(reporting.flow, "directory", lambda *args: Path("unused"))
    request = {"references": {"nozzle": refs["nozzle"].copy(), "reference": refs["reference"].copy()}}
    monkeypatch.setattr(reporting.flow, "read_json", lambda *args: request)
    assert reporting.mesh_status({}, {"id": "rotor_1"}, {"rotor_1": refs}) == "Current"
    refs["reference"]["job"] = "recomputed"
    assert reporting.mesh_status({}, {"id": "rotor_1"}, {"rotor_1": refs}) == "Outdated"


def test_dropdown_scopes_and_report_are_independent(monkeypatch):
    app = application.create_app(CASES / "stator_rotor.yaml")
    layout = components(app.server.test_client().get("/_dash-layout").get_json())
    assert layout["calculation-scope"]["value"] == "all"
    assert layout["plot-scope"]["value"] == "all"
    assert layout["report-scope"]["value"] == "all"
    assert "geometry-configure-display" in layout and "mesh-configure-display" in layout
    sidebar = components(layout["sidebar"]["children"])
    assert "calculation-scope" in sidebar and "plot-scope" in sidebar
    assert "Show" in str(sidebar["interspace-note"]["children"])
    for pane in ("geometry", "mesh"):
        scene_callback = next(v for k, v in app.callback_map.items() if k.startswith(f"..{pane}-scene.children"))
        assert any(item["id"] == "plotted-rows" for item in scene_callback["inputs"])
        assert '"type":"display-visible"' not in str(app.callback_map)
    project = layout["base-project"]["data"]
    response = call(app, "..calculation-scope.options", "plot-scope.value", {
        ("base-project", "data"): project, ("plot-scope", "value"): "rotor_1",
        ("calculation-scope", "value"): "stator_1", ("report-scope", "value"): "all"})
    assert response["plotted-rows"]["value"] == ["rotor_1"]
    assert response["calculation-scope"]["value"] == "stator_1"
    assert response["report-scope"]["value"] == "all"
    # A mode change reconciles unavailable selections to all active rows.
    callback = next(v["callback"].__wrapped__ for k, v in app.callback_map.items()
                    if k.startswith("..calculation-scope.options"))
    changed = callback(projects.set_design_mode(project, "stator"), "rotor_1", "rotor_1", "rotor_1")
    assert changed[1] == changed[5] == changed[7] == "all"
    assert changed[3] == ["stator_1"]
    report_callback = next(v for k, v in app.callback_map.items() if k.startswith("..volume-data.data"))
    assert not any(i["id"] in ("plotted-rows", "selected-row", "calculation-scope")
                   for i in report_callback["inputs"])
    stored = entry("stator_1", [0, .1])
    stored["summary"].update(geometry={"type": "axial", "num_blades": 28}, span_stations=[0, 1], elapsed_s=1)
    stored["report"].update(metrics=reporting.aggregate([stored]), worst_cells={})
    monkeypatch.setattr(application, "volume_artifact", lambda *args: ({}, stored["summary"], stored["report"]))
    monkeypatch.setattr(reporting, "metric_arrays", lambda *args: stored["arrays"])
    monkeypatch.setattr(reporting, "mesh_status", lambda *args: "Current")
    registry = {"stator_1": {"volume": {"session": "a" * 32, "job": "b" * 32}}}
    view = report_callback["callback"].__wrapped__
    data, total = view({"valid": True, "project": project, "row_id": "rotor_1"}, registry, "all")
    assert len(data["rows"]) == 1
    assert "1 of 2 row meshes available" in str(total)
    _, stator = view({"valid": True, "project": project, "row_id": "rotor_1"}, registry, "stator_1")
    assert "Minimum signed inverse condition number" in str(stator) and "P95" in str(stator)
    _, missing = view({"valid": True, "project": project, "row_id": "stator_1"}, registry, "rotor_1")
    assert "Generate a 3D mesh for this row" in missing
