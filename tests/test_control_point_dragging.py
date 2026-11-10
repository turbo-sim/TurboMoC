"""Reconstruction and app boundaries for the installed nurbspy editor."""
from copy import deepcopy
import base64

import numpy as np
import pytest
import yaml

from turbo_moc.app import control_point_editor as editor
from turbo_moc.geometry.stator_blade import parametrize_stator
from turbo_moc.configuration import project as projects

editor.nurbspy_api()


def test_editor_uses_installed_dependency_without_changing_import_paths(monkeypatch):
    import importlib.metadata
    from pathlib import Path
    import sys
    import nurbspy

    # The former local-development override must no longer affect the app.
    monkeypatch.setenv("TURBO_MOC_NURBSPY", str(Path(__file__).resolve().parents[2] / "nurbspy"))
    original_paths = list(sys.path)
    editor.nurbspy_api.cache_clear()
    curve_type, editor_type, assets = editor.nurbspy_api()
    assert sys.path == original_paths
    installed_package = Path(importlib.metadata.distribution("nurbspy").locate_file("nurbspy")).resolve()
    assert Path(nurbspy.__file__).resolve().parent == installed_package
    assert assets == installed_package / "web_assets"
    assert curve_type.__module__.startswith("nurbspy.")
    assert editor_type.__module__.startswith("nurbspy.")


@pytest.fixture
def session(small_design):
    nozzle, config = small_design
    wall = nozzle["wall_final"]
    base = parametrize_stator(wall["x"], wall["y"], **config["blade_geometry"])
    return editor.make_session(base, config["blade_geometry"], "baseline")


@pytest.mark.parametrize("mode", ["full", "semi"])
def test_xy_edits_reproduce_after_yaml_reload_without_accumulation(small_design, mode):
    nozzle, config = small_design
    inputs = {**config["blade_geometry"], "parametrization": mode,
              "control_point_displacements": {5: [0.01234567890123, -0.008], 9: [-0.002, 0.004]}}
    wall = nozzle["wall_final"]
    expected = parametrize_stator(wall["x"], wall["y"], **inputs)
    loaded = yaml.safe_load(yaml.safe_dump(inputs))
    actual = parametrize_stator(wall["x"], wall["y"], **loaded)
    assert actual == expected
    session = editor.make_session(parametrize_stator(wall["x"], wall["y"], **{
        k: v for k, v in inputs.items() if k not in editor.EDIT_FIELDS}), loaded, "base")
    assert editor.edits_from_model(session, session["scene"]["model"]) == session["edits"]


def test_noop_legacy_and_untouched_vectors_keep_exact_values(session):
    session["edits"] = {"control_point_offsets": {5: 0.0134567890123456}}
    model = editor.geometry_editor(session["blade"], editor.initial_blade(session)["control_points"]).initial_state
    assert editor.edits_from_model(session, model) == session["edits"]
    session["edits"] = {"control_point_displacements": {5: [0.0134567890123456, -0.00712345678901234]}}
    model = editor.geometry_editor(session["blade"], editor.initial_blade(session)["control_points"]).initial_state
    model["control_points"][0][9] += 0.002
    assert editor.edits_from_model(session, model)["control_point_displacements"][5] == session["edits"][
        "control_point_displacements"][5]


def test_no_edits_and_reset_remain_zero(session):
    assert editor.edits_from_model(session, session["scene"]["model"]) == session["edits"]
    session["edits"] = {"control_point_offsets": {5: 0.01}}
    baseline_model = editor.geometry_editor(session["blade"]).initial_state
    assert editor.edits_from_model(session, baseline_model) == {"control_point_displacements": {}}


@pytest.mark.parametrize("mutation", ["locked", "z", "weight", "knots", "nonfinite"])
def test_forbidden_model_edits_are_rejected(session, mutation):
    model = deepcopy(session["scene"]["model"])
    if mutation == "locked":
        model["control_points"][0][0] += 0.1
    elif mutation == "z":
        model["control_points"][2][5] = 0.1
    elif mutation == "weight":
        model["weights"][5] = 2.0
    elif mutation == "knots":
        model["knots"][4] += 0.001
    else:
        model["control_points"][0][5] = float("nan")
    with pytest.raises(ValueError):
        editor.edits_from_model(session, model)


def test_intersection_cannot_be_applied(session):
    model = deepcopy(session["scene"]["model"])
    model["control_points"][0][7] += 100.0
    with pytest.raises(ValueError, match="intersects"):
        editor.edits_from_model(session, model)


def test_vector_displacements_survive_project_normalization():
    project = projects.default_project()
    geometry = project["rows"][0]["geometry"]["blade_geometry"]
    geometry["control_point_displacements"] = {"6": [0.01, -0.02], "7": [0, 0]}
    loaded = projects.validate_project(projects.normalize_project(yaml.safe_load(yaml.safe_dump(project))))
    assert loaded["rows"][0]["geometry"]["blade_geometry"]["control_point_displacements"] == {6: [0.01, -0.02]}


def test_app_evaluates_and_applies_drag_and_rejects_stale_commit(monkeypatch, small_design, tmp_path):
    from app_helpers import call, row_state
    from turbo_moc.app import create_app, workflow as flow
    nozzle, config = small_design
    registry = {"stator_1": {"nozzle": {"session": "test", "job": "moc"}}}
    state = row_state(config, registry)
    monkeypatch.setattr(flow, "resolved_configuration", lambda *args: config)
    monkeypatch.setattr(flow, "load_nozzle", lambda *args: nozzle)
    app = create_app(output_dir=tmp_path)
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    opened = call(app, "..cp-session.data", "cp-edit-open.n_clicks", {
        ("cp-edit-open", "n_clicks"): 1, ("config", "data"): state, ("row-references", "data"): registry})
    session = opened["cp-session"]["data"]
    model = deepcopy(session["scene"]["model"])
    model["control_points"][0][5] += 0.01
    packet = {"revision": 1, "action": "edit", "quality": "full",
              "control_points": model["control_points"], "weights": model["weights"]}
    response = client.post("/_turbomoc_cp/evaluate", json={"session": session, "packet": packet})
    assert response.status_code == 200
    result = response.get_json()
    assert result["pitch"] > 0
    assert result["revision"] == 1 and result["quality"] == "full"
    assert result["draw_quality"] == "fine"
    assert result["topologies"]["preview"] == result["topologies"]["full"]
    assert len(base64.b64decode(result["vertices"])) == result["topologies"]["full"]["count"] * 3 * 4
    assert client.get("/_turbomoc_nurbspy/explorer.js").status_code == 200
    apply_callback = next(key for key in app.callback_map if "base-project.data" in key and "cp-message.children" in key)
    applied = call(app, apply_callback, "cp-commit.data", {
        ("cp-commit", "data"): {"identity": session["identity"], "model": model},
        ("cp-session", "data"): session, ("config", "data"): state, ("row-references", "data"): registry})
    project = applied["base-project"]["data"]
    assert project["rows"][0]["geometry"]["blade_geometry"]["control_point_displacements"]["5"][0] != 0
    expected = editor.edits_from_model(session, model)["control_point_displacements"]
    assert np.allclose(expected[5], [0.01 / editor.blade_chord(session["blade"]), 0])
    stale = {**session, "identity": "old"}
    rejected = call(app, apply_callback, "cp-commit.data", {
        ("cp-commit", "data"): {"identity": "old", "model": model},
        ("cp-session", "data"): stale, ("config", "data"): state, ("row-references", "data"): registry})
    assert "base-project" not in rejected
    assert "changed" in rejected["cp-message"]["children"]


def test_neighbour_spacing_uses_production_pitch_at_both_display_qualities(session, small_design):
    from turbo_moc.geometry.blade_row import build_design
    from turbo_moc.app import create_app
    nozzle, config = small_design
    design = build_design(nozzle, config)
    editor.use_design_passage(session, design)
    # This fixture deliberately changes blade count, so the MoC natural pitch
    # would give a misleading passage if used directly.
    assert not np.isclose(session["pitch"], session["blade"]["pitch"])
    model = deepcopy(session["scene"]["model"])
    model["control_points"][1][7] += 0.05
    client = create_app().server.test_client()
    pitches = []
    for quality in ("preview", "full"):
        response = client.post("/_turbomoc_cp/evaluate", json={"session": session, "packet": {
            "revision": 1, "action": "edit", "quality": quality,
            "control_points": model["control_points"], "weights": model["weights"]}})
        assert response.status_code == 200
        pitches.append(response.get_json()["pitch"])
    assert pitches[0] == pitches[1]
    edits = editor.edits_from_model(session, model)
    edited = editor.apply_control_point_displacements(session["blade"], edits["control_point_displacements"])
    actual = build_design(nozzle, {**config, "blade_geometry": {**config["blade_geometry"], **edits}})
    native_chord = np.ptp(editor.blade_contour(edited)[:, 1])
    placed_chord = np.ptp(editor.blade_contour(actual.blade)[:, 1])
    assert np.isclose(pitches[0], actual.blade["pitch"] * native_chord / placed_chord)


def test_automatic_sampling_round_trip_and_rejected_edits(session):
    from turbo_moc.app import create_app
    client = create_app().server.test_client()
    model = deepcopy(session["scene"]["model"])
    packet = {"revision": 1, "action": "edit", "quality": "full", "draw_quality": "fine",
              "sampling": session["scene"]["sampling"],
              "control_points": model["control_points"], "weights": model["weights"]}
    result = client.post("/_turbomoc_cp/evaluate", json={"session": session, "packet": packet}).get_json()
    packet.update(revision=2, quality="preview", sampling=result["sampling"])
    preview = client.post("/_turbomoc_cp/evaluate", json={"session": session, "packet": packet})
    assert preview.status_code == 200
    assert preview.get_json()["quality"] == "preview"
    assert result["sampling"]["preview"] == result["sampling"]["full"]
    assert preview.get_json()["vertices"] == result["vertices"]
    assert len(base64.b64decode(preview.get_json()["vertices"])) == result["topologies"]["preview"]["count"] * 3 * 4

    for index, expected in ((0, "locked"), (7, "intersects")):
        invalid = deepcopy(packet)
        invalid["quality"] = "full"
        invalid["control_points"][0][index] += 100
        response = client.post("/_turbomoc_cp/evaluate", json={"session": session, "packet": invalid})
        assert response.status_code == 422
        assert expected in response.get_json()["error"]
