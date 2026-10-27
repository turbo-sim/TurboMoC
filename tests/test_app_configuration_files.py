"""Exercise YAML load/save through Dash's actual callback endpoint."""
import base64

import pytest
import yaml

from app_helpers import call

pytest.importorskip("dash_vtk")

from turbo_moc.app import create_app
from turbo_moc.configuration import project as projects


def upload(project):
    content = yaml.safe_dump(project, sort_keys=False).encode("utf-8")
    return "data:text/yaml;base64," + base64.b64encode(content).decode("ascii")


@pytest.fixture
def app():
    result = create_app()
    assert result.server.test_client().get("/_dash-layout").status_code == 200
    return result


def test_load_paired_yaml_rebuilds_settings_and_clears_artifacts(app):
    base = projects.default_project()
    project = projects.set_design_mode(projects.default_project("radial"), "stator_rotor")
    project["rows"][0]["mesh"]["numerics"]["blade_size"] = 0.73
    loaded = call(app, "..loaded-project.data", "configuration-upload.contents", {
        ("configuration-upload", "contents"): upload(project),
        ("configuration-upload", "filename"): "stage.yaml",
        ("base-project", "data"): base,
    })
    assert loaded["loaded-project"]["data"] == project
    assert loaded["row-references"]["data"] == {}
    assert loaded["job"]["data"] is None
    assert len(loaded["session"]["data"]) == 32
    changed = call(app, "..base-project.data", "loaded-project.data", {
        ("loaded-project", "data"): project,
        ("design-mode", "value"): "stator",
        ("machine-family", "value"): "axial",
        ("base-project", "data"): base,
        ("selected-row", "value"): "stator_1",
    })
    assert changed["base-project"]["data"] == project
    assert changed["design-mode"]["value"] == "stator_rotor"
    assert changed["machine-family"]["value"] == "radial"
    assert len(changed["selected-row"]["options"]) == 2


@pytest.mark.parametrize("contents", ["invalid", upload({"machine": "bad"}),
    "data:text/yaml;base64," + base64.b64encode(b"rows: [").decode("ascii")])
def test_invalid_upload_preserves_current_project_and_artifacts(app, contents):
    result = call(app, "..loaded-project.data", "configuration-upload.contents", {
        ("configuration-upload", "contents"): contents,
        ("base-project", "data"): projects.default_project(),
    })
    assert set(result) == {"configuration-message"}
    assert "Cannot load configuration" in result["configuration-message"]["children"]


def test_save_preserves_each_row_and_excludes_job_state(app):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    project["rows"][1]["mesh"]["numerics"]["span_layers"] = 17
    project["rows"][0]["geometry"]["blade_geometry"]["metal_angle_out"] = 62.0
    saved = call(app, "..loaded-project.data", "save-configuration.n_clicks", {
        ("save-configuration", "n_clicks"): 1,
        ("base-project", "data"): projects.default_project(),
        ("config", "data"): {"project": project, "valid": True},
    })
    data = saved["configuration-download"]["data"]
    assert data["filename"].endswith(".yaml")
    assert yaml.safe_load(data["content"]) == project
    assert "loaded-project" not in saved and "session" not in saved


def test_loading_during_a_job_preserves_the_running_job(app):
    result = call(app, "..loaded-project.data", "configuration-upload.contents", {
        ("configuration-upload", "contents"): upload(projects.default_project("radial")),
        ("job", "data"): {"job": "a" * 32},
        ("base-project", "data"): projects.default_project(),
    })
    assert set(result) == {"configuration-message"}
    assert "active job" in result["configuration-message"]["children"]


def test_loaded_settings_are_not_overwritten_by_previous_sidebar_values(app):
    project = projects.default_project()
    project["rows"][0]["mesh"]["numerics"]["blade_size"] = 0.73
    key = next(key for key in app.callback_map if key.startswith("..config.data"))
    callback = app.callback_map[key]
    pattern = next(item["id"] for item in callback["inputs"] if item["id"].startswith("{"))
    result = call(app, "..config.data", "base-project.data", {
        ("base-project", "data"): project, ("selected-row", "value"): "stator_1",
        (pattern, "value"): [0.25],
        (pattern, "id"): [{"type": "parameter", "row": "stator_1", "path": "mesh.numerics.blade_size"}],
    })
    assert result["config"]["data"]["config"]["mesh"]["numerics"]["blade_size"] == 0.73


def test_invalid_settings_cannot_be_saved(app):
    project = projects.default_project()
    result = call(app, "..loaded-project.data", "save-configuration.n_clicks", {
        ("save-configuration", "n_clicks"): 1,
        ("base-project", "data"): project,
        ("config", "data"): {"project": project, "valid": False},
    })
    assert set(result) == {"configuration-message"}
    assert "invalid settings" in result["configuration-message"]["children"]


def test_load_yaml_with_control_point_offsets_rebuilds_settings_and_saves_them(app):
    base = projects.default_project()
    project = projects.default_project()
    project["rows"][0]["geometry"]["blade_geometry"]["control_point_offsets"] = {"16": 0.035, "17": 0.065}
    loaded = call(app, "..loaded-project.data", "configuration-upload.contents", {
        ("configuration-upload", "contents"): upload(project),
        ("configuration-upload", "filename"): "edited.yaml",
        ("base-project", "data"): base,
    })["loaded-project"]["data"]
    changed = call(app, "..base-project.data", "loaded-project.data", {
        ("loaded-project", "data"): loaded,
        ("design-mode", "value"): "stator",
        ("machine-family", "value"): "axial",
        ("base-project", "data"): base,
        ("selected-row", "value"): "stator_1",
    })
    assert "2 control points modified" in str(changed["settings"]["children"])
    saved = call(app, "..loaded-project.data", "save-configuration.n_clicks", {
        ("save-configuration", "n_clicks"): 1,
        ("base-project", "data"): base,
        ("config", "data"): {"project": changed["base-project"]["data"], "valid": True},
    })
    offsets = yaml.safe_load(saved["configuration-download"]["data"]["content"])["rows"][0]["geometry"][
        "blade_geometry"]["control_point_offsets"]
    assert {int(key): value for key, value in offsets.items()} == {16: 0.035, 17: 0.065}
