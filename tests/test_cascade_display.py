"""Display controls without solving or generating meshes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("dash_vtk")
from turbo_moc import configuration_project as projects
from turbo_moc.apps.app.application import create_app
from turbo_moc.apps.app import display_panel as display


@pytest.fixture
def app():
    return create_app()


def callback(app, pane, prefix):
    return next(entry["callback"].__wrapped__ for key, entry in app.callback_map.items()
                if prefix in key and f'"pane":"{pane}"' in key)


def test_preferences_are_independent_and_do_not_edit_design(app):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    original = deepcopy(project)
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    update = app.callback_map["geometry-display.data"]["callback"].__wrapped__
    ids = [display.identifier("copies", "geometry", "stator_1")]
    surface = [display.identifier("surface", "geometry", "stator_1", surface="hub")]
    color = [display.identifier("color", "geometry", "stator_1", surface="blade")]
    changed = update([3], [], [[]], ["#ff0000"], ids, [], surface, color, saved, project)
    assert changed["stator_1"]["copies"] == 3
    assert changed["stator_1"]["colors"]["blade"] == "#ff0000"
    assert "hub" not in changed["stator_1"]["surfaces"]
    assert changed["rotor_1"] == saved["rotor_1"]
    assert saved["stator_1"]["colors"]["blade"] != "#ff0000"
    assert project == original


def test_copy_geometry_only_changes_selected_mesh_row(app, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    row = projects.get_row(project, "rotor_1")
    settings = display.defaults(row)
    settings.update(copies=4, surfaces=["blade"], edges=False)
    settings["colors"]["blade"] = "#123456"
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id={"row": "rotor_1", "action": "copy"}))
    ids = [display.identifier("surface", "mesh", "rotor_1", surface=name) for name in display.BOUNDARIES]
    colors = [display.identifier("color", "mesh", "rotor_1", surface=name) for name in display.BOUNDARIES]
    result = callback(app, "mesh", '"type":"display-surface"')([1], ids, colors,
        {"project": project}, None, {"rotor_1": settings})
    assert result[0] == [["blade"] if name == "blade" else [] for name in display.BOUNDARIES]
    assert result[1][0] == "#123456"
    assert result[2:] == (4, [])


@pytest.mark.parametrize("pane,expected", [("geometry", 31), ("mesh", 12)])
def test_counts_clamp_to_each_dataset(app, monkeypatch, pane, expected):
    project = projects.default_project()
    row = projects.get_row(project, "stator_1")
    row["geometry"]["meridional_channel"]["num_blades"] = 31
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id={"type": "display-slider"}))
    result = callback(app, pane, '"type":"display-slider"')(2, 100, {"project": project},
        {"rows": [{"id": "stator_1", "is_linear": False, "num_blades": 12}]},
        display.identifier("copies", pane, "stator_1"))
    assert result[:4] == (expected,) * 4
    row["geometry"]["meridional_channel"]["type"] = "linear"
    assert display.limit(row, "geometry") == 25
