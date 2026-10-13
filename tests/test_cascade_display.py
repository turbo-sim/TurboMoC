"""Display controls without solving or generating meshes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("dash_vtk")
from turbo_moc.configuration import project as projects
from turbo_moc.app.application import create_app
from turbo_moc.app import display_panel as display


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
    changed = update([3], [[]], ["#ff0000"], ids, surface, color, saved, project)
    assert changed["stator_1"]["copies"] == 3
    assert changed["stator_1"]["colors"]["blade"] == "#ff0000"
    assert "hub" not in changed["stator_1"]["surfaces"]
    assert changed["rotor_1"] == saved["rotor_1"]
    assert saved["stator_1"]["colors"]["blade"] != "#ff0000"
    assert project == original


@pytest.mark.parametrize("pane", ["geometry", "mesh"])
def test_single_panel_selects_hidden_rows_and_preserves_settings(app, monkeypatch, pane):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    saved["rotor_1"].update(copies=4, surfaces=["blade"])
    saved["rotor_1"]["colors"]["blade"] = "#123456"
    original = deepcopy(saved)
    structure = next(entry["callback"].__wrapped__ for key, entry in app.callback_map.items()
                     if key.startswith(f"..{pane}-display-editors.children"))
    editors, options, selected = structure(project, saved, "rotor_1")
    assert selected == "rotor_1"
    assert [item["value"] for item in options] == ["stator_1", "rotor_1"]
    identifiers = [display.identifier("editor", pane, row["id"]) for row in projects.active_rows(project)]
    open_panel = callback(app, pane, '"type":"display-editor"')
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id=f"{pane}-configure-display"))
    styles, panel, opened = open_panel(1, 0, selected, identifiers, False)
    assert opened and panel == {} and styles == [{"display": "none"}, {}]
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id=f"{pane}-display-row"))
    styles, panel, opened = open_panel(1, 0, "stator_1", identifiers, opened)
    assert opened and styles == [{}, {"display": "none"}]
    assert saved == original
    # Switching design mode reconciles a removed row without losing preferences.
    _, options, selected = structure(projects.set_design_mode(project, "stator"), saved, "rotor_1")
    assert selected == "stator_1" and len(options) == 1
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id=f"{pane}-close-display"))
    assert open_panel(1, 1, selected, identifiers[:1], True)[1:] == ({"display": "none"}, False)


def test_reset_display_uses_selected_mesh_row_defaults(app, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    row = projects.get_row(project, "rotor_1")
    settings = display.defaults(row)
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id={"row": "rotor_1", "action": "reset"}))
    ids = [display.identifier("surface", "mesh", "rotor_1", surface=name) for name in display.BOUNDARIES]
    colors = [display.identifier("color", "mesh", "rotor_1", surface=name) for name in display.BOUNDARIES]
    result = callback(app, "mesh", '"type":"display-surface"')([1], ids, colors,
        {"project": project}, None)
    assert result[0] == [[name] if name in settings["surfaces"] else [] for name in display.BOUNDARIES]
    assert result[1] == [settings["colors"][name] for name in display.BOUNDARIES]
    assert result[2:] == (settings["copies"], ["edges"] if settings["edges"] else [])


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
