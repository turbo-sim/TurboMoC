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


def test_preferences_are_independent_and_do_not_edit_design(app, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    original = deepcopy(project)
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    update = app.callback_map["geometry-display.data"]["callback"].__wrapped__
    ids = [display.identifier("copies", "geometry", "stator_1")]
    surface = [display.identifier("surface", "geometry", "stator_1", surface="hub")]
    color = [display.identifier("color", "geometry", "stator_1", surface="blade")]
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_prop_ids={str(i): item for i, item in enumerate(ids + surface + color)}))
    changed = update([3], [[]], ["#ff0000"], ids, surface, color, saved, project, ["stator_1"])
    assert changed["stator_1"]["copies"] == 3
    assert changed["stator_1"]["colors"]["blade"] == "#ff0000"
    assert "hub" not in changed["stator_1"]["surfaces"]
    assert changed["rotor_1"] == saved["rotor_1"]
    assert saved["stator_1"]["colors"]["blade"] != "#ff0000"
    assert project == original


@pytest.mark.parametrize("pane", ["geometry", "mesh"])
def test_panel_shows_first_selected_row_without_overwriting_settings(app, pane):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    project["rows"][1]["geometry"]["meridional_channel"]["num_blades"] = 17
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    saved["rotor_1"].update(copies=4, surfaces=["blade"])
    original = deepcopy(saved)
    structure = next(entry["callback"].__wrapped__ for key, entry in app.callback_map.items()
                     if key.startswith(f"..{pane}-display-editors.children"))
    editors, counts = structure(project, ["rotor_1"], saved)
    assert len(editors) == 1 and editors[0].id["row"] == "rotor_1"
    assert len(counts) == 1 and counts[0].children[0].children[1].value == 4
    editors, counts = structure(project, ["all", "stator_1", "rotor_1"], saved)
    assert len(editors) == 1 and editors[0].id["row"] == "stator_1"
    assert [control.children[0].children[1].id["row"] for control in counts] == ["stator_1", "rotor_1"]
    assert [control.children[0].children[1].max for control in counts] == [31, 17]
    assert structure(project, [], saved)[0] == []
    assert saved == original


def test_group_edit_changes_only_the_edited_field(app, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    ids = [display.identifier("copies", "geometry", "stator_1")]
    color = [display.identifier("color", "geometry", "stator_1", surface="blade")]
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_prop_ids={"copies": ids[0]}))
    update = app.callback_map["geometry-display.data"]["callback"].__wrapped__
    changed = update([4], [], ["#ff0000"], ids, [], color, saved, project, ["all", "stator_1", "rotor_1"])
    for row in saved:
        assert changed[row]["copies"] == (4 if row == "stator_1" else saved[row]["copies"])
        assert changed[row]["colors"] == saved[row]["colors"]
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_prop_ids={"color": color[0]}))
    colored = update([4], [], ["#ff0000"], ids, [], color, changed, project, ["all", "stator_1", "rotor_1"])
    assert all(settings["colors"]["blade"] == "#ff0000" for settings in colored.values())
    assert colored["rotor_1"]["copies"] == saved["rotor_1"]["copies"]


@pytest.mark.parametrize("chosen,previous,expected", [
    (["all"], [], ["s", "r"]),
    (["all", "s"], ["s", "r"], ["s"]),
    (["s", "r"], ["s", "r"], []),
    (["s", "r"], ["s"], ["s", "r"]),
    ([], ["s"], []),
    (["all", "s", "r"], ["s", "r"], ["s", "r"]),
])
def test_all_rows_checkbox(chosen, previous, expected):
    assert display.selected_rows(chosen, previous, ["s", "r"]) == expected


def test_paired_boundary_default_colors():
    assert display.COLORS["hub"] == display.COLORS["shroud"]
    assert display.COLORS["periodic_left"] == display.COLORS["periodic_right"]
    assert display.COLORS["inlet"] == display.COLORS["outlet"]


def test_reset_display_uses_each_selected_rows_defaults(app, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    saved = {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    for settings in saved.values():
        settings["colors"]["blade"] = "#123456"
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id={"row": "stator_1", "action": "reset"}))
    reset = next(entry["callback"].__wrapped__ for key, entry in app.callback_map.items()
                 if key.startswith("..mesh-display.data@"))
    result, editors, counts = reset([1], ["all", "stator_1", "rotor_1"], project, saved)
    assert result == {row["id"]: display.defaults(row) for row in projects.active_rows(project)}
    assert result["stator_1"]["colors"]["blade"] != result["rotor_1"]["colors"]["blade"]
    assert len(counts) == 2
    result, _, _ = reset([1], ["stator_1"], project, saved)
    assert result["rotor_1"] == saved["rotor_1"]


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


@pytest.mark.parametrize("value,expected", [("5", 5), ("999", 12), ("0", 1), ("", 1), ("invalid", 1)])
def test_manual_blade_count_is_validated_and_synchronized(app, monkeypatch, value, expected):
    project = projects.default_project()
    monkeypatch.setattr(display, "ctx", SimpleNamespace(triggered_id={"type": "display-copies"}))
    result = callback(app, "mesh", '"type":"display-slider"')(value, 3, {"project": project},
        {"rows": [{"id": "stator_1", "num_blades": 12}]},
        display.identifier("copies", "mesh", "stator_1"))
    assert result[:4] == (expected, 12, expected, 12)
