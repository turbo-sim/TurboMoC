"""Natural passage sizing and one-time editable blade-count defaults."""
from copy import deepcopy
import base64
from math import ceil, log, pi

import pytest
import yaml
from dash import no_update

from turbo_moc.configuration import project as projects
from turbo_moc.app import create_app, blade_counts, calculations, workflow as flow
from turbo_moc.app.application import read_uploaded_project


@pytest.mark.parametrize("family", ["axial", "radial", "linear"])
def test_natural_count_uses_native_pitch_and_conformal_chord(family):
    project = projects.default_project(family)
    config = projects.row_configuration(project, "stator_1")
    dimensions = config["meridional_channel"]
    native = {"pitch": 2., "chord": 4.}
    natural = blade_counts.sizing(config, native)
    if family == "linear":
        assert natural["natural_num_blades"] is None
        assert natural["suggested_num_blades"] is None
        assert natural["natural_pitch_mm"] == .5 * dimensions["meridional_chord"]
    else:
        if family == "axial":
            prime = (.5 * (dimensions["hub_chord"] + dimensions["tip_chord"])) / (
                dimensions["hub_radius"] + dimensions["blade_height"] / 2)
        else:
            prime = abs(log(dimensions["outlet_radius"] / dimensions["inlet_radius"]))
        assert natural["natural_num_blades"] == pytest.approx(2 * pi / (.5 * prime))
        assert natural["suggested_num_blades"] == ceil(natural["natural_num_blades"])
        dimensions["num_blades"] += 7
        assert blade_counts.sizing(config, native) == natural


def test_native_pitch_is_extracted_before_annular_overrides():
    stator = projects.row_configuration(projects.default_project(), "stator_1")
    template = {"pitch": 3., "blade_curve": {"x": [0., 1., 1., 0., 0.], "y": [0., -2., -4., -2., 0.]},
                "trailing_edge": {"x": [], "y": []}}
    assert blade_counts.native_passage(stator, {"converged": True}, template=template) == {"pitch": 3., "chord": 4.}
    rotor = projects.row_configuration(projects.set_design_mode(projects.default_project(), "rotor"), "rotor_1")
    assert blade_counts.native_passage(rotor, {"pitch": 5., "blade": {"x": [0., 8., 0.], "y": [0., 1., 0.]}}) == {
        "pitch": 5., "chord": 8.}


def callbacks():
    app = create_app()
    return {key: value["callback"].__wrapped__ for key, value in app.callback_map.items()
            if "callback" in value}


def test_defaults_apply_once_per_solve_and_manual_edits_survive(monkeypatch):
    callback = next(value for key, value in callbacks().items() if key.startswith(
        '..{"row":["ALL"],"type":"actual-blade-count"}.children'))
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    state = {"project": project, "valid": True}
    identifiers = [{"type": "actual-blade-count", "row": row} for row in ("stator_1", "rotor_1")]
    registry = {row: {"nozzle": {"session": "a" * 32, "job": job * 32}}
                for row, job in (("stator_1", "b"), ("rotor_1", "c"))}
    monkeypatch.setattr(flow, "resolved_configuration", lambda *args: {})
    monkeypatch.setattr(blade_counts, "from_reference", lambda *args: {"suggested_num_blades": 43})
    original = deepcopy(project)
    controls, applied = callback(registry, state, identifiers, {})
    assert all(control.children[1].value == 43 for control in controls)
    assert project == original
    project["rows"][0]["geometry"]["meridional_channel"]["num_blades"] = 37
    registry["stator_1"]["reference"] = {"job": "d" * 32}
    controls, _ = callback(registry, state, identifiers, applied)
    assert controls == [no_update, no_update]
    registry["stator_1"]["nozzle"]["job"] = "e" * 32
    controls, _ = callback(registry, state, identifiers, applied)
    assert controls[0].children[1].value == 43 and controls[1] is no_update


def test_failed_or_stale_results_do_not_change_the_actual_count(monkeypatch):
    callback = next(value for key, value in callbacks().items() if key.startswith(
        '..{"row":["ALL"],"type":"actual-blade-count"}.children'))
    def stale(*args):
        raise ValueError("Compute the current blade design first")
    monkeypatch.setattr(flow, "resolved_configuration", stale)
    controls, applied = callback({"stator_1": {"nozzle": {"session": "a", "job": "b"}}},
        {"valid": True, "project": {}}, [{"row": "stator_1"}], {})
    assert controls == [no_update] and applied == {}


def test_count_controls_update_through_dash_and_save_as_regular_inputs(monkeypatch):
    app = create_app()
    project = projects.default_project()
    registry = {"stator_1": {"nozzle": {"session": "a" * 32, "job": "b" * 32}}}
    monkeypatch.setattr(flow, "resolved_configuration", lambda *args: {})
    monkeypatch.setattr(blade_counts, "from_reference", lambda *args: {"suggested_num_blades": 43})
    key = next(key for key in app.callback_map if key.startswith(
        '..{"row":["ALL"],"type":"actual-blade-count"}.children'))
    identifier = {"row": "stator_1", "type": "actual-blade-count"}
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": key, "outputs": [[{"id": identifier, "property": "children"}],
            {"id": "applied-blade-count-jobs", "property": "data"}],
        "changedPropIds": ["row-references.data"],
        "inputs": [{"id": "row-references", "property": "data", "value": registry}],
        "state": [{"id": "config", "property": "data", "value": {"project": project, "valid": True}},
            {"id": '{"row":["ALL"],"type":"actual-blade-count"}', "property": "id", "value": [identifier]},
            {"id": "applied-blade-count-jobs", "property": "data", "value": {}}],
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    control = response.get_json()["response"]['{"row":"stator_1","type":"actual-blade-count"}']["children"]
    widget = control["props"]["children"][1]["props"]
    edited = flow.edited_project(project, [widget["value"]], [widget["id"]])
    assert projects.validate_project(edited)["rows"][0]["geometry"]["meridional_channel"]["num_blades"] == 43
    upload = "data:text/yaml;base64," + base64.b64encode(yaml.safe_dump(edited).encode()).decode()
    assert read_uploaded_project(upload) == edited


def test_queued_success_applies_the_count_to_the_next_project_snapshot(tmp_path, monkeypatch):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    handle = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_1", "action": "nozzle",
              "queue": {"project": project, "index": 0, "log": ""}}
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    folder = flow.directory(handle)
    folder.mkdir(parents=True)
    flow.write_json(folder / "result.json", {"blade_sizing": {"suggested_num_blades": 43}})
    monkeypatch.setattr(calculations, "cancellation_requested", lambda *args: False)
    captured = []
    monkeypatch.setattr(calculations, "_launch_next", lambda session, queue, registry: captured.append(queue))
    calculations.advance(handle, {}, "")
    assert captured[0]["project"]["rows"][0]["geometry"]["meridional_channel"]["num_blades"] == 43
    assert project["rows"][0]["geometry"]["meridional_channel"]["num_blades"] == 31
