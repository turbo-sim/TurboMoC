"""Fluid selection, pair persistence and real thermodynamic inlet conversion."""
from copy import deepcopy
import json
import math

import CoolProp.CoolProp as CP
import jaxprop as jxp
import pytest
import yaml

from app_helpers import call
from turbo_moc import configuration_project as projects
from turbo_moc.apps.app import create_app, workflow as flow
from turbo_moc.fluid_inputs import INLET_FIELDS, inlet_values, solver_parameters


def parameter(row, section, name):
    return {"type": "parameter", "row": row, "path": f"{section}.{name}"}


def nodes(value):
    if isinstance(value, dict):
        if "props" in value:
            yield value
        for item in value.values():
            yield from nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from nodes(item)


def test_fluid_dropdowns_and_pair_controls_cover_both_rows(tmp_path):
    path = tmp_path / "paired.yaml"
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    # Keep a supported alias from historical files visible in the dropdown.
    project["rows"][0]["geometry"]["method_of_characteristics"]["fluid_name"] = "CO2"
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    app = create_app(path)
    layout = app.server.test_client().get("/_dash-layout").get_json()
    controls = [node for node in nodes(layout) if isinstance(node["props"].get("id"), dict)]
    for row, section, pair, first, second in (
            ("stator_1", "method_of_characteristics", "PQ_INPUTS", 251300., .025),
            ("rotor_1", "rotor_design", "PT_INPUTS", 5e5, 400.)):
        fields = {node["props"]["id"]["path"]: node for node in controls
                  if node["props"]["id"].get("type") == "parameter"
                  and node["props"]["id"]["row"] == row}
        fluid = fields[f"{section}.fluid_name"]
        assert fluid["type"] == "Dropdown"
        assert set(CP.FluidsList()) <= set(fluid["props"]["options"])
        assert fluid["props"]["value"] in fluid["props"]["options"]
        selection = fields[f"{section}.inlet_property_pair"]["props"]
        assert {option["value"] for option in selection["options"]} == set(jxp.INPUT_PAIRS)
        assert selection["value"] == pair
        assert fields[f"{section}.inlet_property_1"]["props"]["value"] == first
        assert fields[f"{section}.inlet_property_2"]["props"]["value"] == second


@pytest.mark.parametrize("section,row", [("method_of_characteristics", "stator_1"), ("rotor_design", "rotor_1")])
def test_pair_selection_updates_labels_and_units(section, row):
    app = create_app()
    key = next(key for key in app.callback_map if "fluid-property-label" in key and section in key)
    callback = app.callback_map[key]
    outputs = [{"id": {**output.component_id, "row": row}, "property": output.component_property}
               for output in callback["output"]]
    for pair, first, second in (
            ("PT_INPUTS", "pressure [Pa]", "temperature [K]"),
            ("HmassP_INPUTS", "specific enthalpy [J/kg]", "pressure [Pa]"),
            ("QT_INPUTS", "vapor quality [kg/kg]", "temperature [K]")):
        response = app.server.test_client().post("/_dash-update-component", json={
            "output": key, "outputs": outputs, "state": [],
            "inputs": [{"id": parameter(row, section, "inlet_property_pair"), "property": "value", "value": pair}],
            "changedPropIds": [],
        })
        assert response.status_code == 200, response.get_data(as_text=True)
        result = response.get_json()["response"]
        def output_value(output):
            identifier = json.dumps(output["id"], sort_keys=True, separators=(",", ":"))
            return result[identifier][output["property"]]
        assert first in output_value(outputs[0])
        assert second in output_value(outputs[1])


def test_edited_pairs_save_and_reload_without_old_inlet_fields(tmp_path):
    app = create_app()
    original = projects.set_design_mode(projects.default_project(), "stator_rotor")
    values, identifiers = [], []
    for row, section, pair, first, second in (
            ("stator_1", "method_of_characteristics", "PT_INPUTS", 5e5, 400.),
            ("rotor_1", "rotor_design", "HmassP_INPUTS", 5e5, 3e5)):
        values.extend([pair, first, second])
        identifiers.extend(parameter(row, section, name) for name in INLET_FIELDS)
    callback = app.callback_map[next(key for key in app.callback_map if key.startswith("..config.data"))]
    pattern = next(item["id"] for item in callback["inputs"] if item["id"].startswith("{"))
    edited = call(app, "..config.data", pattern + ".value", {
        ("base-project", "data"): original, ("selected-row", "value"): "stator_1",
        (pattern, "value"): values, (pattern, "id"): identifiers,
    })["config"]["data"]
    assert edited["valid"]
    assert edited["project"] != original
    assert "P0" in original["rows"][0]["geometry"]["method_of_characteristics"]
    for row, section in (("stator_1", "method_of_characteristics"), ("rotor_1", "rotor_design")):
        inputs = projects.get_row(edited["project"], row)["geometry"][section]
        assert set(INLET_FIELDS) <= inputs.keys()
        assert not {"P0", "T0", "Q0", "P0_rel", "T0_rel", "Q0_rel"} & inputs.keys()
    saved = call(app, "..loaded-project.data", "save-configuration.n_clicks", {
        ("save-configuration", "n_clicks"): 1, ("config", "data"): edited,
        ("base-project", "data"): original,
    })["configuration-download"]["data"]["content"]
    path = tmp_path / "saved.yaml"
    path.write_text(saved, encoding="utf-8")
    reloaded = create_app(path).server.test_client().get("/_dash-layout").get_json()
    base = next(node["props"]["data"] for node in nodes(reloaded) if node["props"].get("id") == "base-project")
    assert base == edited["project"]


@pytest.mark.parametrize("fluid,base_pair,first,second,pair", [
    ("Air", "PT_INPUTS", 5e5, 400., "PT_INPUTS"),
    ("Air", "PT_INPUTS", 5e5, 400., "HmassP_INPUTS"),
    ("Air", "PT_INPUTS", 5e5, 400., "DmassT_INPUTS"),
    ("Air", "PT_INPUTS", 5e5, 400., "PSmass_INPUTS"),
    ("Water", "PQ_INPUTS", 1e5, .25, "PQ_INPUTS"),
    ("Water", "PQ_INPUTS", 1e5, .25, "QT_INPUTS"),
    ("Water", "PQ_INPUTS", 1e5, .25, "HmassP_INPUTS"),
    ("Water", "PQ_INPUTS", 1e5, 0., "PQ_INPUTS"),
    ("Water", "PQ_INPUTS", 1e5, 1., "PQ_INPUTS"),
])
def test_selected_pair_resolves_the_same_real_state(fluid, base_pair, first, second, pair):
    state = jxp.Fluid(fluid).get_state(jxp.INPUT_PAIRS[base_pair], first, second)
    symbols = {"P": state.p, "T": state.T, "Q": state.Q, "Hmass": state.h,
               "Dmass": state.rho, "Smass": state.s}
    from turbo_moc.fluid_inputs import pair_properties
    prop1, prop2 = pair_properties(pair)
    inputs = dict(fluid_name=fluid, inlet_property_pair=pair,
                  inlet_property_1=float(symbols[prop1]), inlet_property_2=float(symbols[prop2]))
    original = deepcopy(inputs)
    for rotor in (False, True):
        resolved = solver_parameters(inputs, rotor=rotor)
        pressure, temperature, quality = ("P0_rel", "T0_rel", "Q0_rel") if rotor else ("P0", "T0", "Q0")
        assert resolved[pressure] == pytest.approx(float(state.p), rel=1e-7)
        assert resolved[temperature] == pytest.approx(float(state.T), rel=1e-7)
        if state.is_two_phase:
            assert resolved[quality] == pytest.approx(float(state.Q), abs=1e-7)
        else:
            assert math.isnan(resolved[quality])
    assert inputs == original


@pytest.mark.parametrize("changes,match", [
    ({"inlet_property_pair": "invalid"}, "supported inlet"),
    ({"inlet_property_1": 0}, "positive"),
    ({"inlet_property_2": 1.5}, "quality"),
    ({"inlet_property_2": float("nan")}, "finite"),
])
def test_invalid_pair_values_block_the_app_before_jobs(changes, match):
    project = projects.default_project()
    inputs = project["rows"][0]["geometry"]["method_of_characteristics"]
    inputs.update(inlet_values(inputs))
    inputs.update(changes)
    with pytest.raises(ValueError, match=match):
        projects.validate_project(project)


def test_new_inlet_values_reach_the_moc_solver(monkeypatch):
    import turbo_moc
    from turbo_moc.geometry.moc import solve_nozzle
    received = []
    def solve(**parameters):
        received.append(parameters)
        return {"converged": True, "wall_final": {"x": [0., 1.], "y": [0., 1.]}}
    monkeypatch.setattr(turbo_moc, "design_nozzle", solve)
    solve_nozzle(dict(fluid_name="Air", inlet_property_pair="PT_INPUTS",
                      inlet_property_1=5e5, inlet_property_2=400., p_back=1e5))
    assert received[0]["P0"] == pytest.approx(5e5)
    assert received[0]["T0"] == pytest.approx(400.)
    assert math.isnan(received[0]["Q0"])
    assert not set(INLET_FIELDS) & received[0].keys()


def test_manual_pair_reaches_the_rotor_worker(tmp_path, monkeypatch):
    from turbo_moc.apps.app import worker
    from turbo_moc import rotor as rotor_api
    project = projects.set_design_mode(projects.default_project(), "rotor")
    design = projects.get_row(project, "rotor_1")["geometry"]["rotor_design"]
    design.pop("P0_rel")
    design.pop("T0_rel")
    design.update(fluid_name="Air", inlet_property_pair="PT_INPUTS",
                  inlet_property_1=5e5, inlet_property_2=400.)
    config = projects.row_configuration(project, "rotor_1")
    received = []
    def solve(**parameters):
        received.append(parameters)
        return {"beta_inlet": 65., "beta_outlet": 65., "pitch": 1.}
    monkeypatch.setattr(rotor_api, "design_rotor_vortex_blade", solve)
    monkeypatch.setattr(flow, "get_design", lambda *args: None)
    handle = {"session": "a" * 32, "job": "b" * 32, "action": "nozzle",
              "row_id": "rotor_1", "row_role": "rotor", "started": 0}
    folder = tmp_path / handle["session"] / handle["job"]
    folder.mkdir(parents=True)
    request = folder / "request.json"
    request.write_text(json.dumps({"handle": handle, "config": config, "references": {}}), encoding="utf-8")
    # run() sets the standalone output root from its request; restore it afterwards.
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    assert worker.run(request) == 0
    assert received[0]["P0_rel"] == pytest.approx(5e5)
    assert received[0]["T0_rel"] == pytest.approx(400.)
    assert math.isnan(received[0]["Q0_rel"])
    assert not set(INLET_FIELDS) & received[0].keys()
