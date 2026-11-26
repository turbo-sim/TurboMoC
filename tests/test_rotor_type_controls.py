"""Which rotor controls the studio shows, for every rotor blade type, with a manual and a coupled inlet."""
import pytest

from app_helpers import call

pytest.importorskip("dash_vtk")

from turbo_moc.app import create_app
from turbo_moc.app.application import BLADE_FIELDS, FLOW_FREE_BLADES, ROTOR_BLADE_OPTIONS, blade_field_visible
from turbo_moc.configuration import project as projects

TYPES = [option["value"] for option in ROTOR_BLADE_OPTIONS]


def _walk(component, found):
    if isinstance(component, dict):
        props = component.get("props", {})
        identifier = props.get("id")
        if isinstance(identifier, dict):
            found.append((identifier, props.get("style") or {}))
        for item in component.values():
            _walk(item, found)
    elif isinstance(component, list):
        for item in component:
            _walk(item, found)
    return found


def _visible(style):
    return style.get("display") != "none"


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.mark.parametrize("mode", ["manual", "stator_exit"])
@pytest.mark.parametrize("kind", TYPES)
def test_rotor_controls_per_type_and_inlet_mode(app, kind, mode):
    project = projects.set_design_mode(projects.default_project("axial"), "stator_rotor")
    rotor = projects.get_row(project, "rotor_1")["geometry"]
    rotor["rotor_design"]["blade_type"] = kind
    rotor["inlet_coupling"]["mode"] = mode
    project = projects.validate_project(projects.normalize_project(project))
    layout = call(app, "..base-project.data", "loaded-project.data", {
        ("loaded-project", "data"): project, ("design-mode", "value"): "stator",
        ("machine-family", "value"): "axial", ("base-project", "data"): projects.default_project(),
        ("selected-row", "value"): "rotor_1"})["settings"]["children"]
    items = [(i, s) for i, s in _walk(layout, []) if i.get("row") == "rotor_1"]
    by = lambda kind_: {i.get("field") or i.get("part") or i.get("kind") or i.get("path"): s
                        for i, s in items if i["type"] == kind_}
    paths = {i["path"] for i, _ in items if i["type"] == "parameter"}
    coupled = mode == "stator_exit"
    geometric = kind in FLOW_FREE_BLADES
    # The coupling selector and the speed are always available
    assert "inlet_coupling.mode" in paths and "inlet_coupling.rotational_speed.value" in paths
    groups = by("rotor-flow-group")
    assert _visible(groups["inlet"])
    assert _visible(groups["outlet"]) == (not geometric)
    # Vortex-solver inputs: only the ones the type reads
    for field, style in by("rotor-field").items():
        assert _visible(style) == blade_field_visible(kind, field), (kind, field)
    # The selected type's geometric group is shown, the other hidden
    for group, style in by("geometric-blade").items():
        assert _visible(style) == (group == kind)
    # Coupled inlet: the angle field of the geometric blades is replaced by a note
    for group, style in by("coupled-angle").items():
        assert _visible(style) == (not coupled)
    for group, style in by("coupled-angle-note").items():
        assert _visible(style) == coupled
    # Manual thermodynamic inlet fields: only for flow-based types with a manual inlet
    state = {"project": project, "row_id": "rotor_1", "valid": True}
    key = next(k for k in app.callback_map if "rotor-manual-inlet" in k)
    ids = [{"type": "rotor-manual-inlet", "row": row["id"]} for row in project["rows"] if row["role"] == "rotor"]
    payload = {"output": key, "outputs": [[{"id": i, "property": prop} for i in ids] for prop in
                                          ("style", "children", "style", "style", "children")],
               "changedPropIds": ["config.data"],
               "inputs": [{"id": "config", "property": "data", "value": state},
                          {"id": "row-references", "property": "data", "value": {}}],
               "state": [[{"id": i, "property": "id", "value": i} for i in ids]]}
    # Output ids per output group differ by type; rebuild them from the callback map
    spec = app.callback_map[key]["output"]
    outputs = spec if isinstance(spec, list) else [spec]
    payload["outputs"] = [[{"id": {**i, "type": o.component_id["type"]},
                            "property": o.component_property} for i in ids] for o in outputs]
    response = app.server.test_client().post("/_dash-update-component", json=payload).get_json()["response"]
    manual = next(v for k, v in response.items() if "rotor-manual-inlet" in k)["style"]
    assert _visible(manual) == (not coupled and not geometric), (kind, mode)
