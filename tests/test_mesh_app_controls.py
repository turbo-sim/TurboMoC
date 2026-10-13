from pathlib import Path
"""Focused browser-input and snapshot checks; no MoC solve or mesh generation."""

from copy import deepcopy

import pytest

pytest.importorskip("dash_vtk")
from turbo_moc.app import workflow as app
from turbo_moc.configuration import row as config_module
from turbo_moc.configuration import project as projects


def components(node):
    """Yield serialized Dash components, including components nested in lists."""
    if isinstance(node, list):
        for item in node:
            yield from components(item)
    elif isinstance(node, dict):
        if "props" in node:
            yield node
        for value in node.values():
            yield from components(value)


def test_browser_numbers_allow_fractional_geometry_but_require_integer_counts():
    config = config_module.read_configuration(CASES["axial_stator"])
    # JavaScript sends an integral float such as 1.0 back to Python as 1.
    config["mesh"]["numerics"]["inflation_layers"]["thickness"] = 1
    identifiers = [
        {"path": "mesh.numerics.inflation_layers.thickness"},
        {"path": "meridional_channel.num_blades"},
    ]
    edited = app.edited_configuration(config, [0.3, 29.0], identifiers)
    assert edited["mesh"]["numerics"]["inflation_layers"]["thickness"] == 0.3
    assert type(edited["meridional_channel"]["num_blades"]) is int
    config_module.validate_configuration(edited)
    with pytest.raises(ValueError, match="requires an integer"):
        app.edited_configuration(config, [0.3, 29.5], identifiers)


def test_axial_interspace_updates_pending_meridional_channel_without_a_moc_result():
    from turbo_moc.app.application import create_app
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"]["axial_interspace_mm"] = 3.0
    config = projects.row_configuration(project, "stator_1")
    state = {"project": project, "row_id": "stator_1", "role": "stator", "valid": True,
             "supported": True, "config": config, **app.keys(config)}
    application = create_app()
    key = next(key for key in application.callback_map if key.startswith("..geometry-data.data"))
    geometry = application.callback_map[key]["callback"].__wrapped__
    first = geometry(state, {}, None)
    assert first[0]["rows"] == [], "A planning channel must not synthesize a blade surface."
    outline = next(trace for trace in first[3].data if trace.name.startswith("Rotor"))
    assert outline.name == "Rotor meridional channel (planning)" and outline.line.dash in (None, "solid")
    upstream = projects.get_row(project, "stator_1")["geometry"]["meridional_channel"]
    assert min(outline.x) == pytest.approx(max(upstream["hub_chord"], upstream["tip_chord"]) + 3)
    assert first[3].layout.yaxis.range[0] == 0
    rotor["geometry"]["axial_interspace_mm"] = 7.0
    second = geometry(state, {}, None, first[1])
    moved = next(trace for trace in second[3].data if trace.name.startswith("Rotor"))
    assert second[1] != first[1] and min(moved.x) == pytest.approx(min(outline.x) + 4)
    assert app.keys(projects.row_configuration(project, "stator_1")) == app.keys(config)


def test_fractional_volume_width_reaches_shader_store_without_changing_mesh_identity():
    from turbo_moc.app.application import create_app
    project = projects.default_project()
    config = projects.row_configuration(project, "stator_1")
    before = app.keys(config)
    config["mesh"]["visualization"]["volume_edge_width"] = 0.15
    project["rows"][0]["mesh"] = deepcopy(config["mesh"])
    state = {"project": project, "config": config, "valid": True, "supported": True,
             "row_id": "stator_1", "role": "stator", **app.keys(config)}
    application = create_app()
    display_settings = application.callback_map["mesh-width.data"]["callback"].__wrapped__
    assert display_settings(state) == 0.15
    assert app.keys(config) == before
    shader_key = next(key for key in application.callback_map if key.startswith("..mesh-scene.children"))
    shader = application.callback_map[shader_key]
    assert {"id": "mesh-width", "property": "data"} in shader["inputs"]


def test_row_selection_uses_only_that_rows_selected_moc_result():
    from turbo_moc.app.application import create_app
    project = projects.default_project()
    second = deepcopy(project["rows"][0])
    second["id"] = "stator_2"
    second["placement"]["streamwise_offset_mm"] = 25
    project["rows"].append(second)
    config = projects.row_configuration(project, "stator_1")
    registry = {identifier: {"nozzle": {"session": "a"*32, "job": job*32,
                    "row_id": identifier, "key": app.keys(config)["nozzle"]}}
                for identifier, job in (("stator_1", "b"), ("stator_2", "c"))}
    original = deepcopy(registry)
    application = create_app()
    key = next(key for key in application.callback_map if key.startswith("..config.data"))
    edit = application.callback_map[key]["callback"].__wrapped__
    first, error = edit([], project, registry, "stator_1", [])
    assert not error and first["valid"] and first["supported"]
    second, error = edit([], project, registry, "stator_2", [])
    assert not error and second["valid"] and second["supported"]
    assert {k: v for k, v in first["config"].items() if k != "assembly"} == {
        k: v for k, v in second["config"].items() if k != "assembly"}
    assert first["geometry"] != second["geometry"]
    assert first["reference"] != second["reference"]
    assert registry == original
    readiness_key = next(key for key in application.callback_map if key.startswith("..nozzle.disabled"))
    readiness = application.callback_map[readiness_key]["callback"].__wrapped__
    assert readiness(first, registry, None, None, False)[-1].startswith("Calculations: Stator.")


def test_rotor_selection_enables_manual_design_and_requires_current_stator_when_coupled():
    from turbo_moc.app.application import create_app
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    application = create_app(CASE_DIR / "stator_rotor.yaml")
    layout = application.server.test_client().get("/_dash-layout").get_json()
    selector = next(node for node in components(layout) if node["props"].get("id") == "selected-row")
    assert selector["props"]["options"] == [
        {"label": "Stator", "value": "stator_1"}, {"label": "Rotor", "value": "rotor_1"}]
    edit_key = next(key for key in application.callback_map if key.startswith("..config.data"))
    edit = application.callback_map[edit_key]["callback"].__wrapped__
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"]["inlet_coupling"]["mode"] = "manual"
    state, error = edit([], project, {}, "rotor_1", [])
    assert not error and state["valid"] and state["supported"]
    assert state["config"]["rotor_design"] == {
        **rotor["geometry"]["rotor_design"], "fluid_name": project["machine"]["fluid_name"]}
    ready_key = next(key for key in application.callback_map if key.startswith("..nozzle.disabled"))
    readiness = application.callback_map[ready_key]["callback"].__wrapped__
    result = readiness(state, {}, None, None, False)
    assert result[:4] == (False, True, True, True)
    assert result[-1].startswith("Calculations: Rotor.")
    rotor["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    state, error = edit([], project, {}, "rotor_1", [])
    assert state["valid"] and not state["supported"]
    assert state["config"] is not None and "upstream stator" in error
    result = readiness(state, {}, None, None, False)
    assert result[:4] == (True,) * 4
    assert "upstream stator" in result[-1]


def test_coupled_rotor_inputs_are_read_only_and_manual_values_survive_toggle(monkeypatch):
    from turbo_moc.app.application import create_app, ROTOR_INLET_FIELDS
    application = create_app()
    callback = lambda prefix: next(value["callback"].__wrapped__
        for key, value in application.callback_map.items() if key.startswith(prefix))
    edit = callback("..config.data")
    display = callback('..{"row":["ALL"],"type":"rotor-manual-inlet"}.style')
    readiness = callback("..nozzle.disabled")
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    manual_inputs = deepcopy(rotor["geometry"]["rotor_design"])
    resolved = projects.row_configuration(project, "rotor_1")
    resolved["rotor_design"].update(P0_rel=123456.0, T0_rel=345.0, M_inlet=0.8, beta_inlet=-25.0)
    resolved["rotor_source"] = {"inlet_radius_mm": 100.0, "triangle": {"U": 110.0}}
    monkeypatch.setattr(app, "resolved_configuration", lambda *args: deepcopy(resolved))
    registry = {"unknown_row": {"nozzle": {"job": "a" * 32}}}
    original_registry = deepcopy(registry)
    state, error = edit([], project, registry, "rotor_1", [])
    assert not error and state["supported"]
    identifiers = [{"type": "rotor-manual-inlet", "row": "rotor_1"}]
    manual_styles, cells, computed_styles, speed_styles, warnings = display(state, registry, identifiers)
    assert manual_styles == [{"display": "none"}] and computed_styles == [{}] and speed_styles == [{}]
    for name, cell in zip(ROTOR_INLET_FIELDS, cells[0]):
        widget = cell.children[1]
        assert widget.disabled and not getattr(widget, "id", None)
        assert widget.value == resolved["rotor_design"][name]
        assert cell.className == "control computed-control"
    assert len(warnings[0]) == 2
    assert "supersonic relative inlet" in warnings[0][0].children
    assert "Negative relative inlet angle" in warnings[0][1].children
    assert all(alert.className == "inlet-warning" and alert.role == "alert" for alert in warnings[0])
    assert "supersonic" not in cells[0][-1].children
    assert readiness(state, registry, None, None, False)[0]
    assert "supersonic relative inlet" in readiness(state, registry, None, None, False)[-1]
    assert rotor["geometry"]["rotor_design"] == manual_inputs
    assert registry == original_registry
    rotor["geometry"]["inlet_coupling"]["mode"] = "manual"
    state["project"] = project
    manual_styles, _, computed_styles, speed_styles, warnings = display(state, registry, identifiers)
    assert manual_styles == [{}] and computed_styles == [{"display": "none"}]
    assert speed_styles == [{"display": "none"}]
    assert warnings == [[]]
    assert rotor["geometry"]["rotor_design"] == manual_inputs


@pytest.mark.parametrize("exception", [ValueError, OSError, RuntimeError])
def test_pending_or_failed_coupling_clears_calculated_inlet_cells(monkeypatch, exception):
    from turbo_moc.app.application import create_app
    application = create_app()
    callback = lambda prefix: next(value["callback"].__wrapped__
        for key, value in application.callback_map.items() if key.startswith(prefix))
    edit = callback("..config.data")
    display = callback('..{"row":["ALL"],"type":"rotor-manual-inlet"}.style')
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "rotor_1")["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    def unavailable(*args):
        raise exception("Upstream inlet unavailable")
    monkeypatch.setattr(app, "resolved_configuration", unavailable)
    state, error = edit([], project, {}, "rotor_1", [])
    assert state["valid"] and not state["supported"] and error == "Upstream inlet unavailable"
    result = display(state, {}, [{"type": "rotor-manual-inlet", "row": "rotor_1"}])
    assert all(cell.children[1].disabled and cell.children[1].value is None for cell in result[1][0][:-1])
    assert result[1][0][-1].children == error


def test_structure_can_leave_a_paired_project_for_a_linear_stator_without_losing_rotor_settings():
    from turbo_moc.app.application import create_app
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    projects.get_row(project, "rotor_1")["geometry"]["parametrization_type"] = "vortex"
    application = create_app()
    key = next(key for key in application.callback_map if key.startswith("..base-project.data"))
    structure = application.callback_map[key]["callback"].__wrapped__
    state = {"project": project, "valid": True}
    result = structure("stator", "linear", state, project, "stator_1")
    assert result[0]["machine"] == {"design_mode": "stator", "geometry_type": "linear", "fluid_name": "Cyclopentane"}
    assert projects.get_row(result[0], "rotor_1")["geometry"]["parametrization_type"] == "vortex"
    assert result[3] == "stator_1" and result[5] == "linear"
    assert not any(option.get("disabled") for option in result[4])


def test_invalid_input_then_mode_switch_keeps_the_last_valid_row_settings():
    from turbo_moc.app.application import create_app
    application = create_app()
    edit_key = next(key for key in application.callback_map if key.startswith("..config.data"))
    structure_key = next(key for key in application.callback_map if key.startswith("..base-project.data"))
    edit = application.callback_map[edit_key]["callback"].__wrapped__
    structure = application.callback_map[structure_key]["callback"].__wrapped__
    base = projects.default_project()
    identifiers = [
        {"type": "parameter", "row": "stator_1", "path": "mesh.numerics.blade_size"},
        {"type": "parameter", "row": "stator_1", "path": "method_of_characteristics.n"}]
    valid, error = edit([0.37, 15], base, {}, "stator_1", identifiers, None)
    assert not error and valid["valid"]
    invalid, error = edit([0.37, 15.5], base, {}, "stator_1", identifiers, valid)
    assert "requires an integer" in error and not invalid["valid"]
    assert invalid["project"] == valid["project"]
    # A second invalid response must carry the same validated snapshot, rather
    # than restoring the original YAML defaults or accepting invalid raw input.
    invalid_again, error = edit([0.37, 16.5], base, {}, "stator_1", identifiers, invalid)
    assert "requires an integer" in error and invalid_again["project"] == valid["project"]
    changed = structure("rotor", "axial", invalid_again, base, "stator_1")
    assert "last valid" in changed[-1].lower()
    restored = structure("stator", "axial", {"project": changed[0], "valid": True},
                         changed[0], "rotor_1")
    stator = projects.get_row(restored[0], "stator_1")
    assert stator["mesh"]["numerics"]["blade_size"] == 0.37
    assert stator["geometry"]["method_of_characteristics"]["n"] == 15


@pytest.mark.parametrize("family", ["radial", "linear", "axial"])
def test_axial_interspace_controls_respect_the_current_machine_family(family):
    mode = "stator_rotor" if family == "axial" else "rotor"
    project = projects.set_design_mode(projects.default_project(family), mode)
    edited = app.edited_project(project, [9.0], [
        {"type": "parameter", "row": "rotor_1", "path": "axial_interspace_mm"}])
    projects.validate_project(edited)
    rotor_geometry = projects.get_row(edited, "rotor_1")["geometry"]
    if family == "axial":
        assert rotor_geometry["axial_interspace_mm"] == 9.0
        assert projects.assembly_placement(edited, "rotor_1")["streamwise_offset_mm"] == 29.0
    else:
        assert "axial_interspace_mm" not in rotor_geometry
        assert edited == project

CASE_DIR = Path(__file__).resolve().parents[1] / "examples/apps/cases"
CASES = {n: CASE_DIR / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}
FLUENT_CONFIG = Path(__file__).resolve().parents[1] / "examples/fluent_2D/workflow_fluent_stator_2D.yaml"
