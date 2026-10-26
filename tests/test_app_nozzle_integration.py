"""Nozzle mode shares the calculation toolbar without exposing workflow state."""
import pytest

from turbo_moc.app import calculations, create_app
from turbo_moc.configuration import project as projects
from app_helpers import call, components


@pytest.mark.parametrize("mode", ["nozzle", "stator", "rotor", "stator_rotor"])
def test_mode_switch_keeps_shared_toolbar_and_independent_workspaces(mode):
    app = create_app()
    layout = components(app.server.test_client().get("/_dash-layout").get_json())
    base = layout["base-project"]["data"]
    changed = call(app, "..base-project.data", "design-mode.value", {
        ("design-mode", "value"): mode,
        ("machine-family", "value"): "axial",
        ("base-project", "data"): base,
        ("selected-row", "value"): "stator_1"})
    project = changed["base-project"]["data"]
    assert project["machine"]["design_mode"] == mode
    selection = call(app, "..calculation-scope.options", "base-project.data", {
        ("base-project", "data"): project,
        ("calculation-scope", "value"): layout["calculation-scope"]["value"],
        ("calculation-scope", "options"): layout["calculation-scope"]["options"]})
    assert selection["calculation-scope"]["value"] == [r["id"] for r in projects.active_rows(project)]
    assert selection["calculation-rows"]["options"] == [
        {"label": "All rows", "value": "all"}, *calculations.scope_options(project)]
    if mode == "nozzle":
        assert selection["calculation-rows"]["options"][1]["label"] == "Nozzle"
        assert "nozzle-section" in layout and "nozzle-mesh-3d" in layout
    visibility = call(app, "..nozzle-views.style", "config.data", {
        ("config", "data"): {"valid": True, "project": project}})
    assert visibility["nozzle-views"]["style"] == ({} if mode == "nozzle" else {"display": "none"})
    assert visibility["blade-views"]["style"] == ({"display": "none"} if mode == "nozzle" else {})
    family = app.callback_map["machine-family-control.style"]["callback"].__wrapped__(mode)
    assert family == ({"display": "none"} if mode == "nozzle" else {})
    assert layout["interspace-note"]["style"] == {"display": "none"}
    assert not any(output.component_id == "interspace-note" and output.component_property == "style"
        for callback in app.callback_map.values()
        for output in (callback["output"] if isinstance(callback["output"], list) else [callback["output"]]))
    assert app.callback_map["nozzle.children"]["callback"].__wrapped__(
        {"project": project}) == "Generate geometry"
