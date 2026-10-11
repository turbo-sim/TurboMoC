"""Shared-domain controls and physical interface plots; no thermodynamic jobs."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("dash_vtk")
from turbo_moc import configuration_project as projects
from turbo_moc.apps.app import application, plots


def paired_project(family="axial"):
    project = projects.set_design_mode(projects.default_project(family), "stator_rotor")
    stator, rotor = project["rows"]
    if family == "radial":
        stator["geometry"]["meridional_channel"].update(inlet_radius=100, outlet_radius=80)
        rotor["geometry"]["meridional_channel"].update(inlet_radius=78, outlet_radius=60)
    return project


def test_internal_extension_controls_are_hidden_but_solo_values_are_retained():
    project = paired_project()
    original = deepcopy(project)
    stator, rotor = project["rows"]
    for row, internal in ((stator, "outlet_chords"), (rotor, "inlet_chords")):
        group = application.flow_domain_controls(project, row)
        for wrapper in group.children[1:-1]:
            widget = wrapper.children[0].children[1]
            path = widget.id["path"].split(".")[-1]
            assert wrapper.style == ({"display": "none"} if path == internal else {})
        solo = projects.set_design_mode(project, row["role"])
        assert not application.internal_domain_sides(solo, row["id"])
    assert project == original


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_meridional_planning_view_shows_the_shared_physical_interface(family):
    project = paired_project(family)
    interfaces = projects.shared_interfaces(project)
    planning = [(application.row_label(row), row["geometry"]["meridional_channel"],
        projects.assembly_placement(project, row["id"])) for row in project["rows"]]
    plot = plots.assembly_meridional_figure([], planning, interfaces=interfaces)
    trace = next(trace for trace in plot.data if trace.name == "Shared row interface")
    coordinate = trace.x if family == "axial" else trace.y
    assert np.allclose(coordinate, interfaces[0]["coordinate_mm"])
    assert plot.layout.yaxis.range[0] == 0
    if family == "radial":
        for row in project["rows"]:
            cells = application.shared_domain_readout(project, row["id"])[:-1]
            assert [cell.children[1].value for cell in cells] == [79, 1, 1, 1]


def test_invalid_interface_keeps_settings_available_and_does_not_crash_views():
    project = paired_project()
    project["rows"][1]["geometry"]["axial_interspace_mm"] = 0
    app = application.create_app()
    callback = lambda prefix: next(value["callback"].__wrapped__
        for key, value in app.callback_map.items() if key.startswith(prefix))
    state, error = callback("..config.data")([], project, {}, "stator_1", [])
    assert state["valid"] and not state["supported"] and error
    scene = callback("..geometry-data.data")(state, {}, None)
    assert scene[0]["rows"] == []
    assert "Shared interface unavailable" in scene[-1]
    assert not any(trace.name == "Shared row interface" for trace in scene[3].data)


def test_stale_mesh_keeps_the_assembly_placement_used_in_its_export(monkeypatch):
    project = paired_project()
    saved = projects.row_assembly(project, "rotor_1")
    project["rows"][1]["geometry"]["axial_interspace_mm"] += 5
    monkeypatch.setattr(application, "volume_artifact", lambda *_: ({}, {
        "geometry": {"type": "axial", "num_blades": 30}, "assembly": saved}, {}))
    app = application.create_app()
    view = next(value["callback"].__wrapped__ for key, value in app.callback_map.items()
        if key.startswith("..volume-data.data"))
    state = {"project": project, "row_id": "stator_1", "valid": True}
    registry = {"rotor_1": {"volume": {"session": "unused", "job": "old"}}}
    data, _ = view(state, registry)
    placement = data["rows"][0]["placement"]
    assert placement == saved["placement"]
    assert placement != projects.assembly_placement(project, "rotor_1")
