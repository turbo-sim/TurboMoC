"""Row migration and rotor inputs, without running a thermodynamic design."""
from copy import deepcopy

import pytest

from turbo_moc import configuration_project as projects
from turbo_moc.configuration_rotor import default_axial_interspace_mm, inlet_radius_mm


def test_planning_row_migrates_without_changing_geometry_dimensions_or_mesh():
    project = projects.set_design_mode(projects.default_project(), "rotor")
    rotor = projects.get_row(project, "rotor_1")
    rotor["geometry"] = {
        "parametrization_type": "conventional",
        "meridional_channel": deepcopy(rotor["geometry"]["meridional_channel"]),
        "axial_interspace_mm": 4.5,
    }
    rotor["mesh"]["numerics"]["span_layers"] = 13
    normalized = projects.validate_project(projects.normalize_project(project))
    migrated = projects.get_row(normalized, "rotor_1")
    assert migrated["geometry"]["parametrization_type"] == "vortex"
    assert migrated["geometry"]["meridional_channel"] == rotor["geometry"]["meridional_channel"]
    assert migrated["geometry"]["axial_interspace_mm"] == 4.5
    assert migrated["mesh"] == rotor["mesh"]
    assert "rotor_design" not in rotor["geometry"]
    assert "axial_interspace_mm" not in projects.row_configuration(normalized, "rotor_1")


def test_missing_axial_gap_uses_rotor_midspan_chord_but_keeps_saved_zero():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    rotor = projects.get_row(project, "rotor_1")
    assert rotor["geometry"]["axial_interspace_mm"] == 2.0
    dimensions = rotor["geometry"]["meridional_channel"]
    dimensions.update(hub_chord=20.0, tip_chord=40.0)
    assert default_axial_interspace_mm(dimensions) == 3.0
    rotor["geometry"].pop("axial_interspace_mm")
    normalized = projects.normalize_project(project)
    assert projects.get_row(normalized, "rotor_1")["geometry"]["axial_interspace_mm"] == 3.0
    # Assembly placement must use the same fallback before normalization too.
    assert projects.assembly_interspaces(project)[0]["gap_mm"] == 3.0
    rotor["geometry"]["axial_interspace_mm"] = 0.0
    preserved = projects.normalize_project(project)
    assert projects.get_row(preserved, "rotor_1")["geometry"]["axial_interspace_mm"] == 0.0


@pytest.mark.parametrize("family,radius", [("linear", None), ("axial", 105.0), ("radial", 100.0)])
def test_manual_rotor_configuration_supports_each_geometry_family(family, radius):
    project = projects.set_design_mode(projects.default_project(family), "rotor")
    if family == "radial":
        # Test the specified physical radius independently of sample defaults.
        projects.get_row(project, "rotor_1")["geometry"]["meridional_channel"].update(
            inlet_radius=100.0, outlet_radius=80.0)
    projects.validate_project(project)
    config = projects.row_configuration(project, "rotor_1")
    assert config["rotor_design"]["M_inlet"] == 1.5
    assert config["inlet_coupling"]["mode"] == "manual"
    assert inlet_radius_mm(config) == radius


def test_coupling_follows_active_order_and_retains_inputs_when_stator_is_hidden():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    stator, rotor = project["rows"]
    stator["id"], rotor["id"] = "stationary", "moving"
    rotor["geometry"]["inlet_coupling"]["mode"] = "stator_exit"
    assert projects.upstream_stator(project, "moving") is stator
    isolated = projects.set_design_mode(project, "rotor")
    projects.validate_project(isolated)
    assert projects.upstream_stator(isolated, "moving") is None
    assert projects.get_row(isolated, "moving")["geometry"]["inlet_coupling"]["mode"] == "stator_exit"


@pytest.mark.parametrize("name,value", [("M_inlet", 1.0), ("P0_rel", -1), ("num_points", 2)])
def test_invalid_rotor_inputs_fail_before_numerical_jobs(name, value):
    project = projects.set_design_mode(projects.default_project(), "rotor")
    projects.get_row(project, "rotor_1")["geometry"]["rotor_design"][name] = value
    with pytest.raises(ValueError, match=name):
        projects.validate_project(project)
