"""Physical domain dependencies and usable defaults for paired blade rows."""
from copy import deepcopy

import pytest
import yaml

from turbo_moc.configuration import project as projects
from turbo_moc.configuration.row import read_configuration
from turbo_moc.app import workflow as flow


def test_tapered_axial_interface_is_in_minimum_gap_and_does_not_overwrite_inputs(tmp_path):
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    stator, rotor = project["rows"]
    stator["geometry"]["meridional_channel"].update(hub_chord=20, tip_chord=24)
    stator["placement"]["streamwise_offset_mm"] = 5
    rotor["geometry"]["axial_interspace_mm"] = 4
    saved = deepcopy(project)
    interface, = projects.shared_interfaces(project)
    assert interface["coordinate_mm"] == 31
    assert projects.row_assembly(project, stator["id"])["interfaces"] == {"outlet": interface}
    assert projects.row_assembly(project, rotor["id"])["interfaces"] == {"inlet": interface}
    assert project == saved
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    assert read_configuration(path)["assembly"] == projects.row_assembly(project, stator["id"])


@pytest.mark.parametrize("up,down,expected", [((100, 80), (76, 60), 78),
                                           ((80, 100), (104, 120), 102)])
def test_radial_interface_is_one_cylinder_in_either_direction(up, down, expected):
    project = projects.set_design_mode(projects.default_project("radial"), "stator_rotor")
    for row, (inlet, outlet) in zip(project["rows"], (up, down)):
        row["geometry"]["meridional_channel"].update(inlet_radius=inlet, outlet_radius=outlet)
    interface, = projects.shared_interfaces(project)
    assert interface["coordinate_mm"] == expected
    assert interface["gap_mm"] == 4


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_new_paired_rows_have_matching_endwalls_and_positive_clearance(family):
    project = projects.default_project(family)
    project["rows"][0]["geometry"]["meridional_channel"]["blade_height"] = 13
    paired = projects.set_design_mode(project, "stator_rotor")
    projects.validate_project(paired)
    assert projects.shared_interfaces(paired)[0]["gap_mm"] > 0
    assert paired["rows"][1]["geometry"]["meridional_channel"]["blade_height"] == 13
    assert projects.set_machine_type(paired, family) == paired
    radial = projects.set_machine_type(paired, "radial")
    assert projects.shared_interfaces(radial)[0]["gap_mm"] > 0


@pytest.mark.parametrize("family,field,value,message", [
    ("axial", "hub_radius", 101, "matching hub_radius"),
    ("axial", "blade_height", 11, "matching blade_height"),
    ("radial", "blade_height", 11, "matching blade_height"),
    ("radial", "outlet_radius", 90, "same flow direction"),
    ("radial", "inlet_radius", 85, "positive blade gap"),
])
def test_incompatible_domains_are_editable_but_cannot_run(family, field, value, message):
    project = projects.set_design_mode(projects.default_project(family), "stator_rotor")
    project["rows"][1]["geometry"]["meridional_channel"][field] = value
    projects.validate_project(project)
    with pytest.raises(ValueError, match=message):
        flow.resolved_configuration(project, "stator_1", {})


def test_gap_change_invalidates_both_meshes_and_retains_both_profile_solutions():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    before = {row["id"]: flow.keys(flow.resolved_configuration(project, row["id"], {}))
              for row in project["rows"]}
    project["rows"][1]["geometry"]["axial_interspace_mm"] = 4
    for row in project["rows"]:
        after = flow.keys(flow.resolved_configuration(project, row["id"], {}))
        assert before[row["id"]]["nozzle"] == after["nozzle"]
        for stage in ("geometry", "reference", "volume"):
            assert before[row["id"]][stage] != after[stage]
    standalone = projects.set_design_mode(project, "stator")
    assert projects.row_assembly(standalone, "stator_1") is None
    assert "assembly" not in flow.resolved_configuration(standalone, "stator_1", {})
