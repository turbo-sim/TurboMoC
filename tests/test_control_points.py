import numpy as np
import pytest
import yaml

from turbo_moc.configuration.project import default_project, get_row, normalize_project, validate_project
from turbo_moc.geometry.stator_blade import (apply_control_point_offsets, blade_self_intersects,
                                             editable_control_points, parametrize_stator)


@pytest.fixture
def blade(small_design):
    wall, config = small_design
    inputs = {k: v for k, v in config["blade_geometry"].items() if k != "control_point_offsets"}
    return parametrize_stator(wall["wall_final"]["x"], wall["wall_final"]["y"], **inputs)


def test_only_interior_points_are_editable(blade):
    count = len(blade["control_points"])
    assert editable_control_points(blade) == list(range(2, count - 2))
    for index in (0, 1, count - 2, count - 1):
        with pytest.raises(ValueError, match="cannot be moved"):
            apply_control_point_offsets(blade, {index: 0.01})


def test_offsets_move_only_the_selected_points_and_keep_the_trailing_edge(blade):
    moved = apply_control_point_offsets(blade, {"5": 0.02, 9: -0.01})
    delta = np.linalg.norm(np.subtract(moved["control_points"], blade["control_points"]), axis=1)
    assert np.nonzero(delta > 1e-12)[0].tolist() == [5, 9]
    assert np.allclose(moved["control_points"][0], blade["control_points"][0])
    assert np.allclose(moved["control_points"][-1], blade["control_points"][-1])
    assert moved["trailing_edge"] == blade["trailing_edge"]
    assert apply_control_point_offsets(blade, {}) is blade


def test_dispatcher_applies_offsets_from_the_configuration(small_design, blade):
    wall, config = small_design
    inputs = {k: v for k, v in config["blade_geometry"].items() if k != "control_point_offsets"}
    edited = parametrize_stator(wall["wall_final"]["x"], wall["wall_final"]["y"],
                                control_point_offsets={6: 0.01}, **inputs)
    assert edited["control_points"] == apply_control_point_offsets(blade, {6: 0.01})["control_points"]


def test_self_intersection_is_detected(blade):
    assert not blade_self_intersects(blade)
    middle = editable_control_points(blade)[len(editable_control_points(blade)) // 2]
    assert blade_self_intersects(apply_control_point_offsets(blade, {middle: -0.5}))


def test_offsets_survive_a_saved_case():
    project = default_project()
    project["rows"][0]["geometry"]["blade_geometry"]["control_point_offsets"] = {"12": 0.04, 20: -0.02}
    project = validate_project(normalize_project(project))
    loaded = validate_project(normalize_project(yaml.safe_load(yaml.safe_dump(project))))
    assert get_row(loaded, "stator_1")["geometry"]["blade_geometry"]["control_point_offsets"] == {12: 0.04, 20: -0.02}


@pytest.mark.parametrize("mode", ["full", "semi"])
def test_vector_displacements_reconstruct_exactly_from_project_yaml(small_design, mode):
    wall, config = small_design
    config["blade_geometry"].update(parametrization=mode, control_point_displacements={5: [0.01, -0.002]})
    project = validate_project(normalize_project(config))
    loaded = validate_project(normalize_project(yaml.safe_load(yaml.safe_dump(project))))
    original_inputs = project["rows"][0]["geometry"]["blade_geometry"]
    loaded_inputs = loaded["rows"][0]["geometry"]["blade_geometry"]
    x, y = wall["wall_final"]["x"], wall["wall_final"]["y"]
    assert parametrize_stator(x, y, **original_inputs) == parametrize_stator(x, y, **loaded_inputs)


@pytest.mark.parametrize("value", [[0.1], [0.1, 0.2, 0.3], [float("nan"), 0], [True, 0.1]])
def test_invalid_vector_displacements_are_rejected(blade, value):
    from turbo_moc.geometry.stator_blade import apply_control_point_displacements
    with pytest.raises(ValueError, match="two finite"):
        apply_control_point_displacements(blade, {5: value})
