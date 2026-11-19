import inspect
import numpy as np
import pytest
import yaml

from turbo_moc.configuration.project import default_project, get_row, normalize_project, validate_project
from turbo_moc.geometry.stator_blade import (apply_control_point_offsets, blade_self_intersects,
                                             apply_geometry_uncertainty, editable_control_points,
                                             blade_chord, control_point_normals,
                                             correct_suction_intersections,
                                             edit_blade_geometry, parametrize_stator)
from turbo_moc.geometry.karhunen_loeve import SigmaProfile, build_kl_expansion


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


def test_dispatcher_replays_geometry_uncertainty_from_xi(small_design):
    wall, config = small_design
    inputs = {k: v for k, v in config["blade_geometry"].items()
              if k not in {"control_point_offsets", "control_point_displacements",
                           "geometry_uncertainty"}}
    expansion = build_kl_expansion(n_quad=20, sigma0=1.5, energy_threshold=0.99)
    uncertainty = {
        "Xi": [0.0] * expansion["n_modes"],
        "sigma0": 1.5,
        "L0": 0.1,
        "L_le": 1e-2,
        "omega": 0.1,
        "energy_threshold": 0.99,
        "n_quad": 20,
    }
    edited = parametrize_stator(
        wall["wall_final"]["x"],
        wall["wall_final"]["y"],
        geometry_uncertainty=uncertainty,
        **inputs,
    )
    assert edited["geometry_uncertainty"] == uncertainty


@pytest.mark.parametrize(
    "profile_settings, expected_pressure, expected_suction",
    [
        ({"sigma1": 0.2}, 0.25, 0.75),
        (
            {"sigma1": 0.2, "cp_te_pressure": 0.1, "cp_te_suction": 0.9},
            0.1,
            0.9,
        ),
    ],
)
def test_dispatcher_initializes_sigma_profile(
    small_design,
    monkeypatch,
    profile_settings,
    expected_pressure,
    expected_suction,
):
    import turbo_moc.geometry.karhunen_loeve as kl

    wall, config = small_design
    inputs = {
        key: value
        for key, value in config["blade_geometry"].items()
        if key not in {
            "control_point_offsets",
            "control_point_displacements",
            "geometry_uncertainty",
        }
    }
    captured = {}

    def fake_build_kl_expansion(**kwargs):
        captured["sigma0"] = kwargs["sigma0"]
        return {"n_modes": 1}

    def fake_define_perturbed_blade(expansion, blade, **kwargs):
        return {"blades": [blade]}

    monkeypatch.setattr(kl, "build_kl_expansion", fake_build_kl_expansion)
    monkeypatch.setattr(kl, "define_perturbed_blade", fake_define_perturbed_blade)

    uncertainty = {
        "Xi": [0.0],
        "sigma0": 1.5,
        "n_quad": 20,
        **profile_settings,
    }
    edited = parametrize_stator(
        wall["wall_final"]["x"],
        wall["wall_final"]["y"],
        geometry_uncertainty=uncertainty,
        **inputs,
    )

    profile = captured["sigma0"]
    assert isinstance(profile, SigmaProfile)
    assert profile.sigma0 == 1.5
    assert profile.sigma1 == 0.2
    assert profile.start_te_pside == expected_pressure
    assert profile.start_te_sside == expected_suction
    assert edited["geometry_uncertainty"] == uncertainty


def test_dispatcher_rejects_invalid_sigma_profile_endpoints(small_design):
    wall, config = small_design
    inputs = {
        key: value
        for key, value in config["blade_geometry"].items()
        if key not in {
            "control_point_offsets",
            "control_point_displacements",
            "geometry_uncertainty",
        }
    }
    uncertainty = {
        "Xi": [0.0],
        "sigma0": 1.5,
        "sigma1": 0.2,
        "cp_te_pressure": 0.8,
        "cp_te_suction": 0.2,
        "n_quad": 20,
    }
    with pytest.raises(ValueError, match="endpoints"):
        parametrize_stator(
            wall["wall_final"]["x"],
            wall["wall_final"]["y"],
            geometry_uncertainty=uncertainty,
            **inputs,
        )


def test_edit_blade_geometry_returns_fixed_plus_random_total_vectors(small_design):
    wall, config = small_design
    blade_parameters = dict(config["blade_geometry"])
    blade_parameters["control_point_displacements"] = {5: [0.01, -0.002]}

    nominal_parameters = {
        key: value
        for key, value in blade_parameters.items()
        if key not in {
            "control_point_offsets",
            "control_point_displacements",
            "geometry_uncertainty",
        }
    }
    nominal = parametrize_stator(
        wall["wall_final"]["x"],
        wall["wall_final"]["y"],
        **nominal_parameters,
    )
    fixed = parametrize_stator(
        wall["wall_final"]["x"],
        wall["wall_final"]["y"],
        **blade_parameters,
    )

    vectors = edit_blade_geometry(
        wall,
        [5],
        np.asarray([0.5]),
        np.asarray([[0.0, 0.02]]),
        blade_parameters,
    )

    chord = blade_chord(nominal)
    fixed_vector = (
        np.asarray(fixed["control_points"][5])
        - np.asarray(nominal["control_points"][5])
    ) / chord
    random_vector = control_point_normals(
        np.asarray(nominal["control_points"], dtype=float)
    )[5] * 0.01 / chord

    assert vectors.shape == (len(nominal["control_points"]), 2)
    assert np.allclose(vectors[5], fixed_vector + random_vector)
    assert np.allclose(vectors[0], [0.0, 0.0])


def test_suction_intersection_correction_defaults_to_two_mm():
    parameter = inspect.signature(correct_suction_intersections).parameters["minimum_gap"]
    assert parameter.default == 2.0


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


def test_geometry_uncertainty_moves_profile_points_along_local_normals():
    blade = {
        "blade_curve": {"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]},
    }
    moved = apply_geometry_uncertainty(blade, {1: 0.1})
    chord = np.sqrt(2.0)
    normal = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    expected = np.array([1.0, 0.0]) + 0.1 * chord * normal
    assert np.allclose([moved["blade_curve"]["x"][1], moved["blade_curve"]["y"][1]], expected)
    assert np.allclose(
        np.asarray(moved["blade_curve"]["x"])[[0, 2, 3]],
        np.asarray(blade["blade_curve"]["x"])[[0, 2, 3]],
    )


def test_dense_geometry_uncertainty_and_zero_values_are_supported():
    blade = {"blade_curve": {"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0]}}
    values = [0.0, 0.01, -0.02]
    moved = apply_geometry_uncertainty(blade, values)
    assert moved["geometry_uncertainty"] == {"1": 0.01, "2": -0.02}
    assert apply_geometry_uncertainty(blade, [0.0, 0.0, 0.0]) is blade


def test_geometry_uncertainty_moves_trailing_edge_arch_with_profile_endpoints():
    blade = {
        "blade_curve": {"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]},
        "trailing_edge": {"x": [1.0, 1.5, 1.0], "y": [0.0, 0.5, 1.0]},
    }
    moved = apply_geometry_uncertainty(blade, {0: 0.1, 1: 0.1, 3: -0.1})
    trailing = np.column_stack((
        blade["trailing_edge"]["x"],
        blade["trailing_edge"]["y"],
    ))
    moved_trailing = np.column_stack((
        moved["trailing_edge"]["x"],
        moved["trailing_edge"]["y"],
    ))
    assert not np.array_equal(moved_trailing, trailing)
    assert np.allclose(
        np.diff(moved_trailing, axis=0),
        np.diff(trailing, axis=0),
    )
    moved_curve = np.column_stack((moved["blade_curve"]["x"], moved["blade_curve"]["y"]))
    for trailing_point in moved_trailing[[0, -1]]:
        assert np.any(np.all(np.isclose(moved_curve, trailing_point), axis=1))


@pytest.mark.parametrize(
    "uncertainty, message",
    [([0.0], "contain"), ({3: 0.1}, "outside"), ({1: float("nan")}, "finite")],
)
def test_invalid_geometry_uncertainty_is_rejected(uncertainty, message):
    blade = {"blade_curve": {"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0]}}
    with pytest.raises(ValueError, match=message):
        apply_geometry_uncertainty(blade, uncertainty)
