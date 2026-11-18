import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from turbo_moc.geometry.karhunen_loeve import (
    SigmaProfile,
    _correct_trailing_edge_intersection,
    _curve_sides,
    _cut_side_at_x,
    _reconcile_trailing_edge,
    _trim_side_before_crossing,
    build_kl_expansion,
    define_perturbed_blade,
)
from turbo_moc.geometry.stator_blade import find_side_intersections


def test_sigma_profile_smoothly_transitions_between_endpoint_values():
    profile = SigmaProfile(0.25, 0.75, sigma0=2.0, sigma1=1.0)
    samples = np.asarray([profile.sigma(value) for value in np.linspace(0.25, 0.75, 101)])
    assert samples[0] == pytest.approx(1.0)
    assert samples[-1] == pytest.approx(1.0)
    assert samples[50] == pytest.approx(2.0)
    assert np.all(np.diff(samples[:51]) >= 0.0)
    assert np.all(np.diff(samples[50:]) <= 0.0)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_sigma_profile_rejects_out_of_range_positions(value):
    profile = SigmaProfile(0.25, 0.75, sigma0=2.0, sigma1=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        profile.sigma(value)


def test_sigma_profile_rejects_invalid_transition_endpoints():
    with pytest.raises(ValueError, match="endpoints"):
        SigmaProfile(0.75, 0.25, sigma0=2.0, sigma1=1.0)


def test_define_perturbed_blade_accepts_one_dimensional_xi():
    nominal = {
        "blade_curve": {
            "x": [0.0, 0.5, 1.0],
            "y": [0.0, 0.1, 0.0],
        },
        "suction": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.1, 0.0]},
        "pressure": {"x": [1.0, 0.5, 0.0], "y": [0.0, -0.1, 0.0]},
        "trailing_edge": {"x": [1.0], "y": [0.0]},
    }
    expansion = build_kl_expansion(n_quad=20, energy_threshold=0.99)

    result = define_perturbed_blade(
        expansion,
        nominal,
        xi=np.zeros(expansion["n_modes"]),
    )

    assert result["coefficients"].shape == (1, expansion["n_modes"])
    assert len(result["blades"]) == 1
    assert len(result["points"]) == 1


def test_generated_project_yaml_contains_control_point_displacements(tmp_path):
    example = (
        Path(__file__).resolve().parents[1]
        / "examples/modifie_blade_control_points/control_points_random_field_perturbation.py"
    )
    spec = importlib.util.spec_from_file_location("cp_kl_example", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    project = yaml.safe_load(
        (example.parent / "blade_design_baseline.yaml").read_text(encoding="utf-8")
    )
    cp_displacements = [
        np.zeros((30, 2))
        for _ in range(5)
    ]
    cp_displacements[0][2] = [0.001, -0.002]
    kl_samples = [
        {"coefficients": np.asarray([[0.01, 0.02, 0.03]])}
        for _ in range(5)
    ]

    paths = module._save_generated_blades(
        project,
        cp_displacements,
        kl_samples,
        tmp_path,
    )

    assert len(paths) == 5
    saved = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    original = yaml.safe_load(
        (example.parent / "blade_design_baseline.yaml").read_text(encoding="utf-8")
    )
    assert saved["machine"] == original["machine"]
    assert saved["workflow"] == original["workflow"]
    assert saved["output_dir"] == original["output_dir"]
    uncertainty_field = (
        saved["rows"][0]["geometry"]["blade_geometry"]
        ["geometry_uncertainty"]
    )
    assert uncertainty_field["Xi"] == pytest.approx([0.01, 0.02, 0.03])
    assert uncertainty_field["sigma0"] == 0.8
    assert uncertainty_field["n_quad"] == 500
    displacement_field = (
        saved["rows"][0]["geometry"]["blade_geometry"]
        ["control_point_displacements"]
    )
    assert displacement_field["2"] == pytest.approx([0.001, -0.002])

    saved_zero = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    zero_displacements = (
        saved_zero["rows"][0]["geometry"]["blade_geometry"]
        ["control_point_displacements"]
    )
    original_displacements = (
        original["rows"][0]["geometry"]["blade_geometry"]
        ["control_point_displacements"]
    )
    assert zero_displacements == original_displacements


def test_curve_sides_include_trailing_edge():
    blade = {
        "blade_curve": {
            "x": [0.0, 1.0, 2.0, 1.5, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, -1.0, -1.5, 0.0],
        },
        "suction": {"x": [999.0, 998.0], "y": [999.0, 998.0]},
        "pressure": {"x": [-999.0, -998.0], "y": [-999.0, -998.0]},
        "trailing_edge": {"x": [2.0, 1.8, 1.6], "y": [2.0, 0.0, -2.0]},
    }

    suction, pressure = _curve_sides(blade)

    assert len(suction) > len(blade["suction"]["x"])
    assert len(pressure) == 3


def test_cut_side_at_x_keeps_only_prefix_and_crossing_point():
    side = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cut = _cut_side_at_x(side, 1.5)

    np.testing.assert_allclose(cut[-1], [1.5, 1.5])
    assert np.all(cut[:, 0] <= 1.5)


def test_reconcile_trailing_edge_matches_control_point_endpoints():
    blade = {
        "blade_curve": {
            "x": [4.0, 3.0, 2.0, 1.0, 0.0, 5.0],
            "y": [0.0, 1.0, 2.0, 1.0, 0.0, -1.0],
        },
        "suction": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        "pressure": {"x": [2.0, 1.0, 0.0], "y": [2.0, 1.0, 0.0]},
        "trailing_edge": {
            "x": [5.0, 4.5, 4.0],
            "y": [-1.0, 0.0, 0.0],
        },
        "control_points": [
            [3.9, 0.1],
            [3.0, 1.0],
            [1.0, 1.0],
            [5.1, -1.1],
        ],
        "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    }

    reconciled = _reconcile_trailing_edge(blade)
    control_points = np.asarray(reconciled["control_points"], dtype=float)
    trailing = np.column_stack((
        reconciled["trailing_edge"]["x"],
        reconciled["trailing_edge"]["y"],
    ))

    assert np.linalg.norm(control_points[-1] - trailing[0]) <= 1e-7
    assert np.linalg.norm(control_points[0] - trailing[-1]) <= 1e-7


def test_trim_side_before_crossing_removes_crossing_and_after_points():
    side = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    trim = _trim_side_before_crossing(side, np.asarray([1.5, 1.5]))

    np.testing.assert_allclose(trim, [[0.0, 0.0], [1.0, 1.0]])


def test_trailing_edge_correction_truncates_and_reconnects():
    blade = {
        "blade_curve": {
            "x": [0.0, 1.0, 2.0, 1.5, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, -1.0, -1.5, 0.0],
        },
        "suction": {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]},
        "pressure": {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, -1.0, -2.0, -3.0]},
        "trailing_edge": {"x": [3.0, 2.8, 2.6], "y": [3.0, -1.0, -3.0]},
    }

    suction, pressure = _curve_sides(
        blade,
        include_trailing_edge=False,
    )
    corrected = _correct_trailing_edge_intersection(
        blade,
        suction,
        pressure,
        np.asarray([[1.5, 0.0], [2.5, 0.0]]),
    )

    pressure = np.column_stack((
        corrected["pressure"]["x"],
        corrected["pressure"]["y"],
    ))
    trailing = np.column_stack((
        corrected["trailing_edge"]["x"],
        corrected["trailing_edge"]["y"],
    ))
    assert np.min(trailing[:, 1]) < np.max(pressure[:, 1])
