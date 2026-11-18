import numpy as np
import pytest

from turbo_moc.geometry.karhunen_loeve import (
    SigmaProfile,
    _correct_trailing_edge_intersection,
    _blade_range_distance,
    _cut_side_at_x,
    _curve_sides,
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


def test_define_perturbed_blade_can_return_uncertainties_only():
    nominal = {
        "blade_curve": {
            "x": [0.0, 0.5, 1.0],
            "y": [0.0, 0.1, 0.0],
        },
    }
    expansion = build_kl_expansion(n_quad=20, energy_threshold=0.99)

    result = define_perturbed_blade(
        expansion,
        nominal,
        n_samples=2,
        random_state=1,
        uncertainties_only=True,
    )

    assert result["uncertainties"].shape == (2, 3)
    assert result["blades"] == []
    assert result["points"] == []


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
