"""Checks for turbo_moc.geometry.radial's conformal-mapping length scale:
conformal_length_scale / mapped_throat_width -- the log-spiral map
preserves angles, not lengths, so a 2D-plane length needs an exact local
magnification factor to be valid once mapped onto the annulus."""

import numpy as np

from turbo_moc.geometry.radial import (
    _reference_point_and_scale,
    apply_conformal_mapping,
    conformal_length_scale,
    mapped_throat_width,
)


def test_conformal_length_scale_is_always_positive():
    # r1 < r2 and r1 > r2 (map can run either direction) -- the
    # magnification factor is a length ratio, never negative.
    for r1, r2 in [(10.0, 20.0), (20.0, 10.0)]:
        x1, c_axial = 5.0, -8.0
        for chord in [0.0, -3.0, -7.9]:
            scale = conformal_length_scale(chord, x1, r1, r2, c_axial)
            assert scale > 0


def test_conformal_length_scale_matches_finite_difference():
    """|dw/du| at a point, checked against a direct finite-difference
    estimate of the mapped distance between two nearby points -- confirms
    the closed-form factor actually matches what the map itself does,
    not just that it's sign-correct."""
    x1, y1, r1, r2, c_axial = 2.0, 0.0, 10.0, 25.0, -6.0
    chord0, pitch0 = -3.0, 1.0
    eps = 1e-5

    X0, Y0 = apply_conformal_mapping(chord0, pitch0, x1, y1, r1, r2, c_axial)
    X1, Y1 = apply_conformal_mapping(chord0 + eps, pitch0, x1, y1, r1, r2, c_axial)
    numerical_scale = np.hypot(X1 - X0, Y1 - Y0) / eps

    analytical_scale = conformal_length_scale(chord0, x1, r1, r2, c_axial)
    assert np.isclose(numerical_scale, analytical_scale, rtol=1e-3)


def _make_stator_blade_data(pitch=10.0, throat_opening=4.0):
    # A simple closed square-ish blade_curve so _reference_point_and_scale
    # has a well-defined chord span and argmax point.
    curve = {"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 20.0, 20.0]}
    return {
        "blade_curve": curve,
        "throat_opening": throat_opening,
        "throat_point": {"x": 0.5, "y": 10.0},
        "pitch": pitch,
    }


def test_mapped_throat_width_uses_the_same_reference_as_the_actual_wrap():
    blade = _make_stator_blade_data(throat_opening=4.0)
    r1, r2 = 15.0, 30.0

    chord_vals = np.asarray(blade["blade_curve"]["y"], dtype=float)
    pitch_vals = np.asarray(blade["blade_curve"]["x"], dtype=float)
    x1, y1, c_axial = _reference_point_and_scale(chord_vals, pitch_vals)

    expected_scale = conformal_length_scale(blade["throat_point"]["y"], x1, r1, r2, c_axial)
    expected = blade["throat_opening"] * expected_scale

    assert np.isclose(mapped_throat_width(blade, r1, r2, source="stator"), expected)


def test_mapped_throat_width_rejects_rotor_source():
    blade = _make_stator_blade_data()
    try:
        mapped_throat_width(blade, 15.0, 30.0, source="rotor")
        assert False, "expected a ValueError for source='rotor'"
    except ValueError:
        pass
