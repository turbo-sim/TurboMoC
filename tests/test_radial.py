"""Checks for turbo_moc.geometry.radial's conformal-mapping length scale:
conformal_length_scale / mapped_throat_width -- the log-spiral map
preserves angles, not lengths, so a 2D-plane length needs an exact local
magnification factor to be valid once mapped onto the annulus."""

import numpy as np
import pytest

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


def _wrapped_swirl(wrap_fn, data, curve_key, p, probe, r1, r2):
    """v_theta / v_r of the step probe -> p (or p -> probe) after wrapping:
    both points are appended to the blade's own contour so they go through
    exactly the map (reference point, c_axial) the blade itself gets."""
    curve = data[curve_key]
    patched = {**data, curve_key: {"x": list(curve["x"]) + [p[0], probe[0]],
                                   "y": list(curve["y"]) + [p[1], probe[1]]}}
    c = wrap_fn(patched, r1, r2, 1)["blades"][0]
    (X0, Y0), (X1, Y1) = (c["x"][-2], c["y"][-2]), (c["x"][-1], c["y"][-1])
    dX, dY = X1 - X0, Y1 - Y0
    return (X0 * dY - Y0 * dX) / (X0 * dX + Y0 * dY)


@pytest.mark.parametrize("stator_r, rotor_r", [((150.0, 300.0), (610.0, 305.0)),   # outward
                                               ((300.0, 200.0), (97.5, 195.0))])   # inward
def test_radial_rotor_inlet_swirl_matches_stator_exit_swirl(stator_r, rotor_r):
    """The rotor's leading edge must face the stator's exit flow: same
    tangential sense at the interface. Regression for the rotor coming out
    with the opposite handedness (concavity against the incoming flow)."""
    from turbo_moc.geometry.blade import parametrize_stator_blade
    from turbo_moc.geometry.radial import wrap_blade_radial, wrap_rotor_blade_radial

    wall_x = np.linspace(0.0, 10.0, 50)
    stator = parametrize_stator_blade(wall_x, 1.0 + 0.1 * wall_x, metal_angle_in=0.0,
                                      metal_angle_out=70.0, r_trailing=0.1143)
    sx, sy = stator["suction"]["x"], stator["suction"]["y"]
    te, upstream = np.array([sx[-1], sy[-1]]), np.array([sx[-2], sy[-2]])
    d_s = (te - upstream) / np.linalg.norm(te - upstream)
    # probe upstream of the TE (larger chord), so the chord-min reference is untouched
    stator_swirl = _wrapped_swirl(wrap_blade_radial, stator, "blade_curve", te - 1e-4 * d_s, te, *stator_r)

    # Rotor convention (design_rotor_vortex_blade): flow along +x, inlet at
    # min x, inlet flow direction (1, tan(beta_inlet)) with beta_inlet > 0.
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    rotor = {"blade": {"x": (2.0 * np.cos(t)).tolist(), "y": (0.3 * np.sin(t)).tolist()}}
    i_in = int(np.argmin(rotor["blade"]["x"]))
    p = np.array([rotor["blade"]["x"][i_in], rotor["blade"]["y"][i_in]])
    d_r = np.array([1.0, np.tan(np.radians(57.0))])
    d_r /= np.linalg.norm(d_r)
    rotor_swirl = _wrapped_swirl(wrap_rotor_blade_radial, rotor, "blade", p, p + 1e-4 * d_r, *rotor_r)

    assert np.sign(stator_swirl) == np.sign(rotor_swirl)
    assert np.isclose(abs(stator_swirl), np.tan(np.radians(70.0)), rtol=1e-3)
    assert np.isclose(abs(rotor_swirl), np.tan(np.radians(57.0)), rtol=1e-3)


def test_mapped_throat_width_rejects_rotor_source():
    blade = _make_stator_blade_data()
    try:
        mapped_throat_width(blade, 15.0, 30.0, source="rotor")
        assert False, "expected a ValueError for source='rotor'"
    except ValueError:
        pass
