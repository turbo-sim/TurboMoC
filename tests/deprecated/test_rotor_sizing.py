"""Checks for turbo_moc.geometry.sizing's rotor-side functions:
rescale_rotor_blade / size_rotor_from_pitch -- the Phase 5 (3D rotor
design) analogue of the stator's size_stator_mode_a/b, deriving a scale
factor from pitch consistency alone against an annulus shared with the
stator (no mass-flow equation at the 2D-cascade level for this tool's
rotor method)."""

import numpy as np

from turbo_moc.geometry import rescale_rotor_blade, size_rotor_from_pitch


def _make_rotor_data(pitch=5.0, chord=8.0):
    return {
        "blade": {"x": [0.0, 1.0, 2.0, 1.0], "y": [0.0, 1.0, 0.0, -1.0]},
        "pitch": pitch,
        "chord": chord,
        "beta_inlet": 65.0,
        "beta_outlet": -68.74,
    }


def test_rescale_rotor_blade_scales_lengths_only():
    rotor = _make_rotor_data()
    scaled = rescale_rotor_blade(rotor, k=3.0)

    assert scaled["blade"]["x"] == [0.0, 3.0, 6.0, 3.0]
    assert scaled["blade"]["y"] == [0.0, 3.0, 0.0, -3.0]
    assert scaled["pitch"] == rotor["pitch"] * 3.0
    assert scaled["chord"] == rotor["chord"] * 3.0

    # Angles untouched, original not mutated.
    assert scaled["beta_inlet"] == rotor["beta_inlet"]
    assert scaled["beta_outlet"] == rotor["beta_outlet"]
    assert scaled["units"] == "mm"
    assert scaled["sizing_k"] == 3.0
    assert rotor["blade"]["x"] == [0.0, 1.0, 2.0, 1.0]


def test_size_rotor_from_pitch_satisfies_pitch_consistency():
    rotor = _make_rotor_data(pitch=5.0)
    r_hub, r_shroud, n_blades = 150.0, 180.0, 34
    result = size_rotor_from_pitch(rotor, r_hub, r_shroud, n_blades)

    assert np.isclose(result["k"] * rotor["pitch"], 2.0 * np.pi * r_hub / n_blades)
    assert np.isclose(result["pitch_scaled"], result["k"] * rotor["pitch"])
    assert result["r_hub"] == r_hub
    assert result["r_shroud"] == r_shroud
    assert result["n_blades"] == n_blades
    assert result["scaled_blade"]["pitch"] == rotor["pitch"] * result["k"]


def test_size_rotor_from_pitch_independent_blade_count_from_stator():
    # Same annulus (r_hub/r_shroud shared with the stator), different N --
    # the whole point of the rotor having its own free n_blades.
    rotor = _make_rotor_data(pitch=5.0)
    r_hub, r_shroud = 150.0, 180.0

    result_15 = size_rotor_from_pitch(rotor, r_hub, r_shroud, n_blades=15)
    result_34 = size_rotor_from_pitch(rotor, r_hub, r_shroud, n_blades=34)

    assert result_15["k"] != result_34["k"]
    assert np.isclose(result_15["k"], (2.0 * np.pi * r_hub / 15) / rotor["pitch"])
    assert np.isclose(result_34["k"], (2.0 * np.pi * r_hub / 34) / rotor["pitch"])
