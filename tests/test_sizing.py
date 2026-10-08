"""Checks for turbo_moc.geometry.sizing: pure post-hoc rescaling of a Phase 2
cascade blade, and the Case A / Case B stator-sizing closed forms."""

import numpy as np

from turbo_moc.geometry import (
    compute_throat_sonic_state,
    rescale_stator_blade,
    size_stator_mode_a,
    size_stator_mode_b,
)

FLUID_NAME = "Nitrogen"


def _make_blade(pitch=10.0, throat_opening=4.0):
    return {
        "suction": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        "pressure": {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.5, 1.0]},
        "blade_curve": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "trailing_edge": {"x": [2.0, 2.1], "y": [1.0, 1.1]},
        "control_points": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        "knots": [0.0, 0.0, 0.5, 1.0, 1.0],
        "pitch": pitch,
        "throat_opening": throat_opening,
        "inlet_opening": 6.0,
        "axial_chord_convergent": 3.0,
        "axial_chord_divergent": 5.0,
        "r_trailing": 0.2,
        "metal_angle_in": 0.0,
        "metal_angle_out": 75.0,
        "mirror": False,
        "units": "mm",
    }


def _make_nozzle_result(p=2.0e5, T=300.0, rho=2.3):
    return {
        "wall_final": {"p": [p], "T": [T], "rho": [rho], "M": [1.0]},
        "fluid_name": FLUID_NAME,
    }


def test_rescale_stator_blade_scales_lengths_only():
    blade = _make_blade()
    scaled = rescale_stator_blade(blade, k=2.0)

    assert scaled["pitch"] == blade["pitch"] * 2.0
    assert scaled["throat_opening"] == blade["throat_opening"] * 2.0
    assert scaled["blade_curve"]["x"] == [0.0, 2.0]
    assert scaled["blade_curve"]["y"] == [0.0, 2.0]
    assert scaled["control_points"] == [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]

    # Dimensionless/angular/metadata fields untouched.
    assert scaled["knots"] == blade["knots"]
    assert scaled["metal_angle_in"] == blade["metal_angle_in"]
    assert scaled["metal_angle_out"] == blade["metal_angle_out"]
    assert scaled["mirror"] is False
    assert scaled["units"] == "mm"
    assert scaled["sizing_k"] == 2.0

    # Original dict not mutated.
    assert blade["pitch"] == 10.0


def test_compute_throat_sonic_state_returns_density_and_sound_speed():
    nozzle_result = _make_nozzle_result()
    rho_star, a_star = compute_throat_sonic_state(nozzle_result)
    assert rho_star == nozzle_result["wall_final"]["rho"][0]
    assert a_star > 0


def test_size_stator_mode_a_satisfies_pitch_and_massflow_consistency():
    blade = _make_blade(pitch=10.0, throat_opening=4.0)
    nozzle_result = _make_nozzle_result()

    mdot = 0.5
    n_blades = 40
    r_hub = 300.0
    result = size_stator_mode_a(blade, nozzle_result, mdot, n_blades, r_hub)

    k = result["k"]
    H = result["H"]
    rho_star, a_star = result["rho_star"], result["a_star"]

    # Pitch consistency: k * pitch_ref == 2*pi*r_hub / n_blades
    assert np.isclose(k * blade["pitch"], 2.0 * np.pi * r_hub / n_blades)

    # Mass-flow consistency: mdot == rho_star * a_star * (k*throat_ref) * H * n_blades * 1e-6
    mdot_check = rho_star * a_star * (k * blade["throat_opening"]) * H * n_blades * 1e-6
    assert np.isclose(mdot_check, mdot)

    assert np.isclose(result["r_shroud"], r_hub + H)
    assert result["scaled_blade"]["pitch"] == blade["pitch"] * k


def test_size_stator_mode_b_residual_zero_when_self_consistent():
    blade = _make_blade(pitch=10.0, throat_opening=4.0)
    nozzle_result = _make_nozzle_result()

    # Build a self-consistent Case B input from a Case A result, so
    # k_pitch and k_massflow must agree exactly (residual ~ 0).
    mode_a = size_stator_mode_a(blade, nozzle_result, mdot=0.5, n_blades=40, r_hub=300.0)

    result_b = size_stator_mode_b(
        blade, nozzle_result,
        r_hub=mode_a["r_hub"], H=mode_a["H"],
        n_blades=mode_a["n_blades"], mdot=mode_a["mdot"],
    )

    assert abs(result_b["residual"]) < 1e-9
    assert np.isclose(result_b["k_pitch"], mode_a["k"])
    assert np.isclose(result_b["k_massflow"], mode_a["k"])


def test_size_stator_mode_b_residual_nonzero_when_height_perturbed():
    blade = _make_blade(pitch=10.0, throat_opening=4.0)
    nozzle_result = _make_nozzle_result()

    mode_a = size_stator_mode_a(blade, nozzle_result, mdot=0.5, n_blades=40, r_hub=300.0)

    # Perturb H away from the self-consistent value -- k_massflow should
    # diverge from k_pitch, flagging that the blade shape (not just scale)
    # may need revisiting.
    perturbed_H = mode_a["H"] * 1.5
    result_b = size_stator_mode_b(
        blade, nozzle_result,
        r_hub=mode_a["r_hub"], H=perturbed_H,
        n_blades=mode_a["n_blades"], mdot=mode_a["mdot"],
    )

    assert abs(result_b["residual"]) > 0.1
