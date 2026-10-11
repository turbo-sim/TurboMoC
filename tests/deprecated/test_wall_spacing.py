"""Checks for turbo_moc.meshing.wall_spacing: the y+ -> first-cell estimate
on a nozzle expansion and the geometric inflation stack."""

import numpy as np
import pytest

from turbo_moc.meshing.wall_spacing import estimate_first_layer_height, inflation_stack


@pytest.fixture(scope="module")
def nozzle():
    # Single-phase nitrogen expansion, three wall stations.
    return {"fluid_name": "Nitrogen",
            "wall_final": {"rho": [3.0, 2.0, 1.2], "T": [280.0, 250.0, 210.0], "M": [1.0, 1.5, 2.0]}}


def test_first_height_matches_the_flat_plate_formula(nozzle):
    import jaxprop as jxp

    result = estimate_first_layer_height(nozzle, 50.0, 1.0, cell_centroid=False)
    w = nozzle["wall_final"]
    state = jxp.Fluid("Nitrogen").get_state(jxp.DmassT_INPUTS, np.asarray(w["rho"]), np.asarray(w["T"]))
    rho, mu = np.asarray(w["rho"]), np.asarray(state.viscosity)
    u = np.asarray(w["M"]) * np.asarray(state.speed_of_sound)
    cf = 0.026 * (rho * u * 0.05 / mu) ** (-1 / 7)
    expected = np.min((mu / rho) / np.sqrt(0.5 * cf * u**2)) * 1e3
    assert np.isclose(result["first_height_mm"], expected, rtol=1e-10)


def test_first_height_scales_with_y_plus_and_centroid_convention(nozzle):
    one = estimate_first_layer_height(nozzle, 50.0, 1.0)["first_height_mm"]
    assert np.isclose(estimate_first_layer_height(nozzle, 50.0, 30.0)["first_height_mm"], 30 * one)
    wall = estimate_first_layer_height(nozzle, 50.0, 1.0, cell_centroid=False)["first_height_mm"]
    assert np.isclose(one, 2 * wall)


def test_fluid_without_viscosity_model_uses_chung():
    """Novec649 has no CoolProp viscosity model (NaN): fall back, don't fail."""
    nozzle = {"fluid_name": "Novec649",
              "wall_final": {"rho": [30.0, 15.0], "T": [380.0, 350.0], "M": [1.0, 1.5]}}
    result = estimate_first_layer_height(nozzle, 5.0, 1.0)
    assert np.isfinite(result["first_height_mm"]) and result["first_height_mm"] > 0
    assert result["viscosity_model"].startswith("Chung")
    assert 5e-6 < result["viscosity"] < 5e-5


def test_chung_matches_coolprop_for_a_nonpolar_vapour():
    import CoolProp.CoolProp as CP
    from turbo_moc.meshing.wall_spacing import chung_dilute_gas_viscosity

    reference = CP.PropsSI("V", "T", 400.0, "P", 1e5, "Cyclopentane")
    assert np.isclose(chung_dilute_gas_viscosity("Cyclopentane", 400.0), reference, rtol=0.03)


def test_inflation_stack_is_the_geometric_series():
    thickness, last = inflation_stack(0.01, 1.2, 10)
    heights = 0.01 * 1.2 ** np.arange(10)
    assert np.isclose(thickness, heights.sum()) and np.isclose(last, heights[-1])
    with pytest.raises(ValueError):
        inflation_stack(0.01, 1.0, 10)
