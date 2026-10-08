"""Checks for turbo_moc.coupling.derive_rotor_inlet_from_stator: the
absolute -> relative velocity triangle at the rotor inlet, from a stator
exit static state plus a chosen blade speed."""

import numpy as np

from turbo_moc.coupling import derive_rotor_inlet_from_stator

FLUID_NAME = "Nitrogen"


def _make_nozzle_result(p=2.0e5, T=300.0, rho=2.3, M=1.4):
    return {
        "wall_final": {"p": [p, p], "T": [T, T], "rho": [rho, rho], "M": [1.0, M]},
        "fluid_name": FLUID_NAME,
    }


def test_zero_blade_speed_leaves_relative_equal_to_absolute():
    nozzle_result = _make_nozzle_result(M=1.4)
    tri = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=0.0, U=0.0)

    # U=0: relative frame == absolute frame -- W_rel == V_abs, M_rel == M_abs.
    assert np.isclose(tri["W_rel"], tri["V_abs"])
    assert np.isclose(tri["M_rel"], tri["M_abs"])
    assert np.isclose(tri["beta_rel_deg"], 0.0, atol=1e-6)


def test_velocity_triangle_matches_manual_vector_subtraction():
    nozzle_result = _make_nozzle_result(M=1.4)
    alpha_deg = 65.0
    U = 80.0
    tri = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=alpha_deg, U=U)

    a = tri["a"]
    V_abs = tri["M_abs"] * a
    alpha = np.radians(alpha_deg)
    V_m, V_t = V_abs * np.cos(alpha), V_abs * np.sin(alpha)
    W_m, W_t = V_m, V_t - U
    expected_W = np.hypot(W_m, W_t)
    expected_beta = np.degrees(np.arctan2(W_t, W_m))

    assert np.isclose(tri["W_rel"], expected_W)
    assert np.isclose(tri["beta_rel_deg"], expected_beta)
    assert np.isclose(tri["M_rel"], expected_W / a)


def test_is_supersonic_flag_tracks_relative_mach():
    nozzle_result = _make_nozzle_result(M=1.4)
    # U close to the tangential absolute velocity component collapses most
    # of W_rel down to just its axial part, dragging M_rel below 1.
    tri_sub = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=70.0, U=300.0)
    assert tri_sub["M_rel"] < 1.0
    assert tri_sub["is_supersonic"] is False

    tri_super = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=70.0, U=50.0)
    assert tri_super["M_rel"] > 1.0
    assert tri_super["is_supersonic"] is True


def test_relative_stagnation_temperature_rises_with_relative_speed():
    nozzle_result = _make_nozzle_result(M=1.4)
    tri = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=65.0, U=80.0)
    # T0_rel = static T plus a positive dynamic contribution from W_rel.
    assert tri["T0_rel"] > tri["T_static"]
    assert tri["P0_rel"] > tri["p_static"]


def test_T0_rel_is_nudged_off_the_raw_isentropic_reconstruction():
    """T0_rel must differ from the raw (unnudged) isentropic state's own T
    by at least the documented margin -- the nudge exists specifically so
    design_rotor_vortex_blade's downstream FluidManager (a SEPARATE
    PT_INPUTS query) never lands exactly on the saturation curve, see
    derive_rotor_inlet_from_stator's docstring."""
    import jaxprop as jxp

    nozzle_result = _make_nozzle_result(M=1.4)
    alpha_deg, U = 65.0, 80.0
    tri = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=alpha_deg, U=U)

    fluid = jxp.Fluid(FLUID_NAME)
    state_static = fluid.get_state(jxp.DmassT_INPUTS, nozzle_result["wall_final"]["rho"][-1],
                                    nozzle_result["wall_final"]["T"][-1])
    h0_rel = state_static.h + 0.5 * tri["W_rel"] ** 2
    raw_state0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, state_static.s)

    assert tri["T0_rel"] - float(raw_state0.T) >= 1e-3


def test_rotor_inlet_state_survives_the_downstream_fluid_manager_query():
    """Regression test for a real reported crash: design_rotor_vortex_blade
    hands (P0_rel, T0_rel) to its own FluidManager, which re-derives the
    state via jxp.FluidJAX's PT_INPUTS (core/classes.py's __init__) -- a
    SEPARATE query from the one used here to derive P0_rel/T0_rel. For a
    two-phase-capable fluid (Cyclopentane, matching the reported case) an
    isentropic reconstruction can coincidentally land almost exactly on the
    saturation curve, where that downstream query raises "Saturation
    pressure ... is within 1e-4% of given p". This replays that exact
    downstream query against our own output to confirm it no longer fails."""
    import jaxprop as jxp

    nozzle_result = {
        "wall_final": {"p": [2.0e5, 95006.4], "T": [320.0, 320.45],
                        "rho": [12.0, 14.095], "M": [1.0, 1.4033]},
        "fluid_name": "Cyclopentane",
    }
    tri = derive_rotor_inlet_from_stator(nozzle_result, alpha_deg=70.0, U=90.0)

    fluid_jax = jxp.FluidJAX("Cyclopentane", "HEOS")
    state = fluid_jax.get_state(jxp.PT_INPUTS, tri["P0_rel"], tri["T0_rel"])
    assert float(state.p) > 0
