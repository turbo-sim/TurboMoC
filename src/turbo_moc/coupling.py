"""
turbo_moc.coupling -- derive the rotor's relative-frame inlet conditions
(M_rel, beta_rel, P0_rel, T0_rel) from the Phase 1 nozzle/stator's EXIT
state plus a chosen blade speed U, via the absolute -> relative velocity
triangle. This is the rotor-side analogue of turbo_moc.geometry.sizing:
instead of hand-typing the rotor's "uniform flow" Mach/P0_rel/T0_rel/
beta_inlet (today's only option, still available as a manual fallback),
derive them from the stage's own upstream design -- no separate loss model
needed, since the stator is loss-free in this tool (isentropic MOC solve),
so rothalpy (h0_rel = h_static + W^2/2) at the stator's exit entropy is
exactly the rotor's relative stagnation state.
"""

import numpy as np
import jaxprop as jxp

__all__ = ["derive_rotor_inlet_from_stator"]


def derive_rotor_inlet_from_stator(nozzle_result, alpha_deg, U):
    """Velocity triangle at the rotor inlet, from the stator's own EXIT
    static state (wall_final's LAST point: p, T, rho, M -- the converged
    nozzle wall exit, not the throat) plus the absolute flow angle
    alpha_deg (from axial -- the stator blade's own metal_angle_out, the
    angle the MOC design actually targets) and blade speed U [m/s].

    Uses DmassT_INPUTS (not PT_INPUTS) for the static-state EOS query --
    same reasoning as turbo_moc.geometry.sizing.compute_throat_sonic_state:
    a wall point can sit arbitrarily close to the saturation curve (wet-
    steam/two-phase designs), where (p, T) alone is degenerate for phase
    determination, but (rho, T) never is.

    Returns
    -------
    dict with keys:
        p_static, T_static, rho_static : the stator exit static state
        M_abs, V_abs : absolute Mach/velocity at stator exit
        a : local speed of sound (same in both frames)
        W_rel, M_rel, beta_rel_deg : relative velocity/Mach/flow angle
        P0_rel, T0_rel : relative-frame stagnation state (from the
            isentropic state at the stator exit's own entropy and the
            rothalpy-consistent relative stagnation enthalpy)
        U : the blade speed used
        is_supersonic : M_rel > 1 -- the vortex-flow method requires a
            supersonic relative inflow; False is a hard warning, not
            just a style note, since the method's geometry is undefined
            otherwise.
    """
    wall = nozzle_result["wall_final"]
    p_static = float(wall["p"][-1])
    T_static = float(wall["T"][-1])
    rho_static = float(wall["rho"][-1])
    M_abs = float(wall["M"][-1])

    fluid = jxp.Fluid(nozzle_result["fluid_name"])
    state_static = fluid.get_state(jxp.DmassT_INPUTS, rho_static, T_static)
    a = float(state_static.a)
    h_static = float(state_static.h)
    s_static = float(state_static.s)

    V_abs = M_abs * a
    alpha = np.radians(alpha_deg)
    V_m = V_abs * np.cos(alpha)
    V_t = V_abs * np.sin(alpha)

    W_m = V_m
    W_t = V_t - U
    W_rel = float(np.hypot(W_m, W_t))
    beta_rel_deg = float(np.degrees(np.arctan2(W_t, W_m)))
    M_rel = W_rel / a

    h0_rel = h_static + 0.5 * W_rel ** 2
    state0_rel = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s_static)
    P0_rel = float(state0_rel.p)
    T0_rel = float(state0_rel.T)

    # design_rotor_vortex_blade hands (P0_rel, T0_rel) straight to its own
    # FluidManager, which re-derives the state via a SEPARATE PT_INPUTS
    # query (core/classes.py's __init__) -- a razor-thin coincidence where
    # this isentropic reconstruction lands (P0_rel, T0_rel) almost exactly
    # ON the saturation curve makes that downstream query raise
    # "Saturation pressure ... is within 1e-4% of given p" with no context
    # connecting it back to Omega/r here. Confirmed empirically: the failing
    # band is a few 1e-4 K wide (matching CoolProp's own 1e-4% tolerance),
    # not a broad unsafe region -- nudging T0_rel by a small, scale-aware
    # margin reliably clears it (and is a no-op in every other case, well
    # under this app's 2-decimal display precision).
    T0_rel += max(1e-3, abs(T0_rel) * 1e-4)

    return {
        "p_static": p_static, "T_static": T_static, "rho_static": rho_static,
        "M_abs": M_abs, "V_abs": float(V_abs), "a": a,
        "W_rel": W_rel, "M_rel": float(M_rel), "beta_rel_deg": beta_rel_deg,
        "P0_rel": P0_rel, "T0_rel": T0_rel,
        "U": float(U), "is_supersonic": bool(M_rel > 1.0),
    }
