"""JAX-jitted steady 2D (x, y) MOC unit process for the single-velocity
5-equation nonequilibrium two-phase model -- fast counterpart of
`nonequilibrium_5eq.py`.

Why this exists
-----------------
`nonequilibrium_5eq.py` calls `jaxprop.Fluid.get_state_metastable` (a
scipy least-squares solve) once or twice per phase, per mixture-property
evaluation, and `solve_p_from_energy` calls that up to ~40 times per
bisection x 3 predictor/corrector passes -- confirmed directly (this
session) to make a single ~45-step, 6-ray nozzle design take several
minutes, and an 8-step debug run still exceed 3 minutes. That made
iterating on a real geometry bug (see below) impractically slow.

This module replaces every per-point scipy call with the SAME table-based
approach `moc_unsteady.fast_5eqn.py`/`fast_5eqn_1d.py` already use and
validated: `jaxprop.FluidBicubic` for the gas phase (direct HmassP_INPUTS
bicubic evaluation) and `fast_5eqn_1d.build_metastable_liquid_table` +
`_lookup_metastable_liquid` (precomputed bilinear (h,p) table) for the
metastable liquid phase -- both pure jnp array ops after one-time host-side
construction, hence jit/vmap-transparent. `solve_p_from_energy`'s bisection
becomes a FIXED-iteration `jax.lax.fori_loop` (same convention as
`moc_unsteady.nonequilibrium.HEMEos.state_hs`'s fixed-iteration Newton
solve, or `fast_5eqn_1d._solve_inlet_u_1d`'s fixed-iteration bisection).

Also fixes a real correctness bug found while debugging the (slow)
`nonequilibrium_5eq.py` path: its `interior_point_prop` computed the
new point's (x, y) geometry using ONLY the two feet's own characteristic
slopes (a single pass through `jitted_moc_predictor_algebra`), unlike
`moc.core.classes.interior_point_prop`'s exact-invariant version, which
iterates the geometry using slopes averaged between each foot and the
(now known) new point, converging in a few passes. Without that
correction, position error compounds step over step -- confirmed directly:
a 6-ray/45-step design's wall position drifted backward in x by the final
steps instead of monotonically advancing. `interior_point_step` below adds
that same iterative slope-averaging corrector (fixed N_GEOM_ITER passes,
approximating the new point's Mach angle from the average of the two
feet's speed of sound, since -- unlike the single-phase exact-invariant
case -- this model has no closed form giving the new point's true sound
speed before the pressure solve).

Usage
-----
Build `fb_g` (FluidBicubic), `liq_table` (metastable liquid table), and
`sat_tables` (log_p grid + liq/vap saturation arrays, e.g. from a
`moc_unsteady.nonequilibrium.SaturationCurve`) ONCE at setup (host-side,
non-jitted, same cost profile as `moc_unsteady`'s own scripts). Every
per-point call below (`mixture_props`, `solve_p_from_energy`,
`interior_point_step`, `wall_point_step`) is then a pure JAX function of
those tables plus plain float/jnp-scalar state -- safe to wrap in
`jax.jit` (done here at module load for the two `_step` entry points) and
call repeatedly from a plain Python driver loop with negligible per-call
Python overhead after the first (tracing) call.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOC_UNSTEADY_SRC = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "moc_unsteady", "src")
)
if os.path.isdir(_MOC_UNSTEADY_SRC) and _MOC_UNSTEADY_SRC not in sys.path:
    sys.path.insert(0, _MOC_UNSTEADY_SRC)

jax.config.update("jax_enable_x64", True)

import jaxprop as jxp  # noqa: E402

from moc_unsteady.nonequilibrium import (  # noqa: E402
    wallis_sound_speed,
    mixture_density,
    five_eq_source_rates,
)
from moc_unsteady.fast_5eqn import _phase_props_ph, _sat_lookup  # noqa: E402
from moc_unsteady.fast_5eqn_1d import (  # noqa: E402
    _lookup_metastable_liquid,
    _lookup_metastable_liquid_transport,
    build_metastable_liquid_table,
)

from moc.core.classes import jitted_moc_predictor_algebra  # noqa: E402


N_P_BISECT = 40   # fixed p-solve bisection iterations
N_MASS_TRANSFER_SUBSTEPS = 20   # see _pathline_integrate_substepped
N_GEOM_ITER = 6    # fixed geometry/velocity slope-averaging corrector passes


class RayState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    p: jnp.ndarray
    alpha_g: jnp.ndarray
    h_g: jnp.ndarray
    h_l: jnp.ndarray


class ClosureParams(NamedTuple):
    Nb: float
    Nd: float
    alpha_b: float
    alpha_d: float
    Nu_l: float
    mass_transfer: bool = True   # False -> sigma_Mg=Q_g=Q_l=0 (frozen
                                 # composition, pure mechanical/acoustic
                                 # problem, no two-phase source terms at
                                 # all). A plain Python bool, resolved as a
                                 # static branch at trace time (like
                                 # five_eq_source_rates' own mass_transfer
                                 # flag), not a runtime jnp.where.


def _gas_transport_at(fb_g, p, h):
    p = jnp.asarray(p)
    h = jnp.asarray(h)
    shape = p.shape
    st = jax.vmap(lambda pp, hh: fb_g._get_state_scalar(jxp.HmassP_INPUTS, hh, pp))(p.ravel(), h.ravel())
    return st.k.reshape(shape), st.cp.reshape(shape)


def mixture_props(fb_g, liq_table, p, alpha_g, h_g, h_l):
    """(rho_mix, a_mix, rho_g, a_g, T_g, rho_l, a_l, T_l) at a local state,
    both phases via jittable table lookups (no scipy)."""
    rho_g, a_g, T_g = _phase_props_ph(fb_g, p, h_g)
    rho_l, a_l, T_l = _lookup_metastable_liquid(liq_table, h_l, p)
    a_mix = wallis_sound_speed(alpha_g, rho_g, a_g, rho_l, a_l)
    rho_mix = mixture_density(alpha_g, rho_g, rho_l)
    return rho_mix, a_mix, rho_g, a_g, T_g, rho_l, a_l, T_l


def h_mix_of(rho_mix, alpha_g, rho_g, h_g, rho_l, h_l):
    alpha_l = 1.0 - alpha_g
    return (alpha_g * rho_g * h_g + alpha_l * rho_l * h_l) / rho_mix


def solve_p_from_energy(fb_g, liq_table, h0, V, alpha_g, h_g, h_l, p_lo, p_hi, n_iter=N_P_BISECT):
    """Fixed-iteration bisection (jax.lax.fori_loop, no Python control
    flow) for p such that h_mix(p, alpha_g, h_g, h_l) + V^2/2 = h0. See
    `nonequilibrium_5eq.FiveEqFluidManager.solve_p_from_energy` for the
    (non-jitted) derivation/rationale -- identical physics, just a
    jit-safe fixed-loop bisection instead of a Python while/for with
    early return."""
    target = h0 - 0.5 * V * V

    def resid(p):
        rho_mix, _, rho_g, _, _, rho_l, _, _ = mixture_props(fb_g, liq_table, p, alpha_g, h_g, h_l)
        return h_mix_of(rho_mix, alpha_g, rho_g, h_g, rho_l, h_l) - target

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        f_lo = resid(lo)
        f_mid = resid(mid)
        bracket_lo = f_lo * f_mid <= 0.0
        new_hi = jnp.where(bracket_lo, mid, hi)
        new_lo = jnp.where(bracket_lo, lo, mid)
        return (new_lo, new_hi)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.asarray(p_lo), jnp.asarray(p_hi)))
    return 0.5 * (lo + hi)


def source_rates(sat_tables, fb_g, liq_table, closure: ClosureParams, p, alpha_g, h_g, h_l):
    rho_g, a_g, T_g = _phase_props_ph(fb_g, p, h_g)
    rho_l, a_l, T_l = _lookup_metastable_liquid(liq_table, h_l, p)
    liq, vap = _sat_lookup(sat_tables["log_p"], sat_tables["liq"], sat_tables["vap"], p)
    T_sat, h_sat_g, h_sat_l = liq["T"], vap["h"], liq["h"]

    if not closure.mass_transfer:
        # Frozen composition: no mass or heat exchange between phases at
        # all (stronger than five_eq_source_rates' own mass_transfer=False,
        # which still exchanges heat -- this is the "disable source terms
        # entirely" simplification for isolating the pure mechanical/
        # acoustic problem, skipping the TA-BD/transport-property calls
        # too since they're unused).
        zero = jnp.zeros_like(jnp.asarray(alpha_g))
        return zero, zero, zero, T_sat, h_sat_g, h_sat_l, rho_g, rho_l

    k_g, cp_g = _gas_transport_at(fb_g, p, h_g)
    k_l, cp_l, mu_l = _lookup_metastable_liquid_transport(liq_table, h_l, p)
    # gas viscosity not tabulated in fb_g by default -> mu_g=None falls
    # back to the model's own Nu=2 (zero-slip conduction) limit, see
    # moc_unsteady.nonequilibrium.five_eq_source_rates/interfacial_heat_flux.
    rates = five_eq_source_rates(
        alpha_g, T_g, T_l, T_sat, h_sat_g, h_sat_l, k_g, k_l,
        mu_g=None, rho_g=rho_g, cp_g=cp_g, mass_transfer=True,
        alpha_b=closure.alpha_b, alpha_d=closure.alpha_d, Nb=closure.Nb, Nd=closure.Nd, Nu_l=closure.Nu_l,
    )
    return rates.sigma_Mg, rates.Q_g, rates.Q_l, T_sat, h_sat_g, h_sat_l, rho_g, rho_l


def _theta_mu(u, v, a_mix):
    V = jnp.hypot(u, v)
    theta = jnp.arctan2(v, u)
    mu = jnp.arcsin(jnp.clip(a_mix / V, -1.0, 1.0))
    return theta, mu, V


def _QRS(u, v, a_mix, y, delta, lam):
    Q = u ** 2 - a_mix ** 2
    R1 = 2.0 * u * v - Q * lam
    S = jnp.where(jnp.abs(y) <= 1e-9, 0.0, delta * (a_mix ** 2 * v) / y)
    return Q, R1, S


def _cot(angle):
    """cos/sin evaluated directly, NOT 1/tan(angle) -- for angle near
    +-90deg (near-sonic characteristics, mu approaching 90deg), tan(angle)
    itself is already huge/imprecise before any reciprocal is taken, so
    1/tan(angle) does NOT recover a well-conditioned small value (this
    exact confusion was confirmed directly earlier this session in the
    Pt-based exact-invariant kernel code -- 1/tan gave bit-identical,
    still-broken output to the plain tan formulation). cos/sin never forms
    the blown-up intermediate: sin(~+-90deg)~=1 (well-behaved) and
    cos(~+-90deg) is small but exact, so the ratio stays accurate exactly
    where tan() itself is not."""
    return jnp.cos(angle) / jnp.sin(angle)


def _intersect_cot(xM, yM, cotM, xP, yP, cotP):
    """Intersection of two lines in inverse-slope (x = x0 + cot*(y-y0))
    form instead of the standard y = y0 + lam*(x-x0) -- the standard
    remedy for near-vertical characteristics, see _cot's docstring."""
    y = (xP - xM + cotM * yM - cotP * yP) / (cotM - cotP)
    x = xM + cotM * (y - yM)
    return x, y


def _geometry_velocity_corrector(xP, yP, thetaP, muP, aP, QP, R1P, SP, uP, vP,
                                 xM, yM, thetaM, muM, aM, QM, R1M, SM, uM, vM,
                                 n_iter=N_GEOM_ITER):
    """Iterative slope-averaging geometry+velocity solve (the corrector
    missing from `nonequilibrium_5eq.interior_point_prop`, see module
    docstring): each pass evaluates the C+/C- slopes at the AVERAGE of
    each foot's own (theta, mu) and the current new-point estimate's,
    then re-solves both the (x, y) intersection and the linear (u, v)
    compatibility system at that x. The new point's mu is approximated
    from the average of the two feet's sound speed (no closed-form sound
    speed is available before the pressure solve in this model, unlike
    the single-phase exact-invariant case classes.py's own corrector
    exploits).

    Geometry intersection uses cot (inverse slope), not tan -- confirmed
    directly (this session) that the original tan-based version produces
    badly corrupted geometry (small/negative-y artifacts) whenever the
    characteristics run near-vertical (mu approaching 90deg, i.e. M close
    to 1 -- exactly the regime a Sauer-line-seeded IVP march starts in).
    """
    cotP0 = _cot(thetaP + muP)
    cotM0 = _cot(thetaM - muM)
    x, y = _intersect_cot(xM, yM, cotM0, xP, yP, cotP0)
    # Velocity predictor still needs lamP/lamM (dy/dx) for the Q/R1/S
    # linear system below -- fine even near-vertical, since that system
    # doesn't divide by (lamM-lamP) the way the geometry intersection did.
    lamP0 = jnp.tan(thetaP + muP)
    lamM0 = jnp.tan(thetaM - muM)
    TP0 = SP * (x - xP) + QP * uP + R1P * vP
    TM0 = SM * (x - xM) + QM * uM + R1M * vM
    denom_v0 = QM * R1P - QP * R1M
    v = (TP0 * QM - TM0 * QP) / denom_v0
    u = (TM0 - R1M * v) / QM
    a_approx = 0.5 * (aP + aM)

    def body(_, carry):
        x, y, u, v = carry
        V_new = jnp.hypot(u, v)
        theta_new = jnp.arctan2(v, u)
        mu_new = jnp.arcsin(jnp.clip(a_approx / V_new, -1.0, 1.0))
        cotP_avg = _cot(0.5 * (thetaP + theta_new) + 0.5 * (muP + mu_new))
        cotM_avg = _cot(0.5 * (thetaM + theta_new) - 0.5 * (muM + mu_new))
        x_new, y_new = _intersect_cot(xM, yM, cotM_avg, xP, yP, cotP_avg)
        TP = SP * (x_new - xP) + QP * uP + R1P * vP
        TM = SM * (x_new - xM) + QM * uM + R1M * vM
        denom_v = QM * R1P - QP * R1M
        v_new = (TP * QM - TM * QP) / denom_v
        u_new = (TM - R1M * v_new) / QM
        return (x_new, y_new, u_new, v_new)

    x, y, u, v = jax.lax.fori_loop(0, n_iter, body, (x, y, u, v))
    return x, y, u, v


def _pathline_step(alpha_S, h_g_S, h_l_S, rho_gS, rho_lS, sigma_Mg, Q_g, Q_l, h_ex, dt,
                   h_g_bounds, h_l_bounds):
    """Mass-conservative pathline update for (alpha_g, h_g, h_l), bounded
    even as either phase's mass -> 0.

    The original version of this update was a naive Euler step, dh =
    dt*(sigma_Mg*(h_ex-h)+Q)/(alpha*rho) -- singular as the phase's own
    mass alpha*rho -> 0. Confirmed directly (this session): a rapidly
    evaporating case drove alpha_g to its 1.0 clip bound (liquid fully
    vaporized) within a handful of steps, and the h_l update's division by
    the vanishing liquid mass blew up, corrupting the Wallis mixture sound
    speed and cascading to NaN. `fast_5eqn_1d.py`'s `_pathline_update_1d`
    hit the identical singularity at the opposite limit (vapor mass -> 0
    for a near-zero-void-fraction reservoir) and fixed it with mass-
    weighted conservative mixing instead of additive Euler: the new phase
    enthalpy is a mass-weighted average of the OLD phase mass (at its own
    h) and the newly transferred mass (entering at h_ex), which stays
    bounded between h_S and h_ex regardless of how small the old mass is.
    Ported here, plus the equivalent treatment for alpha_g itself (updated
    via the phase MASS balance, not a direct alpha_g rate, so it inherits
    the same boundedness).
    """
    alpha_l_S = 1.0 - alpha_S
    mass_g = alpha_S * rho_gS
    mass_l = alpha_l_S * rho_lS
    dm_raw = dt * sigma_Mg
    # Cap the transferred mass at what's actually available in the DONOR
    # phase (mass_l if evaporating, mass_g if condensing) -- without this,
    # a source rate computed as if the donor phase still had plenty of
    # mass (not "aware" it's nearly depleted) can request dm far exceeding
    # what exists. The mass_new = clip(mass +- dm, 1e-9, None) floor below
    # only stops the RESULT going negative; it does nothing to stop dm
    # itself from being wildly oversized, which still blows up the
    # conservative-mixing numerator (mass*h - dm*h_ex) even with that
    # floor in place. Confirmed directly: without this cap, alpha_g still
    # saturated at 1.0 and M still crashed to 0 in the exact same place as
    # before the mixing-formula fix, i.e. that fix alone was insufficient.
    dm = jnp.where(dm_raw > 0.0, jnp.minimum(dm_raw, mass_l), jnp.maximum(dm_raw, -mass_g))
    mass_g_before_floor = mass_g + dm
    mass_l_before_floor = mass_l - dm
    mass_g_new = jnp.clip(mass_g_before_floor, 1e-9, None)
    mass_l_new = jnp.clip(mass_l_before_floor, 1e-9, None)

    # Mixing-ratio floor bug (confirmed directly, this session, porting
    # this closure into the steady 2D MOC design tool): the numerator
    # below (mass*h - dm*h_ex) is computed from the UNCLIPPED mass, while
    # the denominator uses the CLIPPED (floored) mass_new. When a donor
    # phase's mass is small (e.g. ~1e-6, well above the 1e-9 floor but
    # still tiny) and gets ~fully depleted by dm, the numerator stays a
    # small but non-negligible finite value while the denominator is
    # floored to 1e-9 -- a scale mismatch that produces an enthalpy wrong
    # by many orders of magnitude (measured: ~1e8 J/kg vs a physically
    # sane ~1e5-1e6), which then gets silently clipped to h_bounds and
    # corrupts everything downstream (mixture sound speed -> M=0). Once a
    # phase's mass is genuinely driven to (near) zero, its own enthalpy
    # is physically irrelevant (negligible mass) -- just use h_ex
    # directly rather than an ill-conditioned mixing ratio.
    h_g_mix_raw = (mass_g * h_g_S + dm * h_ex) / mass_g_new
    h_l_mix_raw = (mass_l * h_l_S - dm * h_ex) / mass_l_new
    h_g_mix = jnp.where(mass_g_before_floor <= 1e-9, h_ex, h_g_mix_raw)
    h_l_mix = jnp.where(mass_l_before_floor <= 1e-9, h_ex, h_l_mix_raw)
    # Heat-exchange term dt*Q/mass has the SAME donor-mass singularity as
    # dm above (Q_g, Q_l don't know their phase is nearly depleted
    # either); rather than rescale it heuristically, bound the RESULT to
    # the property tables' own valid domain below (h_bounds_*), the same
    # defensive practice fast_5eqn_1d.py uses for exactly this failure
    # mode (its interior_point_1d clips h_g6/h_l6 to fb_g.h_min/h_max and
    # the liquid table's own h_grid bounds after every update).
    h_g_new = h_g_mix + dt * Q_g / jnp.maximum(mass_g, 1e-9)
    h_l_new = h_l_mix + dt * Q_l / jnp.maximum(mass_l, 1e-9)
    h_g_new = jnp.clip(h_g_new, h_g_bounds[0], h_g_bounds[1])
    h_l_new = jnp.clip(h_l_new, h_l_bounds[0], h_l_bounds[1])
    alpha_g_new = jnp.clip(mass_g_new / rho_gS, 1e-9, 1.0 - 1e-9)
    return alpha_g_new, h_g_new, h_l_new


def _pathline_integrate_substepped(fb_g, liq_table, sat_tables, closure: ClosureParams,
                                   h0, V_new, alpha_g_S, h_g_S, h_l_S, dt,
                                   p_lo, p_hi, h_g_bounds, h_l_bounds,
                                   n_sub=N_MASS_TRANSFER_SUBSTEPS):
    """Sub-cycled pathline integration: N_MASS_TRANSFER_SUBSTEPS explicit
    steps of dt/n_sub each, RE-EVALUATING source_rates from the evolving
    (p, alpha_g, h_g, h_l) at every substep, instead of one predictor +
    n_corrector-iteration-averaged single-shot step over the full dt.

    Why this exists: confirmed directly (this session, porting this
    closure into the steady 2D MOC nozzle design tool) that a single dt
    step can be enormously stiff -- one measured case wanted to transfer
    ~700x the actual available donor-phase mass in one step. The old
    predictor/corrector scheme still evaluates the rate at only two
    points (start and one corrector pass) and applies each with the FULL
    dt, so it still jumps straight to the donor-mass cap (full local
    depletion) every time, regardless of corrector count. Sub-cycling
    with a SMALL dt/n_sub and re-evaluating the rate each substep lets
    the driving force (superheat, i.e. T_l - T_sat) relax toward
    saturation as evaporation proceeds -- sigma_Mg naturally shrinks
    substep over substep as h_l cools toward h_sat_l, instead of being
    held at its (large, initial-superheat) value for the whole step. This
    is the standard fix for a stiff source term that an explicit,
    single-evaluation-per-step scheme cannot resolve (matches how
    nozzle_hyperbolic's two-fluid FVM solver instead uses full implicit
    time integration for the same physical stiffness -- sub-cycling is
    the explicit-scheme equivalent, cheaper to retrofit here without
    restructuring this module's whole predictor/corrector architecture).
    """
    dt_sub = dt / n_sub

    def substep(_, carry):
        alpha_g, h_g, h_l, p = carry
        sigma_Mg, Q_g, Q_l, T_sat, h_sat_g, h_sat_l, rho_gS, rho_lS = source_rates(
            sat_tables, fb_g, liq_table, closure, p, alpha_g, h_g, h_l)
        h_ex = jnp.where(sigma_Mg > 0.0, h_sat_l, h_sat_g)
        alpha_g_new, h_g_new, h_l_new = _pathline_step(
            alpha_g, h_g, h_l, rho_gS, rho_lS, sigma_Mg, Q_g, Q_l, h_ex, dt_sub,
            h_g_bounds, h_l_bounds)
        p_new = solve_p_from_energy(fb_g, liq_table, h0, V_new, alpha_g_new, h_g_new, h_l_new, p_lo, p_hi)
        return (alpha_g_new, h_g_new, h_l_new, p_new)

    p0 = solve_p_from_energy(fb_g, liq_table, h0, V_new, alpha_g_S, h_g_S, h_l_S, p_lo, p_hi)
    alpha_g_f, h_g_f, h_l_f, p_f = jax.lax.fori_loop(
        0, n_sub, substep, (alpha_g_S, h_g_S, h_l_S, p0))
    return alpha_g_f, h_g_f, h_l_f, p_f


def _segment_intersection(x1, y1, x2, y2, xr, yr, dx, dy):
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    denom_safe = jnp.where(jnp.abs(denom) < 1e-14, 1.0, denom)
    t_raw = ((xr - x1) * dy - (yr - y1) * dx) / denom_safe
    t = jnp.where(jnp.abs(denom) < 1e-14, 0.5, jnp.clip(t_raw, 0.0, 1.0))
    xi = x1 + t * ex
    yi = y1 + t * ey
    return xi, yi, t


@eqx.filter_jit
def interior_point_step(fb_g, liq_table, sat_tables, closure: ClosureParams, delta, h0,
                        P: RayState, M: RayState, n_corrector: int = 2):
    """C+ foot P (lower), C- foot M (upper) -- see nonequilibrium_5eq.py's
    module docstring for the convention (confirmed by debugging: swapping
    these sends the characteristics backward)."""
    rho_mix_P, a_P, rho_gP, a_gP, T_gP, rho_lP, a_lP, T_lP = mixture_props(
        fb_g, liq_table, P.p, P.alpha_g, P.h_g, P.h_l)
    rho_mix_M, a_M, rho_gM, a_gM, T_gM, rho_lM, a_lM, T_lM = mixture_props(
        fb_g, liq_table, M.p, M.alpha_g, M.h_g, M.h_l)

    thetaP, muP, VP = _theta_mu(P.u, P.v, a_P)
    thetaM, muM, VM = _theta_mu(M.u, M.v, a_M)
    lamP = jnp.tan(thetaP + muP)
    lamM = jnp.tan(thetaM - muM)
    QP, R1P, SP = _QRS(P.u, P.v, a_P, P.y, delta, lamP)
    QM, R1M, SM = _QRS(M.u, M.v, a_M, M.y, delta, lamM)

    x, y, u, v = _geometry_velocity_corrector(
        P.x, P.y, thetaP, muP, a_P, QP, R1P, SP, P.u, P.v,
        M.x, M.y, thetaM, muM, a_M, QM, R1M, SM, M.u, M.v,
    )

    theta_avg = 0.5 * (thetaP + thetaM)
    dx_dir, dy_dir = jnp.cos(theta_avg), jnp.sin(theta_avg)
    xs, ys, t = _segment_intersection(M.x, M.y, P.x, P.y, x, y, -dx_dir, -dy_dir)
    p_S = M.p + t * (P.p - M.p)
    alpha_g_S = M.alpha_g + t * (P.alpha_g - M.alpha_g)
    h_g_S = M.h_g + t * (P.h_g - M.h_g)
    h_l_S = M.h_l + t * (P.h_l - M.h_l)

    ds = jnp.hypot(x - xs, y - ys)
    V_new_guess = jnp.hypot(u, v)
    V_avg = 0.5 * (0.5 * (VM + VP) + V_new_guess)
    dt = ds / jnp.maximum(V_avg, 1e-6)

    h_g_bounds = (fb_g.h_min, fb_g.h_max)
    h_l_bounds = (liq_table.h_grid[0], liq_table.h_grid[-1])
    p_lo = 0.2 * p_S
    p_hi = 1.05 * jnp.maximum(M.p, P.p)

    alpha_g_new, h_g_new, h_l_new, p_new = _pathline_integrate_substepped(
        fb_g, liq_table, sat_tables, closure, h0, V_new_guess, alpha_g_S, h_g_S, h_l_S, dt,
        p_lo, p_hi, h_g_bounds, h_l_bounds)

    return RayState(x=x, y=y, u=u, v=v, p=p_new, alpha_g=alpha_g_new, h_g=h_g_new, h_l=h_l_new)


@eqx.filter_jit
def wall_point_step(fb_g, liq_table, sat_tables, closure: ClosureParams, delta, h0,
                    interior_below: RayState, prev_wall: RayState, theta_wall_new, n_corrector: int = 2):
    rho_mix_i, a_i, *_ = mixture_props(fb_g, liq_table, interior_below.p, interior_below.alpha_g,
                                       interior_below.h_g, interior_below.h_l)
    rho_mix_w, a_w, rho_gw, a_gw, T_gw, rho_lw, a_lw, T_lw = mixture_props(
        fb_g, liq_table, prev_wall.p, prev_wall.alpha_g, prev_wall.h_g, prev_wall.h_l)

    theta_i, mu_i, V_i = _theta_mu(interior_below.u, interior_below.v, a_i)
    theta_w0, mu_w0, V_w0 = _theta_mu(prev_wall.u, prev_wall.v, a_w)
    # C+ (lamP = tan(theta+mu)), NOT C- : an UPPER wall reached from an
    # interior point BELOW it must be connected by the characteristic with
    # the LARGE POSITIVE slope (C+), matching moc.core.classes.Inv_Wall_pnt
    # (which uses p2.lamP for exactly this construction). Using C- (large
    # NEGATIVE slope near M~1) was confirmed (this session, direct
    # numerical debugging) to send the wall geometrically BACKWARD in x --
    # this was the actual root cause of the wall-freeze/drift bug, not the
    # narrow-fan conditioning (that made the symptom easier to see, but
    # this sign/family choice was the underlying error).
    lamP = jnp.tan(theta_i + mu_i)
    QP, R1P, SP = _QRS(interior_below.u, interior_below.v, a_i, interior_below.y, delta, lamP)

    # Geometry: cot (inverse-slope) intersection, not tan -- same fix as
    # _geometry_velocity_corrector, for the same reason (near-sonic wall
    # marching keeps mu close to 90deg, where tan(theta+mu) is already
    # imprecise before any further use).
    cotP = _cot(theta_i + mu_i)
    cot_wall = _cot(0.5 * (theta_w0 + theta_wall_new))
    x, y = _intersect_cot(interior_below.x, interior_below.y, cotP, prev_wall.x, prev_wall.y, cot_wall)

    TP = SP * (x - interior_below.x) + QP * interior_below.u + R1P * interior_below.v
    denom_V = QP * jnp.cos(theta_wall_new) + R1P * jnp.sin(theta_wall_new)
    V_new = TP / denom_V
    u = V_new * jnp.cos(theta_wall_new)
    v = V_new * jnp.sin(theta_wall_new)

    ds = jnp.hypot(x - prev_wall.x, y - prev_wall.y)
    V_avg = 0.5 * (V_w0 + V_new)
    dt = ds / jnp.maximum(V_avg, 1e-6)

    alpha_g_0, h_g_0, h_l_0 = prev_wall.alpha_g, prev_wall.h_g, prev_wall.h_l
    h_g_bounds = (fb_g.h_min, fb_g.h_max)
    h_l_bounds = (liq_table.h_grid[0], liq_table.h_grid[-1])
    p_lo = 0.2 * prev_wall.p
    p_hi = 1.05 * prev_wall.p

    alpha_g_new, h_g_new, h_l_new, p_new = _pathline_integrate_substepped(
        fb_g, liq_table, sat_tables, closure, h0, V_new, alpha_g_0, h_g_0, h_l_0, dt,
        p_lo, p_hi, h_g_bounds, h_l_bounds)

    return RayState(x=x, y=y, u=u, v=v, p=p_new, alpha_g=alpha_g_new, h_g=h_g_new, h_l=h_l_new)
