"""Steady, spatial 2D (x, y) Method-of-Characteristics unit process for the
single-velocity ("zero-slip") 5-equation nonequilibrium two-phase model.

Physics/closure provenance
---------------------------
This module does NOT re-derive the two-phase physics. It reuses, unmodified,
the exact closure functions already validated in the `moc_unsteady` package
(the same TA-BD interfacial-area + Ranz-Marshall/energy-jump mass-transfer
model used in the NICFD 2026 preliminary saddle-point study, see
`moc_unsteady/src/moc_unsteady/two_phase_closure.py` and `nonequilibrium.py`,
and derived from Staedtke Ch. 4.47-4.52 in
`moc_unsteady/docs/5eqn_ph_derivation.md`):

  - `wallis_sound_speed`, `mixture_density`  (nonequilibrium.py)
  - `five_eq_source_rates` -> (sigma_Mg, Q_g, Q_l, a_int, d)  (nonequilibrium.py)
  - `SaturationCurve` (fast jnp.interp saturation-property lookup)

State per point: (x, y, u, v, p, alpha_g, h_g, h_l) -- phase ENTHALPIES
(not entropies), matching moc_unsteady's own convention (direct HmassP-style
property lookups, no reverse Newton solve inside the hot loop).

Derivation: unsteady (x,t) -> steady (x,y)
-------------------------------------------
`5eqn_ph_derivation.md` Sec. 4-5 derives, for the QUASI-1D UNSTEADY duct
system, three real characteristic families with eigenvalues
lambda = u, u, u, u +- a_mix (a_mix the Wallis frozen mixture sound speed,
Sec. 4.1), i.e. two acoustic characteristics dx/dt = u +- a_mix plus a
repeated (multiplicity-3) pathline root dx/dt = u carrying (alpha_g, h_g,
h_l). That eigenstructure is the SAME one classical gasdynamics MOC theory
already generalizes from unsteady 1D to steady 2D (x, y): the acoustic pair
becomes the C+/C- Mach lines dy/dx = tan(theta +- mu), mu = asin(a_mix/V),
and the repeated pathline root becomes the streamline dy/dx = tan(theta).

Two simplifications this steady-2D reduction gets "for free" relative to
the unsteady 1D solver in `fast_5eqn_1d.py`:

1. No explicit duct-area source term is needed in the acoustic relations.
   In the unsteady duct model, d(ln A)/dx has to be injected by hand into
   the mass equations (Sec. 6) because a quasi-1D duct cannot otherwise
   "see" cross-sectional spreading. In genuine 2D flow, that same physics
   (streamtube divergence) is automatically carried by how theta and V vary
   from point to point across the characteristics net -- exactly the same
   reason existing single-phase planar MOC (`moc.core.classes`) needs no
   area term at all, and its axisymmetric variant needs only the algebraic
   `S = delta*a^2*v/y` correction, not a differential ODE state. The
   acoustic velocity solve below reuses `moc.core.classes.
   jitted_moc_predictor_algebra` UNCHANGED -- that function already IS the
   general (non-barotropic, EOS-agnostic) 2D compatibility algebra; the
   only change from its single-phase use is what "a" gets fed into it
   (a_mix from the Wallis formula instead of a single-phase real-gas a(rho,s)).

2. Pressure is not solved as an independent unknown from the acoustic pair.
   `fast_5eqn_1d.py` needs the additional "D_p^mt" mass-transfer pressure
   source (Sec. 7.1) because its (p, u) pair is solved directly from the
   C+/C- relations with NO separate energy closure. Here, because the
   mixture is adiabatic and frictionless (Q_g+Q_l=0 external heat, per
   Staedtke Sec. 2 -- interfacial exchange is internal to the mixture),
   mixture stagnation enthalpy h0 = h_mix(p, alpha_g, h_g, h_l) + V^2/2 is
   EXACTLY conserved along every streamline, same as single-phase flow.
   That means once (alpha_g, h_g, h_l) are known at a point (from the
   streamline/pathline update below) and V is known (from the acoustic
   solve), p follows directly from a 1D root-find on h0 -- no separate
   mass-transfer pressure-source bookkeeping is needed at all.

Pathline (alpha_g, h_g, h_l) update
------------------------------------
Rather than porting `fast_5eqn_1d.py`'s beta/gamma/Den-transformed pathline
equations (engineered specifically to keep ITS (p, u) acoustic solve
consistent when p is a coupled unknown -- not needed here, see point 2
above), this module integrates the RAW physical rate equations directly
(Staedtke Eqs. 4.47/4.50/4.51, steady form, along path length ds with
V = ds/dt):

    d(alpha_g)/ds ~= sigma_Mg / (V * rho_g)         [mass, leading order]
    d(h_g)/ds = [sigma_Mg*(h_ex - h_g) + Q_g] / (V * alpha_g * rho_g)
    d(h_l)/ds = [-sigma_Mg*(h_ex - h_l) + Q_l] / (V * alpha_l * rho_l)

with h_ex = h_sat_l when evaporating (sigma_Mg > 0, mass leaves the liquid
at its own saturated enthalpy) else h_sat_g -- the same donor-phase
convention `five_eq_source_rates`/`mass_transfer_rate` already assume.
This is a first-order (in step size) update; a corrector pass (re-evaluate
the source rates at the predicted new-point state and average) is applied
by default, matching the predictor-corrector spirit already used
throughout `moc.core.classes`.

What this module does NOT do
------------------------------
- No throat/sonic-point treatment: like the brief given, this starts
  marching from an already-supersonic initial data line supplied by the
  caller (e.g. exported from a converged `fast_5eqn_1d.py` quasi-1D run,
  or a uniform flat front), exactly mirroring how single-phase/HEM MOC's
  Stage 1 throat construction is a separate, untouched piece of code.
- The existing single-phase/HEM solver files (`moc.core.classes`,
  `moc.solvers.conventional`, `moc.app`) are only ever IMPORTED FROM here,
  never modified.
- Not executed/tested in this environment: this sandbox has neither
  `moc`/`moc_unsteady` installed as packages nor `jax`/`jaxprop` available,
  so nothing in this module has been run. `sys.path` shims below make it
  importable from a normal dev checkout (same directory layout as this
  repo), but correctness must be checked by the caller's own environment,
  starting with the quasi-1D validation described in the module docstring
  of `moc/examples/validate_nonequilibrium_5eq_1d.py`.
"""
from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make `moc_unsteady` importable from a sibling checkout without requiring
# either project to be pip-installed (neither has an egg-link/editable
# install in this environment -- see this module's docstring). This mirrors
# the directory layout every other file in `method_of_characteristics/`
# already assumes (moc/, moc_unsteady/, space_marching/ as siblings).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOC_UNSTEADY_SRC = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "moc_unsteady", "src")
)
if os.path.isdir(_MOC_UNSTEADY_SRC) and _MOC_UNSTEADY_SRC not in sys.path:
    sys.path.insert(0, _MOC_UNSTEADY_SRC)

import jaxprop as jxp  # noqa: E402

from moc_unsteady.eos import Eos  # noqa: E402
from moc_unsteady.nonequilibrium import (  # noqa: E402
    SaturationCurve,
    wallis_sound_speed,
    mixture_density,
    five_eq_source_rates,
)

from moc.core.classes import jitted_moc_predictor_algebra, Intersection  # noqa: E402


def _f(x) -> float:
    """Scalar float out of a possibly-jnp 0-d array."""
    return float(np.asarray(x))


# ---------------------------------------------------------------------------
# Metastable per-phase (rho, a, T, k, cp, mu) at (p, h) -- same technique
# and retry ladder as space_marching/functions.py's `_metastable_state`
# (jaxprop.Fluid.get_state_metastable, a direct Helmholtz-energy (rho, T)
# solve, NOT CoolProp's own (p, h) equilibrium flash, which silently
# collapses a metastable single-phase query inside the local dome to a
# two-phase MIXTURE state -- see fast_5eqn_1d.py's module docstring for the
# same failure mode diagnosed independently there).
# ---------------------------------------------------------------------------
def _metastable_phase_state(fluid_obj, p: float, h: float, T_hint: float, rho_hint: float) -> dict:
    attempts = [
        dict(solver_algorithm="lm", rhoT_guess=[rho_hint, T_hint]),
        dict(solver_algorithm="hybr", rhoT_guess=[rho_hint, T_hint]),
        dict(solver_algorithm="lm", rhoT_guess=[rho_hint * 1.02, T_hint * 1.001]),
    ]
    st = None
    last_exc = None
    for kwargs in attempts:
        try:
            st = fluid_obj.get_state_metastable(
                prop_1="p", prop_1_value=float(p), prop_2="h", prop_2_value=float(h), **kwargs
            )
            break
        except Exception as exc:  # noqa: BLE001 -- see space_marching's identical choice
            last_exc = exc
    if st is None:
        raise RuntimeError(
            f"get_state_metastable did not converge at p={p:.4g} Pa, h={h:.4g} J/kg "
            f"after {len(attempts)} attempts."
        ) from last_exc
    return dict(
        rho=float(st.rho), a=float(st.speed_sound), T=float(st.T),
        k=float(st.conductivity), cp=float(st.isobaric_heat_capacity), mu=float(st.viscosity),
    )


# ---------------------------------------------------------------------------
# Fluid/closure manager -- analogous role to classes.py's FluidManager, but
# for the two-phase (p, alpha_g, h_g, h_l) state instead of a barotropic
# single-phase (V, s0) state.
# ---------------------------------------------------------------------------
@dataclass
class FiveEqFluidManager:
    fluid_name: str
    backend: str = "HEOS"
    Nb: float = 1e10
    Nd: float = 1e10
    alpha_b: float = 0.3
    alpha_d: float = 0.7
    Nu_l: float = 6.0
    mass_transfer: bool = True
    sat_n: int = 400

    def __post_init__(self):
        self.fluid = jxp.Fluid(self.fluid_name, backend=self.backend)
        self.eos = Eos(self.fluid_name, backend=self.backend)
        self._sat_curve: Optional[SaturationCurve] = None

    # -- setup: build the saturation-property table over the pressure range
    # this run will actually visit (call once, after the caller knows the
    # rough inlet/exit pressure bracket -- same "build once" split every
    # other table in this codebase uses). --
    def build_saturation_curve(self, p_min: float, p_max: float):
        self._sat_curve = SaturationCurve(self.eos, p_min=p_min, p_max=p_max, n=self.sat_n)

    def _sat(self, p: float):
        if self._sat_curve is None:
            raise RuntimeError("Call build_saturation_curve(p_min, p_max) before using the manager.")
        liq, vap = self._sat_curve(p)
        return liq, vap

    def phase_state(self, p: float, h: float, is_gas: bool, T_hint: float, rho_hint: float) -> dict:
        return _metastable_phase_state(self.fluid, p, h, T_hint, rho_hint)

    def mixture_props(self, p: float, alpha_g: float, h_g: float, h_l: float,
                      T_hint_g: float, rho_hint_g: float,
                      T_hint_l: float, rho_hint_l: float) -> dict:
        """Full local mixture state at (p, alpha_g, h_g, h_l): both phases'
        (rho, a, T, k, cp, mu), the mixture density, and the Wallis frozen
        mixture sound speed a_mix (the MOC characteristic speed)."""
        pg = self.phase_state(p, h_g, True, T_hint_g, rho_hint_g)
        pl = self.phase_state(p, h_l, False, T_hint_l, rho_hint_l)
        alpha_l = 1.0 - alpha_g
        rho_mix = _f(mixture_density(alpha_g, pg["rho"], pl["rho"]))
        a_mix = _f(wallis_sound_speed(alpha_g, pg["rho"], pg["a"], pl["rho"], pl["a"]))
        return dict(rho_mix=rho_mix, a_mix=a_mix,
                   rho_g=pg["rho"], a_g=pg["a"], T_g=pg["T"], k_g=pg["k"], cp_g=pg["cp"], mu_g=pg["mu"],
                   rho_l=pl["rho"], a_l=pl["a"], T_l=pl["T"], k_l=pl["k"], cp_l=pl["cp"], mu_l=pl["mu"])

    def h_mix(self, mix: dict, alpha_g: float, h_g: float, h_l: float) -> float:
        alpha_l = 1.0 - alpha_g
        num = alpha_g * mix["rho_g"] * h_g + alpha_l * mix["rho_l"] * h_l
        return num / mix["rho_mix"]

    def source_rates(self, p: float, alpha_g: float, mix: dict):
        """TA-BD + Ranz-Marshall/energy-jump closure (moc_unsteady.
        nonequilibrium.five_eq_source_rates, unmodified) at the given
        local state. Returns (sigma_Mg, Q_g, Q_l, T_sat, h_sat_g, h_sat_l)."""
        liq, vap = self._sat(p)
        T_sat = _f(liq.T)
        h_sat_g, h_sat_l = _f(vap.h), _f(liq.h)
        rates = five_eq_source_rates(
            alpha_g, mix["T_g"], mix["T_l"], T_sat, h_sat_g, h_sat_l,
            mix["k_g"], mix["k_l"], mu_g=mix["mu_g"], rho_g=mix["rho_g"], cp_g=mix["cp_g"],
            mass_transfer=self.mass_transfer,
            alpha_b=self.alpha_b, alpha_d=self.alpha_d, Nb=self.Nb, Nd=self.Nd, Nu_l=self.Nu_l,
        )
        return _f(rates.sigma_Mg), _f(rates.Q_g), _f(rates.Q_l), T_sat, h_sat_g, h_sat_l

    def solve_p_from_energy(self, h0: float, V: float, alpha_g: float, h_g: float, h_l: float,
                            p_lo: float, p_hi: float,
                            T_hint_g: float, rho_hint_g: float, T_hint_l: float, rho_hint_l: float,
                            tol: float = 1e-6, max_iter: int = 40) -> float:
        """Bisection on p so that h_mix(p, alpha_g, h_g, h_l) + V^2/2 = h0.
        h_mix decreases with p at fixed (alpha_g, h_g, h_l) along the same
        physical branch (higher p -> higher phase density -> for fixed
        alpha_g the gas-mass fraction drops -- monotonic in the regimes
        this model targets), mirroring `classes.jax_solve_critical_state`'s
        bracketed root-find pattern for the single-phase barotropic EOS."""
        target = h0 - 0.5 * V * V

        def resid(p):
            mix = self.mixture_props(p, alpha_g, h_g, h_l, T_hint_g, rho_hint_g, T_hint_l, rho_hint_l)
            return self.h_mix(mix, alpha_g, h_g, h_l) - target

        lo, hi = p_lo, p_hi
        f_lo, f_hi = resid(lo), resid(hi)
        if f_lo * f_hi > 0.0:
            # Fall back to whichever bound is closer instead of raising --
            # a genuinely supersonic/near-choked point can legitimately sit
            # just outside a loosely-guessed bracket; the caller widens the
            # bracket on the next predictor-corrector pass.
            return lo if abs(f_lo) < abs(f_hi) else hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = resid(mid)
            if abs(f_mid) < tol * abs(target):
                return mid
            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# MOC point
# ---------------------------------------------------------------------------
@dataclass
class FiveEqPoint:
    manager: FiveEqFluidManager
    x: float = 0.0
    y: float = 0.0
    u: float = 0.0
    v: float = 0.0
    p: float = 0.0
    alpha_g: float = 0.0
    h_g: float = 0.0
    h_l: float = 0.0
    delta: float = 0.0  # 0 planar, 1 axisymmetric (geometric S-source only)

    # derived, filled by refresh_thermo()
    V: float = field(default=0.0, init=False)
    theta: float = field(default=0.0, init=False)
    mix: dict = field(default_factory=dict, init=False)
    a_mix: float = field(default=0.0, init=False)
    M: float = field(default=0.0, init=False)
    mu: float = field(default=0.0, init=False)  # Mach angle
    lamP: float = field(default=0.0, init=False)
    lamM: float = field(default=0.0, init=False)

    def refresh_thermo(self, T_hint_g: float = 350.0, rho_hint_g: float = 5.0,
                       T_hint_l: float = 340.0, rho_hint_l: float = 600.0):
        self.V = math.hypot(self.u, self.v)
        self.theta = math.atan2(self.v, self.u)
        self.mix = self.manager.mixture_props(
            self.p, self.alpha_g, self.h_g, self.h_l, T_hint_g, rho_hint_g, T_hint_l, rho_hint_l
        )
        self.a_mix = self.mix["a_mix"]
        self.M = self.V / self.a_mix if self.a_mix > 0 else float("nan")
        self.mu = math.asin(min(max(self.a_mix / self.V, -1.0), 1.0)) if self.V > 0 else 0.0
        self.lamP = math.tan(self.theta + self.mu)
        self.lamM = math.tan(self.theta - self.mu)
        return self

    def _QRS(self, other_side_sign: int):
        """Q, R1, S for THIS point acting as a foot on the C+ (sign=+1) or
        C- (sign=-1) characteristic, in the same convention as
        `moc.core.classes`'s interior_point_prop/jitted_moc_predictor_algebra."""
        a = self.a_mix
        lam = self.lamP if other_side_sign > 0 else self.lamM
        Q = self.u ** 2 - a ** 2
        R1 = 2.0 * self.u * self.v - Q * lam
        S = 0.0 if abs(self.y) <= 1e-9 else self.delta * (a ** 2 * self.v) / self.y
        return lam, Q, R1, S


def _segment_intersection(x1, y1, x2, y2, xr, yr, dx, dy):
    """Intersection of segment (x1,y1)-(x2,y2) with the ray/line through
    (xr,yr) with direction (dx,dy). Returns (xi, yi, t) with t the fraction
    along the segment (1->2), or None if parallel. Used to locate the
    streamline foot point by projecting the new point backward along the
    local flow direction onto the initial-data segment between the C- and
    C+ feet -- the standard construction for a third ("entropy"/pathline)
    characteristic in rotational/non-homentropic MOC (Zucrow & Hoffman,
    Vol. 2, unit processes for rotational flow)."""
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-14:
        return None
    t = ((xr - x1) * dy - (yr - y1) * dx) / denom
    xi = x1 + t * ex
    yi = y1 + t * ey
    return xi, yi, t


class FiveEqSolverPoint(FiveEqPoint):
    """FiveEqPoint plus the unit-process methods (interior, axis)."""

    def interior_point_prop(self, paramP: "FiveEqSolverPoint", paramM: "FiveEqSolverPoint",
                            n_corrector: int = 2):
        paramP.refresh_thermo()
        paramM.refresh_thermo()

        lamP, QP, R1P, SP = paramP._QRS(+1)
        lamM, QM, R1M, SM = paramM._QRS(-1)

        x, y, u, v = jitted_moc_predictor_algebra(
            xM=paramM.x, yM=paramM.y, lamM=lamM, SM=SM, QM=QM, R1M=R1M, uM=paramM.u, vM=paramM.v,
            xP=paramP.x, yP=paramP.y, lamP=lamP, SP=SP, QP=QP, R1P=R1P, uP=paramP.u, vP=paramP.v,
        )
        x, y, u, v = _f(x), _f(y), _f(u), _f(v)

        theta_avg = 0.5 * (paramP.theta + paramM.theta)
        dx_dir, dy_dir = math.cos(theta_avg), math.sin(theta_avg)
        hit = _segment_intersection(paramM.x, paramM.y, paramP.x, paramP.y, x, y, -dx_dir, -dy_dir)
        if hit is None or not (0.0 <= hit[2] <= 1.0):
            # Streamline foot falls outside the M-P segment (can happen on
            # a coarse net); fall back to the midpoint-in-position
            # interpolation as a robust default.
            t = 0.5
            xs = 0.5 * (paramM.x + paramP.x)
            ys = 0.5 * (paramM.y + paramP.y)
        else:
            xs, ys, t = hit
        p_S = paramM.p + t * (paramP.p - paramM.p)
        alpha_g_S = paramM.alpha_g + t * (paramP.alpha_g - paramM.alpha_g)
        h_g_S = paramM.h_g + t * (paramP.h_g - paramM.h_g)
        h_l_S = paramM.h_l + t * (paramP.h_l - paramM.h_l)

        ds = math.hypot(x - xs, y - ys)
        V_new_guess = math.hypot(u, v)
        V_avg = 0.5 * (0.5 * (paramM.V + paramP.V) + V_new_guess)

        mix_S = self.manager.mixture_props(
            p_S, alpha_g_S, h_g_S, h_l_S,
            paramM.mix.get("T_g", 350.0), paramM.mix.get("rho_g", 5.0),
            paramM.mix.get("T_l", 340.0), paramM.mix.get("rho_l", 600.0),
        )
        sigma_Mg, Q_g, Q_l, T_sat, h_sat_g, h_sat_l = self.manager.source_rates(p_S, alpha_g_S, mix_S)
        alpha_l_S = 1.0 - alpha_g_S
        h_ex = h_sat_l if sigma_Mg > 0.0 else h_sat_g

        def _step(dt_scale):
            dalpha_g = dt_scale * sigma_Mg / (mix_S["rho_g"])
            dh_g = dt_scale * (sigma_Mg * (h_ex - h_g_S) + Q_g) / (alpha_g_S * mix_S["rho_g"])
            dh_l = dt_scale * (-sigma_Mg * (h_ex - h_l_S) + Q_l) / (alpha_l_S * mix_S["rho_l"])
            return (min(max(alpha_g_S + dalpha_g, 1e-9), 1.0 - 1e-9),
                   h_g_S + dh_g, h_l_S + dh_l)

        dt = ds / max(V_avg, 1e-6)
        alpha_g_new, h_g_new, h_l_new = _step(dt)

        # Corrector: re-evaluate source rates at the predicted new-point
        # state and average with the foot values (trapezoidal), matching
        # the predictor-corrector convention used throughout this codebase
        # (e.g. classes.interior_point_prop's slope averaging).
        p_new = self.manager.solve_p_from_energy(
            self.h0, V_new_guess, alpha_g_new, h_g_new, h_l_new,
            p_lo=0.2 * p_S, p_hi=1.05 * max(paramM.p, paramP.p),
            T_hint_g=mix_S["T_g"], rho_hint_g=mix_S["rho_g"],
            T_hint_l=mix_S["T_l"], rho_hint_l=mix_S["rho_l"],
        )
        for _ in range(n_corrector):
            mix_new = self.manager.mixture_props(
                p_new, alpha_g_new, h_g_new, h_l_new,
                mix_S["T_g"], mix_S["rho_g"], mix_S["T_l"], mix_S["rho_l"],
            )
            sigma_Mg_n, Q_g_n, Q_l_n, T_sat_n, h_sat_g_n, h_sat_l_n = self.manager.source_rates(
                p_new, alpha_g_new, mix_new
            )
            h_ex_n = h_sat_l_n if sigma_Mg_n > 0.0 else h_sat_g_n
            sigma_Mg_avg = 0.5 * (sigma_Mg + sigma_Mg_n)
            Q_g_avg = 0.5 * (Q_g + Q_g_n)
            Q_l_avg = 0.5 * (Q_l + Q_l_n)
            h_ex_avg = 0.5 * (h_ex + h_ex_n)

            dalpha_g = dt * sigma_Mg_avg / (0.5 * (mix_S["rho_g"] + mix_new["rho_g"]))
            dh_g = dt * (sigma_Mg_avg * (h_ex_avg - h_g_S) + Q_g_avg) / (alpha_g_S * mix_S["rho_g"])
            dh_l = dt * (-sigma_Mg_avg * (h_ex_avg - h_l_S) + Q_l_avg) / (alpha_l_S * mix_S["rho_l"])
            alpha_g_new = min(max(alpha_g_S + dalpha_g, 1e-9), 1.0 - 1e-9)
            h_g_new = h_g_S + dh_g
            h_l_new = h_l_S + dh_l

            p_new = self.manager.solve_p_from_energy(
                self.h0, V_new_guess, alpha_g_new, h_g_new, h_l_new,
                p_lo=0.2 * p_S, p_hi=1.05 * max(paramM.p, paramP.p),
                T_hint_g=mix_new["T_g"], rho_hint_g=mix_new["rho_g"],
                T_hint_l=mix_new["T_l"], rho_hint_l=mix_new["rho_l"],
            )

        self.x, self.y, self.u, self.v = x, y, u, v
        self.p, self.alpha_g, self.h_g, self.h_l = p_new, alpha_g_new, h_g_new, h_l_new
        self.refresh_thermo(T_hint_g=mix_S["T_g"], rho_hint_g=mix_S["rho_g"],
                            T_hint_l=mix_S["T_l"], rho_hint_l=mix_S["rho_l"])
        return self

    def wall_point_prop(self, interior_below: "FiveEqSolverPoint", theta_wall_new: float,
                        prev_wall: "FiveEqSolverPoint", n_corrector: int = 2):
        """Wall unit process: prescribed wall flow angle `theta_wall_new`
        (the design turning schedule) as the boundary condition, closed by
        the ONE usable characteristic reaching the wall from below (C-,
        from `interior_below`) -- the standard MOC wall-point construction
        (Zucrow & Hoffman Vol. 2; Anderson, Modern Compressible Flow, wall-
        point unit process) generalized to this model's linearized (no
        closed-form invariant) compatibility algebra, same as
        `interior_point_prop`'s C+/C- pair but with only one line since
        theta itself is no longer an unknown here.

        `prev_wall`: the previous wall point (already thermo-fresh). The
        wall itself IS a streamline (no mass crosses it by definition), so
        unlike `interior_point_prop` the pathline foot for the (alpha_g,
        h_g, h_l) update is simply `prev_wall` directly -- no streamline-
        foot interpolation needed.
        """
        interior_below.refresh_thermo()
        prev_wall.refresh_thermo()

        # C+ (not C-): an UPPER wall reached from an interior point BELOW
        # it must be connected by the characteristic with the LARGE
        # POSITIVE slope, matching moc.core.classes.Inv_Wall_pnt (uses
        # p2.lamP for exactly this construction). Using C- (large
        # NEGATIVE slope near M~1) sends the wall geometrically BACKWARD
        # in x -- confirmed directly via numerical debugging of the fast
        # (jitted) counterpart of this function; this was the actual root
        # cause of an observed wall-drift/freeze bug, not a narrow-fan
        # conditioning issue as first suspected.
        lamP, QP, R1P, SP = interior_below._QRS(+1)

        # Wall tangent line from the previous wall point, slope evaluated
        # at the average of the old and new wall angles (predictor value;
        # refined implicitly by the corrector loop below re-deriving V,
        # which does not change theta_wall_new itself since that is a
        # prescribed BC, not solved for).
        lam_wall = math.tan(0.5 * (prev_wall.theta + theta_wall_new))
        denom = lamP - lam_wall
        if abs(denom) < 1e-10:
            raise ValueError("wall_point_prop: wall tangent nearly parallel to C+.")
        x = (lamP * interior_below.x - lam_wall * prev_wall.x + prev_wall.y - interior_below.y) / denom
        y = interior_below.y + lamP * (x - interior_below.x)

        # Single compatibility equation (C+ only) with theta FIXED at
        # theta_wall_new: Q*u + R1*v = T, u=V*cos(theta), v=V*sin(theta)
        # => V*(Q*cos(theta)+R1*sin(theta)) = T.
        TP = SP * (x - interior_below.x) + QP * interior_below.u + R1P * interior_below.v
        denom_V = QP * math.cos(theta_wall_new) + R1P * math.sin(theta_wall_new)
        if abs(denom_V) < 1e-8:
            raise ValueError("wall_point_prop: singular V-solve (QP*cos+R1P*sin ~ 0).")
        V_new = TP / denom_V
        u, v = V_new * math.cos(theta_wall_new), V_new * math.sin(theta_wall_new)

        ds = math.hypot(x - prev_wall.x, y - prev_wall.y)
        V_avg = 0.5 * (prev_wall.V + V_new)

        sigma_Mg, Q_g, Q_l, T_sat, h_sat_g, h_sat_l = self.manager.source_rates(
            prev_wall.p, prev_wall.alpha_g, prev_wall.mix
        )
        alpha_g_0, h_g_0, h_l_0 = prev_wall.alpha_g, prev_wall.h_g, prev_wall.h_l
        alpha_l_0 = 1.0 - alpha_g_0
        h_ex = h_sat_l if sigma_Mg > 0.0 else h_sat_g

        dt = ds / max(V_avg, 1e-6)
        rho_g0, rho_l0 = prev_wall.mix["rho_g"], prev_wall.mix["rho_l"]
        alpha_g_new = min(max(alpha_g_0 + dt * sigma_Mg / rho_g0, 1e-9), 1.0 - 1e-9)
        h_g_new = h_g_0 + dt * (sigma_Mg * (h_ex - h_g_0) + Q_g) / (alpha_g_0 * rho_g0)
        h_l_new = h_l_0 + dt * (-sigma_Mg * (h_ex - h_l_0) + Q_l) / (alpha_l_0 * rho_l0)

        p_new = self.manager.solve_p_from_energy(
            self.h0, V_new, alpha_g_new, h_g_new, h_l_new,
            p_lo=0.2 * prev_wall.p, p_hi=1.05 * prev_wall.p,
            T_hint_g=prev_wall.mix["T_g"], rho_hint_g=rho_g0,
            T_hint_l=prev_wall.mix["T_l"], rho_hint_l=rho_l0,
        )
        for _ in range(n_corrector):
            mix_new = self.manager.mixture_props(
                p_new, alpha_g_new, h_g_new, h_l_new,
                prev_wall.mix["T_g"], rho_g0, prev_wall.mix["T_l"], rho_l0,
            )
            sigma_Mg_n, Q_g_n, Q_l_n, T_sat_n, h_sat_g_n, h_sat_l_n = self.manager.source_rates(
                p_new, alpha_g_new, mix_new
            )
            h_ex_n = h_sat_l_n if sigma_Mg_n > 0.0 else h_sat_g_n
            sigma_Mg_avg = 0.5 * (sigma_Mg + sigma_Mg_n)
            Q_g_avg = 0.5 * (Q_g + Q_g_n)
            Q_l_avg = 0.5 * (Q_l + Q_l_n)
            h_ex_avg = 0.5 * (h_ex + h_ex_n)

            alpha_g_new = min(max(alpha_g_0 + dt * sigma_Mg_avg / (0.5 * (rho_g0 + mix_new["rho_g"])),
                                  1e-9), 1.0 - 1e-9)
            h_g_new = h_g_0 + dt * (sigma_Mg_avg * (h_ex_avg - h_g_0) + Q_g_avg) / (alpha_g_0 * rho_g0)
            h_l_new = h_l_0 + dt * (-sigma_Mg_avg * (h_ex_avg - h_l_0) + Q_l_avg) / (alpha_l_0 * rho_l0)

            p_new = self.manager.solve_p_from_energy(
                self.h0, V_new, alpha_g_new, h_g_new, h_l_new,
                p_lo=0.2 * prev_wall.p, p_hi=1.05 * prev_wall.p,
                T_hint_g=mix_new["T_g"], rho_hint_g=mix_new["rho_g"],
                T_hint_l=mix_new["T_l"], rho_hint_l=mix_new["rho_l"],
            )

        self.x, self.y, self.u, self.v = x, y, u, v
        self.p, self.alpha_g, self.h_g, self.h_l = p_new, alpha_g_new, h_g_new, h_l_new
        self.refresh_thermo(T_hint_g=prev_wall.mix["T_g"], rho_hint_g=rho_g0,
                            T_hint_l=prev_wall.mix["T_l"], rho_hint_l=rho_l0)
        return self


# h0 (mixture stagnation enthalpy) is a run-wide constant, not a per-point
# attribute in the original derivation -- attached to instances via this
# tiny mixin-style default so `FiveEqSolverPoint.h0` can be set once on the
# manager-shared value without threading it through every constructor call.
FiveEqSolverPoint.h0 = 0.0
