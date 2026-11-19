"""Rotor at unique incidence: the blade metal angle for which the unique-incidence model returns the inflow angle.

As ``TurboExpo2027/meanline/stage_model._sharp_incidence``: the blade is built at ``beta_rel - incidence``; the model
uses the blade spacing (vortex pitch + translate) and an entrance blockage ``blockage * spacing`` (the calibrated
sharp-edge value 0.0056 of the CFD rotors) or the blade's own LE radius (``blockage=None``); the incidence solves

    g(incidence; p) = unique_incidence(blade(beta_rel - incidence; p)) - beta_rel = 0,

then the measured model miss (meanline ``incidence_miss``) is added. ``make_unique_design`` returns f(p) -> (blade,
incidence) differentiable in every continuous design input, with d(incidence)/dp from the implicit function theorem
(-g_p / g_incidence), so sensitivities include the re-staggering that keeps the rotor at unique incidence.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import incidence as ui
from . import vortex_blade as vb


def incidence_miss(mach):
    """Measured miss of the unique-incidence model vs M_rel (design_evolution s.32, s.34)."""
    return jnp.clip(1.4 * (1.49 - mach) / (1.49 - 1.30), 0.0, 1.4)


def _model_angle(iso, blade, p, mach, blockage, bracket):
    spacing = blade["pitch"]  # vortex pitch + translate (the reference's spacing)
    radius = blockage * spacing if blockage is not None else blade["blade"]["le_radius"]
    return ui.unique_incidence(iso, p["beta_inlet"], spacing, radius, mach, None, *bracket)[0]


def make_unique_design(iso, inputs, beta_rel, mach_rel, blockage=0.0056, miss=True, x0=(0.5, 2.5)):
    """inputs: vortex-blade inputs WITHOUT beta_inlet (M_inlet = mach_rel). Returns (f, p0, info).

    f(p) -> {"blade", "incidence", "model_incidence"}; p holds the continuous blade inputs plus "beta_rel" and
    "M_rel" (the inflow), so derivatives in the inflow angle and Mach number are available too.

    LOCAL: the incidence is solved once, at p0; ``f`` returns that value with the exact implicit DERIVATIVE at p0.
    Use f only at p0 (jax.grad / jax.jacfwd / jax.jvp there); for another design call make_unique_design again.
    """
    base = dict(inputs, M_inlet=mach_rel)

    def concrete_residual(inc):
        blade = vb.design_rotor_vortex_blade(iso, **base, beta_inlet=beta_rel - inc)
        spacing = float(blade["pitch"])
        radius = blockage * spacing if blockage is not None else float(blade["blade"]["le_radius"])
        beta = float(ui.unique_incidence(iso, beta_rel - inc, spacing, radius, mach_rel, float(iso.V_min))[0])
        return beta - beta_rel

    # Secant on concrete values, as the reference (one step suffices for sharp blades)
    a, b = x0
    ra, rb = concrete_residual(a), concrete_residual(b)
    for _ in range(12):
        a, b, ra = b, b - rb * (b - a) / (rb - ra), rb
        rb = concrete_residual(b)
        if abs(rb) < 1e-10:
            break
    inc0 = b
    # Static layout and unique-incidence bracket frozen at the solution
    f_blade, p_blade, s = vb.make_design_function(iso, dict(base, beta_inlet=beta_rel - inc0))
    blade0 = f_blade(p_blade)
    spacing0 = float(blade0["pitch"])
    radius0 = blockage * spacing0 if blockage is not None else float(blade0["blade"]["le_radius"])
    bracket = ui.bracket(iso, jnp.radians(beta_rel - inc0), spacing0, radius0, mach_rel, float(iso.V_min))

    def blade_at(p, inc):
        q = {k: v for k, v in p.items() if k not in ("beta_rel", "M_rel")}
        q.update(beta_inlet=p["beta_rel"] - inc, M_inlet=p["M_rel"])
        return vb.design_core(iso, q, s), q

    def g(inc, p):
        blade, q = blade_at(p, inc)
        return _model_angle(iso, blade, q, p["M_rel"], blockage, bracket) - p["beta_rel"]

    @jax.custom_jvp
    def solve(p):
        return jnp.float64(inc0) + 0.0 * p["M_rel"]

    @solve.defjvp
    def solve_jvp(primals, tangents):
        (p,), (dp,) = primals, tangents
        inc = solve(p)
        dg_dinc = jax.grad(g, 0)(inc, p)
        _, dg = jax.jvp(lambda q: g(inc, q), (p,), (dp,))
        return inc, -dg / dg_dinc

    def f(p):
        inc = solve(p)
        blade, _ = blade_at(p, inc)
        total = inc + (incidence_miss(p["M_rel"]) if miss else 0.0)
        if miss:  # the miss re-staggers the blade the same way (meanline: the blade is built at the total incidence)
            blade, _ = blade_at(p, total)
        return {"blade": blade, "incidence": total, "model_incidence": inc}

    p0 = dict(p_blade, beta_rel=jnp.float64(beta_rel), M_rel=jnp.float64(mach_rel))
    p0.pop("beta_inlet", None)
    p0.pop("M_inlet", None)  # driven by M_rel
    return f, p0, {"structure": s, "bracket": bracket, "residual": rb}
