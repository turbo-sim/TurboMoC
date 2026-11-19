"""Unique incidence of a started supersonic rotor (Starken / Bufi, as ``core.rotor_starting.unique_incidence``) in JAX.

    rho_1 W_1 cos(beta_1) s = rho_E W_E (s cos(beta_s) - r),     beta_1 - beta_s = nu(V_E) - nu(V_1)

solved for the entrance velocity V_E. The reference evaluates nu with a 400-point trapezoid per call on its own
CoolProp isentrope; here nu, rho, a come from the ``JaxIsentrope`` table (same integrand sqrt(1/a^2 - 1/V^2) on
10000 nodes), so the two agree to the discretization error of the reference's integral (~1e-6 rad), not to round-off.

Root selection as the reference (the sign change closest to the metal angle) is made on concrete values by
``bracket``; ``solve_entrance`` then bisects inside that bracket with an implicit-function derivative, so the result is
differentiable in the metal angle, the spacing, the edge radius and the inflow Mach number.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _angle(iso, metal, nu1, v):
    return metal + iso.nu_of_V(v) - nu1


def _residual(iso, metal, spacing, radius, mach, v):
    v1 = iso.V_of_M(mach)
    nu1, flux1 = iso.nu_of_V(v1), iso.rho_of_V(v1) * v1
    width = spacing * jnp.cos(metal) - radius
    return flux1 * jnp.cos(_angle(iso, metal, nu1, v)) * spacing - iso.rho_of_V(v) * v * width


def bracket(iso, metal, spacing, radius, mach, v_star):
    """Concrete pre-pass: the reference's 40-point grid and its root choice; returns (lo, hi) around that root."""
    v_top = float(iso.V_of_M(jnp.minimum(2.5 * mach, float(iso.mach_of_V(iso.V_max)) - 1e-9)))
    grid = np.linspace(1.0001 * v_star, v_top, 40)
    values = np.array([float(_residual(iso, metal, spacing, radius, mach, v)) for v in grid])
    pairs = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0]
    if float(mach) * np.cos(float(metal)) >= 1.0 or not pairs:
        raise ValueError("No unique incidence: the axial inflow is supersonic or the passage cannot pass the flow.")
    nu1 = float(iso.nu_of_V(iso.V_of_M(mach)))
    roots = [_bisect(iso, metal, spacing, radius, mach, lo, hi) for lo, hi in pairs]
    k = int(np.argmin([abs(float(_angle(iso, metal, nu1, r)) - float(metal)) for r in roots]))
    return pairs[k]


def _bisect(iso, metal, spacing, radius, mach, lo, hi, steps=100):
    f_lo = _residual(iso, metal, spacing, radius, mach, lo)

    def body(_, b):
        a, c, fa = b
        m = 0.5 * (a + c)
        fm = _residual(iso, metal, spacing, radius, mach, m)
        same = jnp.sign(fm) == jnp.sign(fa)
        return jnp.where(same, m, a), jnp.where(same, c, m), jnp.where(same, fm, fa)

    a, c, _ = jax.lax.fori_loop(0, steps, body, (jnp.float64(lo), jnp.float64(hi), f_lo))
    return 0.5 * (a + c)


def solve_entrance(iso, metal, spacing, radius, mach, lo, hi):
    """V_E in [lo, hi], differentiable in (metal, spacing, radius, mach) by the implicit function theorem."""
    return _solve(iso, lo, hi, metal, spacing, radius, mach)


@jax.custom_jvp
def _solve(iso, lo, hi, metal, spacing, radius, mach):
    return _bisect(iso, metal, spacing, radius, mach, lo, hi)


@_solve.defjvp
def _solve_jvp(primals, tangents):
    iso, lo, hi, metal, spacing, radius, mach = primals
    _, _, _, dmetal, dspacing, dradius, dmach = tangents
    v = _bisect(iso, metal, spacing, radius, mach, lo, hi)
    f = lambda vv, m, s, r, M: _residual(iso, m, s, r, M, vv)
    dfdv = jax.grad(f, 0)(v, metal, spacing, radius, mach)
    _, df = jax.jvp(lambda m, s, r, M: f(v, m, s, r, M), (metal, spacing, radius, mach),
                    (dmetal, dspacing, dradius, dmach))
    return v, -df / dfdv


def unique_incidence(iso, metal_deg, spacing, radius, mach, v_star, lo=None, hi=None):
    """Inflow angle [deg] the started passage imposes; returns (beta_inlet, incidence, M_E). Bracket from ``bracket``
    on concrete inputs when not given (pass it explicitly inside traced code)."""
    metal = jnp.radians(metal_deg)
    if lo is None:
        lo, hi = bracket(iso, metal, spacing, radius, mach, v_star)
    v_e = solve_entrance(iso, metal, spacing, radius, mach, lo, hi)
    nu1 = iso.nu_of_V(iso.V_of_M(mach))
    angle = _angle(iso, metal, nu1, v_e)
    return jnp.degrees(angle), jnp.degrees(angle - metal), v_e / iso.a_of_V(v_e)
