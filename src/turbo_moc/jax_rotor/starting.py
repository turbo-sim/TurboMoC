"""Kantrowitz starting check of a JAX vortex blade (counterpart of ``turbo_moc.core.rotor_starting``).

- ``geometric_throat``: smallest distance between the closed outline and its neighbour one pitch away, with the
  reference's sampling (20 points per outline segment, every third point on the neighbour). Exact in JAX; its gradient
  is that of the closest pair (non-smooth only where the closest pair switches).
- ``kantrowitz_ratio`` / ``isentropic_ratio``: they depend on the inlet state and M_inlet only (normal shock and sonic
  flux on the real-gas isentropes, CoolProp root finding, which JAX cannot trace). Values are the reference's, called
  through ``jax.pure_callback`` (identical by construction). The derivative in M_inlet (2026-11-17) is that of a
  quintic spline through the reference values on a fixed Mach grid (step ``GRID_STEP``, ``HALF_WIDTH`` nodes on each
  side of the query, cached per inlet state): truncation ~1e-10, and the ~1e-11 root-solve noise of the reference is
  amplified only to ~1e-9 (a central difference at a small step amplified it to ~1e-4). Verified against
  Richardson-extrapolated differences of the reference (Cyclopentane, Water, CO2, Novec649; two-phase and vapour):
  <= 3e-8. Where the post-shock state crosses the dew line the ratio's curvature jumps (slope continuous; CO2 at
  0.3 pc: M 1.357) and within ~0.08 in Mach of that point the spline derivative is good to ~1e-5 only. Not valid
  across a kink of the ratio itself (inlet Mach at the dew line of a wet-to-dry expansion): the spline smooths it.
  Derivatives in the inlet state are not provided (phase 1b).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from turbo_moc.core import rotor_starting as ref

GRID_STEP = 0.01   # Mach spacing of the derivative grid
HALF_WIDTH = 8     # grid nodes on each side of the query (quintic spline through 2 * HALF_WIDTH + 2 nodes)


def geometric_throat(blade_x, blade_y, pitch):
    outline = jnp.stack([blade_x, blade_y], axis=1)
    fractions = jnp.linspace(0.0, 1.0, 20, endpoint=False)[:, None]
    a, b = outline[:-1], outline[1:]
    dense = (a[:, None, :] + fractions[None] * (b - a)[:, None, :]).reshape(-1, 2)
    neighbour = dense[::3] + jnp.array([0.0, 1.0]) * pitch

    def closest(point):
        return jnp.min(jnp.sum((dense - point) ** 2, axis=1))

    return jnp.sqrt(jnp.min(jax.lax.map(closest, neighbour, batch_size=256)))


class _GridDerivative:
    """d fn / d M from a quintic spline through fn on the fixed grid M = k * GRID_STEP (values cached by k)."""

    def __init__(self, fn):
        self.fn, self.values = fn, {}

    def _value(self, k):
        if k not in self.values:
            try:
                self.values[k] = float(self.fn(k * GRID_STEP))
            except (ValueError, RuntimeError):  # outside the isentrope or the shock solution (e.g. M < 1)
                self.values[k] = np.nan
        return self.values[k]

    def __call__(self, mach):
        from scipy.interpolate import make_interp_spline
        k0 = int(np.floor(mach / GRID_STEP))
        ks = np.arange(k0 - HALF_WIDTH, k0 + HALF_WIDTH + 2)
        vals = np.array([self._value(int(k)) for k in ks])
        ok = np.isfinite(vals)
        if not ok.all():  # near the end of the valid range: keep the finite nodes on the query's side only
            run = np.flatnonzero(ok)
            if run.size < 6 or not (ks[run[0]] * GRID_STEP <= mach <= ks[run[-1]] * GRID_STEP):
                raise ValueError(f"Mach {mach}: too close to the end of the valid range for the spline derivative")
            ks, vals = ks[run], vals[run]
        return float(make_interp_spline(ks * GRID_STEP, vals, k=5).derivative()(mach))


def _scalar_callback(fn, mach):
    return jax.pure_callback(lambda m: np.float64(fn(float(m))), jax.ShapeDtypeStruct((), jnp.float64), mach,
                             vmap_method="sequential")


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _ratio(fn, slope_fn, mach):
    return _scalar_callback(fn, mach)


@_ratio.defjvp
def _ratio_jvp(fn, slope_fn, primals, tangents):
    (mach,), (dmach,) = primals, tangents
    return _scalar_callback(fn, mach), _scalar_callback(slope_fn, mach) * dmach


class StartingModel:
    """Relative isentrope of one inlet state (reference object, built once) with JAX-callable ratios."""

    def __init__(self, fluid_name, P0_rel, T0_rel, Q0_rel=float("nan"), backend="HEOS"):
        self.isentrope = ref.relative_isentrope(fluid_name, P0_rel, T0_rel, Q0_rel, backend)
        self._kantrowitz = lambda m: ref.kantrowitz_ratio(self.isentrope, m)
        self._isentropic = lambda m: ref.isentropic_ratio(self.isentrope, m)
        self._dkantrowitz = _GridDerivative(self._kantrowitz)
        self._disentropic = _GridDerivative(self._isentropic)

    def kantrowitz_ratio(self, mach):
        return _ratio(self._kantrowitz, self._dkantrowitz, mach)

    def isentropic_ratio(self, mach):
        return _ratio(self._isentropic, self._disentropic, mach)

    def check(self, blade, M_inlet, beta_inlet):
        """Same outputs as ``rotor_starting.starting_check`` (arrays), for a JAX blade dict."""
        beta = jnp.radians(beta_inlet)
        throat_ratio = geometric_throat(blade["blade"]["x"], blade["blade"]["y"], blade["pitch"]) / (
            blade["pitch"] * jnp.cos(beta))
        kantrowitz = self.kantrowitz_ratio(M_inlet)
        return {"M_inlet": M_inlet, "beta_inlet": beta_inlet, "axial_mach": M_inlet * jnp.cos(beta),
                "throat_ratio": throat_ratio, "kantrowitz_ratio": kantrowitz,
                "isentropic_ratio": self.isentropic_ratio(M_inlet), "margin": throat_ratio / kantrowitz,
                "starts": throat_ratio >= kantrowitz,
                "leading_edge_radius_over_pitch": blade["blade"]["le_radius"] / blade["pitch"]}
