"""Kantrowitz starting check of a JAX vortex blade (counterpart of ``turbo_moc.core.rotor_starting``).

- ``geometric_throat``: smallest distance between the closed outline and its neighbour one pitch away, with the
  reference's sampling (20 points per outline segment, every third point on the neighbour). Exact in JAX; its gradient
  is that of the closest pair (non-smooth only where the closest pair switches).
- ``kantrowitz_ratio`` / ``isentropic_ratio``: they depend on the inlet state and M_inlet only (normal shock and sonic
  flux on the real-gas isentropes, CoolProp root finding). Values are the reference's, called through
  ``jax.pure_callback``; the derivative in M_inlet is a central difference of the reference (step ``FD_STEP``), i.e.
  finite-difference accurate (~1e-8 relative), not automatic differentiation. Derivatives in the inlet state are not
  provided (phase 1b).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from turbo_moc.core import rotor_starting as ref

FD_STEP = 1e-5


def geometric_throat(blade_x, blade_y, pitch):
    outline = jnp.stack([blade_x, blade_y], axis=1)
    fractions = jnp.linspace(0.0, 1.0, 20, endpoint=False)[:, None]
    a, b = outline[:-1], outline[1:]
    dense = (a[:, None, :] + fractions[None] * (b - a)[:, None, :]).reshape(-1, 2)
    neighbour = dense[::3] + jnp.array([0.0, 1.0]) * pitch

    def closest(point):
        return jnp.min(jnp.sum((dense - point) ** 2, axis=1))

    return jnp.sqrt(jnp.min(jax.lax.map(closest, neighbour, batch_size=256)))


def _scalar_callback(fn, mach):
    return jax.pure_callback(lambda m: np.float64(fn(float(m))), jax.ShapeDtypeStruct((), jnp.float64), mach)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _fd_scalar(fn, mach):
    return _scalar_callback(fn, mach)


@_fd_scalar.defjvp
def _fd_scalar_jvp(fn, primals, tangents):
    (mach,), (dmach,) = primals, tangents
    value = _scalar_callback(fn, mach)
    h = FD_STEP * jnp.maximum(1.0, jnp.abs(mach))
    slope = (_scalar_callback(fn, mach + h) - _scalar_callback(fn, mach - h)) / (2 * h)
    return value, slope * dmach


class StartingModel:
    """Relative isentrope of one inlet state (reference object, built once) with JAX-callable ratios."""

    def __init__(self, fluid_name, P0_rel, T0_rel, Q0_rel=float("nan"), backend="HEOS"):
        self.isentrope = ref.relative_isentrope(fluid_name, P0_rel, T0_rel, Q0_rel, backend)
        self._kantrowitz = lambda m: ref.kantrowitz_ratio(self.isentrope, m)
        self._isentropic = lambda m: ref.isentropic_ratio(self.isentrope, m)

    def kantrowitz_ratio(self, mach):
        return _fd_scalar(self._kantrowitz, mach)

    def isentropic_ratio(self, mach):
        return _fd_scalar(self._isentropic, mach)

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
