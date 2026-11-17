"""JAX lookups along the relative isentrope, identical to ``FluidManager``'s SciPy splines.

``JaxIsentrope.from_manager`` takes the table nodes and the piecewise-cubic coefficients of the SciPy ``CubicSpline``
objects that ``FluidManager._build_nu_table`` already built, and evaluates them in JAX (same interval search, same
Horner form, same clipping). Values therefore agree with the NumPy path to round-off, and every lookup is
differentiable in its argument.

The table itself depends on the inlet state (P0_rel, T0_rel); derivatives with respect to the inlet state are not
provided here (phase 1b of the plan): the table is a constant of the traced functions.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PPoly:
    """Piecewise cubic of ``scipy.interpolate.CubicSpline`` (breakpoints ``x``, coefficients ``c`` of shape (4, n-1))."""
    x: jnp.ndarray
    c: jnp.ndarray

    @classmethod
    def from_scipy(cls, spline):
        return cls(jnp.asarray(spline.x, dtype=jnp.float64), jnp.asarray(spline.c, dtype=jnp.float64))

    def __call__(self, value):
        # SciPy: interval i with x[i] <= value < x[i + 1]; the last interval also takes value == x[-1]
        i = jnp.clip(jnp.searchsorted(self.x, value, side="right") - 1, 0, self.x.shape[0] - 2)
        dx = value - self.x[i]
        c = self.c[:, i]
        return ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]

    def tree_flatten(self):
        return (self.x, self.c), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxIsentrope:
    """nu(V), a(V), rho(V), V(nu) on the relative isentrope, plus the critical speed a*."""
    nu_of_V_poly: PPoly
    a_of_V_poly: PPoly
    rho_of_V_poly: PPoly
    V_of_nu_poly: PPoly
    V_min: jnp.ndarray
    V_max: jnp.ndarray
    nu_max: jnp.ndarray
    a_star: jnp.ndarray

    @classmethod
    def from_manager(cls, manager):
        """From a built ``thermo_characteristics.FluidManager`` (cached by ``get_fluid_manager``)."""
        table = manager._nu_V_table
        return cls(PPoly.from_scipy(manager._spline_nu_of_V), PPoly.from_scipy(manager._spline_a_of_V),
                   PPoly.from_scipy(manager._spline_rho_of_V), PPoly.from_scipy(manager._spline_V_of_nu),
                   jnp.float64(table[0]), jnp.float64(table[-1]), jnp.float64(manager._nu_max),
                   jnp.float64(float(manager.a_star)))

    def _clip(self, V):
        return jnp.clip(V, self.V_min, self.V_max)

    def nu_of_V(self, V):
        return self.nu_of_V_poly(self._clip(V))

    def a_of_V(self, V):
        return self.a_of_V_poly(self._clip(V))

    def rho_of_V(self, V):
        return self.rho_of_V_poly(self._clip(V))

    def V_of_nu(self, nu):
        # Reference: V_min for nu <= 0, nu capped at nu_max
        return jnp.where(nu <= 0.0, self.V_min, self.V_of_nu_poly(jnp.minimum(nu, self.nu_max)))

    def mach_of_V(self, V):
        return V / self.a_of_V(V)

    def V_of_M(self, M):
        return _V_of_M(self, M)

    def tree_flatten(self):
        return (self.nu_of_V_poly, self.a_of_V_poly, self.rho_of_V_poly, self.V_of_nu_poly, self.V_min, self.V_max,
                self.nu_max, self.a_star), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _bisect_V_of_M(iso, M):
    """80-step bisection of M(V) = V / a(V), as ``vortex_blade._V_of_M`` (same iterates): in the first table
    interval where M(V) reaches M (first crossing along the expansion; M(V) is not monotone wet-to-dry)."""
    nodes = iso.a_of_V_poly.x  # the table velocities (CubicSpline breakpoints)
    k = jnp.argmax(nodes / iso.a_of_V(nodes) - M >= 0)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        high = iso.mach_of_V(mid) - M > 0
        return jnp.where(high, lo, mid), jnp.where(high, mid, hi)

    lo, hi = jax.lax.fori_loop(0, 80, body, (nodes[jnp.maximum(k - 1, 0)], nodes[k]))
    return jnp.where(k == 0, nodes[0], 0.5 * (lo + hi))


@jax.custom_jvp
def _V_of_M(iso, M):
    return _bisect_V_of_M(iso, M)


@_V_of_M.defjvp
def _V_of_M_jvp(primals, tangents):
    # Implicit function theorem on M(V) - M = 0: dV = dM / (dM/dV). The table (iso) is a constant here.
    iso, M = primals
    _, dM = tangents
    V = _bisect_V_of_M(iso, M)
    dMdV = jax.grad(iso.mach_of_V)(V)
    return V, dM / dMdV


def check_range(iso, M):
    """The reference raises outside the table; JAX cannot raise inside a trace, so callers check concrete inputs."""
    machs = np.asarray(iso.mach_of_V(iso.a_of_V_poly.x))  # M(V) need not be monotone: reachable = [M(V_min), max M]
    m_lo, m_hi = float(machs[0]), float(machs.max())
    if np.any(np.asarray(M) < m_lo) or np.any(np.asarray(M) > m_hi):
        raise ValueError(f"Mach number outside the isentrope table range [{m_lo:.4f}, {m_hi:.4f}].")


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(np.asarray, tree)
