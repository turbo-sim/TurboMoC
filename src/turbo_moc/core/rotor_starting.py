"""Starting check of a supersonic vortex rotor passage (Kantrowitz criterion, real gas).

A supersonic passage starts only if a normal shock standing at its entrance can
be swallowed: the throat must pass the inlet mass flow at sonic conditions on
the post-shock isentrope. With A_1 the captured inlet width (pitch times
cos(beta_in)) and A_t the geometric throat, the passage starts when

    A_t / A_1 >= (rho V)_1 / (rho V)*_2,

the right-hand side being the Kantrowitz ratio evaluated with the real-gas
equation of state, including two-phase equilibrium states. Its isentropic
counterpart (rho V)_1 / (rho V)*_0 is the smallest throat the started flow
needs.
"""

import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq


class Isentrope:
    """States along one isentrope at a fixed total enthalpy, parametrized by velocity."""

    def __init__(self, fluid, h0, s, T_min):
        self.fluid, self.h0, self.s = fluid, float(h0), float(s)
        self.v_max = 0.999 * np.sqrt(2 * (self.h0 - float(fluid.get_state(jxp.SmassT_INPUTS, s, T_min).h)))
        self.v_star = brentq(lambda v: v - float(self.state(v).a), 1e-3, self.v_max)

    def state(self, v):
        return self.fluid.get_state(jxp.HmassSmass_INPUTS, self.h0 - 0.5 * v * v, self.s)

    def velocity(self, mach):
        return brentq(lambda v: v / float(self.state(v).a) - mach, 1e-3, self.v_max)

    def flux(self, v):
        return float(self.state(v).rho) * v


def minimum_temperature(fluid_name):
    import CoolProp.CoolProp as CP
    return CP.PropsSI("Tmin", fluid_name) + 5.0


def relative_isentrope(fluid_name, P0_rel, T0_rel, Q0_rel=float("nan")):
    fluid = jxp.Fluid(fluid_name)
    total = (fluid.get_state(jxp.PQ_INPUTS, P0_rel, Q0_rel) if np.isfinite(Q0_rel)
             else fluid.get_state(jxp.PT_INPUTS, P0_rel, T0_rel))
    return Isentrope(fluid, total.h, total.s, minimum_temperature(fluid_name))


def isentropic_ratio(isentrope, mach):
    return isentrope.flux(isentrope.velocity(mach)) / isentrope.flux(isentrope.v_star)


def kantrowitz_ratio(isentrope, mach):
    """Inlet mass flux over the sonic mass flux behind a normal shock at the inlet Mach number."""
    fluid = isentrope.fluid
    v1 = isentrope.velocity(mach)
    upstream = isentrope.state(v1)
    rho1, p1, h1 = float(upstream.rho), float(upstream.p), float(upstream.h)

    def downstream(v2):
        return fluid.get_state(jxp.DmassHmass_INPUTS, rho1 * v1 / v2, h1 + 0.5 * (v1 ** 2 - v2 ** 2))

    def momentum(v2):
        return p1 + rho1 * v1 ** 2 - float(downstream(v2).p) - rho1 * v1 * v2

    def safe_momentum(v2):
        try:
            return momentum(v2)
        except RuntimeError:  # Post-shock density beyond the liquid limit (dense two-phase fluids)
            return np.nan

    grid = np.linspace(0.05 * v1, 0.998 * v1, 120)
    values = [safe_momentum(v) for v in grid]
    k = next(i for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0)
    v2 = brentq(momentum, grid[k], grid[k + 1])
    behind = Isentrope(fluid, isentrope.h0, float(downstream(v2).s), minimum_temperature(fluid.name))
    return rho1 * v1 / behind.flux(behind.v_star)


def geometric_throat(rotor):
    """Smallest distance between the closed blade outline and its neighbour one pitch away."""
    outline = np.column_stack([rotor["blade"]["x"], rotor["blade"]["y"]])
    fractions = np.linspace(0, 1, 20, endpoint=False)[:, None]
    dense = np.vstack([a + fractions * (b - a) for a, b in zip(outline[:-1], outline[1:])])
    neighbour = dense[::3] + np.array([0.0, rotor["pitch"]])
    return float(np.min(np.linalg.norm(dense[:, None, :] - neighbour[None, :, :], axis=2)))


def starting_check(rotor):
    """Throat, Kantrowitz and isentropic ratios of a vortex rotor result (as written by the app worker)."""
    isentrope = relative_isentrope(rotor["fluid_name"], rotor["P0_rel"], rotor["T0_rel"],
                                   rotor.get("Q0_rel", float("nan")))
    mach, beta = float(rotor["M_inlet"]), float(rotor["beta_inlet"])
    throat_ratio = geometric_throat(rotor) / (rotor["pitch"] * np.cos(np.radians(beta)))
    kantrowitz = kantrowitz_ratio(isentrope, mach)
    return {"M_inlet": mach, "beta_inlet": beta, "axial_mach": mach * np.cos(np.radians(beta)),
            "throat_ratio": throat_ratio, "kantrowitz_ratio": kantrowitz,
            "isentropic_ratio": isentropic_ratio(isentrope, mach), "margin": throat_ratio / kantrowitz,
            "starts": bool(throat_ratio >= kantrowitz),
            "leading_edge_radius_over_pitch": rotor["blade"]["le_radius"] / rotor["pitch"]}


def starting_curves(rotor, machs):
    """Isentropic and Kantrowitz ratios along the rotor's relative isentrope, for the starting chart."""
    isentrope = relative_isentrope(rotor["fluid_name"], rotor["P0_rel"], rotor["T0_rel"],
                                   rotor.get("Q0_rel", float("nan")))
    return np.array([[isentropic_ratio(isentrope, m), kantrowitz_ratio(isentrope, m)] for m in machs])


def prandtl_meyer(isentrope, v, count=400):
    """Real-gas Prandtl-Meyer angle [rad]: the integral of sqrt(M^2 - 1) dV / V from the sonic speed."""
    speeds = np.linspace(isentrope.v_star, v, count)
    machs = np.array([u / float(isentrope.state(u).a) for u in speeds])
    return float(np.trapezoid(np.sqrt(np.maximum(machs ** 2 - 1, 0.0)) / speeds, speeds))


def unique_incidence(rotor, mach):
    """Inflow angle [deg] a started rotor imposes on a supersonic inflow with subsonic axial component.

    Simplified form of Bufi (thesis, Sec. 3.3.4, after Starken et al. 1984): the limiting characteristic meets
    the adjacent suction surface on its straight entrance (phi_E = 0), the flow there runs along the inlet metal
    angle beta_s, and the bow-shock loss is neglected. Continuity across the passage entrance (Eq. 3.31)

        rho_1 W_1 cos(beta_1) s = rho_E W_E (s cos(beta_s) - r)

    and the simple-wave relation along the right-running characteristic (Eq. 3.34)

        beta_1 - beta_s = nu(M_E) - nu(M_1)

    fix beta_1 and M_E; s is the blade spacing and r the leading-edge radius.
    """
    isentrope = relative_isentrope(rotor["fluid_name"], rotor["P0_rel"], rotor["T0_rel"],
                                   rotor.get("Q0_rel", float("nan")))
    spacing, radius = float(rotor["pitch"]), float(rotor["blade"]["le_radius"])
    metal = np.radians(float(rotor["beta_inlet"]))
    v1 = isentrope.velocity(mach)
    nu1, flux1 = prandtl_meyer(isentrope, v1), isentrope.flux(v1)
    width = spacing * np.cos(metal) - radius

    def angle(v):
        return metal + prandtl_meyer(isentrope, v) - nu1

    def residual(v):
        return flux1 * np.cos(angle(v)) * spacing - isentrope.flux(v) * width

    try:
        v_top = isentrope.velocity(min(2.5 * mach, 4.0))
    except ValueError:  # Mach not reachable before the triple point (dense fluids): search the whole isentrope
        v_top = isentrope.v_max
    grid = np.linspace(1.0001 * isentrope.v_star, v_top, 40)
    values = [residual(v) for v in grid]
    roots = [brentq(residual, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0]
    if mach * np.cos(metal) >= 1.0 or not roots:
        raise ValueError("No unique incidence: the axial inflow is supersonic or the passage cannot pass the flow.")
    vE = min(roots, key=lambda v: abs(angle(v) - metal))
    return {"beta_inlet": float(np.degrees(angle(vE))), "incidence": float(np.degrees(angle(vE) - metal)),
            "M_E": vE / float(isentrope.state(vE).a), "width_ratio": float(width / (spacing * np.cos(metal)))}
