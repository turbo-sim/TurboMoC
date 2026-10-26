"""Leading-edge bow shock of a supersonic rotor cascade (Moeckel 1949, real gas).

Moeckel (NACA TN 1921) places the detached shock ahead of a blunt body with three
assumptions: the shock is a hyperbola asymptotic to the free-stream Mach lines,

    eta = sqrt(x^2 - x0^2) / B,    B = sqrt(M1^2 - 1),

the sonic line joining the sonic point on the shock to the sonic point on the
body is straight, and continuity across that sonic line fixes the shock
standoff. Here the sonic line is taken normal to the mean flow direction
across it, so that

    rho1 V1 y_s = (rho a)* (y_s - y_b) / cos(theta_m),

with y_b the half-height of the body at its sonic point (where the surface
deflection equals the sonic deflection of an oblique shock) and (rho a)* the
sonic flux behind the normal shock on the stagnation streamline. The ratio
rho1 V1 / (rho a)* is the Kantrowitz ratio of ``rotor_starting``.

All jump conditions use the real-gas (equilibrium two-phase) equation of state.
In a cascade the shock branches are followed until they meet a neighbouring
blade: the first hit on a suction surface is where the bow shock impinges
(Sousa & Paniagua, Entropy 17 (2015) 5593, Figs. 2b and 5). The local shock
strength there follows from the hyperbola slope with the uniform inflow
upstream, which ignores the waves the shock crosses on its way.
"""

import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq

from .rotor_starting import Isentrope, minimum_temperature, relative_isentrope


def normal_shock(fluid, rho1, p1, h1, vn1):
    """Downstream state and normal velocity of a normal shock with upstream normal velocity ``vn1``."""

    def downstream(vn2):
        return fluid.get_state(jxp.DmassHmass_INPUTS, rho1 * vn1 / vn2, h1 + 0.5 * (vn1 ** 2 - vn2 ** 2))

    def momentum(vn2):
        try:
            return p1 + rho1 * vn1 ** 2 - float(downstream(vn2).p) - rho1 * vn1 * vn2
        except RuntimeError:  # Post-shock density beyond the liquid limit
            return np.nan

    grid = np.linspace(0.05 * vn1, 0.998 * vn1, 120)
    values = [momentum(v) for v in grid]
    k = next(i for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0)
    vn2 = brentq(momentum, grid[k], grid[k + 1])
    return downstream(vn2), vn2


class ObliqueShock:
    """Real-gas oblique shocks for one upstream state on an isentrope."""

    def __init__(self, isentrope, mach):
        self.isentrope, self.mach = isentrope, float(mach)
        self.v1 = isentrope.velocity(mach)
        upstream = isentrope.state(self.v1)
        self.rho1, self.p1, self.h1, self.s1 = (float(upstream.rho), float(upstream.p), float(upstream.h),
                                                float(upstream.s))
        self.mach_angle = np.arcsin(1.0 / self.mach)
        self.weakest = np.arcsin(min(1.01 / self.mach, 1.0))  # Normal Mach 1.01: weaker waves are taken as Mach waves

    def jump(self, wave_angle):
        """Deflection [rad], downstream Mach, pressure ratio and entropy rise for a wave angle [rad]."""
        vn1, vt = self.v1 * np.sin(wave_angle), self.v1 * np.cos(wave_angle)
        state, vn2 = normal_shock(self.isentrope.fluid, self.rho1, self.p1, self.h1, vn1)
        return {"deflection": wave_angle - np.arctan2(vn2, vt), "mach": np.hypot(vn2, vt) / float(state.a),
                "pressure_ratio": float(state.p) / self.p1, "entropy_rise": float(state.s) - self.s1}

    def sonic(self):
        """Wave angle and deflection [rad] at which the flow behind the shock is sonic."""
        low, high = self.weakest, 0.5 * np.pi * 0.9995
        angle = brentq(lambda a: self.jump(a)["mach"] - 1.0, low, high, xtol=1e-6)
        return angle, self.jump(angle)["deflection"]

    def attached_angle(self, deflection):
        """Weak-branch wave angle [rad] for a deflection [rad]; None if the shock detaches."""
        sonic_angle, sonic_deflection = self.sonic()
        if deflection > sonic_deflection:
            return None
        if deflection <= self.jump(self.weakest)["deflection"]:
            return self.weakest  # A Mach wave in practice
        return brentq(lambda a: self.jump(a)["deflection"] - deflection, self.weakest, sonic_angle,
                      xtol=1e-7)


def _densify(points, count=4000):
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    samples = np.linspace(0.0, length[-1], count)
    return np.column_stack([np.interp(samples, length, points[:, i]) for i in range(2)])


def _body_sonic_points(outline, flow, sonic_deflection):
    """Stagnation point and, on each side of it, the first outline point whose deflection falls to the sonic one.

    Returns (stagnation, sides) with sides a list of (point, deflection at the nose side of it).
    """
    closed = _densify(np.vstack([outline, outline[:1]]))[:-1]
    count = len(closed)
    stagnation = int(np.argmin(closed @ flow))
    sides = []
    for step in (1, -1):
        found = None
        for i in range(1, count // 4):
            a, b = closed[(stagnation + step * (i - 1)) % count], closed[(stagnation + step * i) % count]
            d = b - a
            deflection = abs(np.arctan2(flow[0] * d[1] - flow[1] * d[0], flow @ d))
            if deflection <= sonic_deflection:
                found = (a, deflection)
                break
        sides.append(found)
    return closed[stagnation], sides


def _segment_hits(path, curve):
    """Parameters along ``path`` (cumulative length) where it crosses the polyline ``curve``, with the curve index."""
    hits = []
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for i in range(len(path) - 1):
        p, r = path[i], path[i + 1] - path[i]
        q, s = curve[:-1], np.diff(curve, axis=0)
        denominator = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((q[:, 0] - p[0]) * s[:, 1] - (q[:, 1] - p[1]) * s[:, 0]) / denominator
            u = ((q[:, 0] - p[0]) * r[1] - (q[:, 1] - p[1]) * r[0]) / denominator
        valid = np.flatnonzero((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))
        hits += [(length[i] + t[j] * (length[i + 1] - length[i]), int(j)) for j in valid]
    return hits


def bow_shock(rotor, mach=None, beta_inlet=None, branch_length=None, count=400):
    """Bow shock of a vortex rotor result (as written by the app worker) and where it hits the neighbours.

    ``mach`` and ``beta_inlet`` [deg, from axial] default to the design inflow; pass the CFD values to compare
    with a simulation. Lengths are in the rotor's units. Returns the shock geometry, its standoff, and the
    first impingement on a neighbouring suction surface (chord fraction from the leading edge, local wave
    angle, pressure ratio, downstream Mach and entropy rise of the local oblique shock).
    """
    isentrope = relative_isentrope(rotor["fluid_name"], rotor["P0_rel"], rotor["T0_rel"],
                                   rotor.get("Q0_rel", float("nan")))
    mach = float(rotor["M_inlet"] if mach is None else mach)
    beta = np.radians(float(rotor["beta_inlet"] if beta_inlet is None else beta_inlet))
    flow, normal = np.array([np.cos(beta), np.sin(beta)]), np.array([-np.sin(beta), np.cos(beta)])
    shock = ObliqueShock(isentrope, mach)
    sonic_angle, sonic_deflection = shock.sonic()

    outline = np.column_stack([rotor["blade"]["x"], rotor["blade"]["y"]])
    stagnation, sides = _body_sonic_points(outline, flow, sonic_deflection)
    flank = [side[1] if side else 0.0 for side in sides]
    B = np.sqrt(mach ** 2 - 1.0)
    pitch = float(rotor["pitch"])
    x_le, chord_axial = outline[:, 0].min(), np.ptp(outline[:, 0])

    attached = [shock.attached_angle(d) if side and np.allclose(side[0], stagnation) else None
                for side, d in zip(sides, flank)]
    if all(a is not None for a in attached):
        # Sharp edge with both flanks below the sonic deflection: straight oblique shocks from the tip.
        standoff, x0 = 0.0, None
        length = branch_length or 3.0 * pitch
        branches = [stagnation + np.outer(np.linspace(0, length, count),
                                          np.cos(a) * flow + sign * np.sin(a) * normal)
                    for a, sign in zip(attached, (1, -1))]
        kind = "attached"
    else:
        # Moeckel: half-heights of the body sonic points about the stagnation streamline, averaged over the sides.
        heights = [abs((side[0] - stagnation) @ normal) for side in sides if side]
        depths = [(side[0] - stagnation) @ flow for side in sides if side]
        y_b, depth = float(np.mean(heights)), float(np.mean(depths))
        flux_behind = Isentrope(isentrope.fluid, isentrope.h0, shock.s1 + shock.jump(0.5 * np.pi)["entropy_rise"],
                                minimum_temperature(isentrope.fluid.name))
        ratio = isentrope.flux(shock.v1) / flux_behind.flux(flux_behind.v_star)
        theta_m = 0.5 * (sonic_deflection + shock.jump(sonic_angle)["deflection"])
        y_s = y_b / (1.0 - ratio * np.cos(theta_m))
        t = np.tan(sonic_angle)
        x0 = y_s * B * np.sqrt(B * B * t * t - 1.0)
        x_s = x0 * B * t / np.sqrt(B * B * t * t - 1.0)
        x_b = x_s + (y_s - y_b) * np.tan(theta_m)
        standoff = x_b - depth - x0
        length = branch_length or 3.0 * pitch
        x = x0 * np.cosh(np.linspace(0.0, np.arccosh(1.0 + length * B / x0 + length / x0), count))
        eta = np.sqrt(np.maximum(x * x - x0 * x0, 0.0)) / B
        vertex = stagnation - standoff * flow
        branches = [vertex + np.outer(x - x0, flow) + sign * np.outer(eta, normal) for sign in (1, -1)]
        kind = "detached"

    surfaces = {"suction": np.column_stack([rotor["blade_upper"]["x"], rotor["blade_upper"]["y"]]),
                "pressure": np.column_stack([rotor["blade_lower"]["x"], rotor["blade_lower"]["y"]])}
    hits = []
    for b, path in enumerate(branches):
        for k in (-2, -1, 1, 2):
            for name, curve in surfaces.items():
                for along, j in _segment_hits(path, curve + np.array([0.0, k * pitch])):
                    point = curve[j] + np.array([0.0, k * pitch])
                    hits.append({"branch": b, "neighbour": k, "surface": name, "along": along, "point": point})
    impact = None
    first = sorted(hits, key=lambda h: h["along"])
    suction = [h for h in first if h["surface"] == "suction"]
    if suction:
        hit = suction[0]
        path = branches[hit["branch"]]
        i = int(np.argmin(np.linalg.norm(path - hit["point"], axis=1)))
        tangent = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        wave = abs(np.arctan2(flow[0] * tangent[1] - flow[1] * tangent[0], flow @ tangent))
        wave = min(max(wave, shock.weakest), 0.5 * np.pi)
        local = shock.jump(wave)
        impact = {"chord_fraction": float((hit["point"][0] - x_le) / chord_axial), "neighbour": hit["neighbour"],
                  "wave_angle_deg": float(np.degrees(wave)), "pressure_ratio": local["pressure_ratio"],
                  "mach_behind": local["mach"], "entropy_rise": local["entropy_rise"],
                  "first_hit_surface": first[0]["surface"]}
    return {"kind": kind, "mach": mach, "beta_inlet": float(np.degrees(beta)),
            "sonic_deflection_deg": float(np.degrees(sonic_deflection)),
            "flank_deflection_deg": [float(np.degrees(d)) for d in flank],
            "standoff_over_pitch": float(standoff / pitch), "branches": branches, "impact": impact}
