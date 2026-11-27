"""Mixed-out state behind a blade row of the planar (one cell thick) stage, for the barotropic model.

Prasad (J. Turbomach. 127, 2005), rectangular geometry: the non-uniform flow on a pitchwise line behind the trailing
edge is replaced by the uniform state that carries the same mass flux, axial momentum flux (with pressure) and
tangential momentum flux. With a barotropic fluid the energy equation is not needed (and not solved by Fluent): the
equation of state rho(p) closes the system, as in Romei et al. (Energy 255, 2022, App. B) for an arbitrary equation of
state. Per unit span and pitch length A,

    m = rho(p) c_x A,      X = m c_x + p A,      Y = m c_t
    ->  m^2 / (rho(p) A) + p A - X = 0   for p (subsonic axial branch, the higher root), c_x = m / (rho A), c_t = Y/m.

The mixing loss is the drop of total head H0 = H(p) + c^2/2 (H(p) = int dp/rho, the isentropic head) from the
mass-averaged value on the line to the mixed-out value; energy is not conserved by construction, the dissipated
mechanical energy is the loss. Total-to-total efficiency at the line: eta_tt = Dh0 / (H(p01) - H0_mixed).
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq


def pitchwise_line(flat, axial, pitch, samples=400):
    """Node values of an unwrapped midspan mesh (x = axial, y = arc, both in mm; analysis.unwrapped) interpolated on a
    pitchwise line of one ``pitch`` [mm] at ``axial`` [mm]. The periodic edges are slanted behind the blades, so the
    lower edge is taken from a narrow axial band. Returns arc positions [mm] and a dict of the point fields."""
    xy = flat.points[:, :2]
    for width in (0.1, 0.2, 0.4, 0.8):
        band = np.abs(xy[:, 0] - axial) < width
        if band.sum() >= 5:
            break
    else:
        raise ValueError(f"No mesh points near the axial position {axial:.2f} mm.")
    lo = xy[band, 1].min()
    y = lo + pitch * (np.arange(samples) + 0.5) / samples  # Cell-centred samples over exactly one pitch
    points = np.column_stack([np.full_like(y, axial), y])
    fields = {}
    for name in flat.point_data:
        values = LinearNDInterpolator(xy, np.asarray(flat.point_data[name], dtype=float))(points)
        fields[name] = values
    keep = np.all([np.isfinite(v) for v in fields.values()], axis=0)
    if keep.mean() < 0.95:
        raise ValueError("The pitchwise line leaves the mesh (more than 5 % of the samples outside).")
    return y[keep], {k: v[keep] for k, v in fields.items()}


def mixed_out(y_mm, p, rho, c_x, c_t, c_r, model):
    """Mass averages and the mixed-out state of one pitchwise line (velocities in m/s, pressures in Pa)."""
    y = y_mm * 1e-3
    dy = np.gradient(y)  # Uniform spacing: trapezoid-equivalent weights inside the line
    A = float(np.sum(dy))
    flux = rho * c_x * dy
    m = float(flux.sum())
    X = float(np.sum((rho * c_x ** 2 + p) * dy))
    Y = float(np.sum(rho * c_x * c_t * dy))
    c2 = c_x ** 2 + c_t ** 2 + c_r ** 2
    H0_mass = float(np.sum(flux * (model.head(p) + 0.5 * c2)) / m)
    p_mass = float(np.sum(flux * p) / m)

    def residual(q):
        return m ** 2 / (model.density(q) * A) + q * A - X

    # The momentum function m^2/(rho A) + p A has its minimum at the axial sonic point; the subsonic root lies above.
    grid = np.geomspace(0.3 * p.min(), 3.0 * p.max(), 400)
    values = np.array([residual(q) for q in grid])
    turn = int(np.argmin(values))
    above = np.where(np.sign(values[turn:-1]) != np.sign(values[turn + 1:]))[0]
    if values[turn] > 0 or not len(above):
        raise ValueError("No subsonic mixed-out state: the momentum flux is below the sonic minimum.")
    k = turn + above[0]
    p_mix = brentq(residual, grid[k], grid[k + 1], xtol=1e-6)
    rho_mix = float(model.density(p_mix))
    cx_mix, ct_mix = m / (rho_mix * A), Y / m
    H0_mix = float(model.head(p_mix) + 0.5 * (cx_mix ** 2 + ct_mix ** 2))
    return {"mass_flux": m / A, "p_mass": p_mass, "H0_mass": H0_mass,
            "p_mix": float(p_mix), "rho_mix": rho_mix, "c_x_mix": cx_mix, "c_t_mix": ct_mix,
            "alpha_mix": float(np.degrees(np.arctan2(ct_mix, cx_mix))),
            "M_x_mix": cx_mix / float(model.speed_of_sound(p_mix)), "H0_mix": H0_mix,
            "mixing_loss": H0_mass - H0_mix,
            "momentum_check": (m * cx_mix + p_mix * A - X) / X}
