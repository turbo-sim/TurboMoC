"""First-cell height for a target y+ from the Phase-1 MoC expansion.

A rough, pre-CFD estimate: local turbulent flat-plate skin friction at the
blade chord, with the flow state taken from the nozzle's own isentropic
solution (wall_final: rho, T, M per station). Properties come from jaxprop at
(rho, T), so a two-phase (wet) expansion uses the mixture viscosity and the
equilibrium speed of sound. Accurate to roughly a factor of two -- the usual
precision of a y+ estimate; check the actual y+ in the CFD.

    Re_c  = rho U c / mu,          U = M a
    C_f   = 0.026 Re_c^(-1/7)      (turbulent flat plate, local; White)
    tau_w = C_f rho U^2 / 2,       u_tau = sqrt(tau_w / rho)
    y     = y+ nu / u_tau,         nu = mu / rho

The station with the highest wall shear (smallest y) governs, so the
target is met everywhere along the expansion.

Fluids without a CoolProp viscosity model (e.g. Novec649, MDM) fall back to
the Chung et al. (1988) dilute-gas correlation from the critical constants
(within ~1% of CoolProp for non-polar vapours, ~10% for polar ones such as
R245fa; dipole and dense-gas corrections are omitted). In a two-phase state
that is the vapour-phase value at the local temperature.
"""

import CoolProp.CoolProp as CP
import jaxprop as jxp
import numpy as np


def chung_dilute_gas_viscosity(fluid_name, temperature):
    """Chung et al. (1988) dilute-gas viscosity [Pa s], non-polar form."""
    t_crit = CP.PropsSI("Tcrit", fluid_name)
    v_crit = 1e6 / CP.PropsSI("rhomolar_critical", fluid_name)  # cm^3/mol
    omega = CP.PropsSI("acentric", fluid_name)
    molar_mass = CP.PropsSI("molar_mass", fluid_name) * 1e3  # g/mol
    t_star = 1.2593 * np.asarray(temperature, dtype=float) / t_crit
    collision = (1.16145 * t_star**-0.14874 + 0.52487 * np.exp(-0.77320 * t_star)
                 + 2.16178 * np.exp(-2.43787 * t_star))
    micropoise = (40.785 * (1 - 0.2756 * omega) * np.sqrt(molar_mass * np.asarray(temperature, dtype=float))
                  / (v_crit ** (2 / 3) * collision))
    return micropoise * 1e-7


def blade_chord_mm(blade):
    """True chord [mm]: leading edge (axial maximum of the closed contour)
    to trailing edge (axial minimum), flow toward -y."""
    x = np.asarray(blade["blade_curve"]["x"], dtype=float)
    y = np.asarray(blade["blade_curve"]["y"], dtype=float)
    le, te = np.argmax(y), np.argmin(y)
    return float(np.hypot(x[le] - x[te], y[le] - y[te]))


def estimate_first_layer_height(nozzle_result, chord_mm, y_plus, cell_centroid=True):
    """First wall-normal cell height [mm] giving `y_plus` at the most
    sheared station of the MoC expansion.

    cell_centroid=True: y+ is measured at the first cell's centroid (the
    convention Fluent reports), so the cell height is twice that distance.
    Returns a dict with first_height_mm and the governing station's
    diagnostics (Re, C_f, wall shear, friction velocity, U, rho, mu, M,
    vapour quality).
    """
    if not np.isfinite(y_plus) or y_plus <= 0:
        raise ValueError("Target y+ must be finite and positive.")
    if not np.isfinite(chord_mm) or chord_mm <= 0:
        raise ValueError("Chord must be finite and positive.")
    wall = nozzle_result["wall_final"]
    fluid_name = nozzle_result["fluid_name"]
    rho = np.asarray(wall["rho"], dtype=float)
    mach = np.asarray(wall["M"], dtype=float)
    temperature = np.asarray(wall["T"], dtype=float)
    state = jxp.Fluid(fluid_name).get_state(jxp.DmassT_INPUTS, rho, temperature)
    mu = np.asarray(state.viscosity, dtype=float)
    viscosity_model = "CoolProp (mixture value in two-phase states)"
    if not np.all(np.isfinite(mu)):  # no transport model for this fluid
        mu = chung_dilute_gas_viscosity(fluid_name, temperature)
        viscosity_model = f"Chung et al. (1988) dilute-gas correlation (CoolProp has no viscosity model for {fluid_name})"
    velocity = mach * np.asarray(state.speed_of_sound, dtype=float)
    chord = chord_mm * 1e-3
    reynolds = rho * velocity * chord / mu
    skin_friction = 0.026 * reynolds ** (-1.0 / 7.0)
    wall_shear = 0.5 * skin_friction * rho * velocity**2
    friction_velocity = np.sqrt(wall_shear / rho)
    distance = y_plus * (mu / rho) / friction_velocity
    height = distance * (2.0 if cell_centroid else 1.0)
    for name, values in (("density", rho), ("Mach number", mach), ("viscosity", mu),
                         ("speed of sound", velocity / np.where(mach == 0, 1, mach))):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} along the nozzle wall ({fluid_name}).")
    i = int(np.argmin(height))
    return {
        "first_height_mm": float(height[i] * 1e3),
        "y_plus": float(y_plus),
        "cell_centroid": bool(cell_centroid),
        "chord_mm": float(chord_mm),
        "station": i,
        "reynolds": float(reynolds[i]),
        "skin_friction": float(skin_friction[i]),
        "wall_shear_Pa": float(wall_shear[i]),
        "friction_velocity": float(friction_velocity[i]),
        "velocity": float(velocity[i]),
        "density": float(rho[i]),
        "viscosity": float(mu[i]),
        "viscosity_model": viscosity_model,
        "mach": float(mach[i]),
        "vapor_quality": float(np.asarray(state.vapor_quality, dtype=float)[i]),
    }


def inflation_stack(first_height, growth_ratio, num_layers):
    """Total thickness [mm] of `num_layers` geometric layers and the last
    layer's height: h1 (r^N - 1) / (r - 1) and h1 r^(N-1)."""
    if growth_ratio <= 1 or num_layers < 1 or first_height <= 0:
        raise ValueError("Need first_height > 0, growth_ratio > 1 and at least one layer.")
    thickness = first_height * (growth_ratio**num_layers - 1.0) / (growth_ratio - 1.0)
    return float(thickness), float(first_height * growth_ratio ** (num_layers - 1))
