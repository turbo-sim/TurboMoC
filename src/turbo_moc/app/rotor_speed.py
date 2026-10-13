"""Convert rotor speed specifications using a total-to-static reference drop."""
from math import cos, isfinite, pi, radians, sqrt

import jaxprop as jxp

from turbo_moc.configuration.fluid_inputs import inlet_values


def stator_reference(config, nozzle, alpha_deg):
    """Reference the stator back pressure and estimate flow by exit continuity.

    Mass flow uses a uniform midspan exit velocity over the full annular area,
    without a blade-blockage correction. This is a meanline reference estimate.
    """
    inputs = config["method_of_characteristics"]
    values = inlet_values(inputs)
    fluid = jxp.Fluid(inputs["fluid_name"])
    inlet = fluid.get_state(jxp.INPUT_PAIRS[values["inlet_property_pair"]],
                           values["inlet_property_1"], values["inlet_property_2"])
    outlet = fluid.get_state(jxp.PSmass_INPUTS, inputs["p_back"], inlet.s)
    drop = float(inlet.h - outlet.h)
    if not isfinite(drop) or drop <= 0:
        raise ValueError("Speed reference requires an expansion to the stator back pressure.")
    channel = config["meridional_channel"]
    if channel["type"] == "axial":
        area = pi * ((channel["hub_radius"] + channel["blade_height"]) ** 2
                     - channel["hub_radius"] ** 2) * 1e-6
    elif channel["type"] == "radial":
        area = 2 * pi * channel["outlet_radius"] * channel["blade_height"] * 1e-6
    else:
        raise ValueError("Dimensionless rotor speed requires an annular stator.")
    wall = nozzle["wall_final"]
    density, temperature = float(wall["rho"][-1]), float(wall["T"][-1])
    static = fluid.get_state(jxp.DmassT_INPUTS, density, temperature)
    meridional_velocity = float(wall["M"][-1]) * float(static.a) * cos(radians(alpha_deg))
    return {"enthalpy_drop": drop, "exit_density": float(outlet.rho),
            "mass_flow": density * meridional_velocity * area,
            "reference_pressure": inputs["p_back"], "c0": sqrt(2 * drop)}


def resolve_speed(specification, radius_mm, *, enthalpy_drop=None, exit_density=None, mass_flow=None):
    """Return RPM and inlet blade speed; specific speed follows TurboDash."""
    kind, value = specification["type"], specification["value"]
    radius = radius_mm / 1000
    if not isfinite(radius) or radius <= 0:
        raise ValueError("Rotor speed conversion requires a positive inlet radius.")
    if kind == "actual":
        omega = value * 2 * pi / 60
    else:
        if enthalpy_drop is None or not isfinite(enthalpy_drop) or enthalpy_drop <= 0:
            raise ValueError("Dimensionless rotor speed requires a positive isentropic enthalpy drop.")
        if kind == "blade_velocity_ratio":
            omega = value * sqrt(2 * enthalpy_drop) / radius
        elif kind == "specific":
            if any(x is None or not isfinite(x) or x <= 0 for x in (exit_density, mass_flow)):
                raise ValueError("Specific speed requires positive mass flow and isentropic outlet density.")
            omega = value * enthalpy_drop ** .75 / sqrt(mass_flow / exit_density)
        else:
            raise ValueError(f"Unknown rotational speed type {kind!r}.")
    return {"rpm": omega * 60 / (2 * pi), "U": omega * radius}
