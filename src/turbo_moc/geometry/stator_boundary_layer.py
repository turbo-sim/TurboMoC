"""Displacement-thickness correction of the MoC stator nozzle wall.

The MoC wall is inviscid: in the viscous blade passage the boundary layers grow from the throat and shrink the
effective area ratio of the divergent part, so the nozzle under-expands. The correction moves the wall outwards
(away from the nozzle axis) by the turbulent displacement thickness

    delta* = 0.046 s Re_s^-0.2,   Re_s = rho V s / mu,

with s the wall arc length from the throat and rho, V the MoC wall state. The half-nozzle wall stands for both
walls of the symmetric passage, so each wall opens by its own delta*. The throat (s = 0) is unchanged: the CFD
of the 30 C stage passed exactly the geometric throat's sonic flow (discharge coefficient about 1).

Settings (``method_of_characteristics.boundary_layer``): ``viscosity`` [Pa s] and optionally ``factor``. The
physical length scale comes from the blade chord: the stator template is the wall in millimetres, sized to the
midspan chord of the meridional channel.
"""
from copy import deepcopy

import jaxprop as jxp
import numpy as np


def displacement_thickness(wall, fluid_name, length_scale, viscosity, factor=1.0):
    """delta* [wall units] along the wall; ``length_scale`` is metres per wall unit."""
    x, y = np.asarray(wall["x"], dtype=float), np.asarray(wall["y"], dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) * length_scale
    fluid = jxp.Fluid(fluid_name)
    rho = np.asarray(wall["rho"], dtype=float)
    sound = np.array([float(fluid.get_state(jxp.DmassT_INPUTS, d, t).a) for d, t in zip(rho, wall["T"])])
    speed = np.asarray(wall["M"], dtype=float) * sound
    reynolds = np.maximum(rho * speed * s / viscosity, 1.0)
    return factor * 0.046 * s * reynolds ** -0.2 / length_scale


def displaced_nozzle(nozzle, config):
    """The nozzle with its wall opened by delta*, or the nozzle itself when no correction is configured."""
    settings = config["method_of_characteristics"].get("boundary_layer")
    if not settings:
        return nozzle
    from turbo_moc.app.blade_counts import native_passage
    channel = config["meridional_channel"]
    chord_mm = 0.5 * (float(channel["hub_chord"]) + float(channel["tip_chord"]))
    native_chord = native_passage(config, nozzle)["chord"]          # Template units: wall millimetres
    length_scale = chord_mm / native_chord                           # Metres per wall metre
    wall = nozzle["wall_final"]
    delta = displacement_thickness(wall, nozzle["fluid_name"], length_scale, float(settings["viscosity"]),
                                   float(settings.get("factor", 1.0)))
    x, y = np.asarray(wall["x"], dtype=float), np.asarray(wall["y"], dtype=float)
    tx, ty = np.gradient(x), np.gradient(y)
    norm = np.hypot(tx, ty)
    nx, ny = -ty / norm, tx / norm                                   # Left normal of a wall marching in +x: +y side
    result = deepcopy(nozzle)
    result["wall_final"] = dict(wall, x=(x + delta * nx).tolist(), y=(y + delta * ny).tolist())
    result["wall_inviscid"] = {"x": wall["x"], "y": wall["y"]}
    result["boundary_layer"] = {"delta_star_m": (delta * length_scale).tolist(),
                                "exit_delta_star_m": float(delta[-1] * length_scale),
                                "exit_area_ratio_gain": float((y[-1] + delta[-1] * ny[-1]) / y[-1])}
    return result
