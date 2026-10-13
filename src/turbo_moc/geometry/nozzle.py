"""Nozzle contour with a convergent inlet, and its planar or axisymmetric solid."""
import numpy as np

from turbo_moc.core.thermo_characteristics import build_convergent_inlet


NOZZLE_DEFAULTS = {
    "convergent_length": 0.04,
    "planar_depth": 0.02,
    "axial_cells": 120,
    "radial_cells": 40,
    "wall_growth_ratio": 1.1,
    "circumferential_cells": 64,
    "depth_cells": 16,
    "boundary_types": {
        "inlet": "BCInflowSubsonic",
        "outlet": "BCOutflowSupersonic",
        "wall": "BCWallViscous",
        "axis": "BCSymmetryPlane",
        "side": "BCWallViscous",
    },
}


def axisymmetric(parameters):
    return bool(parameters.get("delta_flow", 0.0))


def validate_settings(settings):
    for name in ("convergent_length", "planar_depth", "wall_growth_ratio"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"mesh.nozzle.{name} must be a positive number.")
    if settings["wall_growth_ratio"] < 1:
        raise ValueError("mesh.nozzle.wall_growth_ratio must be at least 1.")
    for name in ("axial_cells", "radial_cells", "circumferential_cells", "depth_cells"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"mesh.nozzle.{name} must be an integer of at least 2.")
    if settings["circumferential_cells"] % 4 or settings["circumferential_cells"] < 8:
        raise ValueError("mesh.nozzle.circumferential_cells must be a multiple of 4 (at least 8).")
    if set(settings["boundary_types"]) != set(NOZZLE_DEFAULTS["boundary_types"]):
        raise ValueError(f"mesh.nozzle.boundary_types requires {sorted(NOZZLE_DEFAULTS['boundary_types'])}.")


def contour(result, parameters, settings):
    wall = result["wall_final"]
    x, y = np.asarray(wall["x"], float), np.asarray(wall["y"], float)
    keep = np.concatenate([[True], np.diff(x) > 1e-12 * max(1.0, abs(x).max())])
    x, y = x[keep], y[keep]
    if np.any(np.diff(x) <= 0):
        raise ValueError("The nozzle wall is not single-valued in x.")
    length = float(settings["convergent_length"])
    rho_d, y_t = float(parameters["rho_d"]), float(parameters["y_t"])
    if not 0 < 0.5 * length < rho_d:
        raise ValueError(f"The convergent length must be below twice the throat downstream radius "
                         f"({2 * rho_d:.4g} m).")
    inlet = build_convergent_inlet(y_t, rho_d, length)
    x = np.concatenate([inlet["x"], x])
    y = np.concatenate([inlet["y"], y])
    return {"x": x, "y": y, "throat_index": len(inlet["x"]), "axisymmetric": axisymmetric(parameters)}


def surfaces(shape, depth, n_theta=72):
    x, y = shape["x"], shape["y"]
    if shape["axisymmetric"]:
        theta = np.linspace(0, 2 * np.pi, n_theta + 1)
        return [{"name": "wall", "x": np.repeat(x[None, :], len(theta), 0),
                 "y": np.outer(np.cos(theta), y), "z": np.outer(np.sin(theta), y)}]
    z = np.array([0.0, depth]) - 0.5 * depth
    sides = [{"name": name, "x": np.repeat(x[None, :], 2, 0), "y": np.repeat(sign * y[None, :], 2, 0),
              "z": np.repeat(z[:, None], len(x), 1)} for name, sign in (("upper wall", 1), ("lower wall", -1))]
    plates = [{"name": "side plate", "x": np.repeat(x[None, :], 2, 0), "y": np.vstack([y, -y]),
               "z": np.full((2, len(x)), value)} for value in z]
    return sides + plates
