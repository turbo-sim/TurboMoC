"""Solve and validate a MoC nozzle wall for the geometry workflows."""

import numpy as np
import turbo_moc

from turbo_moc import reporting as log
from turbo_moc.configuration.fluid_inputs import outlet_parameters, solver_parameters

logger = log.get_logger("moc")


def solve_nozzle(parameters, *, progress_callback=None):
    """Always solve the supplied inputs and return a converged, finite wall.

    Solver progress is forwarded directly. Saving a successful result for an
    active workflow is the caller's responsibility; this function performs no
    disk access or cache lookup.
    """
    logger.info("  Calculating a new MoC solution (solver output follows).")
    result = turbo_moc.design_nozzle(
        **outlet_parameters(solver_parameters(parameters)), progress_callback=progress_callback
    )

    # A successful status alone is insufficient: downstream blade fitting needs
    # at least two finite coordinate pairs, with x and y of matching length.
    try:
        wall = result["wall_final"]
        points = np.column_stack((wall["x"], wall["y"]))
        usable = bool(
            result["converged"]
            and points.ndim == 2
            and points.shape[1] == 2
            and len(points) >= 2
            and np.isfinite(points).all()
        )
    except (KeyError, TypeError, ValueError):
        usable = False
    if not usable:
        raise RuntimeError("The nozzle design did not converge to a usable wall.")
    return result
