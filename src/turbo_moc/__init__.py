"""
TurboMoC: Method of Characteristics solvers for nozzle/stator design.

Two solver variants share one thermodynamic/characteristics core
(``turbo_moc.core.classes``):

* ``turbo_moc.solvers.conventional.MOCSolver`` is the rounded-arc kernel. It
  supports subcooled-liquid flashing inlets (with a flat post-jump throat
  front), ordinary single-phase inlets, and saturated two-phase inlets.
* ``turbo_moc.solvers.mln.MOCSolverMLN`` is the minimum-length, sharp-corner
  nozzle with a centred expansion fan from the throat corner.

Both accept the same style of config dict and ``export_full_result_data()``
schema, so results are directly comparable.
"""

from turbo_moc.core.classes import FluidManager, InternalPoint, get_fluid_manager
from turbo_moc.solvers.conventional import MOCSolver as ConventionalSolver
from turbo_moc.solvers.mln import MOCSolverMLN
from turbo_moc.api import design_nozzle, list_supported_fluids, NozzleDesignError
from turbo_moc.progress import make_progress_reporter
import turbo_moc.plotting_mpl as mpl
import turbo_moc.plotting_plotly as plotly

__all__ = [
    "FluidManager", "InternalPoint", "get_fluid_manager",
    "ConventionalSolver", "MOCSolverMLN",
    "design_nozzle", "list_supported_fluids", "NozzleDesignError",
    "make_progress_reporter",
    "mpl", "plotly",
]
__version__ = "0.1.0"
