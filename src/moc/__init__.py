"""
moc: Method of Characteristics solvers for nozzle/stator design.

Two solver variants sharing one thermodynamic/characteristics core
(moc.core.classes):
  - moc.solvers.conventional.MOCSolver    -- rounded-arc kernel, supports
    subcooled-liquid flashing inlets (flat post-jump throat front) as well
    as ordinary single-phase and saturated-two-phase inlets.
  - moc.solvers.mln.MOCSolverMLN          -- minimum-length (sharp-corner)
    nozzle, centred expansion fan from the throat corner.

Both accept the same style of config dict and export_full_paper_data()
schema, so results are directly comparable.
"""

from moc.core.classes import FluidManager, InternalPoint, get_fluid_manager
from moc.solvers.conventional import MOCSolver as ConventionalSolver
from moc.solvers.mln import MOCSolverMLN
from moc.api import design_nozzle, list_supported_fluids, NozzleDesignError
from moc.progress import make_progress_reporter
import moc.plotting_mpl as mpl
import moc.plotting_plotly as plotly

__all__ = [
    "FluidManager", "InternalPoint", "get_fluid_manager",
    "ConventionalSolver", "MOCSolverMLN",
    "design_nozzle", "list_supported_fluids", "NozzleDesignError",
    "make_progress_reporter",
    "mpl", "plotly",
]
__version__ = "0.1.0"
