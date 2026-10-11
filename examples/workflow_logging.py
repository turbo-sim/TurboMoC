"""Reporting for the Gmsh and Fluent examples: a thin wrapper around
turbo_moc.reporting (formatter, logging context, headings, summaries).

Workflow loggers live under ``turbo_moc.workflows.<name>``, next to the
library's own (``turbo_moc.api``, ``turbo_moc.solvers.*``,
``turbo_moc.meshing.*``), so the single ``turbo_moc`` logger configured by
``workflow_logging`` sends MoC, meshing and workflow records to the same
console and log file exactly once. Only entry points call it.
"""

from pathlib import Path

from turbo_moc import reporting
from turbo_moc.reporting import RULE_WIDTH, heading, section, step, summary  # noqa: F401 (re-exported)

LOGGER_NAME = "turbo_moc.workflows"
PROJECT_DIR = Path(__file__).resolve().parent.parent


def get_logger(name):
    """Logger ``turbo_moc.workflows.<name>`` (e.g. "gmsh.runner")."""
    return reporting.get_logger(f"workflows.{name}")


def workflow_logging(log_file, level="INFO"):
    """Console + fresh UTF-8 log file for one workflow run; see
    turbo_moc.reporting.logging_context (handlers, levels and propagation
    are restored afterwards, the root logger is never touched)."""
    return reporting.logging_context(log_file, level)


def display_path(path):
    """Short repository-relative path for generated artifacts."""
    return reporting.display_path(path, PROJECT_DIR)
