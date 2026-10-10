"""Shared console and file reporting for the Gmsh and Fluent examples.

Only entry points install handlers. Imported helpers obtain child loggers, so
both workflows share one reporting style without configuring external packages.
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "stator_workflow"
# Library modules the examples drive (e.g. turbo_moc.meshing) log under their
# own package name; route them to the same handlers.
LIBRARY_LOGGER_NAMES = ("turbo_moc",)
PROJECT_DIR = Path(__file__).resolve().parent.parent
RULE_WIDTH = 72


class WorkflowFormatter(logging.Formatter):
    """Keep progress readable and distinguish diagnostics by severity."""

    def __init__(self):
        super().__init__("%(message)s")

    def format(self, record):
        message = super().format(record)
        if record.levelno != logging.INFO:
            return f"[{record.levelname}] {message}"
        return message


def get_logger(name):
    """Return a module logger under the examples' dedicated namespace."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


@contextmanager
def workflow_logging(log_file, level="INFO"):
    """Stream each message to stdout and a fresh UTF-8 log for this run.

    Handlers flush after every record. Restore the logger on exit so importing
    or running an example again does not duplicate messages or leak open files.
    """
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    configured = [logger] + [logging.getLogger(name) for name in LIBRARY_LOGGER_NAMES]
    previous = [(item, item.handlers[:], item.level, item.propagate) for item in configured]
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(WorkflowFormatter())
    for item in configured:
        item.handlers = handlers
        item.setLevel(level)
        item.propagate = False
    try:
        yield logger
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by the user.")
        raise
    except Exception:
        logger.exception("Workflow failed; see the last step and traceback below.")
        raise
    finally:
        for item, item_handlers, item_level, item_propagation in previous:
            item.handlers = item_handlers
            item.setLevel(item_level)
            item.propagate = item_propagation
        for handler in handlers:
            handler.close()


def heading(logger, title):
    """Start a workflow or its final results with a prominent heading."""
    rule = "=" * RULE_WIDTH
    logger.info("\n%s\n%s\n%s", rule, title.upper(), rule)


def section(logger, title):
    """Separate a group of related details without adding another step."""
    logger.info("\n%s\n%s", title, "-" * RULE_WIDTH)


def step(logger, number, total, title):
    """Announce a top-level operation before starting it."""
    section(logger, f"[{number}/{total}] {title}")


def summary(logger, title, values):
    """Print an aligned label/value block using caller-supplied units."""
    width = max((len(str(label)) for label in values), default=0) + 3
    rows = "\n".join(f"  {label:<{width}}{value}" for label, value in values.items())
    logger.info("\n%s\n%s", title, rows)


def display_path(path):
    """Prefer short repository-relative paths for generated artifacts."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(resolved)
