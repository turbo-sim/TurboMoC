"""Shared console/file reporting for TurboMoC, its Gmsh and Fluent workflows.

Library modules only obtain loggers (``logging.getLogger(__name__)``) under
the ``turbo_moc`` hierarchy (``turbo_moc.api``, ``turbo_moc.solvers.*``,
``turbo_moc.meshing.*``, ``turbo_moc.workflows.*``); they never install
handlers. An entry point opts in to output explicitly::

    from turbo_moc import design_nozzle
    from turbo_moc.reporting import logging_context

    with logging_context("nozzle.log", level="INFO"):
        result = design_nozzle(...)

``logging_context`` configures only the ``turbo_moc`` logger (console and,
optionally, a fresh UTF-8 file), never the root logger, and restores the
previous handlers, level and propagation on exit -- also after exceptions
and interruptions -- so repeated runs neither duplicate records nor leak
file handles. Without it, Python's default applies: warnings and errors
reach stderr, routine INFO progress stays silent.

Levels:

* INFO -- inputs, stage start/completion, throttled progress of long stages
  (at most once per second per stage) and a short result summary.
* DEBUG -- per-iteration diagnostics, adaptive step sizes, critical-state
  details, fluid-manager reuse. Computed only when DEBUG is enabled.
* WARNING -- recoverable failures, premature stops, fallbacks to the last
  valid state, non-converged results and automatic retries.

Progress callbacks (``progress_callback(fraction, stage_label)``, e.g. for
the Dash app) are driven by the same solver state as the log, but are
delivered on every update regardless of the log level or console
throttling. Fractions of counted loops are exact (work done / total); the
conventional kernel reports the fraction of the initial gap to its
termination target closed so far -- an estimate of convergence, not of
remaining runtime; the minimum-length kernel reports its prescribed
angular sweep. Elapsed times use ``time.perf_counter`` at stage
boundaries; they include first-call setup and JAX compilation where those
happen (not measured separately).
"""

import logging
import math
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path

ROOT_LOGGER_NAME = "turbo_moc"
RULE_WIDTH = 72
LINE_WIDTH = 100
PROGRESS_INTERVAL_S = 1.0
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

__all__ = [
    "ReportFormatter", "logging_context", "get_logger", "heading", "section", "step", "summary",
    "display_path", "format_pressure", "StageProgress",
]


class ReportFormatter(logging.Formatter):
    """Plain INFO messages, ``[LEVEL]`` prefixes otherwise; long lines are
    wrapped (not truncated) at LINE_WIDTH with an indented continuation.
    Words, paths and tracebacks are never broken."""

    def __init__(self, width=LINE_WIDTH):
        super().__init__("%(message)s")
        self.width = width

    def _wrap(self, line):
        if len(line) <= self.width:
            return [line]
        indent = " " * (len(line) - len(line.lstrip()))
        return textwrap.wrap(line, self.width, initial_indent="", subsequent_indent=indent + "  ",
                             break_long_words=False, break_on_hyphens=False) or [line]

    def format(self, record):
        message = record.getMessage().expandtabs(4)
        if record.levelno != logging.INFO:
            message = f"[{record.levelname}] {message}"
        lines = [wrapped for line in message.split("\n") for wrapped in self._wrap(line)]
        text = "\n".join(lines)
        if record.exc_info:
            text = f"{text}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            text = f"{text}\n{self.formatStack(record.stack_info)}"
        return text


def get_logger(name):
    """Logger ``turbo_moc.<name>`` (for scripts and workflows outside the package)."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


@contextmanager
def logging_context(log_file=None, level="INFO", stream=None):
    """Send ``turbo_moc`` records to `stream` (default stdout) and, if given,
    a fresh UTF-8 `log_file`, each exactly once; restore everything on exit.

    The root logger is untouched. An exception escaping the block is logged
    once here, with its traceback, and re-raised; Ctrl+C is reported as an
    interruption.
    """
    if level not in LEVELS:
        raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = (logger.handlers[:], logger.level, logger.propagate)
    handlers = [logging.StreamHandler(stream or sys.stdout)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    formatter = ReportFormatter()
    for handler in handlers:
        handler.setFormatter(formatter)
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield logger
    except KeyboardInterrupt:
        logger.warning("Interrupted by the user.")
        raise
    except Exception:
        logger.exception("Failed; see the last step and the traceback below.")
        raise
    finally:
        logger.handlers, logger.propagate = previous[0], previous[2]
        logger.setLevel(previous[1])
        for handler in handlers:
            handler.close()


def heading(logger, title):
    """Prominent top-level heading (workflow start, final results)."""
    rule = "=" * RULE_WIDTH
    logger.info("\n%s\n%s\n%s", rule, title.upper(), rule)


def section(logger, title):
    """Top-level workflow section."""
    logger.info("\n%s\n%s", title, "-" * RULE_WIDTH)


def step(logger, number, total, title):
    """Announce a numbered top-level workflow step before starting it."""
    section(logger, f"[{number}/{total}] {title}")


def summary(logger, title, values, indent=""):
    """Aligned label/value block (caller supplies units), one blank line before."""
    width = max((len(str(label)) for label in values), default=0) + 3
    rows = "\n".join(f"{indent}  {label:<{width}}{value}" for label, value in values.items())
    logger.info("\n%s%s\n%s", indent, title, rows)


def display_path(path, base_dir=None):
    """`path` relative to `base_dir` (default: the working directory) when
    inside it, else absolute; always with forward slashes."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(base_dir or Path.cwd()).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def format_pressure(p):
    """Readable pressure with its unit: Pa, kPa or MPa."""
    p = float(p)
    if not math.isfinite(p):
        return "n/a"
    if abs(p) >= 1e6:
        return f"{p / 1e6:.3f} MPa"
    if abs(p) >= 1e3:
        return f"{p / 1e3:.1f} kPa"
    return f"{p:.0f} Pa"


class StageProgress:
    """Reporting for one MoC stage: a start line, at most one progress line
    per `interval` seconds, and one completion line with the elapsed time.

    ``update(fraction, detail)`` forwards the clamped, monotonic fraction to
    the progress callback on every call (independent of the log level);
    `detail` -- a string or a zero-argument callable, evaluated only when a
    line is actually logged -- describes the measurements. `clock` is
    injectable for tests.
    """

    def __init__(self, logger, index, title, label, callback=None, total=4, name=None,
                 clock=time.perf_counter, interval=PROGRESS_INTERVAL_S):
        self.logger, self.index, self.title, self.label = logger, index, title, label
        self.callback, self.total, self.name = callback, total, name or title.lower()
        self.clock, self.interval = clock, interval
        self.fraction = 0.0
        self.start = self.last_line = None

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and self.start is not None:
            self.finish()
        return False

    def begin(self):
        self.start = self.last_line = self.clock()
        self.fraction = 0.0
        self.logger.info("  MoC [%d/%d] %s", self.index, self.total, self.title)
        self._notify()

    @property
    def elapsed(self):
        return self.clock() - self.start

    def _notify(self):
        if self.callback is not None:
            self.callback(self.fraction, self.label)

    def update(self, fraction, detail=None):
        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            fraction = self.fraction
        if math.isfinite(fraction):
            self.fraction = max(self.fraction, min(1.0, max(0.0, fraction)))
        self._notify()
        now = self.clock()
        if detail is not None and now - self.last_line >= self.interval and self.logger.isEnabledFor(logging.INFO):
            self.last_line = now
            text = detail() if callable(detail) else detail
            self.logger.info("  MoC %s: %s; %.2f s", self.name, text, now - self.start)

    def finish(self, outcome="completed"):
        """Completion line, e.g. outcome "completed", "reached the target",
        "stopped early"; a completed stage reports fraction 1 to the callback."""
        elapsed = self.elapsed
        self.fraction = 1.0
        self._notify()
        self.logger.info("  MoC [%d/%d] %s %s in %.2f s", self.index, self.total, self.title, outcome, elapsed)
        self.start = None
