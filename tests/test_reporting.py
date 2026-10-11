"""Checks for turbo_moc.reporting and the MoC/workflow logging built on it:
formatting limits, handler hygiene, throttled stage progress (fake clock),
retry/warning reporting (mocked solver), callback compatibility, and the
shape of a small real solver run's output."""

import io
import logging
import math

import pytest

import turbo_moc.api as api
from turbo_moc.progress import make_progress_reporter
from turbo_moc.reporting import (
    LINE_WIDTH,
    ReportFormatter,
    StageProgress,
    format_pressure,
    get_logger,
    logging_context,
)


def _record(message, level=logging.INFO, exc_info=None):
    return logging.LogRecord("turbo_moc.test", level, __file__, 1, message, None, exc_info)


# ------------------------------------------------------------------ formatting
def test_formatter_prefixes_severity_and_wraps_without_breaking_words():
    formatter = ReportFormatter()
    assert formatter.format(_record("  plain info")) == "  plain info"
    assert formatter.format(_record("careful", logging.WARNING)) == "[WARNING] careful"
    path = "C:/" + "very_long_directory_name/" * 6 + "file.json"
    text = formatter.format(_record("  MoC solution saved after a long explanation of what happened: " + path
                                    + " and some more words to wrap\tafter it"))
    lines = text.split("\n")
    assert len(lines) > 1 and "\t" not in text and path in text  # the path is never split
    assert all(len(line) <= LINE_WIDTH for line in lines if path not in line)
    assert all(line.startswith("    ") for line in lines[1:])  # indented continuation


def test_formatter_keeps_tracebacks_intact():
    try:
        raise ValueError("boom " + "x" * 150)
    except ValueError:
        import sys
        text = ReportFormatter().format(_record("failed", logging.ERROR, sys.exc_info()))
    assert text.startswith("[ERROR] failed\nTraceback (most recent call last):")
    assert "ValueError: boom " + "x" * 150 in text  # not wrapped or truncated


def test_pressure_units():
    assert format_pressure(95006.4) == "95.0 kPa"
    assert format_pressure(2.513e6) == "2.513 MPa"
    assert format_pressure(512.0) == "512 Pa"
    assert format_pressure(float("nan")) == "n/a"


# ------------------------------------------------------------------ handlers
def test_logging_context_restores_state_and_never_touches_root(tmp_path):
    root = logging.getLogger()
    root_state = (root.handlers[:], root.level)
    package = logging.getLogger("turbo_moc")
    package_state = (package.handlers[:], package.level, package.propagate)
    for run in range(2):  # repeated runs: no duplicated records
        stream = io.StringIO()
        log_file = tmp_path / f"run{run}.log"
        with logging_context(log_file, stream=stream):
            get_logger("workflows.gmsh.runner").info("workflow record")
            logging.getLogger("turbo_moc.solvers.conventional").info("  MoC record")
        assert stream.getvalue().splitlines() == ["workflow record", "  MoC record"]
        assert log_file.read_text(encoding="utf-8").splitlines() == ["workflow record", "  MoC record"]
        log_file.unlink()  # the handle was closed (Windows refuses to delete open files)
    assert (root.handlers, root.level) == root_state
    assert (package.handlers, package.level, package.propagate) == package_state


def test_logging_context_logs_an_escaping_exception_once_and_reraises():
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with logging_context(stream=stream):
            raise RuntimeError("solver exploded")
    text = stream.getvalue()
    assert text.count("Traceback") == 1 and "RuntimeError: solver exploded" in text


def test_workflow_and_meshing_loggers_share_the_handlers():
    stream = io.StringIO()
    with logging_context(stream=stream):
        get_logger("workflows.fluent.runner").info("fluent")
        logging.getLogger("turbo_moc.meshing.stator_mesh_2d").info("mesh")
    assert stream.getvalue().splitlines() == ["fluent", "mesh"]


# ------------------------------------------------------------------ stage progress
class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_stage_progress_throttles_lines_but_not_callbacks():
    clock, calls, evaluated = FakeClock(), [], []
    stream = io.StringIO()
    with logging_context(stream=stream):
        stage = StageProgress(get_logger("test"), 3, "Kernel region", "3 of 4", lambda f, l: calls.append((f, l)),
                              name="kernel", clock=clock)
        stage.begin()
        for i, fraction in enumerate([0.1, 0.05, float("nan"), 0.4, 1.7, 0.9]):
            clock.now += 0.3

            def detail(i=i):
                evaluated.append(i)
                return f"step {i}"
            stage.update(fraction, detail)
        clock.now = 2.5
        stage.finish("reached the target")
    lines = stream.getvalue().splitlines()
    assert lines[0] == "  MoC [3/4] Kernel region"
    assert lines[-1] == "  MoC [3/4] Kernel region reached the target in 2.50 s"
    progress = lines[1:-1]
    assert progress == ["  MoC kernel: step 3; 1.20 s"]  # once per second: only at t = 1.2 s
    assert evaluated == [3]  # details are built only for logged lines
    fractions = [f for f, _ in calls]
    assert fractions == [0.0, 0.1, 0.1, 0.1, 0.4, 1.0, 1.0, 1.0]  # clamped, monotonic, finite
    assert all(label == "3 of 4" for _, label in calls)


def test_callbacks_do_not_depend_on_the_log_level():
    calls = []
    with logging_context(level="WARNING", stream=io.StringIO()) as logger:
        stage = StageProgress(logging.getLogger("turbo_moc.x"), 1, "Stage", "1 of 4", lambda f, l: calls.append(f))
        stage.begin()
        stage.update(0.5, "ignored")
        stage.finish()
        assert not logger.isEnabledFor(logging.INFO)
    assert calls == [0.0, 0.5, 1.0]


# ------------------------------------------------------------------ progress reporter (Dash)
def test_reporter_routes_all_label_variants():
    seen = []
    reporter = make_progress_reporter(seen.append)
    reporter(0.5, "3 of 4")
    reporter(1.0, "4A of 4")
    reporter(0.2, "1 of 4 (retry 2 of 6)")
    reporter(float("nan"), "2 of 4")
    reporter(1.0, "4 of 4 (finished, not converged)")
    assert seen[0] == ("53%", "Stage 3 of 4")
    assert seen[1] == ("100%", "Stage 4 of 4")
    assert seen[2] == ("1%", "Stage 1 of 4 (retry 2 of 6)")
    assert seen[3] == ("3%", "Stage 2 of 4")
    assert seen[4] == ("100%", "Stage 4 of 4 (finished, not converged)")


# ------------------------------------------------------------------ api: retries and warnings
class _Point:
    def __init__(self, M, p):
        self.M, self.p = M, p


def _fake_solver(fail_attempts, exit_mach):
    attempts = []

    class FakeSolver:
        def __init__(self, config, show_plot=False, fluid_manager=None):
            attempts.append(config["Q0"])
            self.config = config
            self.axis_points_vec = []

        def run_stage_1_sauer(self, use_true_sauer_line=False, progress_callback=None):
            progress_callback(0.5, "1 of 4")
            if len(attempts) <= fail_attempts:
                raise RuntimeError("near-sonic stall")

        def run_stage_2_ivp(self, progress_callback=None):
            progress_callback(1.0, "2 of 4")

        def run_stage_3_kernel(self, progress_callback=None):
            progress_callback(1.0, "3 of 4")

        def run_stage_4_reflex(self, progress_callback=None):
            progress_callback(1.0, "4A of 4")
            design = self.config["Noz_Mach"]
            self.axis_points_vec = [_Point(exit_mach(design), self.config["P_back"])]

        def build_result_data(self, extra_meta=None):
            return {"meta": extra_meta}

    return FakeSolver, attempts


def _design(monkeypatch, solver_class):
    monkeypatch.setattr(api, "ConventionalSolver", solver_class)
    monkeypatch.setattr(api, "get_fluid_manager", lambda *args, **kwargs: object())
    calls, stream = [], io.StringIO()
    with logging_context(stream=stream):
        try:
            data = api.design_nozzle(fluid_name="Nitrogen", P0=5e5, Q0=0.0, p_back=1e5,
                                     use_true_sauer_line=True, progress_callback=lambda f, l: calls.append((f, l)))
        except api.NozzleDesignError as error:
            data = error
    return data, calls, stream.getvalue()


def test_retries_are_reported_and_tagged(monkeypatch):
    solver, attempts = _fake_solver(fail_attempts=1, exit_mach=lambda design: design)
    data, calls, text = _design(monkeypatch, solver)
    assert attempts == [0.0, 0.005] and data["converged"]
    assert "[WARNING]   MoC attempt 1 of 6 failed with Q0 = 0 (near-sonic stall). Retrying with Q0 = 0.005." in text
    assert ("1 of 4", 0.5) == (calls[0][1], calls[0][0])  # first attempt untagged
    assert calls[1] == (0.5, "1 of 4 (retry 2 of 6)")
    assert calls[-1] == (1.0, "4 of 4 (finished; retry 2 of 6)")
    assert "Inlet quality     0.005 (retried from 0)" in text


def test_nonconverged_result_is_a_warning_not_a_success(monkeypatch):
    solver, _ = _fake_solver(fail_attempts=0, exit_mach=lambda design: 0.5 * design)
    data, calls, text = _design(monkeypatch, solver)
    assert not data["converged"]
    assert "[WARNING]   MoC result not converged" in text and "Not converged" in text
    assert calls[-1] == (1.0, "4 of 4 (finished, not converged; retry 6 of 6)")


def test_final_failure_propagates_without_being_logged_by_the_library(monkeypatch):
    solver, _ = _fake_solver(fail_attempts=99, exit_mach=lambda design: design)
    error, calls, text = _design(monkeypatch, solver)
    assert isinstance(error, api.NozzleDesignError)
    assert "Traceback" not in text and text.count("[WARNING]   MoC attempt") == 5
    assert all(f < 1.0 or "finished" not in l for f, l in calls)  # no success implied


# ------------------------------------------------------------------ one small real run
@pytest.mark.parametrize("solver", ["conventional", "mln"])
def test_real_run_output_is_compact(solver):
    stream = io.StringIO()
    with logging_context(stream=stream):
        data = api.design_nozzle(fluid_name="Nitrogen", P0=5e5, T0=300.0, p_back=1e5, solver=solver, n=7)
    text = stream.getvalue()
    lines = text.splitlines()
    assert data["converged"] and math.isfinite(data["exit_M"])
    assert "\r" not in text and "\t" not in text and "####" not in text and "\x1b" not in text
    assert all(len(line) <= LINE_WIDTH for line in lines)
    # Fixed lines are bounded; throttled progress lines (at most one per second
    # per stage) are added only when the machine is slow.
    progress = [line for line in lines if "% of the" in line]
    assert "\n\n\n" not in text and len(lines) - len(progress) <= 20
    assert len(progress) <= data["runtime_s"] + 2
    starts = [line for line in lines if line.startswith("  MoC [") and " in " not in line]
    assert [line[:11] for line in starts] == ["  MoC [1/4]", "  MoC [2/4]", "  MoC [3/4]", "  MoC [4/4]"]
    assert lines[-1].startswith("    Total runtime")
