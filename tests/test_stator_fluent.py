"""Fast Fluent workflow checks; no Fluent license or companion checkout needed."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import jaxprop as jxp
import pandas as pd
import pytest


@pytest.fixture
def workflow_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.fluent.run_fluent_case")


def test_barotropic_fit_preserves_moc_inlet_and_has_positive_properties(
    workflow_module,
):
    model = importlib.import_module("examples.fluent.barotropic_model")
    workflow, config = workflow_module.read_workflow(workflow_module.CONFIG_FILE)
    inputs = {**config["nozzle"], "Q0": 0.0}
    # Preserve the quality actually used by MoC, rather than its original input.
    fluid, inlet, quality = model.resolve_inlet_state(
        jxp, inputs, {"meta": {"Q0": 0.025}}
    )
    assert quality == 0.025
    p_min = float(inlet.p) / workflow["barotropic"]["pressure_ratio"]
    fitter = SimpleNamespace(
        p_in=float(inlet.p), p_out=p_min, poly_breakpoints=[1.0, p_min / float(inlet.p)]
    )
    report = model.fit_wide_isentrope(
        jxp, fitter, fluid, float(inlet.s), workflow["barotropic"]["polynomial_degree"]
    )
    assert report["p_min"] == p_min
    for name in model.POSITIVE_PROPERTIES:
        assert report["max_relative_error"][name] <= 0.01
        assert all(
            polynomial(edge) > 0
            for polynomial, edge in zip(
                fitter.poly_handles[name], fitter.poly_breakpoints[1:]
            )
        )


@pytest.mark.parametrize(
    "setup_fails", [False, True], ids=["create-only", "setup-error"]
)
def test_case_creation_never_solves_and_always_closes_fluent(
    workflow_module, monkeypatch, tmp_path, setup_fails
):
    workflow, _ = workflow_module.read_workflow(workflow_module.CONFIG_FILE)
    workflow.update(output_dir=tmp_path, run_case=False)
    options = workflow["fluent"]
    folder = tmp_path / "fluent"
    folder.mkdir()
    case = folder / options["case_filename"]
    case.write_text("previous case")
    transcript = folder / f"{case.name.removesuffix('.cas.h5')}.trn"
    transcript.write_text("previous residual history")
    solver = MagicMock()
    solver.get_fluent_version.return_value = "24.2"
    solver.settings.solution.methods.high_order_term_relaxation.get_state.return_value = (
        {}
    )
    solver.settings.file.write_case.side_effect = lambda file_name: Path(
        file_name
    ).write_bytes(b"case")
    configure = Mock(return_value={"inlet": "inlet"})
    if setup_fails:
        configure.side_effect = RuntimeError("setup failure")
    monkeypatch.setattr(workflow_module, "configure_fluent", configure)
    pyfluent = SimpleNamespace(launch_fluent=Mock(return_value=solver))
    if setup_fails:
        with pytest.raises(RuntimeError, match="setup failure"):
            workflow_module.save_fluent_case(
                pyfluent, workflow, tmp_path / "mesh.cgns", {}, {}, 0.05
            )
        assert case.read_text() == "previous case"
        solver.settings.file.write_case.assert_not_called()
    else:
        workflow_module.save_fluent_case(
            pyfluent, workflow, tmp_path / "mesh.cgns", {}, {}, 0.05
        )
        solver.settings.file.write_case.assert_called_once_with(file_name=str(case))
        assert (folder / "setup_summary.json").is_file()
        solver.settings.file.stop_transcript.assert_called_once()
    assert transcript.read_text() == ""
    assert list(folder.glob("*.cas.h5")) == [case]
    solver.settings.solution.initialization.initialize.assert_not_called()
    solver.settings.solution.run_calculation.iterate.assert_not_called()
    solver.settings.file.write_data.assert_not_called()
    solver.settings.file.write_case_data.assert_not_called()
    solver.exit.assert_called_once()


@pytest.mark.parametrize("outcome", ["converged", "exception", "silent-error"])
def test_solver_strategy_and_final_exports(
    workflow_module, monkeypatch, tmp_path, outcome
):
    workflow, _ = workflow_module.read_workflow(workflow_module.CONFIG_FILE)
    options = workflow["fluent"]
    # Two blocks exercise order switching and advancing after early convergence.
    # Keep the scenario independent of the user-editable example stage budgets.
    options["solution_strategy"] = [
        {"iterations": 100, "time_scale_factor": 0.5, "order": "first"},
        {"iterations": 200, "time_scale_factor": 2.0, "order": "second"},
    ]
    options["plot_residuals"] = False
    options["periodic_instancing"]["enabled"] = True
    for name in ("pressure_contour", "mach_contour"):
        options[name]["enabled"] = True
    state = {"P0": 251300.0, "p_min": 31412.5}
    solver = MagicMock()
    diagnostics = Mock(
        return_value={
            "min-density": 5.0,
            "min-viscosity": 1e-5,
            "min-pressure": 45000.0,
            "max-pressure": 251300.0,
            "relative_mass_imbalance": 1e-8,
        }
    )
    monkeypatch.setattr(workflow_module, "solution_diagnostics", diagnostics)
    rows = []
    # Mock barotropy's reader, keeping our real CSV/summary processing and final
    # transcript checks. No external Fluent or barotropy installation is needed.
    reader = Mock(
        side_effect=lambda filename: (pd.DataFrame(rows), outcome != "silent-error")
    )
    monkeypatch.setitem(
        sys.modules, "barotropy", SimpleNamespace(read_residual_file=reader)
    )
    stem = options["case_filename"].removesuffix(".cas.h5")
    transcript = tmp_path / f"{stem}.trn"
    observed = []

    def iterate(iter_count):
        initialization = solver.settings.solution.initialization
        assert initialization.initialization_type == "hybrid"
        initialization.initialize.assert_called_once()
        diagnostics.assert_not_called()
        reader.assert_not_called()
        if outcome == "exception":
            raise RuntimeError("floating point exception")
        settings = solver.settings.solution
        checks = [
            settings.monitor.residual.equations[name].check_convergence
            for name in options["residual_equations"]
        ]
        observed.append(
            (
                iter_count,
                settings.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor,
                settings.methods.discretization_scheme["mom"],
                checks,
            )
        )
        # Simulate Fluent reaching tolerance before each block's iteration limit.
        iteration = (len(rows) + 1) * 3 if all(checks) else iter_count
        rows.append(
            {
                "iter": iteration,
                **{name: 1e-8 for name in options["residual_equations"]},
            }
        )
        with transcript.open("a") as stream:
            equations = options["residual_equations"]
            stream.write("iter " + " ".join(equations) + " time/iter\n")
            stream.write(
                f" {iteration} " + " ".join("1e-8" for _ in equations) + " 0:00:00 0\n"
            )
            if outcome == "silent-error":
                stream.write("Error: floating point exception\n")

    solver.settings.solution.run_calculation.iterate.side_effect = iterate

    def save_final_pair(file_name):
        diagnostics.assert_called_once()
        reader.assert_called_once()
        settings = solver.settings.solution
        assert settings.run_calculation.iterate.call_count == 2
        assert settings.methods.discretization_scheme["mom"] == "second-order-upwind"
        assert (
            settings.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor
            == 2.0
        )
        Path(file_name).write_bytes(b"final case")
        (tmp_path / f"{stem}.dat.h5").write_bytes(b"final data")

    solver.settings.file.write_case_data.side_effect = save_final_pair
    if outcome == "converged":
        workflow_module.run_stages(solver, options, tmp_path, state)
        checks = [True] * len(options["residual_equations"])
        assert observed == [
            (100, 0.5, "first-order-upwind", checks),
            (200, 2.0, "second-order-upwind", checks),
        ]
        diagnostics.assert_called_once()
        reader.assert_called_once()
        assert (tmp_path / "residuals.csv").is_file()
        solver.settings.file.write_case_data.assert_called_once_with(
            file_name=str(tmp_path / f"{stem}.cas.h5")
        )
        assert (tmp_path / f"{stem}.cas.h5").read_bytes() == b"final case"
        assert (tmp_path / f"{stem}.dat.h5").read_bytes() == b"final data"
        # Cover both final images and periodic display copies in the same run.
        solver.settings.setup.cell_zone_conditions.fluid.get_object_names.return_value = [
            "fluid_cells"
        ]
        graphics = solver.settings.results.graphics
        graphics.picture.save_picture.side_effect = lambda file_name: Path(
            file_name
        ).write_bytes(b"image")
        images = workflow_module.export_contours(solver, options, tmp_path)
        assert [path.name for path in images] == [
            options[name]["filename"] for name in ("pressure_contour", "mach_contour")
        ]
        assert all(path.stat().st_size > 0 for path in images)
        assert graphics.periodic_instancing["fluid_cells"].display.call_count == 2
    else:
        with pytest.raises(RuntimeError, match="floating point|Fluent diverged"):
            workflow_module.run_stages(solver, options, tmp_path, state)
        diagnostics.assert_not_called()
        solver.settings.file.write_data.assert_not_called()
        solver.settings.file.write_case_data.assert_not_called()
        expected_calls = 1 if outcome == "exception" else 2
        assert (
            solver.settings.solution.run_calculation.iterate.call_count
            == expected_calls
        )
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["status"] == ("completed" if outcome == "converged" else "failed")
    if outcome == "converged":
        assert summary["converged"] is True
        assert summary["last_iteration"] == 6
        assert [stage["actual_iterations"] for stage in summary["stage_results"]] == [
            3,
            3,
        ]
        assert summary["case_file"] == str(tmp_path / f"{stem}.cas.h5")
        assert summary["data_file"] == str(tmp_path / f"{stem}.dat.h5")
    else:
        assert "case_file" not in summary and "data_file" not in summary
    solver.settings.file.write_case.assert_not_called()
    solver.settings.file.write_data.assert_not_called()


@pytest.mark.parametrize(
    "outcome", ["iteration_limit", "invalid_fields", "write_error"]
)
def test_final_pair_save_reports_completion_and_failures(
    workflow_module, monkeypatch, tmp_path, outcome
):
    workflow, _ = workflow_module.read_workflow(workflow_module.CONFIG_FILE)
    options = workflow["fluent"]
    solver = MagicMock()
    stem = options["case_filename"].removesuffix(".cas.h5")
    (tmp_path / f"{stem}.trn").write_text("")
    monkeypatch.setattr(
        workflow_module,
        "residual_history",
        lambda *args, **kwargs: {
            "last_residuals": {
                equation: 1e-2 for equation in options["residual_equations"]
            }
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "solution_diagnostics",
        lambda *args: {
            "min-density": -1.0 if outcome == "invalid_fields" else 1.0,
            "min-viscosity": 1e-5,
            "min-pressure": 95000.0,
            "max-pressure": 250000.0,
            "relative_mass_imbalance": 1e-5,
        },
    )
    state = {"p_min": 50260.0, "P0": 251300.0}
    if outcome == "write_error":
        solver.settings.file.write_case_data.side_effect = RuntimeError("write failed")
    if outcome == "iteration_limit":
        workflow_module.run_stages(solver, options, tmp_path, state)
    else:
        with pytest.raises(RuntimeError, match="Invalid property|write failed"):
            workflow_module.run_stages(solver, options, tmp_path, state)

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    if outcome == "invalid_fields":
        solver.settings.file.write_case_data.assert_not_called()
    else:
        solver.settings.file.write_case_data.assert_called_once_with(
            file_name=str(tmp_path / f"{stem}.cas.h5")
        )
    if outcome == "iteration_limit":
        assert summary["status"] == "completed"
        assert summary["converged"] is False
        assert summary["case_file"] == str(tmp_path / f"{stem}.cas.h5")
        assert summary["data_file"] == str(tmp_path / f"{stem}.dat.h5")
    else:
        assert summary["status"] == "failed"
        assert "case_file" not in summary and "data_file" not in summary
        if outcome == "write_error":
            assert summary["failed_stage"] == "saving_final_solution"
