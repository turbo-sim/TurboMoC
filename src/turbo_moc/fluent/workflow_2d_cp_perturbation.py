"""Freeze a MoC baseline, perturb its blade, and mesh the perturbed design."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from turbo_moc.configuration.planar import build_geometry
from turbo_moc.fluent.barotropic_model import generate_barotropic_model
from turbo_moc.fluent.common import validate_reporting
from turbo_moc.fluent.frozen_geometry import (
    case_from_baseline,
    case_thermodynamic_nozzle,
    freeze_baseline,
    save_baseline,
)
from turbo_moc.fluent.workflow_2d import load_dependencies, save_fluent_case
from turbo_moc.geometry import passage_2d as geometry
from turbo_moc.meshing.mesh_2d import create_mesh
from turbo_moc.reporting import (
    display_path,
    get_logger,
    heading,
    logging_context,
    step,
    summary as log_summary,
)

BASE_LINE_CPS_FILE = "frozen_baseline.json"
LOG_FILE = "workflow.log"
logger = get_logger("fluent.perturbation_runner")


@dataclass(frozen=True)
class NewCaseResult:
    """Outputs produced while building and preparing a perturbed case."""

    case_project: dict[str, Any]
    case_config: dict[str, Any]
    thermodynamic_nozzle: dict[str, Any]
    mesh_dir: Path
    fluent_case: Any
    figure_path: Path
    moved_count: int


def _read_workflow(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        workflow = yaml.safe_load(stream)
    if not isinstance(workflow, dict):
        raise ValueError("Workflow YAML must be a mapping.")
    return workflow


def _plot_geometry_comparison(
    baseline_blade: dict[str, Any],
    case_blade: dict[str, Any],
    output: Path,
) -> int:
    baseline_points = np.asarray(baseline_blade["control_points"], dtype=float)
    case_points = np.asarray(case_blade["control_points"], dtype=float)
    figure, axis = plt.subplots(figsize=(12, 6))
    for blade, label, style in (
        (baseline_blade, "Frozen baseline", "k--"),
        (case_blade, "Perturbed case", "r-"),
    ):
        curve = np.column_stack((blade["blade_curve"]["x"], blade["blade_curve"]["y"]))
        axis.plot(curve[:, 0], curve[:, 1], style, linewidth=1.2, label=label)
    moved = np.linalg.norm(case_points - baseline_points, axis=1) > 1e-12
    axis.scatter(
        baseline_points[:, 0], baseline_points[:, 1],
        color="black", s=12, label="Baseline control points", zorder=3,
    )
    axis.scatter(
        case_points[moved, 0], case_points[moved, 1],
        color="red", s=18, label="Moved control points", zorder=4,
    )
    for start, end in zip(baseline_points[moved], case_points[moved]):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 0.8})
    axis.set_aspect("equal")
    axis.set_xlabel("Pitchwise coordinate [mm]")
    axis.set_ylabel("Axial coordinate [mm]")
    axis.set_title("Frozen baseline and case control-point perturbation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in (".png", ".svg"):
        figure.savefig(output.with_suffix(suffix), dpi=240)
    plt.close(figure)
    return int(moved.sum())


def define_baseline_case(baseline_file: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze and save the baseline case."""
    baseline_file = Path(baseline_file).resolve()
    baseline_project = _read_workflow(baseline_file)
    baseline = freeze_baseline(baseline_project)
    save_baseline(baseline, output_dir / BASE_LINE_CPS_FILE)
    return baseline


def _save_verification_file(
    baseline: dict[str, Any],
    baseline_file: Path,
    case_file: Path,
    case_result: NewCaseResult,
    output_dir: Path,
) -> dict[str, Any]:
    """Create and save the verification manifest for the generated case."""
    manifest = {
        "baseline_file": str(baseline_file),
        "case_file": str(case_file),
        "frozen_baseline": str(output_dir / BASE_LINE_CPS_FILE),
        "mesh": str(case_result.mesh_dir / case_result.case_config["mesh"]["cgns"]["filename"]),
        "fluent_case": str(case_result.fluent_case) if case_result.fluent_case else None,
        "geometry_figure": str(case_result.figure_path.with_suffix(".svg")),
        "moved_control_points": case_result.moved_count,
        "moc_reference": {
            "inlet_property_1": baseline["project"]["rows"][0]["geometry"][
                "method_of_characteristics"
            ]["inlet_property_1"],
            "inlet_property_2": baseline["project"]["rows"][0]["geometry"][
                "method_of_characteristics"
            ]["inlet_property_2"],
        },
        "case_operating_conditions": deepcopy(case_result.case_config["nozzle"]),
        "barotropic": deepcopy(case_result.case_project.get("barotropic")),
        "fluent": deepcopy(case_result.case_project.get("fluent")),
        "wall_frozen": bool(
            np.array_equal(
                case_result.thermodynamic_nozzle["wall_final"]["x"],
                baseline["nozzle"]["wall_final"]["x"],
            )
            and np.array_equal(
                case_result.thermodynamic_nozzle["wall_final"]["y"],
                baseline["nozzle"]["wall_final"]["y"],
            )
        ),
    }
    (output_dir / "verification.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_new_case(
    baseline: dict[str, Any],
    case_file: Path,
    output_dir: Path,
) -> NewCaseResult:
    """Build, mesh, and prepare the perturbed case."""
    case_file = Path(case_file).resolve()
    case_project = _read_workflow(case_file)
    case_config = case_from_baseline(baseline, case_project)
    thermodynamic_nozzle = case_thermodynamic_nozzle(baseline, case_config)
    baseline_blade, _ = build_geometry(baseline["nozzle"], baseline["config"])
    case_blade, case_curves = build_geometry(baseline["nozzle"], case_config)
    if not geometry.assess_clearance(case_blade, case_curves)["contained"]:
        raise ValueError("Cannot mesh: a periodic boundary intersects the perturbed blade.")

    figure_path = output_dir / "baseline_vs_perturbed_geometry"
    moved_count = _plot_geometry_comparison(baseline_blade, case_blade, figure_path)
    mesh_dir = output_dir / "gmsh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    create_mesh(
        case_blade,
        case_curves,
        case_config["mesh"],
        mesh_dir,
        show_gui=bool(case_config["mesh"].get("show_gmsh", False)),
    )

    workflow = {
        "output_dir": output_dir,
        "run_case": bool(case_project["run_case"]),
        "prepare_only": bool(case_project["prepare_only"]),
        "fluent": deepcopy(case_project["fluent"]),
        "barotropic": deepcopy(case_project["barotropic"]),
    }
    fluent_case = None
    if not workflow["prepare_only"]:
        barotropy, pyfluent = load_dependencies(False)
        expressions, state = generate_barotropic_model(
            barotropy, case_config, thermodynamic_nozzle,
            workflow["barotropic"], output_dir / "barotropy",
        )
        fluent_case = save_fluent_case(
            pyfluent, workflow,
            mesh_dir / case_config["mesh"]["cgns"]["filename"],
            state, expressions, float(case_blade["pitch"]) * 0.001,
        )

    return NewCaseResult(
        case_project, case_config, thermodynamic_nozzle, mesh_dir,
        fluent_case, figure_path, moved_count,
    )


def main(
    baseline_file: Path,
    case_file: Path,
    output_dir: Path | None = None,
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the workflow with optional reuse of an already defined baseline."""
    baseline_file = Path(baseline_file).resolve()
    case_file = Path(case_file).resolve()
    output_dir = Path(output_dir or case_file.parent / "output").resolve()
    case_project = _read_workflow(case_file)
    workflow_settings = case_project.get("workflow")
    if not isinstance(workflow_settings, dict):
        raise ValueError("Workflow YAML must define a workflow mapping.")
    reporting = {"level": workflow_settings.get("log_level", "INFO"), "filename": LOG_FILE}
    validate_reporting(reporting)
    log_file = output_dir / reporting["filename"]
    reusing_baseline = baseline is not None
    total_steps = 2 if reusing_baseline else 3

    with logging_context(log_file, reporting["level"]):
        heading(logger, "Fluent perturbation workflow")
        log_summary(logger, "Run settings", {
            "Baseline configuration": display_path(baseline_file),
            "Case configuration": display_path(case_file),
            "Output directory": display_path(output_dir),
            "Execution log": display_path(log_file),
        })
        if not reusing_baseline:
            step(logger, 1, total_steps, "Define and save the baseline case")
            baseline = define_baseline_case(baseline_file, output_dir)
        else:
            logger.info("Using the provided frozen baseline.")
        step_number = 1 if reusing_baseline else 2
        step(logger, step_number, total_steps, "Generate the perturbed case")
        case_result = run_new_case(baseline, case_file, output_dir)
        step(logger, step_number + 1, total_steps, "Save verification results")
        results = _save_verification_file(
            baseline, baseline_file, case_file, case_result, output_dir
        )
        heading(logger, "Results")
        log_summary(logger, "Fluent perturbation workflow completed", {
            "Frozen baseline": display_path(output_dir / BASE_LINE_CPS_FILE),
            "Mesh": display_path(
                case_result.mesh_dir / case_result.case_config["mesh"]["cgns"]["filename"]
            ),
            "Verification file": display_path(output_dir / "verification.json"),
            "Moved control points": case_result.moved_count,
            "Output directory": display_path(output_dir),
            "Execution log": display_path(log_file),
        })
        return results
