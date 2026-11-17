"""Freeze a MoC wall and apply DOE geometry changes without re-solving MoC."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.configuration.planar import from_project
from turbo_moc.workflows.nozzle import solve_nozzle


_PERTURBATION_KEYS = (
    "control_point_displacements",
    "control_point_offsets",
    "geometry_uncertainty",
)


def _row(project, row_id):
    for row in project.get("rows", []):
        if row.get("id") == row_id:
            return row
    raise ValueError(f"Unknown row id: {row_id!r}.")


def _project_only(project):
    if not isinstance(project, dict) or not {"machine", "workflow", "rows"} <= set(project):
        raise ValueError("A frozen geometry input must contain machine, workflow and rows.")
    return {key: deepcopy(project[key]) for key in ("machine", "workflow", "rows")}


def _without_perturbations(project, row_id):
    result = _project_only(project)
    geometry = _row(result, row_id)["geometry"]["blade_geometry"]
    for key in _PERTURBATION_KEYS:
        geometry.pop(key, None)
    return result


def _wall(nozzle):
    try:
        x = np.asarray(nozzle["wall_final"]["x"], dtype=float)
        y = np.asarray(nozzle["wall_final"]["y"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen baseline must contain a finite wall_final contour.") from exc
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Frozen baseline must contain a finite wall_final contour.")
    return x, y


def freeze_baseline(project, *, row_id="stator_1", solver=solve_nozzle):
    """Solve MoC once and retain the reference project's perturbed blade.

    The CP displacement in the baseline YAML is a blade-design input, not a
    MoC input. It is therefore removed only from the temporary solver
    configuration and retained in the frozen artifact used to build the
    baseline mesh and all DOE cases.
    """
    source_project = _project_only(project)
    solver_project = _without_perturbations(source_project, row_id)
    config = from_project(source_project, row_id=row_id)
    solver_config = from_project(solver_project, row_id=row_id)
    nozzle = solver(solver_config["nozzle"])
    _wall(nozzle)
    return {
        "version": 1,
        "row_id": row_id,
        "project": source_project,
        "config": config,
        "solver_config": solver_config,
        "nozzle": nozzle,
    }


def save_baseline(baseline, path):
    """Atomically publish a frozen baseline artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_baseline(path):
    """Load and validate a previously frozen baseline artifact."""
    try:
        baseline = json.loads(Path(path).read_text(encoding="utf-8"))
        _wall(baseline["nozzle"])
        if baseline.get("version") != 1:
            raise ValueError("Unsupported frozen baseline version.")
        return baseline
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read frozen baseline {path}.") from exc


def case_from_baseline(baseline, project, *, row_id=None):
    """Resolve a case using the frozen wall and case-only perturbations/conditions."""
    row_id = row_id or baseline["row_id"]
    project = _project_only(project)
    case_config = from_project(project, row_id=row_id)
    baseline_nozzle = baseline["config"]["nozzle"]
    if "backend" in baseline_nozzle and "backend" not in case_config["nozzle"]:
        case_config["nozzle"]["backend"] = baseline_nozzle["backend"]
    blade = deepcopy(baseline["config"]["blade"])
    case_blade = case_config["blade"]
    for key in _PERTURBATION_KEYS:
        if key in case_blade:
            blade[key] = deepcopy(case_blade[key])
    case_config["blade"] = blade
    if "row_configuration" in case_config:
        case_config["row_configuration"]["blade_geometry"] = deepcopy(blade)
    return case_config


def case_thermodynamic_nozzle(baseline, case_config):
    """Attach case inlet state to the frozen result for barotropy/Fluent only."""
    nozzle = deepcopy(baseline["nozzle"])
    parameters = solver_parameters(case_config["nozzle"])
    metadata = dict(nozzle.get("meta", {}))
    for name in ("P0", "T0", "Q0"):
        metadata.pop(name, None)
    for name in ("P0", "T0", "Q0"):
        if name in parameters:
            metadata[name] = float(parameters[name])
    nozzle["meta"] = metadata
    return nozzle


def wall_is_unchanged(baseline, nozzle):
    """Return whether a downstream case still references the frozen wall."""
    base_x, base_y = _wall(baseline["nozzle"])
    case_x, case_y = _wall(nozzle)
    return bool(np.array_equal(base_x, case_x) and np.array_equal(base_y, case_y))
