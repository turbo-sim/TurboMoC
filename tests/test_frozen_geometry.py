from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from turbo_moc.fluent.frozen_geometry import (
    case_from_baseline,
    case_thermodynamic_nozzle,
    freeze_baseline,
    wall_is_unchanged,
)


HERE = Path(__file__).resolve().parents[1]
BASELINE = HERE / "examples" / "fluent_2D_geometry_perturbations" / "base_line_fluent_stator_2D.yaml"
CASE = HERE / "examples" / "fluent_2D_geometry_perturbations" / "cases" / "case_1" / "case_1.yaml"


def project(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def nozzle():
    return {
        "converged": True,
        "wall_final": {"x": [0.0, 0.001], "y": [0.001, 0.002]},
        "meta": {"P0": 1090000.0, "Q0": 0.42},
    }


def test_freeze_retains_baseline_cp_displacements_but_solves_moc_once():
    calls = []

    def solver(parameters):
        calls.append(deepcopy(parameters))
        return nozzle()

    baseline = freeze_baseline(project(BASELINE), solver=solver)

    assert len(calls) == 1
    reference_blade = project(BASELINE)["rows"][0]["geometry"]["blade_geometry"]
    assert baseline["config"]["blade"]["control_point_displacements"] == reference_blade[
        "control_point_displacements"
    ]
    assert "control_point_displacements" not in baseline["solver_config"]["blade"]
    assert "geometry_uncertainty" not in baseline["solver_config"]["blade"]


def test_case_perturbations_use_frozen_baseline_parameters():
    baseline = freeze_baseline(project(BASELINE), solver=lambda _: nozzle())
    case = case_from_baseline(baseline, project(CASE))

    assert case["nozzle"]["backend"] == baseline["config"]["nozzle"]["backend"]
    assert case["blade"]["metal_angle_out"] == baseline["config"]["blade"]["metal_angle_out"]
    assert case["blade"]["control_point_displacements"] == project(CASE)["rows"][0]["geometry"]["blade_geometry"][
        "control_point_displacements"
    ]
    assert wall_is_unchanged(baseline, baseline["nozzle"])


def test_case_conditions_are_attached_only_to_thermodynamic_nozzle():
    baseline = freeze_baseline(project(BASELINE), solver=lambda _: nozzle())
    case = case_from_baseline(baseline, project(CASE))
    thermodynamic = case_thermodynamic_nozzle(baseline, case)

    assert np.array_equal(thermodynamic["wall_final"]["x"], baseline["nozzle"]["wall_final"]["x"])
    assert np.array_equal(thermodynamic["wall_final"]["y"], baseline["nozzle"]["wall_final"]["y"])
    assert thermodynamic["meta"]["P0"] == case["nozzle"]["inlet_property_1"]
    assert thermodynamic["meta"]["Q0"] == case["nozzle"]["inlet_property_2"]
