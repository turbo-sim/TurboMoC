"""Configuration migration, validation and stage invalidation for quad quality."""
from copy import deepcopy

import pytest
import yaml

from turbo_moc.configuration import project as projects
from turbo_moc.app import workflow
from turbo_moc.meshing.mesh_2d import validate_mesh_settings


def test_older_project_defaults_both_rows_and_retains_explicit_quality():
    original = projects.set_design_mode(projects.default_project(), "stator_rotor")
    original["rows"][0]["mesh"]["numerics"].pop("recombine_minimum_quality")
    original["rows"][1]["mesh"]["numerics"]["recombine_minimum_quality"] = 0.2
    before = deepcopy(original)
    normalized = projects.validate_project(projects.normalize_project(original))
    assert original == before
    assert [r["mesh"]["numerics"]["recombine_minimum_quality"] for r in normalized["rows"]] == [0.1, 0.2]
    assert projects.normalize_project(yaml.safe_load(yaml.safe_dump(normalized))) == normalized
    normalized["rows"][1]["mesh"]["numerics"].pop("recombine_minimum_quality")
    assert projects.normalize_project(normalized)["rows"][1]["mesh"]["numerics"]["recombine_minimum_quality"] == 0.1


@pytest.mark.parametrize("value", [True, False, None, "0.1", float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_recombination_quality_fails_before_meshing(value):
    mesh = projects.default_project()["rows"][0]["mesh"]
    mesh["numerics"]["recombine_minimum_quality"] = value
    with pytest.raises(ValueError, match="recombine_minimum_quality"):
        validate_mesh_settings(mesh)


@pytest.mark.parametrize("value", [0, 0.01, 0.1, 0.2, 1])
def test_quality_accepts_range_and_only_invalidates_mesh_stages(value):
    config = projects.row_configuration(projects.default_project(), "stator_1")
    before = workflow.keys(config)
    config["mesh"]["numerics"]["recombine_minimum_quality"] = value
    from turbo_moc.configuration.row import validate_mesh_configuration
    validate_mesh_configuration(config["mesh"])
    after = workflow.keys(config)
    assert before["nozzle"] == after["nozzle"]
    assert before["geometry"] == after["geometry"]
    assert (before["reference"] == after["reference"]) == (value == 0.1)
    assert (before["volume"] == after["volume"]) == (value == 0.1)


def test_app_quality_control_has_bounds_and_hover_help():
    from turbo_moc.app.application import controls
    label, = controls({"recombine_minimum_quality": 0.1}, "mesh.numerics", "rotor_1")
    assert len(label.children) == 2  # Label + input, without an explanatory text block.
    widget = label.children[1]
    assert widget.id["path"] == "mesh.numerics.recombine_minimum_quality"
    assert widget.value == 0.1 and widget.min == 0 and widget.max == 1
    assert "Gmsh" in label.title and "not a maximum skewness" in label.title
