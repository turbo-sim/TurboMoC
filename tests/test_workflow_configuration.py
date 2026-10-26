"""App downloads are reusable workflow inputs; Fluent adds only automation."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from turbo_moc.configuration import planar, project, row
from turbo_moc.configuration.automation import PROJECT_FIELDS
from turbo_moc.fluent import workflow_2d, workflow_3d
from turbo_moc.geometry import blade_row

ROOT = Path(__file__).resolve().parents[1]


def content(family, dimension):
    folder = ROOT / f"examples/{family}_{dimension}"
    if (folder / "stator").is_dir():
        folder = folder / "stator"
    path = folder / f"workflow_{family}_stator_{dimension}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("dimension,case", [("2D", "linear_stator"), ("3D", "axial_stator")])
def test_examples_share_exact_app_project(dimension, case):
    app = yaml.safe_load((ROOT / f"examples/apps/cases/{case}.yaml").read_text(encoding="utf-8"))
    gmsh, _ = content("gmsh", dimension)
    fluent, _ = content("fluent", dimension)
    assert gmsh == app
    assert PROJECT_FIELDS < fluent.keys()
    assert {k: fluent[k] for k in PROJECT_FIELDS} == app
    assert "design_file" not in fluent
    project.validate_project(project.normalize_project(gmsh))


@pytest.mark.parametrize("dimension,module", [("2D", workflow_2d), ("3D", workflow_3d)])
def test_inline_loading_preserves_app_fields_and_input(dimension, module):
    raw, path = content("fluent", dimension)
    before = deepcopy(raw)
    options, config = module.validate_workflow(raw, path)
    assert raw == before
    canonical = config.get("row_configuration", config)
    expected = row.configuration_from_data({k: raw[k] for k in PROJECT_FIELDS}, row_id="stator_1")
    assert canonical == expected
    assert options["design_file"] == path
    assert options["output_dir"] == path.parent / "output"


@pytest.mark.parametrize("module,dimension", [(workflow_2d, "2D"), (workflow_3d, "3D")])
def test_inline_and_design_file_cannot_conflict(module, dimension):
    raw, path = content("fluent", dimension)
    raw["design_file"] = "elsewhere.yaml"
    with pytest.raises(ValueError, match="never both"):
        module.validate_workflow(raw, path)


@pytest.mark.parametrize("family", ["axial", "radial", "linear"])
def test_app_download_is_direct_gmsh_input(family):
    path = ROOT / f"examples/apps/cases/{family}_stator.yaml"
    adapted = planar.read_configuration(path)
    assert adapted["row_configuration"] == row.read_configuration(path)
    assert set(adapted["mesh"]["cgns"]["boundary_types"]) == {
        "blade", "inlet", "outlet", "periodic_left", "periodic_right"}


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_fluent_2d_rejects_nonphysical_conformal_grid(family):
    raw, path = content("fluent", "2D")
    raw.update(yaml.safe_load((ROOT / f"examples/apps/cases/{family}_stator.yaml").read_text()))
    with pytest.raises(ValueError, match="conformal references"):
        workflow_2d.validate_workflow(raw, path)


def test_2d_geometry_preserves_app_chord_pitch_and_construction():
    raw, _ = content("gmsh", "2D")
    raw["rows"][0]["geometry"]["meridional_channel"]["meridional_chord"] = 27.
    config = planar.from_project(raw)
    x = np.linspace(0., .01, 80)
    nozzle = {"converged": True, "wall_final": {
        "x": x.tolist(), "y": (.002 + .001 * (x / x[-1]) ** 2).tolist()}}
    blade, curves = planar.build_geometry(nozzle, config)
    design = blade_row.build_design(nozzle, config["row_configuration"])
    assert blade["pitch"] == pytest.approx(design.blade["pitch"])
    np.testing.assert_allclose(curves["blade_curve"], design.curves["blade_curve"])
    np.testing.assert_allclose(curves["periodic_left"], design.curves["periodic_left"])
    assert np.ptp(blade_row.planar.blade_contour(blade)[:, 1]) == pytest.approx(27.)
    assert len(curves["construction_camberline"]) > 2


def test_stage_download_retains_shared_interface_and_rejects_rotor():
    raw, path = content("fluent", "3D")
    raw.update(yaml.safe_load((ROOT / "examples/apps/cases/stator_rotor.yaml").read_text()))
    _, config = workflow_3d.validate_workflow(raw, path)
    assert "outlet" in config["assembly"]["interfaces"]
    raw["row_id"] = "rotor_1"
    with pytest.raises(ValueError, match="stator"):
        workflow_3d.validate_workflow(raw, path)
