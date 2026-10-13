import pytest

pytest.importorskip("cadquery")

from turbo_moc.geometry import cad_export, nozzle, nozzle_cad
from turbo_moc.geometry.blade_row import build_design


def _assert_closed(report):
    for item in report["shapes"].values():
        assert item["solids"] >= 1 and item["valid"] and item["watertight"]


def test_axial_row_exports_closed_solids(small_design, tmp_path):
    wall, config = small_design
    design = build_design(wall, config)
    settings = dict(cad_export.CAD_DEFAULTS, span_sections=5)
    report = cad_export.export_row(design, settings, tmp_path, row_id="stator_1")
    _assert_closed(report)
    assert report["volume_checks"]["fluid_plus_blade_over_passage"] == pytest.approx(1, abs=5e-3)
    assert report["volume_checks"]["fused_over_expected"] == pytest.approx(1, abs=5e-3)
    assert report["blade_count"] == 12
    assert report["reimport"]["assembly_parts"]["solids"] == 13
    assert report["reimport"]["assembly_fused"]["solids"] == 1
    assert all((tmp_path / name).is_file() for name in report["files"].values())


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_nozzle_exports_fluid_and_body(small_design, tmp_path, delta):
    wall, _ = small_design
    settings = dict(nozzle.NOZZLE_DEFAULTS, convergent_length=0.004, planar_depth=0.004, wall_thickness=0.0005)
    report = nozzle_cad.export_nozzle(wall, {"delta_flow": delta, "rho_d": 0.1, "y_t": 0.002}, settings, tmp_path)
    _assert_closed(report)
    assert report["volume_checks"]["fluid_over_contour_integral"] == pytest.approx(1, abs=1e-2)
    assert all(item["solids"] == 1 for item in report["reimport"].values())


@pytest.mark.parametrize("name,value", [("span_sections", 2), ("disk_thickness", 0), ("linear_blades", 0),
                                        ("fluid_annulus", "yes")])
def test_invalid_cad_settings(name, value):
    with pytest.raises(ValueError, match=name):
        cad_export.validate_settings(dict(cad_export.CAD_DEFAULTS, **{name: value}))
