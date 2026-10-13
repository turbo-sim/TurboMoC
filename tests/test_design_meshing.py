"""Reference placement, volume quality and real Gmsh sweeps/CGNS exports."""
from pathlib import Path
import json

import gmsh
import numpy as np
import pytest
from scipy.interpolate import BSpline

from turbo_moc.geometry import blade_row as geo
from turbo_moc.app import workflow as flow
from turbo_moc.meshing import mesh_3d as mesh
from turbo_moc.meshing.quality import analyze_volume

CASE_DIR = Path(__file__).resolve().parents[1] / "examples/apps/cases"
CASES = {n: CASE_DIR / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}


def test_reference_placement_only_once_and_periodic_pitch_preserved(small_design, monkeypatch):
    nozzle, config = small_design
    config["meridional_channel"]["tip_chord"] = 22
    calls = []
    real = geo.planar.auto_center_flow_domain
    def count(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(geo.planar, "auto_center_flow_domain", count)
    design = geo.build_design(nozzle, config)
    for eta in np.linspace(0, 1, 15):
        blade, curves = design.section(eta)
        assert blade["pitch"] == pytest.approx(design.meridional_channel.pitch)
        assert curves["shift_mm"] == design.curves["shift_mm"]
        assert np.allclose(curves["periodic_right"] - curves["periodic_left"],
                           [design.meridional_channel.pitch, 0])
    assert len(calls) == 1
    reconstructed = geo.load_design(geo.design_data(design))
    assert np.allclose(reconstructed.section(0)[1]["periodic_left"],
                       design.section(0)[1]["periodic_left"])


def test_quality_cube_and_skewed_cube():
    xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    blocks, tags = {5: np.arange(8)[None]}, {5: np.array([101])}
    report, metrics = analyze_volume(xyz, blocks, tags)
    assert metrics["skewness"] == pytest.approx([0])
    assert metrics["orthogonality"] == pytest.approx([1])
    assert metrics["aspect_ratio"] == pytest.approx([1])
    assert report["worst_cells"]["skewness"][0]["cell_id"] == 101
    xyz[4:, 0] += 0.6
    _, skewed = analyze_volume(xyz, blocks, tags)
    assert skewed["skewness"][0] > 0.1
    assert skewed["orthogonality"][0] < 0.95


@pytest.mark.parametrize("case", ["axial", "axial_taper", "radial_inflow", "radial_outflow", "curved"])
def test_actual_reference_volume_and_cgns_export(case, small_design, tmp_path):
    nozzle, config = small_design
    definition = None
    if case == "axial_taper":
        config["meridional_channel"]["tip_chord"] = 22
    if case.startswith("radial"):
        radial = flow.read_configuration(CASES["radial_stator"])["meridional_channel"]
        if case == "radial_outflow":
            radial["inlet_radius"], radial["outlet_radius"] = radial["outlet_radius"], radial["inlet_radius"]
        config["meridional_channel"] = radial
    elif case == "curved":
        knots = [0,0,0,0,1,1,1,1]
        definition = geo.GeometryDefinition(
            BSpline(knots, [[100,0],[100,8],[101,15],[102,20]], 3),
            BSpline(knots, [[110,0],[110,8],[111,16],[112,22]], 3), 12)
    design = geo.build_design(nozzle, config, definition=definition)
    messages = []
    progress = lambda message, fraction: messages.append((message, fraction))
    summary = mesh.reference_mesh(design, config, tmp_path, progress=progress)
    section, loaded = mesh.load_section(tmp_path)
    assert summary["section_cells"] > 0
    preview = geo.meridional_grid(loaded, config["mesh"]["numerics"], section)
    assert preview["basis"] == "reference mesh boundary stations"
    assert preview["span_layers"] == 3
    volume = mesh.volume_mesh(section, loaded, config, tmp_path, progress=progress)
    assert volume["quality"]["min_determinant_jacobian"] > 0
    assert volume["quality"]["max_periodic_error_mm"] < 1e-8
    assert sum(volume["cells"].values()) == 3 * summary["section_cells"]
    assert not gmsh.isInitialized()
    report = json.loads((tmp_path / "mesh_report.json").read_text())
    assert all(sum(v["counts"]) == sum(volume["cells"].values()) for v in report["metrics"].values())
    result = mesh.export_mesh(tmp_path, config, tmp_path, progress=progress)
    assert result["validation"]["boundary_partition_complete"]
    # Meshing reports completion; export may then report an indeterminate stage.
    # The worker final status, rather than this callback, declares job success.
    assert any(fraction == 1 for _, fraction in messages)
    assert all(message and (fraction is None or 0 <= fraction <= 1)
               for message, fraction in messages)
