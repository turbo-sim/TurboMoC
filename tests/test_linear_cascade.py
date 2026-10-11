"""Physical linear cascades share reference topology and retain true 2D export."""

from copy import deepcopy
import json

import gmsh
import numpy as np
import pytest

from turbo_moc.configuration import read_configuration
from turbo_moc.geometry import geometry_3d as geo
from turbo_moc.geometry.linear_cascade import LinearCascadeDefinition, LinearCascadeMapping
from turbo_moc.geometry.meridional_channel import validate_machine
from turbo_moc.meshing import meshing_3d as mesh
from turbo_moc.meshing.quality import analyze_volume
from turbo_moc.geometry import parametrize_stator_blade
from turbo_moc.io.cgns_adf import read_adf


@pytest.fixture
def linear_case():
    config = read_configuration("examples/apps/cases/axial_stator.yaml")
    config["meridional_channel"] = {"type": "linear", "meridional_chord": 20.0, "blade_height": 10.0}
    config["blade_geometry"].update(r_trailing=0.1, num_baseline_points=100, num_curve_points=100, n_cp=15)
    config["flow_domain"].update(inlet_chords=0.5, outlet_chords=0.5, num_points=40)
    config["mesh"]["numerics"].update(
        terminal_output=False, blade_size=0.7, domain_size=3.0,
        curvature_elements=16, smoothing_steps=1, span_layers=3, span_stretch=0.8,
    )
    config["mesh"]["numerics"]["inflation_layers"].update(first_height=0.03, growth_ratio=1.4, thickness=0.15)
    x = np.linspace(0, 0.01, 80)
    nozzle = {"converged": True, "wall_final": {
        "x": x.tolist(), "y": (0.002 + 0.001 * (x / x[-1]) ** 2).tolist(),
    }}
    return nozzle, config


def test_linear_mapping_uses_lengths_and_translation():
    mapping = LinearCascadeMapping(LinearCascadeDefinition(20.0, 10.0, 5.0))
    q = np.array([[1.0, 0.0], [3.0, -20.0]])
    assert np.allclose(mapping.map_points(q, 0.4), [[0, 1, 4], [20, 3, 4]])
    transform = mapping.periodic_transform()
    assert np.allclose(transform[:3, :3], np.eye(3))
    assert np.allclose(transform[:3, 3], [0, 5, 0])
    assert mapping.section_scale(0) == mapping.section_scale(1) == 1
    assert not hasattr(mapping, "reference_radius")
    assert not hasattr(mapping, "num_blades")


def test_periodic_quality_applies_translation_to_neighbor_centroids():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    transform = np.eye(4)
    transform[1, 3] = 1.0
    _, metrics = analyze_volume(
        xyz, {5: np.arange(8)[None]}, {5: np.array([1])},
        periodic_faces={"right": np.array([[2, 3, 7, 6]])},
        slave=np.array([2, 3, 7, 6]), master=np.array([1, 0, 4, 5]),
        periodic_transform=transform,
    )
    # The neighbor is the same cell shifted by one pitch, rather than a zero
    # connector obtained by treating the identity rotation as the whole map.
    assert metrics["orthogonality"] == pytest.approx([1.0])


@pytest.mark.parametrize("name,value", [("meridional_chord", 0), ("blade_height", -1),
                                       ("meridional_chord", np.nan), ("blade_height", True)])
def test_invalid_linear_dimensions(name, value):
    dimensions = {"type": "linear", "meridional_chord": 20, "blade_height": 10}
    dimensions[name] = value
    with pytest.raises(ValueError, match=name):
        validate_machine(dimensions)


def test_linear_chord_preserves_template_pitch_and_one_placement(linear_case, monkeypatch):
    nozzle, config = linear_case
    template = parametrize_stator_blade(**{
        "wall_x": nozzle["wall_final"]["x"], "wall_y": nozzle["wall_final"]["y"],
        **config["blade_geometry"],
    })
    source_chord = np.ptp(geo.planar.blade_contour(template)[:, 1])
    calls = []
    real = geo.planar.auto_center_flow_domain
    def count(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(geo.planar, "auto_center_flow_domain", count)
    design = geo.build_design(nozzle, config, template=template)
    assert design.meridional_channel.pitch == pytest.approx(template["pitch"] * 20 / source_chord)
    assert np.ptp(geo.planar.blade_contour(design.blade)[:, 1]) == pytest.approx(20)
    for eta in np.linspace(0, 1, 15):
        assert design.section(eta)[0] is design.blade
    assert len(calls) == 1
    restored = geo.load_design(geo.design_data(design))
    assert restored.meridional_channel.definition.to_dict() == design.meridional_channel.definition.to_dict()
    assert np.allclose(restored.meridional_channel.map_points(restored.curves["blade_curve"], 1),
                       design.meridional_channel.map_points(design.curves["blade_curve"], 1))
    grid = geo.meridional_grid(restored, config["mesh"]["numerics"])
    assert grid["coordinate_basis"] == "linear_streamwise_span"
    assert grid["estimated_first_height_mm"] == [0.03, 0.03]
    assert geo.section_data(restored)["coordinate_units"] == "mm"


@pytest.mark.parametrize("layers", [1, 4])
def test_actual_linear_reference_volume_and_both_exports(linear_case, tmp_path, layers, monkeypatch):
    nozzle, config = linear_case
    config["mesh"]["numerics"]["span_layers"] = layers
    design = geo.build_design(nozzle, config)
    reference = mesh.reference_mesh(design, config, tmp_path)
    section, restored = mesh.load_section(tmp_path)
    original = section["xy"].copy()
    volume = mesh.volume_mesh(section, restored, config, tmp_path)
    assert volume["quality"]["min_determinant_jacobian"] > 0
    assert sum(volume["cells"].values()) == reference["section_cells"] * layers
    assert "reference_radius_mm" not in volume
    grid = geo.meridional_grid(design, config["mesh"]["numerics"], section)
    assert grid["basis"] == "reference mesh boundary stations"
    assert grid["streamwise_stations"] == len(np.unique(section["boundaries"]["periodic_left"]))
    with monkeypatch.context() as patch:
        def forbid_mesh_generation(*args, **kwargs):
            pytest.fail("Export must reuse the existing mesh.")
        patch.setattr(gmsh.model.mesh, "generate", forbid_mesh_generation)
        exported_2d = mesh.export_reference_mesh(tmp_path, config, tmp_path)
        exported_3d = mesh.export_mesh(tmp_path, config, tmp_path)
    assert np.array_equal(mesh.load_section(tmp_path)[0]["xy"], original)
    assert exported_2d["coordinate_basis"] == "linear_physical"
    assert exported_2d["file"] != exported_3d["file"]
    gmsh.initialize(["verify_linear_2d_artifact", "-v", "0"])
    try:
        gmsh.open(exported_2d["native_mesh_file"])
        for dim, tag in gmsh.model.getPhysicalGroups(1):
            if gmsh.model.getPhysicalName(dim, tag) == "periodic_right":
                for right in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    _, _, _, affine = gmsh.model.mesh.getPeriodicNodes(1, right)
                    assert np.allclose(np.asarray(affine).reshape(4, 4), design.meridional_channel.periodic_transform())
    finally:
        gmsh.finalize()
    base_2d = read_adf(exported_2d["file"]).get(label="CGNSBase_t")
    assert np.array_equal(base_2d.data, [2, 2])
    grid_2d = base_2d.get(label="Zone_t").get(label="GridCoordinates_t")
    points_2d = np.column_stack([grid_2d.get(f"Coordinate{axis}").data for axis in "XY"])
    physical = np.column_stack((-original[:, 1], original[:, 0])) * config["mesh"]["cgns"]["length_scale"]
    assert np.allclose(points_2d, physical)
    for exported, dimensions in ((exported_2d, 2), (exported_3d, 3)):
        base = read_adf(exported["file"]).get(label="CGNSBase_t")
        assert np.array_equal(base.data, [dimensions, dimensions])
        zone = base.get(label="Zone_t")
        grid = zone.get(label="GridCoordinates_t")
        points = np.column_stack([grid.get(f"Coordinate{axis}").data for axis in "XYZ"[:dimensions]])
        for interface in zone.get(label="ZoneGridConnectivity_t").children:
            current = interface.get("PointList").data.ravel() - 1
            donor = interface.get("PointListDonor").data.ravel() - 1
            periodic = interface.get(label="GridConnectivityProperty_t").get(label="Periodic_t")
            assert np.allclose(periodic.get("RotationAngle").data, 0)
            translation = periodic.get("Translation").data
            assert np.allclose(points[current] + translation, points[donor], atol=1e-8)
    quality = json.loads((tmp_path / "mesh_report.json").read_text(encoding="utf-8"))
    assert quality["metrics"]["orthogonality"]["min"] > 0
    assert np.isfinite(quality["metrics"]["aspect_ratio"]["max"])


def test_annular_reference_export_labels_conformal_basis(linear_case, tmp_path):
    nozzle, linear_config = linear_case
    config = deepcopy(linear_config)
    config["meridional_channel"] = {
        "type": "axial", "hub_radius": 100.0, "blade_height": 10.0,
        "hub_chord": 20.0, "tip_chord": 20.0, "num_blades": 12,
    }
    design = geo.build_design(nozzle, config)
    mesh.reference_mesh(design, config, tmp_path)
    exported = mesh.export_reference_mesh(tmp_path, config, tmp_path)
    assert exported["coordinate_basis"] == "annular_conformal_reference"
    assert "not physical Cartesian" in exported["description"]


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_annular_assembly_offset_is_axial_only(linear_case, family):
    from turbo_moc.apps.app import plots
    nozzle, config = linear_case
    config["meridional_channel"] = read_configuration(
        f"examples/apps/cases/{family}_stator.yaml"
    )["meridional_channel"]
    config["meridional_channel"]["num_blades"] = 12
    design = geo.build_design(nozzle, config)
    assembly = plots.assembly_meridional_figure([
        ("Stator 1", design, {"streamwise_offset_mm": 0}),
        ("Stator 2", design, {"streamwise_offset_mm": 30}),
    ])
    expected = 30 if family == "axial" else 0
    blades = {trace.name: trace for trace in assembly.data}
    assert np.allclose(np.asarray(blades["Stator 2"].x) - np.asarray(blades["Stator 1"].x), expected)
    assert assembly.layout.yaxis.range[0] == 0


def test_linear_geometry_preview_remains_closed_without_annular_parameters(linear_case):
    from turbo_moc.apps.app.preview import preview_geometry
    nozzle, config = linear_case
    design = geo.build_design(nozzle, config)
    preview = preview_geometry(design, span_sections=5, curve_points=60)
    assert preview["is_linear"]
    assert preview["pitch_mm"] == pytest.approx(design.meridional_channel.pitch)
    assert "num_blades" not in preview and "reference_radius_mm" not in preview
    blade = preview["surfaces"]["blade"]
    points = np.asarray(blade["points"]).reshape(-1, 3)
    normals = np.asarray(blade["normals"]).reshape(-1, 3)
    assert np.isfinite(points).all() and np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    # Normal splitting duplicates coordinates along sharp edges. Weld them to
    # audit the closed shell, including both visible blade-tip caps.
    welded, inverse = np.unique(points, axis=0, return_inverse=True)
    triangles = inverse[np.asarray(blade["polys"]).reshape(-1, 4)[:, 1:]]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert len(welded) - len(unique_edges) + len(triangles) == 2
