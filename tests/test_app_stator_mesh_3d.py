"""Checks for turbo_moc.meshing.stator_mesh_3d: the app's 2D hub mesh swept
hub -> shroud into a hex/prism volume matching the exported annular solid,
with y+-based endwall layers and rotational periodicity."""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from turbo_moc.geometry import parametrize_stator_blade, size_stator_mode_a, wrap_blade_annular  # noqa: E402
from turbo_moc.io.cgns_adf import read_adf  # noqa: E402
from turbo_moc.meshing.stator_mesh_2d import mesh_sized_stator  # noqa: E402
from turbo_moc.meshing.stator_mesh_3d import (  # noqa: E402
    load_patch_surfaces,
    mesh_swept_stator,
    span_distribution,
)

R_HUB, N_BLADES = 100.0, 24


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    # Same realistic Cyclopentane wall sample as test_app_stator_mesh.py.
    wall_x = [0.0, 0.006976, 0.013917, 0.020012, 0.022465, 0.023241, 0.023495, 0.023577, 0.023603,
              0.032041, 0.045601, 0.051617, 0.053625, 0.054286, 0.054498, 0.054589]
    wall_y = [0.01, 0.010244, 0.010973, 0.012023, 0.012556, 0.012738, 0.012799, 0.012819, 0.012825,
              0.014577, 0.016172, 0.016432, 0.01646, 0.016463, 0.016463, 0.016463]
    blade = parametrize_stator_blade(wall_x, wall_y, metal_angle_in=0.0, metal_angle_out=65.0,
                                     r_trailing=0.5, inlet_opening_ratio=1.3)
    nozzle = {"wall_final": {"rho": [2.3, 2.3], "T": [300.0, 300.0]}, "fluid_name": "Nitrogen"}
    sizing = size_stator_mode_a(blade, nozzle, mdot=1.0, n_blades=N_BLADES, r_hub=R_HUB)
    pitch = sizing["scaled_blade"]["pitch"]
    mesh = mesh_sized_stator(sizing["scaled_blade"], {
        "blade_size": pitch / 40, "domain_size": pitch / 10, "first_height": pitch / 500,
        "growth_ratio": 1.3, "num_layers": 3, "transition_ratio": 1.2})
    height = sizing["r_shroud"] - R_HUB
    span = {"first_height": height / 200, "growth_ratio": 1.3, "num_layers": 3,
            "transition_ratio": 1.3, "core_size": height / 4}
    output = tmp_path_factory.mktemp("swept")
    result = mesh_swept_stator(mesh, R_HUB, sizing["r_shroud"], N_BLADES, span, output,
                               boundary_types={"blade": "BCWallViscous", "hub": "BCWallViscous",
                                               "shroud": "BCWallViscous"})
    assert not gmsh.isInitialized()
    return sizing, mesh, span, result


def test_span_distribution_has_symmetric_endwall_layers():
    stations = span_distribution(10.0, 0.01, 1.2, 5, 1.3, 1.0)
    sizes = np.diff(stations)
    assert stations[0] == 0 and np.isclose(stations[-1], 10.0)
    assert np.allclose(sizes, sizes[::-1])
    assert np.allclose(sizes[:5], 0.01 * 1.2 ** np.arange(5))
    assert np.all(sizes <= 1.0 + 1e-12) and np.all(np.diff(sizes[: len(sizes) // 2]) >= -1e-12)


def test_span_distribution_fits_a_short_span():
    # Layers that would overrun midspan stop there; still exact and symmetric.
    stations = span_distribution(0.05, 0.01, 1.5, 10, 1.2, 1.0)
    assert np.isclose(stations[-1], 0.05) and np.all(np.diff(stations) > 0)


def test_swept_volume_cell_counts_and_quality(swept):
    _, mesh, span, result = swept
    n_2d = len(mesh["cells"])
    assert sum(result["cells"].values()) == n_2d * result["span_cells"]
    assert result["quality"]["min_determinant_jacobian"] > 0 and result["quality"]["min_sicn"] > 0
    assert np.isclose(result["span_first_height_mm"], span["first_height"])
    assert np.isclose(result["pitch_angle_deg"], 360.0 / N_BLADES)


def test_swept_blade_lies_on_the_exported_solid_at_the_shroud(swept):
    """The app's solid is ruled along radial lines, so the sweep is exact:
    shroud blade nodes coincide with wrap_blade_annular's shroud section."""
    sizing, mesh, _, result = swept
    wrapped = wrap_blade_annular(sizing["scaled_blade"], R_HUB, sizing["r_shroud"], n_blades=N_BLADES,
                                 flare="pitch_scale", n_span=2, source="stator")
    starts, ends = [], []
    for key in ("curve", "trailing_edge"):
        polyline = np.column_stack([wrapped[key][axis][0, -1] for axis in ("x", "y", "z")])
        starts.append(polyline[:-1])
        ends.append(polyline[1:])
    starts, ends = np.vstack(starts), np.vstack(ends)
    points = load_patch_surfaces(result["output_dir"], ["blade"])["blade"]["points"].astype(float)
    shroud = points[np.isclose(np.hypot(points[:, 0], points[:, 1]), sizing["r_shroud"], rtol=1e-6)]
    assert len(shroud) > 10
    edge = ends - starts
    t = np.clip(np.einsum("pij,ij->pi", shroud[:, None] - starts, edge)
                / np.einsum("ij,ij->i", edge, edge), 0.0, 1.0)
    distances = np.min(np.linalg.norm(shroud[:, None] - (starts + t[..., None] * edge), axis=2), axis=1)
    assert distances.max() < 1e-3 * mesh["pitch"]


def test_swept_cgns_has_seven_patches_and_rotational_periodicity(swept):
    _, _, _, result = swept
    from pathlib import Path

    base = read_adf(str(Path(result["output_dir"]) / "stator_mesh_3d.cgns")).get(label="CGNSBase_t")
    zone = base.get(label="Zone_t")
    assert {bc.name for bc in zone.get(label="ZoneBC_t").children} == {
        "blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right"}
    grid = zone.get(label="GridCoordinates_t")
    points = np.column_stack([grid.get(f"Coordinate{axis}").data for axis in "XYZ"])
    for interface in zone.get(label="ZoneGridConnectivity_t").children:
        current = interface.get("PointList").data.ravel() - 1
        donor = interface.get("PointListDonor").data.ravel() - 1
        angle = interface.get(label="GridConnectivityProperty_t").get(label="Periodic_t").get("RotationAngle").data[2]
        assert np.isclose(abs(angle), 2 * np.pi / N_BLADES)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        assert np.allclose(points[current] @ rotation.T, points[donor], atol=1e-9)
