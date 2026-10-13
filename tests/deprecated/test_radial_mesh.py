"""Checks for radial-row meshing: consistent_radial_wrap (N and r2 that keep
the designed passage) and stator_mesh_3d.mesh_radial_passage (2D design-
plane mesh -> radial wrap's log-spiral map -> extrusion through the depth),
for the stator and the rotor."""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from turbo_moc.legacy.geometry.radial import consistent_radial_wrap, implied_n_blades_radial, wrap_blade_radial, wrap_rotor_blade_radial
from turbo_moc.geometry.stator_blade import parametrize_stator_blade
from turbo_moc.legacy.meshing.rotor_mesh_2d import mesh_sized_rotor  # noqa: E402
from turbo_moc.legacy.meshing.stator_mesh_2d import mesh_sized_stator  # noqa: E402
from turbo_moc.legacy.meshing.stator_mesh_3d import load_patch_surfaces, mesh_radial_passage, radial_map_reference
from turbo_moc.core.vortex_blade import design_rotor_vortex_blade  # noqa: E402
from turbo_moc.configuration.fluid_inputs import solver_parameters

DEPTH = 5.0
SPAN = {"first_height": 0.05, "growth_ratio": 1.3, "num_layers": 3, "transition_ratio": 1.3, "core_size": DEPTH / 4}


@pytest.fixture(scope="module")
def stator():
    # The realistic Cyclopentane wall sample of test_app_stator_mesh.py (raw mm).
    wall_x = [0.0, 0.006976, 0.013917, 0.020012, 0.022465, 0.023241, 0.023495, 0.023577, 0.023603,
              0.032041, 0.045601, 0.051617, 0.053625, 0.054286, 0.054498, 0.054589]
    wall_y = [0.01, 0.010244, 0.010973, 0.012023, 0.012556, 0.012738, 0.012799, 0.012819, 0.012825,
              0.014577, 0.016172, 0.016432, 0.01646, 0.016463, 0.016463, 0.016463]
    return parametrize_stator_blade(wall_x, wall_y, metal_angle_in=0.0, metal_angle_out=65.0,
                                    r_trailing=0.5, inlet_opening_ratio=1.3)


@pytest.fixture(scope="module")
def rotor():
    inputs = solver_parameters(dict(fluid_name="Novec649", inlet_property_pair="PT_INPUTS",
                                    inlet_property_1=5e5, inlet_property_2=400.0), rotor=True)
    return design_rotor_vortex_blade(**inputs, M_inlet=1.5,
                                     M_outlet=1.5, M_lower=1.05, M_upper=2.0, beta_inlet=65.0,
                                     backend="HEOS", num_points=60)


@pytest.mark.parametrize("r1, r2", [(100.0, 200.0), (200.0, 100.0)])
def test_consistent_wrap_makes_the_implied_count_an_integer(rotor, r1, r2):
    wrap = consistent_radial_wrap(rotor, r1, r2, source="rotor")
    assert wrap["n_blades"] == round(wrap["n_implied_at_target"])
    assert np.sign(wrap["r2"] - r1) == np.sign(r2 - r1)  # same flow direction
    assert np.isclose(implied_n_blades_radial(rotor, r1, wrap["r2"], source="rotor"), wrap["n_blades"])


def _mesh(blade, source, r1, r2_target, tmp_path):
    wrap = consistent_radial_wrap(blade, r1, r2_target, source=source)
    reference = radial_map_reference(blade, r1, wrap["r2"], source)
    pitch = float(blade["pitch"])
    sizes = {"blade_size": pitch / 40, "domain_size": pitch / 10, "first_height": pitch / 1000,
             "growth_ratio": 1.3, "num_layers": 3, "transition_ratio": 1.2}
    mesh = (mesh_sized_stator if source == "stator" else mesh_sized_rotor)(blade, sizes)
    result = mesh_radial_passage(mesh, reference, wrap["n_blades"], DEPTH, SPAN, tmp_path, name="radial")
    assert not gmsh.isInitialized()
    return wrap, mesh, result


def _max_distance(points, polyline):
    starts, edge = polyline[:-1], np.diff(polyline, axis=0)
    t = np.clip(np.einsum("pij,ij->pi", points[:, None] - starts, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
    return np.min(np.linalg.norm(points[:, None] - (starts + t[..., None] * edge), axis=2), axis=1).max()


def test_radial_stator_mesh_lands_on_the_radial_solid(stator, tmp_path):
    wrap, mesh, result = _mesh(stator, "stator", 50.0, 100.0, tmp_path)
    assert result["quality"]["min_determinant_jacobian"] > 0
    assert np.isclose(result["pitch_angle_deg"], 360.0 / wrap["n_blades"])
    # A quad the map makes non-convex is split in two (prisms): one more cell per split.
    assert sum(result["cells"].values()) == (len(mesh["cells"]) + result["split_quads"]) * result["span_cells"]
    wrapped = wrap_blade_radial(stator, 50.0, wrap["r2"], wrap["n_blades"])
    outline = np.column_stack([wrapped["blades"][0]["x"] + wrapped["trailing_edges"][0]["x"],
                               wrapped["blades"][0]["y"] + wrapped["trailing_edges"][0]["y"]])
    points = load_patch_surfaces(result["output_dir"], ["blade"], result["files"]["surfaces"])["blade"]["points"]
    hub = points[np.isclose(points[:, 2], 0.0)][:, :2].astype(float)
    mapped_pitch = 2 * np.pi * 50.0 / wrap["n_blades"]
    assert _max_distance(hub, outline) < 1e-3 * mapped_pitch


def test_radial_rotor_mesh_lands_on_the_radial_solid(rotor, tmp_path):
    wrap, mesh, result = _mesh(rotor, "rotor", 100.0, 200.0, tmp_path)
    assert result["quality"]["min_determinant_jacobian"] > 0
    assert np.isclose(result["pitch_angle_deg"], 360.0 / wrap["n_blades"])
    # Densify the contour first: its straight LE/TE stubs map to spiral arcs.
    x, y = np.asarray(rotor["blade"]["x"]), np.asarray(rotor["blade"]["y"])
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    t = np.linspace(0.0, s[-1], 20000)
    fine = {**rotor, "blade": {"x": np.interp(t, s, x).tolist(), "y": np.interp(t, s, y).tolist()}}
    wrapped = wrap_rotor_blade_radial(fine, 100.0, wrap["r2"], wrap["n_blades"])
    outline = np.column_stack([wrapped["blades"][0]["x"], wrapped["blades"][0]["y"]])
    points = load_patch_surfaces(result["output_dir"], ["blade"], result["files"]["surfaces"])["blade"]["points"]
    hub = points[np.isclose(points[:, 2], 0.0)][:, :2].astype(float)
    assert _max_distance(hub, outline) < 1e-3 * 2 * np.pi * 100.0 / wrap["n_blades"]


def test_inconsistent_blade_count_is_rejected(rotor, tmp_path):
    wrap = consistent_radial_wrap(rotor, 100.0, 200.0, source="rotor")
    reference = radial_map_reference(rotor, 100.0, 200.0, "rotor")  # target r2: not exactly periodic
    pitch = float(rotor["pitch"])
    mesh = mesh_sized_rotor(rotor, {"blade_size": pitch / 30, "domain_size": pitch / 8, "first_height": pitch / 500,
                                    "growth_ratio": 1.3, "num_layers": 2, "transition_ratio": 1.2})
    with pytest.raises(ValueError, match="does not match the blade count"):
        mesh_radial_passage(mesh, reference, wrap["n_blades"], DEPTH, SPAN, tmp_path)
