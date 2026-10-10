"""Checks for the rotor 3D mesh: the hub section (turbo_moc.meshing.
rotor_mesh_2d, mid-channel periodic passage) swept hub -> shroud with
stator_mesh_3d.mesh_swept_passage onto the app's rotor solid, and the
rotor's own relative-frame y+ states (wall_spacing.rotor_wall_states)."""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from turbo_moc.geometry import wrap_blade_annular  # noqa: E402
from turbo_moc.geometry.sizing import size_rotor_from_pitch  # noqa: E402
from turbo_moc.meshing.rotor_mesh_2d import mesh_sized_rotor, rotor_flow_domain  # noqa: E402
from turbo_moc.meshing.stator_mesh_3d import load_patch_surfaces, mesh_swept_passage  # noqa: E402
from turbo_moc.meshing.wall_spacing import estimate_first_layer_height, rotor_wall_states  # noqa: E402
from turbo_moc.rotor import design_rotor_vortex_blade  # noqa: E402

R_HUB, R_SHROUD, N_BLADES = 100.0, 112.0, 40
P0, T0 = 5e5, 400.0


@pytest.fixture(scope="module")
def rotor():
    data = design_rotor_vortex_blade(fluid_name="Novec649", P0_rel=P0, T0_rel=T0, M_inlet=1.5, M_outlet=1.5,
                                     M_lower=1.05, M_upper=2.0, beta_inlet=65.0, backend="HEOS", num_points=60)
    data.update(P0_rel=P0, T0_rel=T0)
    return data, size_rotor_from_pitch(data, R_HUB, R_SHROUD, N_BLADES)["scaled_blade"]


@pytest.fixture(scope="module")
def swept(rotor, tmp_path_factory):
    _, blade = rotor
    pitch = blade["pitch"]
    sizes = {"blade_size": pitch / 40, "domain_size": pitch / 10, "first_height": pitch / 2000,
             "growth_ratio": 1.3, "num_layers": 3, "transition_ratio": 1.2}
    mesh = mesh_sized_rotor(blade, sizes)
    span = {"first_height": 0.02, "growth_ratio": 1.3, "num_layers": 3, "transition_ratio": 1.3,
            "core_size": (R_SHROUD - R_HUB) / 4}
    result = mesh_swept_passage(mesh, R_HUB, R_SHROUD, N_BLADES, span, tmp_path_factory.mktemp("rotor"),
                                name="rotor_mesh_3d")
    assert not gmsh.isInitialized()
    return mesh, result


def test_rotor_passage_is_periodic_and_contains_the_blade(rotor):
    _, blade = rotor
    domain = rotor_flow_domain(blade)
    assert np.allclose(domain["periodic_right"] - domain["periodic_left"], [blade["pitch"], 0.0])
    assert domain["clearance_mm"] > 0.05 * blade["pitch"]
    # Every blade point lies pitchwise between the two periodic boundaries.
    contour = domain["contour"]
    order = np.argsort(domain["periodic_left"][:, 1])
    left = np.interp(contour[:, 1], domain["periodic_left"][order, 1], domain["periodic_left"][order, 0])
    assert np.all(contour[:, 0] > left) and np.all(contour[:, 0] < left + blade["pitch"])


def test_rotor_hub_mesh_is_valid_and_periodic(rotor, swept):
    _, blade = rotor
    mesh, _ = swept
    assert mesh["statistics"]["minimum_scaled_jacobian"] > 0
    assert mesh["statistics"]["quadrilateral_fraction"] > 0.9
    xy = np.asarray(mesh["xy"])
    offset = xy[np.asarray(mesh["periodic"]["donor"]) - 1] - xy[np.asarray(mesh["periodic"]["current"]) - 1]
    assert np.allclose(np.abs(offset[:, 0]), blade["pitch"]) and np.allclose(offset[:, 1], 0)


def test_rotor_sweep_is_valid_and_rotationally_periodic(swept):
    mesh, result = swept
    assert sum(result["cells"].values()) == len(mesh["cells"]) * result["span_cells"]
    assert result["quality"]["min_determinant_jacobian"] > 0
    assert np.isclose(result["pitch_angle_deg"], 360.0 / N_BLADES)
    assert result["files"]["cgns"] == "rotor_mesh_3d.cgns"


def test_swept_rotor_blade_lies_on_the_rotor_solid(rotor, swept):
    """The app's 3D rotor is the mirrored blade wrapped with pitch_scale; the
    mesh frame (X = y, Y = -x) must land on it. The contour is resampled
    finely first: its straight 20 mm LE/TE stubs wrap to helical arcs that
    the coarse polyline's chords miss by ~0.5 mm."""
    _, blade = rotor
    _, result = swept
    x, y = np.asarray(blade["blade"]["x"]), np.asarray(blade["blade"]["y"])
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    t = np.linspace(0.0, s[-1], 20000)
    fine = {**blade, "blade": {"x": (-np.interp(t, s, x)).tolist(), "y": np.interp(t, s, y).tolist()}}
    wrapped = wrap_blade_annular(fine, R_HUB, R_SHROUD, n_blades=N_BLADES, flare="pitch_scale",
                                 n_span=2, source="rotor")
    polyline = np.column_stack([wrapped["curve"][axis][0, -1] for axis in "xyz"])
    starts, edge = polyline[:-1], np.diff(polyline, axis=0)
    points = load_patch_surfaces(result["output_dir"], ["blade"], result["files"]["surfaces"])["blade"]["points"]
    points = points.astype(float)
    shroud = points[np.isclose(np.hypot(points[:, 0], points[:, 1]), R_SHROUD, rtol=1e-6)]
    assert len(shroud) > 10
    distances = []
    for chunk in np.array_split(shroud, max(1, len(shroud) // 200)):
        f = np.clip(np.einsum("pij,ij->pi", chunk[:, None] - starts, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
        distances.append(np.min(np.linalg.norm(chunk[:, None] - (starts + f[..., None] * edge), axis=2), axis=1))
    assert np.concatenate(distances).max() < 1e-3 * blade["pitch"]


def test_rotor_wall_states_expand_isentropically(rotor):
    data, blade = rotor
    states = rotor_wall_states(data)["wall_final"]
    rho, temperature, mach = (np.asarray(states[k]) for k in ("rho", "T", "M"))
    assert np.allclose(rho[0], rho[3]) and np.allclose(temperature[0], temperature[3])  # M_inlet == M_outlet
    order = np.argsort(mach)
    # Higher Mach -> lower density and temperature (M_inlet == M_outlet repeat a state).
    assert np.all(np.diff(rho[order]) <= 0) and np.all(np.diff(temperature[order]) <= 0)
    assert rho[order][0] > rho[order][-1] and temperature.max() < T0
    wall = estimate_first_layer_height(rotor_wall_states(data), blade["chord"], 1.0)
    assert wall["first_height_mm"] > 0 and wall["mach"] in mach
