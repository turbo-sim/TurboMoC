"""Physical interface placement and CAD-consistent multirow span deformation."""
import numpy as np
import pytest

from turbo_moc.configuration.project import default_project, row_configuration
from turbo_moc.geometry import blade_row as geometry
from turbo_moc.geometry.meridional_channel import corresponding_boundary
from turbo_moc.meshing import mesh_3d as meshing


def small_design_config():
    config = row_configuration(default_project(), "stator_1")
    config["meridional_channel"].update(num_blades=12, hub_chord=20., tip_chord=22.)
    config["blade_geometry"].update(r_trailing=.1, num_baseline_points=100,
                                  num_curve_points=100, n_cp=15)
    config["flow_domain"].update(inlet_chords=.5, outlet_chords=.5, num_points=40)
    config["mesh"]["numerics"].update(terminal_output=False, blade_size=.7,
        domain_size=3., curvature_elements=16, smoothing_steps=1,
        span_layers=3, span_stretch=.8)
    config["mesh"]["numerics"]["inflation_layers"].update(
        first_height=.03, growth_ratio=1.4, thickness=.15)
    x = np.linspace(0., .01, 80)
    nozzle = {"converged": True, "wall_final": {
        "x": x.tolist(), "y": (.002 + .001 * (x / x[-1]) ** 2).tolist()}}
    return nozzle, config


@pytest.mark.parametrize("family,radii", [("axial", None), ("radial", (100., 80.)),
                                          ("radial", (80., 100.))])
def test_caps_are_physical_at_every_span_and_survive_serialization(family, radii):
    nozzle, config = small_design_config()
    if family == "radial":
        config["meridional_channel"] = {"type": "radial", "num_blades": 12,
            "inlet_radius": radii[0], "outlet_radius": radii[1], "blade_height": 10.}
        direction = np.sign(radii[1] - radii[0])
        upstream, downstream = radii[0] - 2 * direction, radii[1] + 2 * direction
        offset = 0.
    else:
        upstream, downstream, offset = 5., 31., 7.
    config["assembly"] = {"row_id": "test", "placement": {"streamwise_offset_mm": offset},
        "interfaces": {side: {"id": side, "geometry_type": family, "coordinate_mm": value}
                       for side, value in (("inlet", upstream), ("outlet", downstream))}}
    design = geometry.build_design(nozzle, config)
    restored = geometry.load_design(geometry.design_data(design))
    for eta in np.linspace(0., 1., 7):
        _, curves = restored.section(eta)
        for side, target in (("inlet", upstream), ("outlet", downstream)):
            points = restored.meridional_channel.map_points(curves[side], eta)
            coordinate = points[:, 2] + offset if family == "axial" else np.linalg.norm(points[:, :2], axis=1)
            np.testing.assert_allclose(coordinate, target, atol=1e-10)
        np.testing.assert_allclose(curves["periodic_right"] - curves["periodic_left"],
                                   np.tile([restored.meridional_channel.pitch, 0.],
                                           (len(curves["periodic_left"]), 1)), atol=1e-12)


def test_boundary_correspondence_retains_native_spline_parameters():
    nozzle, config = small_design_config()
    config["assembly"] = {"interfaces": {"outlet": {
        "id": "pair", "geometry_type": "axial", "coordinate_mm": 24.}}}
    design = geometry.build_design(nozzle, config)
    target = design.section(0.)[1]
    parameters = np.array([.13, .41, .88])
    for name in ("inlet_spline", "convergent_spline", "outlet_spline"):
        for side, sign in (("periodic_left", -1), ("periodic_right", 1)):
            offset = [sign * design.meridional_channel.pitch / 2, 0.]
            nodes = design.curves[name](parameters) + offset
            moved = corresponding_boundary(nodes, side, design.curves, target)
            np.testing.assert_allclose(moved, target[name](parameters) + offset, atol=1e-11)


def test_refined_preview_retains_physical_axial_cap():
    from turbo_moc.visualization.geometry_preview import preview_geometry

    nozzle, config = small_design_config()
    config["assembly"] = {"interfaces": {"outlet": {
        "id": "pair", "geometry_type": "axial", "coordinate_mm": 24.}}}
    design = geometry.build_design(nozzle, config)
    surfaces = preview_geometry(design, span_sections=3, curve_points=40)["surfaces"]
    cap = np.asarray(surfaces["outlet"]["points"]).reshape(-1, 3)
    np.testing.assert_allclose(cap[:, 2], 24., atol=1e-10)
    for side in ("periodic_left", "periodic_right"):
        points = np.asarray(surfaces[side]["points"]).reshape(-1, 3)
        assert points[:, 2].max() == pytest.approx(24., abs=1e-10)


def test_shared_cap_tapered_axial_reference_and_volume(tmp_path):
    nozzle, config = small_design_config()
    config["assembly"] = {"interfaces": {"outlet": {
        "id": "pair", "geometry_type": "axial", "coordinate_mm": 24.}}}
    design = geometry.build_design(nozzle, config)
    summary = meshing.reference_mesh(design, config, tmp_path)
    section, restored = meshing.load_section(tmp_path)
    volume = meshing.volume_mesh(section, restored, config, tmp_path)
    assert summary["section_cells"] > 0
    assert volume["quality"]["min_determinant_jacobian"] > 0
    assert volume["quality"]["max_periodic_error_mm"] < 1e-8
