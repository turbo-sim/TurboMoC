"""The vortex blade uses the canonical annular/linear mapping and mesh topology."""
from copy import deepcopy

import gmsh
import numpy as np
import pytest

from turbo_moc.configuration_project import default_project, row_configuration
from turbo_moc.geometry.geometry_3d import design_data, load_design
from turbo_moc.geometry.rotor_geometry import build_rotor_design
from turbo_moc.geometry.meridional_channel import corresponding_boundary
from turbo_moc.meshing import meshing_3d
from turbo_moc.rotor import design_rotor_vortex_blade


def test_periodic_transfer_preserves_exact_cad_nodes_between_plot_samples():
    sample = np.array([0., .5, 1.])
    boundary = np.column_stack((sample ** 2, -sample))
    station = np.array([.13, .41, .88])
    cad_nodes = np.column_stack((station ** 2, -station))
    reference = {"periodic_left": boundary}
    for scale in (1., 1.07):
        shift = np.array([.3, 0.])
        target = {"periodic_left": boundary * scale + shift}
        moved = corresponding_boundary(cad_nodes, "periodic_left", reference, target)
        np.testing.assert_allclose(moved, cad_nodes * scale + shift, atol=1e-14)


@pytest.fixture(scope="module")
def vortex_result():
    return design_rotor_vortex_blade(
        fluid_name="Nitrogen", P0_rel=5e5, T0_rel=400.,
        M_inlet=1.5, M_outlet=1.5, M_lower=1.05, M_upper=2.,
        beta_inlet=65., num_points=30,
    )


@pytest.mark.parametrize("family", ["axial", "radial", "linear"])
def test_vortex_adapter_preserves_closed_contour_and_mapping(vortex_result, family):
    original = deepcopy(vortex_result)
    config = row_configuration(default_project(family), "stator_1")
    if family == "axial":
        config["meridional_channel"]["tip_chord"] = 17.
    design = build_rotor_design(vortex_result, config)
    restored = load_design(design_data(design))
    for eta in (0., .5, 1.):
        blade, curves = restored.section(eta)
        assert blade["provider"] == "vortex_rotor"
        np.testing.assert_allclose(curves["blade_curve"][0], curves["blade_curve"][-1])
        np.testing.assert_allclose(curves["periodic_right"] - curves["periodic_left"],
                                   np.tile([design.meridional_channel.pitch, 0.],
                                           (len(curves["periodic_left"]), 1)))
        left = restored.meridional_channel.map_points(curves["periodic_left"], eta)
        right = restored.meridional_channel.map_points(curves["periodic_right"], eta)
        transform = restored.meridional_channel.periodic_transform()
        np.testing.assert_allclose(left @ transform[:3, :3].T + transform[:3, 3], right, atol=1e-10)
    assert vortex_result == original


def test_rotor_cad_generates_conforming_inflated_section_and_volume(vortex_result):
    config = row_configuration(default_project("linear"), "stator_1")
    design = build_rotor_design(vortex_result, config)
    numerics = config["mesh"]["numerics"]
    # Frontal-Delaunay is less aggressive than the quad-oriented algorithm in
    # this narrow, strongly curved natural-pitch passage; recombination stays on.
    numerics.update(blade_size=.35, domain_size=1., span_layers=2, span_stretch=0., algorithm=6)
    numerics["inflation_layers"].update(first_height=.02, thickness=.1, growth_ratio=1.2)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        section = meshing_3d.generate_section(design.blade, design.curves, numerics)
        assert len(section["slave"]) > 0
        assert set(section["boundaries"]) == {"blade", "inlet", "outlet", "periodic_left", "periodic_right"}
        xyz = np.array([design.meridional_channel.map_points(section["xy"], eta) for eta in (0., .5, 1.)])
        meshing_3d.assemble_volume(section, xyz, design.meridional_channel)
        quality = meshing_3d.validate_volume(section, xyz, design.meridional_channel)
        assert quality["min_sicn"] > 0
    finally:
        gmsh.finalize()
