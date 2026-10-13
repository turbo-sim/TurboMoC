"""The vortex blade uses the canonical annular/linear mapping and mesh topology."""
from copy import deepcopy

import gmsh
import numpy as np
import pytest

from turbo_moc.configuration.project import default_project, row_configuration
from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.geometry.blade_row import design_data, load_design
from turbo_moc.geometry.rotor_blade import _bezier_reference, build_rotor_design
from turbo_moc.geometry import passage_2d as planar
from turbo_moc.geometry.meridional_channel import corresponding_boundary
from turbo_moc.meshing import mesh_3d as meshing_3d
from turbo_moc.core.vortex_blade import design_rotor_vortex_blade


@pytest.mark.parametrize("angles", [(60., -60.), (0., 0.), (-60., 60.), (5., -5.)])
def test_bezier_reference_handles_parallel_and_incompatible_guides_without_folding(angles):
    station = np.linspace(0, 1, 101)
    center = 1.2 * station * (1 - station)
    upper = np.column_stack((center + .05, -station))
    lower = np.column_stack((center - .05, -station))
    contour = np.vstack((upper, lower[::-1], upper[:1]))
    samples, spline, guides = _bezier_reference(contour, *angles, 120)
    assert len(spline.c) == 4
    assert np.all(np.diff(samples[:, 1]) < 0)
    np.testing.assert_allclose(samples[[0, -1]], guides[[0, -1]])
    for endpoint, angle in zip((0., 1.), angles):
        tangent = spline(endpoint, 1)
        np.testing.assert_allclose(tangent / np.linalg.norm(tangent), planar.metal_direction(angle), atol=1e-12)
    if angles == (60., -60.):
        np.testing.assert_allclose(spline(.5), guides[1])
    # Sampling resolution changes plotting density, not the cubic geometry.
    _, fine, _ = _bezier_reference(contour, *angles, 480)
    np.testing.assert_allclose(spline.c, fine.c)


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
    inputs = solver_parameters(dict(fluid_name="Nitrogen", inlet_property_pair="PT_INPUTS",
                                    inlet_property_1=5e5, inlet_property_2=400.), rotor=True)
    return design_rotor_vortex_blade(
        **inputs,
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
        assert len(curves["blade_reference_spline"].c) == 4
        assert np.all(np.diff(curves["passage_reference"][:, 1]) < 0)
        guides = curves["blade_reference_guides"]
        np.testing.assert_allclose(curves["blade_reference"][[0, -1]], guides[[0, -1]])
        for side, reference_u, extension_u in (("inlet", 0., 1.), ("outlet", 1., 0.)):
            reference_tangent = curves["blade_reference_spline"](reference_u, 1)
            extension_tangent = curves[f"{side}_spline"](extension_u, 1)
            np.testing.assert_allclose(reference_tangent / np.linalg.norm(reference_tangent),
                                       extension_tangent / np.linalg.norm(extension_tangent), atol=1e-12)
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
