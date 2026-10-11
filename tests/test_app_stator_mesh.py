"""Checks for the app's 2D mesh of the sized stator
(turbo_moc.meshing.stator_mesh_2d: mesh_sized_stator, section_coordinates,
write_fluent_cgns_2d): one hub mesh, mapped to any span unrolled or on the
cylinder, consistent with the exported annular solid."""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from turbo_moc.geometry import parametrize_stator_blade, size_stator_mode_a, wrap_blade_annular  # noqa: E402
from turbo_moc.io.cgns_adf import read_adf  # noqa: E402
from turbo_moc.meshing.stator_mesh_2d import (  # noqa: E402
    mesh_sized_stator,
    section_coordinates,
    write_fluent_cgns_2d,
)

R_HUB, N_BLADES = 100.0, 24


@pytest.fixture(scope="module")
def sized_mesh():
    # 16-point sample of the examples' Cyclopentane MoC wall (metres): a real
    # design shape. Made-up walls easily give unrealistic blades (TE radius
    # near half the throat, bulbous leading edges) that neither mesher can
    # handle well, which is not what these tests are about.
    wall_x = [0.0, 0.006976, 0.013917, 0.020012, 0.022465, 0.023241, 0.023495, 0.023577, 0.023603,
              0.032041, 0.045601, 0.051617, 0.053625, 0.054286, 0.054498, 0.054589]
    wall_y = [0.01, 0.010244, 0.010973, 0.012023, 0.012556, 0.012738, 0.012799, 0.012819, 0.012825,
              0.014577, 0.016172, 0.016432, 0.01646, 0.016463, 0.016463, 0.016463]
    blade = parametrize_stator_blade(wall_x, wall_y, metal_angle_in=0.0, metal_angle_out=65.0,
                                     r_trailing=0.5, inlet_opening_ratio=1.3)
    nozzle = {"wall_final": {"rho": [2.3, 2.3], "T": [300.0, 300.0]}, "fluid_name": "Nitrogen"}
    sizing = size_stator_mode_a(blade, nozzle, mdot=1.0, n_blades=N_BLADES, r_hub=R_HUB)
    mesh = mesh_sized_stator(sizing["scaled_blade"], _coarse_sizes(sizing))
    assert not gmsh.isInitialized()
    return sizing, mesh


def _coarse_sizes(sizing):
    pitch = sizing["scaled_blade"]["pitch"]
    return {"blade_size": pitch / 60, "domain_size": pitch / 15, "first_height": pitch / 1000,
            "growth_ratio": 1.3, "num_layers": 5, "transition_ratio": 1.2}


@pytest.fixture(scope="module")
def structured_mesh(sized_mesh):
    sizing, _ = sized_mesh
    mesh = mesh_sized_stator(sizing["scaled_blade"], _coarse_sizes(sizing), topology="structured")
    assert not gmsh.isInitialized()
    return sizing, mesh


def test_structured_mesh_is_all_quads_valid_and_periodic(structured_mesh):
    sizing, mesh = structured_mesh
    assert mesh["topology"] == "structured" and mesh["fallback_reason"] is None
    stats = mesh["statistics"]
    assert stats["quadrilateral_fraction"] == 1.0
    assert min(stats["block_min_scaled_jacobian"].values()) > 0.3
    offset = _periodic_offset(np.asarray(mesh["xy"]), mesh)
    assert np.isclose(abs(offset[0]), 2 * np.pi * R_HUB / N_BLADES) and np.isclose(offset[1], 0)


def test_structured_wall_cells_have_the_requested_height_and_are_orthogonal(structured_mesh):
    """First cell: wall-normal height = first_height (what y+ depends on),
    and wall-normal grid lines (the O-grid is chunked so they don't drift)."""
    sizing, mesh = structured_mesh
    xy, cells = np.asarray(mesh["xy"]), mesh["cells"]
    heights, slants = [], []
    for a, b, cell, _ in mesh["named_edges"]["blade"]:
        others = [n for n in cells[cell - 1] if n not in (a, b)]
        pa, pb = xy[a - 1], xy[b - 1]
        outer = xy[np.asarray(others) - 1].mean(axis=0)
        edge = pb - pa
        normal_height = abs(edge[0] * (outer - pa)[1] - edge[1] * (outer - pa)[0]) / np.linalg.norm(edge)
        heights.append(normal_height)
        slants.append(np.degrees(np.arccos(min(1.0, normal_height / np.linalg.norm(outer - 0.5 * (pa + pb))))))
    ratio = np.asarray(heights) / _coarse_sizes(sizing)["first_height"]
    assert 0.9 < ratio.min() and ratio.max() < 1.02
    assert np.median(slants) < 3.0


def test_structured_failure_falls_back_to_unstructured(sized_mesh, monkeypatch):
    from turbo_moc.meshing import stator_structured_2d

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(stator_structured_2d, "mesh_structured_passage", fail)
    sizing, _ = sized_mesh
    mesh = mesh_sized_stator(sizing["scaled_blade"], _coarse_sizes(sizing), topology="structured")
    assert mesh["topology"] == "unstructured" and "forced" in mesh["fallback_reason"]
    assert mesh["statistics"]["minimum_scaled_jacobian"] > 0


@pytest.mark.parametrize("recombine", [False, True])
def test_folded_cells_are_detected_and_untangled(recombine):
    """Gmsh's minSJ misses a flipped cell on a plane surface; the corner-turn
    check catches it and the guarded relaxation repairs it."""
    from turbo_moc.meshing.stator_mesh_2d import inverted_cell_count, untangle_surface

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        surface = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        for curve in gmsh.model.getBoundary([(2, surface)], oriented=False):
            gmsh.model.mesh.setTransfiniteCurve(curve[1], 6)
        gmsh.model.mesh.setTransfiniteSurface(surface)
        if recombine:
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        assert inverted_cell_count(surface) == 0
        tags, xyz, _ = gmsh.model.mesh.getNodes(2, surface, includeBoundary=False)
        xyz = np.asarray(xyz).reshape(-1, 3)
        node = int(np.argmin(np.linalg.norm(xyz - [0.4, 0.4, 0], axis=1)))
        gmsh.model.mesh.setNode(int(tags[node]), [0.75, 0.75, 0.0], [])  # past its neighbours
        assert inverted_cell_count(surface) > 0
        assert untangle_surface(surface) == 0 and inverted_cell_count(surface) == 0
        element_tags = np.concatenate(gmsh.model.mesh.getElements(2, surface)[1])
        assert np.min(gmsh.model.mesh.getElementQualities(element_tags, "minSJ")) > 0
    finally:
        gmsh.finalize()


def _periodic_offset(points, mesh):
    current = np.asarray(mesh["periodic"]["current"]) - 1
    donor = np.asarray(mesh["periodic"]["donor"]) - 1
    offsets = points[donor] - points[current]
    assert np.allclose(offsets, offsets[0], atol=1e-8)
    return offsets[0]


def test_hub_mesh_is_valid_and_periodic_by_the_sized_pitch(sized_mesh):
    sizing, mesh = sized_mesh
    stats = mesh["statistics"]
    assert stats["minimum_scaled_jacobian"] > 0
    assert stats["quadrilateral_fraction"] > 0.5 and stats["inflation_layers"] > 0
    offset = _periodic_offset(np.asarray(mesh["xy"]), mesh)
    assert np.isclose(abs(offset[0]), 2 * np.pi * R_HUB / N_BLADES) and np.isclose(offset[1], 0)


@pytest.mark.parametrize("span", [0.5, 1.0])
def test_unrolled_section_is_periodic_by_the_local_pitch(sized_mesh, span):
    sizing, mesh = sized_mesh
    radius = R_HUB + span * (sizing["r_shroud"] - R_HUB)
    points = section_coordinates(mesh["xy"], R_HUB, radius, "unrolled")
    offset = _periodic_offset(points, mesh)
    assert np.isclose(abs(offset[0]), 2 * np.pi * radius / N_BLADES) and np.isclose(offset[1], 0)


def test_cylinder_blade_nodes_lie_on_the_solid_section(sized_mesh):
    """The cylinder view must coincide with the annular solid the app exports
    (wrap_blade_annular, flare='pitch_scale') at the same radius."""
    sizing, mesh = sized_mesh
    wrapped = wrap_blade_annular(sizing["scaled_blade"], R_HUB, sizing["r_shroud"],
                                 n_blades=N_BLADES, flare="pitch_scale", n_span=3, source="stator")
    radius = wrapped["curve"]["r"][0, 1, 0]
    starts, ends = [], []
    for key in ("curve", "trailing_edge"):
        polyline = np.column_stack([wrapped[key][axis][0, 1] for axis in ("x", "y", "z")])
        starts.append(polyline[:-1])
        ends.append(polyline[1:])
    starts, ends = np.vstack(starts), np.vstack(ends)

    points = section_coordinates(mesh["xy"], R_HUB, radius, "cylinder")
    assert np.allclose(np.hypot(points[:, 0], points[:, 1]), radius)
    blade_nodes = points[np.unique(np.asarray(mesh["named_edges"]["blade"])[:, :2]) - 1]
    # distance from each blade node to the solid's sampled section (segments)
    edge = ends - starts
    t = np.clip(np.einsum("pij,ij->pi", blade_nodes[:, None] - starts, edge)
                / np.einsum("ij,ij->i", edge, edge), 0.0, 1.0)
    nearest = starts + t[..., None] * edge
    distances = np.min(np.linalg.norm(blade_nodes[:, None] - nearest, axis=2), axis=1)
    # a mapping with the wrong theta (-x/r instead of -x/r_hub) is ~1.4 mm off here
    assert distances.max() < 1e-3 * mesh["pitch"]


def test_step_fluid_domain_is_the_meshed_domain(sized_mesh, tmp_path):
    """The STEP export and the mesh use one domain definition. Areas come from
    densely sampled boundary curves: OpenCASCADE's own face-area integration
    is ~0.4% off on these B-spline-bounded faces, which would mask a real
    mismatch."""
    cq = pytest.importorskip("cadquery")
    from turbo_moc.meshing.stator_mesh_2d import export_flow_domain_step

    sizing, mesh = sized_mesh
    path = tmp_path / "domain.step"
    export_flow_domain_step(sizing["scaled_blade"], path)
    face = cq.importers.importStep(str(path)).faces().vals()
    assert len(face) == 1 and len(face[0].innerWires()) == 1

    def wire_area(wire):
        # Chain the edges head-to-tail, then shoelace the dense polyline.
        segments = [np.array([[p.x, p.y] for p in (edge.positionAt(t) for t in np.linspace(0, 1, 2000))])
                    for edge in wire.Edges()]
        chain = [segments.pop(0)]
        while segments:
            end = chain[-1][-1]
            k = min(range(len(segments)), key=lambda i: min(np.linalg.norm(segments[i][[0, -1]] - end, axis=1)))
            seg = segments.pop(k)
            chain.append(seg if np.linalg.norm(seg[0] - end) <= np.linalg.norm(seg[-1] - end) else seg[::-1])
        p = np.vstack(chain)
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    step_area = wire_area(face[0].outerWire()) - wire_area(face[0].innerWires()[0])
    xy = np.asarray(mesh["xy"])
    mesh_area = sum(
        0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        for q in (xy[np.asarray(cell) - 1] for cell in mesh["cells"])
    )
    assert np.isclose(step_area, mesh_area, rtol=1e-4)


def test_cgns_written_from_arrays_round_trips(sized_mesh, tmp_path):
    sizing, mesh = sized_mesh
    radius = sizing["r_shroud"]
    points = section_coordinates(mesh["xy"], R_HUB, radius, "unrolled")
    types = {"blade": "BCWallViscous", "inlet": "BCInflowSubsonic", "outlet": "BCOutflowSubsonic",
             "periodic_left": "BCGeneral", "periodic_right": "BCGeneral"}
    path = tmp_path / "mesh.cgns"
    write_fluent_cgns_2d(path, points, mesh["cells"], mesh["named_edges"],
                         (mesh["periodic"]["current"], mesh["periodic"]["donor"]),
                         {"length_scale": 0.001, "boundary_types": types})

    def find(node, label):
        return ([node] if node.label == label else []) + [n for c in node.children for n in find(c, label)]

    root = read_adf(str(path))
    assert sorted(n.name for n in find(root, "BC_t")) == sorted(types)
    assert len(find(root, "GridConnectivity_t")) == 2
    translations = [n for n in find(root, "DataArray_t") if n.name == "Translation"]
    assert np.allclose([abs(np.asarray(n.data)[0]) for n in translations],
                       2 * np.pi * radius / N_BLADES * 0.001, rtol=1e-5)
