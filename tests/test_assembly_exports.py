"""Small real CGNS exports check assembly placement and physical interfaces."""
import json

import gmsh
import numpy as np
import pytest

from turbo_moc.configuration_project import default_project, row_configuration
from turbo_moc.geometry import meridional_channel as meridional
from turbo_moc.io.cgns_adf import read_adf
from turbo_moc.io.cgns_adf import read_adf_bytes
from turbo_moc.io.cgns_stage import combine_rows
from turbo_moc.meshing import meshing_3d as mesh


def _tiny_section(mapping):
    """Four fluid quads around a blade hole; this retains all seven 3D BCs."""
    pitch, chord = mapping.pitch, mapping.reference_chord
    return {
        "xy": np.array([[-.5*pitch, -chord], [.5*pitch, -chord],
                        [.5*pitch, .25*chord], [-.5*pitch, .25*chord],
                        [-.1*pitch, -.45*chord], [.1*pitch, -.45*chord],
                        [.1*pitch, -.25*chord], [-.1*pitch, -.25*chord]]),
        "cells": {3: np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])},
        "boundaries": {
            "blade": np.array([[4, 7], [7, 6], [6, 5], [5, 4]]),
            "inlet": np.array([[2, 3]]), "outlet": np.array([[0, 1]]),
            "periodic_left": np.array([[3, 0]]), "periodic_right": np.array([[1, 2]])},
        "slave": np.array([1, 2]), "master": np.array([0, 3])}


def test_close_shared_cap_preserves_collar_and_slides_only_tangentially():
    """A close cap needs less artificial core protection, not thinner layers."""
    points = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3)], dtype=float)
    cells = np.array([[5*j+i, 5*j+i+1, 5*(j+1)+i+1, 5*(j+1)+i]
                      for j in range(4) for i in range(4)
                      if not (i in (1, 2) and j in (1, 2))])
    used = np.unique(cells)
    points, cells = points[used], np.searchsorted(used, cells)
    edges = np.sort(np.concatenate([cells[:, [j, (j+1) % 4]] for j in range(4)]), axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    boundaries = {name: [] for name in ("blade", "inlet", "outlet", "periodic_left", "periodic_right")}
    for edge in edges[counts == 1]:
        ends = points[edge]
        name = ("inlet" if np.all(ends[:, 1] == 2) else
                "outlet" if np.all(ends[:, 1] == -2) else
                "periodic_left" if np.all(ends[:, 0] == -2) else
                "periodic_right" if np.all(ends[:, 0] == 2) else "blade")
        boundaries[name].append(edge)
    section = {"xy": points, "cells": {3: cells},
               "boundaries": {name: np.array(edges) for name, edges in boundaries.items()},
               "profile": {"enabled": True, "stack_thickness": .8},
               "slave": np.flatnonzero(points[:, 0] == 2),
               "master": np.flatnonzero(points[:, 0] == -2)}
    reference = {"inlet": np.array([[-2, 2], [2, 2]], float),
                 "outlet": np.array([[-2, -2], [2, -2]], float),
                 "periodic_left": np.array([[-2, 2], [-2, -2]], float),
                 "periodic_right": np.array([[2, 2], [2, -2]], float)}
    scale = 1.05
    target = {name: values * [1, scale] for name, values in reference.items()}
    deformation = mesh.CoreDeformation(section, {"blade_size": .25}, sliding_boundaries=("inlet",))
    result = deformation.deform(scale, reference, target, 4.)
    # Full prescribed collar stays a similarity-scaled copy of the reference.
    assert np.allclose(result[deformation.collar], points[deformation.collar] * scale)
    inlet = np.flatnonzero(points[:, 1] == 2)
    interior = inlet[np.abs(points[inlet, 0]) < 2]
    assert np.allclose(result[inlet, 1], 2 * scale)
    assert not np.allclose(result[interior, 0], points[interior, 0])
    assert np.all(np.diff(result[inlet, 0]) > 0)
    outlet = np.flatnonzero(points[:, 1] == -2)
    assert np.allclose(result[outlet, 0], points[outlet, 0])
    section["profile"]["stack_thickness"] = 1.1
    with pytest.raises(ValueError, match="Insufficient sampled wall clearance"):
        mesh.CoreDeformation(section, {"blade_size": .25}, sliding_boundaries=("inlet",))


def test_narrow_periodic_core_strip_can_move_without_erasing_inflation_layers():
    """Extra pinned core rows would cross a periodic side under 12% scaling."""
    x = np.array([-2.3, -2.1, -1.8, -1.5, -1, 0, 1, 1.5, 1.8, 2.1, 2.3])
    y = np.array([-3, -2.5, -2, -1, 0, 1, 2, 2.5, 3])
    points = np.array([(a, b) for b in y for a in x])
    cells = np.array([[len(x)*j+i, len(x)*j+i+1, len(x)*(j+1)+i+1, len(x)*(j+1)+i]
                      for j in range(len(y)-1) for i in range(len(x)-1)
                      if not (-1 <= x[i] < 1 and -1 <= y[j] < 1)])
    used = np.unique(cells)
    points, cells = points[used], np.searchsorted(used, cells)
    edges = np.sort(np.concatenate([cells[:, [j, (j+1) % 4]] for j in range(4)]), axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    boundaries = {name: [] for name in ("blade", "inlet", "outlet", "periodic_left", "periodic_right")}
    for edge in edges[counts == 1]:
        ends = points[edge]
        name = ("inlet" if np.all(ends[:, 1] == 3) else "outlet" if np.all(ends[:, 1] == -3) else
                "periodic_left" if np.all(ends[:, 0] == -2.3) else
                "periodic_right" if np.all(ends[:, 0] == 2.3) else "blade")
        boundaries[name].append(edge)
    section = {"xy": points, "cells": {3: cells},
               "boundaries": {name: np.array(items) for name, items in boundaries.items()},
               "profile": {"enabled": True, "stack_thickness": .8},
               "slave": np.flatnonzero(points[:, 0] == 2.3),
               "master": np.flatnonzero(points[:, 0] == -2.3)}
    reference = {"inlet": np.array([[-2.3, 3], [2.3, 3]]),
                 "outlet": np.array([[-2.3, -3], [2.3, -3]]),
                 "periodic_left": np.array([[-2.3, 3], [-2.3, -3]]),
                 "periodic_right": np.array([[2.3, 3], [2.3, -3]])}
    scale = 1.12
    target = {name: values * [1, scale] for name, values in reference.items()}
    deformation = mesh.CoreDeformation(section, {"blade_size": .25})
    result = deformation.deform(scale, reference, target, 4.6)
    # The node at the nominal outer layer retains the complete scaled stack.
    layer = np.flatnonzero(np.all(np.isclose(points, [1.8, 0]), axis=1))[0]
    core = np.flatnonzero(np.all(np.isclose(points, [2.1, 0]), axis=1))[0]
    np.testing.assert_allclose(result[layer], points[layer] * scale)
    assert result[layer, 0] < result[core, 0] < 2.3
    np.testing.assert_allclose(result[section["slave"]] - result[section["master"]],
                               np.tile([4.6, 0], (len(section["slave"]), 1)))


@pytest.mark.parametrize("family,outflow", [("axial", False), ("radial", False), ("radial", True)])
def test_export_applies_assembly_placement_and_preserves_periodicity(tmp_path, family, outflow):
    config = row_configuration(default_project(family), "stator_1")
    channel = config["meridional_channel"]
    if family == "axial":
        channel["tip_chord"] = channel["hub_chord"]
    elif outflow:
        channel["inlet_radius"], channel["outlet_radius"] = channel["outlet_radius"], channel["inlet_radius"]
    mapping = meridional.MeridionalChannel(meridional.geometry_from_machine(channel))
    section = _tiny_section(mapping)
    points = []
    for eta in (0, .5, 1):
        section_xy = section["xy"].copy()
        # Equal axial physical chords need different m' at different radii.
        section_xy[:, 1] *= mapping.section_scale(eta)
        points.append(mapping.map_points(section_xy, eta))
    points = np.asarray(points)
    placement = {"angular_phase_deg": 31.0, "streamwise_offset_mm": 43.0 if family == "axial" else 0.0}
    target = (points[0, 0, 2] + placement["streamwise_offset_mm"] if family == "axial"
              else float(np.linalg.norm(points[0, 0, :2])))
    assembly = {"row_id": "stator_1", "placement": placement, "interfaces": {"outlet": {
        "id": "stator_1_to_rotor_1", "geometry_type": family, "coordinate_mm": target,
        "upstream_id": "stator_1", "downstream_id": "rotor_1"}}}
    gmsh.initialize(["assembly_export_test", "-v", "0"], readConfigFiles=False)
    try:
        mesh.assemble_volume(section, points, mapping)
        _, before, _ = gmsh.model.mesh.getNodes()
        result = mesh.export_cgns(tmp_path / "placed.cgns", config["mesh"]["cgns"], mapping,
                                  assembly=assembly)
        _, after, _ = gmsh.model.mesh.getNodes()
        assert np.array_equal(before, after), "Export must not reposition native/viewer mesh arrays."
        assert result["validation"]["boundary_normals_outward"]
        assert result["validation"]["shared_interfaces"]["outlet"]["max_error_mm"] < 1e-8
        base = read_adf(result["file"]).get(label="CGNSBase_t")
        assert json.loads(base.get("Assembly").text) == assembly
        zone = base.get(label="Zone_t")
        grid = zone.get(label="GridCoordinates_t")
        exported = np.column_stack([grid.get(f"Coordinate{axis}").data for axis in "XYZ"])
        angle = np.deg2rad(placement["angular_phase_deg"])
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        expected = points.reshape(-1, 3) @ rotation.T + [0, 0, placement["streamwise_offset_mm"]]
        assert np.allclose(exported, expected * config["mesh"]["cgns"]["length_scale"], atol=1e-12)
        bc = zone.get(label="ZoneBC_t").get("outlet")
        assert json.loads(bc.get("AssemblyInterface").text) == assembly["interfaces"]["outlet"]
        for interface in zone.get(label="ZoneGridConnectivity_t").children:
            current = interface.get("PointList").data.ravel() - 1
            donor = interface.get("PointListDonor").data.ravel() - 1
            periodic = interface.get(label="GridConnectivityProperty_t").get(label="Periodic_t")
            angle = periodic.get("RotationAngle").data[2]
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            transformed = exported[current] @ rotation.T + periodic.get("Translation").data
            assert np.allclose(transformed, exported[donor], atol=1e-8)
        assembly["interfaces"]["outlet"]["coordinate_mm"] += .01
        with pytest.raises(ValueError, match="does not lie on shared interface"):
            mesh.export_cgns(tmp_path / "invalid.cgns", config["mesh"]["cgns"], mapping,
                             assembly=assembly)
        assert not (tmp_path / "invalid.cgns").exists()
    finally:
        gmsh.finalize()


def test_real_two_row_stage_preserves_periodicity_and_separate_interface_faces(tmp_path):
    """Independent actual Gmsh volumes can be combined without renumbering."""
    sources, source_zones, interface_z = [], [], None
    gmsh.initialize(["stage_export_test", "-v", "0"], readConfigFiles=False)
    try:
        for row_id, blades, phase, boundary in (("stator_1", 12, 8., "outlet"),
                                                 ("rotor_1", 18, 19., "inlet")):
            config = row_configuration(default_project("axial"), "stator_1")
            config["meridional_channel"].update(num_blades=blades, tip_chord=20., hub_chord=20.)
            mapping = meridional.MeridionalChannel(meridional.geometry_from_machine(config["meridional_channel"]))
            section = _tiny_section(mapping)
            points = np.asarray([mapping.map_points(section["xy"] * [1, mapping.section_scale(eta)], eta)
                                 for eta in (0., .5, 1.)])
            cap_node = section["boundaries"][boundary][0, 0]
            cap_z = points[0, cap_node, 2]
            if interface_z is None:
                interface_z = cap_z + 43.
            assembly = {"row_id": row_id, "placement": {
                "angular_phase_deg": phase, "streamwise_offset_mm": interface_z - cap_z},
                "interfaces": {boundary: {"id": "stator_1_to_rotor_1", "geometry_type": "axial",
                    "coordinate_mm": interface_z, "upstream_id": "stator_1", "downstream_id": "rotor_1"}}}
            mesh.assemble_volume(section, points, mapping)
            path = tmp_path / f"{row_id}.cgns"
            mesh.export_cgns(path, config["mesh"]["cgns"], mapping, assembly=assembly)
            sources.append((row_id, path))
            source_zones.append(read_adf(path).get(label="CGNSBase_t").get(label="Zone_t"))
            gmsh.clear()
    finally:
        gmsh.finalize()
    data, manifest = combine_rows(sources)
    zones = read_adf_bytes(data).get(label="CGNSBase_t").find_all("Zone_t")
    assert len(zones) == 2
    for (row_id, _), original, zone in zip(sources, source_zones, zones):
        # No cross-row connectivity and no welded vertices, even on the cap.
        assert np.array_equal(zone.data, original.data)
        grid = zone.get(label="GridCoordinates_t")
        coordinates = np.column_stack([grid.get(f"Coordinate{axis}").data for axis in "XYZ"])
        original_grid = original.get(label="GridCoordinates_t")
        assert np.array_equal(coordinates, np.column_stack([
            original_grid.get(f"Coordinate{axis}").data for axis in "XYZ"]))
        for before, after in zip(original.find_all("Elements_t"), zone.find_all("Elements_t")):
            assert np.array_equal(before.get("ElementConnectivity").data, after.get("ElementConnectivity").data)
        for connection in zone.get(label="ZoneGridConnectivity_t").children:
            assert connection.text == zone.name
            current = connection.get("PointList").data.ravel() - 1
            donor = connection.get("PointListDonor").data.ravel() - 1
            periodic = connection.get(label="GridConnectivityProperty_t").get(label="Periodic_t")
            angle = periodic.get("RotationAngle").data[2]
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            assert np.allclose(coordinates[current] @ rotation.T + periodic.get("Translation").data,
                               coordinates[donor], atol=1e-8)
        side = "outlet" if row_id == "stator_1" else "inlet"
        bc = zone.get(label="ZoneBC_t").get(manifest["rows"][row_id]["boundaries"][side])
        assert json.loads(bc.get("AssemblyInterface").text)["coordinate_mm"] == interface_z
