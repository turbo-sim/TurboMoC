"""Small mapped meshes: coordinate endpoints, orientations, CGNS and periodics."""

from pathlib import Path
import sys

import gmsh
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.gmsh_3D import geometry_source as geo
from examples.gmsh_3D import meshing_source as msh
from examples.gmsh_3D.run_mesh_generation import read_configuration
from turbo_moc.io.cgns_adf import read_adf


def test_physical_chords_and_radial_endpoints():
    config = read_configuration()
    channel = geo.Channel(geo.geometry_from_machine(config["machine"]))
    for eta, chord in (
        (0, config["machine"]["hub_chord"]),
        (1, config["machine"]["tip_chord"]),
    ):
        q = np.array(
            [[0, 0], [0, -channel.reference_radius * channel.prime_chord(eta)]]
        )
        xyz = channel.map_points(q, eta)
        assert xyz[0, 2] == 0
        assert xyz[1, 2] == pytest.approx(chord)
    config = read_configuration("examples/gmsh_3D/radial_stator.yaml")
    channel = geo.Channel(geo.geometry_from_machine(config["machine"]))
    q = np.array([[0, 0], [0, -channel.reference_chord]])
    for eta in (0, 0.5, 1):
        xyz = channel.map_points(q, eta)
        assert np.linalg.norm(xyz[:, :2], axis=1) == pytest.approx(
            [config["machine"]["inlet_radius"], config["machine"]["outlet_radius"]]
        )
        assert xyz[:, 2] == pytest.approx(
            np.full(2, eta * config["machine"]["blade_height"])
        )


def curved_geometry():
    """A curved mixed-direction channel specified without a machine type."""
    knots = [0, 0, 0, 0, 1, 1, 1, 1]
    return geo.GeometryDefinition(
        hub=BSpline(knots, [[100, 0], [100, 12], [102, 28], [104, 40]], 3),
        shroud=BSpline(knots, [[120, 0], [120, 10], [122, 22], [124, 32]], 3),
        num_blades=12,
    )


def test_general_curved_meridional_mapping():
    definition = curved_geometry()
    channel = geo.Channel(definition)
    for eta in (0, 0.5, 1):

        def wall(u, derivative=False):
            values = []
            for curve in (definition.hub, definition.shroud):
                station = np.clip(u, 0, 1)
                tangent = curve.derivative()(station)
                values.append(
                    tangent if derivative else curve(station) + (u - station) * tangent
                )
            return (1 - eta) * values[0] + eta * values[1]

        stations = np.array([-0.15, 0, 0.4, 1, 1.2])
        prime = np.array(
            [
                quad(
                    lambda u: np.linalg.norm(wall(u, derivative=True)) / wall(u)[0],
                    0,
                    end,
                    epsabs=1e-12,
                    epsrel=1e-12,
                    points=[0, 1] if end > 1 else None,
                )[0]
                for end in stations
            ]
        )
        q = np.column_stack((np.zeros(len(prime)), -channel.reference_radius * prime))
        xyz = channel.map_points(q, eta)
        expected = np.array([wall(u) for u in stations])
        assert np.allclose(xyz[:, (0, 2)], expected, rtol=2e-9, atol=2e-8)
        assert np.allclose(xyz[:, 1], 0)
        assert channel.prime_chord(eta) == pytest.approx(prime[3], rel=1e-10)

    with pytest.raises(ValueError, match="degenerate"):
        geo.Channel(geo.GeometryDefinition(definition.hub, definition.hub, 12))


@pytest.mark.parametrize("case", ["axial", "radial_inflow", "radial_outflow", "curved"])
def test_tiny_volume_export_and_periodic_rotation(case, tmp_path):
    name = "axial_stator" if case == "axial" else "radial_stator"
    config = read_configuration(f"examples/gmsh_3D/{name}.yaml")
    if case == "radial_outflow":
        m = config["machine"]
        m["inlet_radius"], m["outlet_radius"] = m["outlet_radius"], m["inlet_radius"]
    definition = (
        curved_geometry()
        if case == "curved"
        else geo.geometry_from_machine(config["machine"])
    )
    channel = geo.Channel(definition)
    p, c = channel.pitch, channel.reference_chord
    # Four CCW quads surrounding a blade hole, with opposite sides translated.
    xy = np.array(
        [
            [-0.5 * p, -c],
            [0.5 * p, -c],
            [0.5 * p, 0.25 * c],
            [-0.5 * p, 0.25 * c],
            [-0.1 * p, -0.45 * c],
            [0.1 * p, -0.45 * c],
            [0.1 * p, -0.25 * c],
            [-0.1 * p, -0.25 * c],
        ]
    )
    section = {
        "xy": xy,
        "cells": {
            3: np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
        },
        "boundaries": {
            "blade": np.array([[4, 7], [7, 6], [6, 5], [5, 4]]),
            "inlet": np.array([[2, 3]]),
            "outlet": np.array([[0, 1]]),
            "periodic_left": np.array([[3, 0]]),
            "periodic_right": np.array([[1, 2]]),
        },
        "slave": np.array([1, 2]),
        "master": np.array([0, 3]),
    }
    xyz = np.array([channel.map_points(xy, eta) for eta in (0, 0.5, 1)])
    gmsh.initialize(["test_3d", "-v", "0"])
    try:
        counts = msh.assemble_volume(section, xyz, channel)
        quality = msh.validate_volume(section, xyz, channel)
        assert counts == {"hexahedra": 8, "prisms": 0}
        assert quality["min_determinant_jacobian"] > 0
        assert quality["periodic_node_pairs"] == 6
        path = tmp_path / f"{case}.cgns"
        result = msh.export_cgns(path, config["mesh"]["cgns"], channel)
        assert result["validation"]["boundary_partition_complete"]
        base = read_adf(path).get(label="CGNSBase_t")
        assert np.array_equal(base.data, [3, 3])
        zone = base.get(label="Zone_t")
        assert {bc.name for bc in zone.get(label="ZoneBC_t").children} == set(
            msh.BOUNDARIES
        )
        grid = zone.get(label="GridCoordinates_t")
        points = np.column_stack([grid.get(f"Coordinate{axis}").data for axis in "XYZ"])
        assert np.allclose(points, xyz.reshape(-1, 3) * 0.001)
        for interface in zone.get(label="ZoneGridConnectivity_t").children:
            current = interface.get("PointList").data.ravel() - 1
            donor = interface.get("PointListDonor").data.ravel() - 1
            periodic = interface.get(label="GridConnectivityProperty_t").get(
                label="Periodic_t"
            )
            angle = periodic.get("RotationAngle").data[2]
            cs, sn = np.cos(angle), np.sin(angle)
            rotation = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
            assert np.allclose(points[current] @ rotation.T, points[donor], atol=1e-8)
    finally:
        gmsh.finalize()
