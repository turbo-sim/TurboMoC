"""Swept passage volumes: discrete hex/prism assembly, checks and CGNS export.

Geometry-independent core shared by the app's swept stator mesh
(turbo_moc.meshing.stator_mesh_3d) and examples/gmsh_3D: a 2D passage
section (quads/triangles, named boundary edges, periodic node pairs) plus
its node coordinates on each span station become hexahedra/prisms with
seven named patches and rotational periodicity about Z.

`channel` is any object with `orientation` (+1/-1: cell winding that gives
positive volumes), `pitch_angle` [rad], `reference_radius` [mm] and
`rotation()` (4x4 homogeneous periodic_left -> periodic_right transform).
Volume assembly and the CGNS audit follow ParaBlade's mesh_examples/3D and
3D_v2, without a ParaBlade dependency.
"""

from pathlib import Path

import gmsh
import numpy as np

from turbo_moc.io.cgns_adf import AdfNode, new_root, read_adf, write_adf

BOUNDARIES = (
    "blade",
    "inlet",
    "outlet",
    "periodic_left",
    "periodic_right",
    "hub",
    "shroud",
)
SURFACES = {name: tag for tag, name in enumerate(BOUNDARIES, 1)}


# =============================================================================
# 3. Discrete volume assembly and rotational periodic correspondence
# =============================================================================


def assemble_volume(section, xyz_sections, channel):
    """Turn section quads/triangles into oriented hexes/prisms and surface faces."""
    gmsh.model.add("moc_3d_passage")
    for name, tag in SURFACES.items():
        gmsh.model.addDiscreteEntity(2, tag)
        gmsh.model.setEntityName(2, tag, name)
    gmsh.model.addDiscreteEntity(3, 1, list(SURFACES.values()))
    n_span, n, _ = xyz_sections.shape
    xyz = xyz_sections.reshape(-1, 3)
    gmsh.model.mesh.addNodes(3, 1, np.arange(1, len(xyz) + 1), xyz.ravel())
    offsets = np.arange(n_span - 1)[:, None, None] * n
    counts = {"hexahedra": 0, "prisms": 0}
    for kind, cells in section["cells"].items():
        if channel.orientation < 0:
            cells = cells[:, ::-1]
        bottom = cells[None] + offsets + 1
        volumes = np.concatenate((bottom, bottom + n), axis=2)
        gmsh.model.mesh.addElementsByType(1, 5 if kind == 3 else 6, [], volumes.ravel())
        counts["hexahedra" if kind == 3 else "prisms"] = int(np.prod(volumes.shape[:2]))
        gmsh.model.mesh.addElementsByType(
            SURFACES["hub"], kind, [], (cells[:, ::-1] + 1).ravel()
        )
        gmsh.model.mesh.addElementsByType(
            SURFACES["shroud"], kind, [], (cells + (n_span - 1) * n + 1).ravel()
        )
    for name, edges in section["boundaries"].items():
        if channel.orientation < 0:
            edges = edges[:, ::-1]
        bottom = edges[None] + offsets + 1
        faces = np.concatenate((bottom, bottom[:, :, ::-1] + n), axis=2)
        gmsh.model.mesh.addElementsByType(SURFACES[name], 3, [], faces.ravel())
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.mesh.createTopology(makeSimplyConnected=False)
    for name, tag in SURFACES.items():
        gmsh.model.addPhysicalGroup(2, [tag], tag, name)
    gmsh.model.addPhysicalGroup(3, [1], 1, "fluid_domain")
    gmsh.model.mesh.setPeriodic(
        2,
        [SURFACES["periodic_right"]],
        [SURFACES["periodic_left"]],
        channel.rotation().ravel(),
    )
    return counts


def validate_volume(section, xyz_sections, channel):
    """Check Jacobians, seven exterior patches and a complete rotational bijection."""
    tags = np.concatenate(gmsh.model.mesh.getElements(3, 1)[1])
    jacobian = gmsh.model.mesh.getElementQualities(tags, "minDetJac")
    quality = gmsh.model.mesh.getElementQualities(tags, "minSICN")
    if (
        not np.isfinite(jacobian).all()
        or min(jacobian) <= 0
        or not np.isfinite(quality).all()
        or min(quality) <= 0
    ):
        raise ValueError(
            "The swept volume contains inverted/degenerate cells; refine or reduce taper."
        )
    master_entity, slave, master, transform = gmsh.model.mesh.getPeriodicNodes(
        2, SURFACES["periodic_right"]
    )
    n_span, n, _ = xyz_sections.shape
    offsets = np.arange(n_span)[:, None] * n + 1
    expected_slave = (section["slave"] + offsets).ravel()
    expected_master = (section["master"] + offsets).ravel()
    order, expected_order = np.argsort(slave), np.argsort(expected_slave)
    if (
        master_entity != SURFACES["periodic_left"]
        or not np.array_equal(slave[order], expected_slave[expected_order])
        or not np.array_equal(master[order], expected_master[expected_order])
    ):
        raise ValueError("Incomplete swept rotational periodic node map.")
    xyz = xyz_sections.reshape(-1, 3)
    error = np.linalg.norm(
        xyz[slave - 1] - xyz[master - 1] @ channel.rotation()[:3, :3].T, axis=1
    )
    if (
        not np.allclose(np.asarray(transform).reshape(4, 4), channel.rotation())
        or max(error) > 1e-8 * channel.reference_radius
    ):
        raise ValueError("Periodic surfaces do not match a one-pitch rotation.")
    exterior = {tag for _, tag in gmsh.model.getBoundary([(3, 1)], oriented=False)}
    if exterior != set(SURFACES.values()):
        raise ValueError(
            "The volume does not have the seven expected exterior patches."
        )
    return {
        "min_determinant_jacobian": float(min(jacobian)),
        "min_sicn": float(min(quality)),
        "periodic_node_pairs": len(slave),
        "max_periodic_error_mm": float(max(error)),
    }


# =============================================================================
# 4. Fluent-oriented CGNS/ADF export (adapted from ParaBlade 3D_v2)
# =============================================================================


ELEMENTS = {
    2: (5, 3, "TRI_3"),
    3: (7, 4, "QUAD_4"),
    5: (17, 8, "HEXA_8"),
    6: (14, 6, "PENTA_6"),
}
BC_TYPES = {
    "Null",
    "UserDefined",
    "BCGeneral",
    "BCDirichlet",
    "BCNeumann",
    "BCExtrapolate",
    "BCWall",
    "BCWallInviscid",
    "BCWallViscous",
    "BCWallViscousHeatFlux",
    "BCWallViscousIsothermal",
    "BCInflow",
    "BCInflowSubsonic",
    "BCInflowSupersonic",
    "BCOutflow",
    "BCOutflowSubsonic",
    "BCOutflowSupersonic",
    "BCFarfield",
    "BCSymmetryPlane",
    "BCSymmetryPolar",
    "BCTunnelInflow",
    "BCTunnelOutflow",
    "BCDegenerateLine",
    "BCDegeneratePoint",
}


def boundary_types(overrides=None):
    """Default to reference-compatible Null; do not infer unspecified physics."""
    result = dict.fromkeys(BOUNDARIES, "Null")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(
                "BC overrides must be a dictionary of boundary name -> CGNS BC type."
            )
        for name, value in overrides.items():
            if name not in result:
                raise ValueError(
                    f"Unknown boundary {name!r}; expected one of {BOUNDARIES}."
                )
            if not isinstance(value, str) or value not in BC_TYPES:
                raise ValueError(
                    f"Unsupported CGNS BC type {value!r}. Use on-disk names such as Null or BCWall."
                )
            result[name] = value
    return result


def _blocks(dim, entities, sorted_tags):
    blocks = {}
    for entity in entities:
        types, _, node_blocks = gmsh.model.mesh.getElements(dim, int(entity))
        for kind, nodes in zip(types, node_blocks):
            kind = int(kind)
            if kind not in ELEMENTS or (dim == 3) != (kind in (5, 6)):
                raise ValueError(
                    f"Unsupported Gmsh element type {kind} on dimension {dim}."
                )
            indices = np.searchsorted(sorted_tags, nodes)
            if np.any(indices >= len(sorted_tags)) or not np.array_equal(
                sorted_tags[indices], nodes
            ):
                raise ValueError("Connectivity refers to absent nodes.")
            blocks.setdefault(kind, []).append(
                (indices + 1).astype(np.int32).reshape(-1, ELEMENTS[kind][1])
            )
    return {kind: np.concatenate(parts) for kind, parts in sorted(blocks.items())}


def _face_keys(nodes):
    """Sortable padded vertex sets, without changing exported face orientation."""
    result = np.zeros((len(nodes), 4), dtype=np.int32)
    result[:, : nodes.shape[1]] = nodes
    return np.sort(result, axis=1)


def _face_audit(coordinates, volumes, boundaries):
    """Check the physical patches are exactly the outward external cell faces.

    Fixed-width int32 keys keep this whole-mesh audit reasonably small for the
    default 578k-cell case. No vertex-based BC inference is used.
    """
    patterns = {
        5: (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
        6: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    }
    keys, owners = [], []
    centers, offset = [], 0
    for kind, cells in volumes.items():
        centers.append(coordinates[cells - 1].mean(axis=1))
        for pattern in patterns[kind]:
            keys.append(_face_keys(cells[:, pattern]))
            owners.append(np.arange(offset, offset + len(cells), dtype=np.int32))
        offset += len(cells)
    keys, owners, centers = (
        np.concatenate(keys),
        np.concatenate(owners),
        np.concatenate(centers),
    )
    row_type = np.dtype([("a", "<i4"), ("b", "<i4"), ("c", "<i4"), ("d", "<i4")])
    records = keys.view(row_type).ravel()
    unique, first, count = np.unique(records, return_index=True, return_counts=True)
    if np.any(count > 2):
        raise ValueError(
            "Nonmanifold volume connectivity: a face has more than two owners."
        )
    outer = count == 1
    external = unique[outer]
    owner_centers = centers[owners[first[outer]]]
    boundary_keys, face_coordinates = [], []
    for blocks in boundaries.values():
        for nodes in blocks.values():
            boundary_keys.append(_face_keys(nodes))
            face_coordinates.append(coordinates[nodes - 1])
    boundary_records = np.concatenate(boundary_keys).view(row_type).ravel()
    if len(np.unique(boundary_records)) != len(boundary_records):
        raise ValueError("A surface face appears in more than one boundary patch.")
    order = np.argsort(boundary_records)
    if not np.array_equal(boundary_records[order], external):
        raise ValueError(
            "Named boundary faces do not exactly cover the volume exterior."
        )
    boundary_centers, normals = [], []
    for vertices in face_coordinates:
        local = vertices - vertices[:, :1]
        normals.append(np.cross(local, np.roll(local, -1, axis=1)).sum(axis=1))
        boundary_centers.append(vertices.mean(axis=1))
    dot = np.einsum(
        "ij,ij->i",
        np.concatenate(normals)[order],
        np.concatenate(boundary_centers)[order] - owner_centers,
    )
    if not np.all(np.isfinite(dot)) or np.any(dot <= 0):
        raise ValueError("CGNS boundary faces must point out of the fluid volume.")
    return {
        "exterior_faces": len(external),
        "interior_faces": int(np.count_nonzero(count == 2)),
        "boundary_partition_complete": True,
        "boundary_normals_outward": True,
    }


def _section(zone, name, blocks, start):
    """Write one contiguous element section, mixing cell types when needed."""
    parts, n_elements = [], 0
    mixed = len(blocks) > 1
    for gmsh_type, cells in blocks.items():
        n_elements += len(cells)
        cgns_type = ELEMENTS[gmsh_type][0]
        if mixed:
            packed = np.column_stack(
                (np.full(len(cells), cgns_type, dtype=np.int32), cells)
            )
            parts.append(packed.ravel())
        else:
            parts.append(cells.ravel())
    if not n_elements:
        raise ValueError(f"Empty element section {name!r}.")
    kind = 20 if mixed else ELEMENTS[next(iter(blocks))][0]
    end = start + n_elements - 1
    section = zone.add(AdfNode(name, "Elements_t", np.array([kind, 0], dtype=np.int32)))
    section.add(
        AdfNode("ElementRange", "IndexRange_t", np.array([start, end], dtype=np.int32))
    )
    section.add(
        AdfNode(
            "ElementConnectivity", "DataArray_t", np.concatenate(parts).astype(np.int32)
        )
    )
    return end + 1, {
        "first_element": start,
        "last_element": end,
        "elements": n_elements,
        "element_type": "MIXED" if mixed else ELEMENTS[next(iter(blocks))][2],
        "triangles": len(blocks.get(2, [])),
        "quadrilaterals": len(blocks.get(3, [])),
    }


def export_cgns(path, settings, channel):
    """Write the current 3D Gmsh mesh as CGNS 3.4 ADF, and audit its BC tree.

    Requires exactly one fluid physical volume plus the seven named patches.
    Returns statistics suitable for the example's JSON/Markdown report. CGNS
    periodic patches remain named BCs; Fluent's rotational pairing is a solver
    setup step, with the transform recorded in the mesh summary and MSH file.
    """
    types = boundary_types(settings["boundary_types"])
    if not gmsh.isInitialized() or gmsh.model.getDimension() != 3:
        raise ValueError("A generated three-dimensional Gmsh model is required.")
    path = Path(path).resolve()
    if path.suffix.lower() != ".cgns":
        raise ValueError("CGNS output requires a .cgns extension.")
    groups = {
        gmsh.model.getPhysicalName(dim, tag): (
            dim,
            gmsh.model.getEntitiesForPhysicalGroup(dim, tag),
        )
        for dim, tag in gmsh.model.getPhysicalGroups()
    }
    if set(groups) != set(BOUNDARIES) | {"fluid_domain"}:
        raise ValueError(
            "Expected fluid_domain and exactly the seven named surface groups."
        )
    if groups["fluid_domain"][0] != 3 or any(groups[n][0] != 2 for n in BOUNDARIES):
        raise ValueError("Physical groups have incorrect dimensions.")
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    coordinates = coordinates.reshape(-1, 3)[order] * settings["length_scale"]
    if len(node_tags) > np.iinfo(np.int32).max or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "CGNS export requires finite coordinates and fewer than 2^31 nodes."
        )
    volumes = _blocks(3, groups["fluid_domain"][1], node_tags)
    boundaries = {name: _blocks(2, groups[name][1], node_tags) for name in BOUNDARIES}
    n_cells = sum(len(c) for c in volumes.values())
    n_boundary = sum(len(c) for b in boundaries.values() for c in b.values())
    if not n_cells or n_cells + n_boundary > np.iinfo(np.int32).max:
        raise ValueError("Invalid cell count for the 32-bit CGNS export.")
    audit = _face_audit(coordinates, volumes, boundaries)
    root = new_root()
    # Pre-4.0 MIXED representation is deliberately chosen for broad importer
    # compatibility: type-prefixed connectivity, without ElementStartOffset.
    root.add(
        AdfNode(
            "CGNSLibraryVersion",
            "CGNSLibraryVersion_t",
            np.array([3.4], dtype=np.float32),
        )
    )
    base = root.add(AdfNode("TurboMoC", "CGNSBase_t", np.array([3, 3], dtype=np.int32)))
    base.add(
        AdfNode(
            "ExportDescription",
            "Descriptor_t",
            "Full 3D swept mesh. Boundary BC_t PointRange values are face-element IDs at FaceCenter. "
            "Periodic vertex maps and rotation about Z are included; pair the patches in the CFD solver.",
        )
    )
    base.add(AdfNode("fluid_family", "Family_t"))
    zone = base.add(
        AdfNode(
            "fluid_domain",
            "Zone_t",
            np.array([[len(node_tags), n_cells, 0]], dtype=np.int32),
        )
    )
    zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
    zone.add(AdfNode("FamilyName", "FamilyName_t", "fluid_family"))
    grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
    for i, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        array = grid.add(
            AdfNode(name, "DataArray_t", coordinates[:, i].astype(np.float64))
        )
        array.add(
            AdfNode(
                "DimensionalExponents",
                "DimensionalExponents_t",
                np.array([0, 1, 0, 0, 0], dtype=np.float32),
            )
        )
    # Fluent creates cell zones from element sections, not just Zone_t. A
    # single MIXED volume section avoids splitting the fluid and its BC patches
    # into separate hexahedron/prism zones on import.
    next_id, _ = _section(zone, "fluid_domain", volumes, 1)
    patches = {}
    zbc = AdfNode("ZoneBC", "ZoneBC_t")
    for name, blocks in boundaries.items():
        next_id, patch = _section(zone, name, blocks, next_id)
        bc = zbc.add(AdfNode(name, "BC_t", types[name]))
        bc.add(
            AdfNode(
                "PointRange",
                "IndexRange_t",
                np.array(
                    [[patch["first_element"], patch["last_element"]]], dtype=np.int32
                ),
            )
        )
        bc.add(AdfNode("GridLocation", "GridLocation_t", "FaceCenter"))
        patches[name] = {
            **patch,
            "bc_type": types[name],
            "grid_location": "FaceCenter",
        }
    zone.add(zbc)
    add_cgns_periodicity(zone, node_tags, channel)
    add_cgns_units(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_adf(str(path), root)
    # Re-read the serialized tree and verify all patches against intended IDs.
    loaded = read_adf(str(path)).get(label="CGNSBase_t")
    loaded_zone = loaded.get(label="Zone_t")
    if not np.array_equal(loaded.data, [3, 3]) or not np.array_equal(
        loaded_zone.data, [[len(node_tags), n_cells, 0]]
    ):
        raise ValueError("CGNS base/zone dimensions did not survive serialization.")
    loaded_bc = loaded_zone.get(label="ZoneBC_t")
    for name, patch in patches.items():
        bc = loaded_bc.get(name)
        if (
            bc.text != types[name]
            or bc.get("GridLocation").text != "FaceCenter"
            or not np.array_equal(
                bc.get("PointRange").data,
                [[patch["first_element"], patch["last_element"]]],
            )
        ):
            raise ValueError(f"CGNS boundary {name} did not survive serialization.")
    return {
        "file": str(path),
        "format": "CGNS 3.4 / ADF",
        "cell_dimension": 3,
        "physical_dimension": 3,
        "nodes": len(node_tags),
        "volume_cells": n_cells,
        "boundary_faces": n_boundary,
        "boundaries": patches,
        "validation": audit,
        "periodic_pairing": "Conforming periodic_left/periodic_right patches; configure rotational pairing in Fluent.",
    }


def add_cgns_units(base):
    """CGNS coordinates are metres, while rotational angles are radians."""
    base.add(AdfNode("DataClass", "DataClass_t", "Dimensional"))
    units = np.full((32, 5), b" ", dtype="S1")
    for column, unit in enumerate(("Kilogram", "Meter", "Second", "Kelvin", "Radian")):
        units[: len(unit), column] = np.frombuffer(unit.encode("ascii"), dtype="S1")
    base.add(AdfNode("DimensionalUnits", "DimensionalUnits_t", units))


def add_cgns_periodicity(zone, node_tags, channel):
    """Store reciprocal vertex correspondences and current-to-donor rotations."""
    _, slave, master, _ = gmsh.model.mesh.getPeriodicNodes(
        2, SURFACES["periodic_right"]
    )
    current = (np.searchsorted(node_tags, slave) + 1).astype(np.int32)
    donor = (np.searchsorted(node_tags, master) + 1).astype(np.int32)
    interfaces = zone.add(AdfNode("ZoneGridConnectivity", "ZoneGridConnectivity_t"))
    # Gmsh stores master -> slave; CGNS specifies current -> connecting patch.
    for name, nodes, donors, angle in (
        ("right_to_left", current, donor, channel.pitch_angle),
        ("left_to_right", donor, current, -channel.pitch_angle),
    ):
        interface = interfaces.add(AdfNode(name, "GridConnectivity_t", "fluid_domain"))
        interface.add(
            AdfNode("GridConnectivityType", "GridConnectivityType_t", "Abutting1to1")
        )
        interface.add(AdfNode("GridLocation", "GridLocation_t", "Vertex"))
        interface.add(AdfNode("PointList", "IndexArray_t", nodes.reshape(1, -1)))
        interface.add(AdfNode("PointListDonor", "IndexArray_t", donors.reshape(1, -1)))
        properties = interface.add(
            AdfNode("GridConnectivityProperty", "GridConnectivityProperty_t")
        )
        periodic = properties.add(AdfNode("Periodic", "Periodic_t"))
        for key, values in (
            ("RotationCenter", [0, 0, 0]),
            ("RotationAngle", [0, 0, angle]),
            ("Translation", [0, 0, 0]),
        ):
            array = periodic.add(
                AdfNode(key, "DataArray_t", np.asarray(values, dtype=np.float32))
            )
            array.add(
                AdfNode(
                    "DimensionalExponents",
                    "DimensionalExponents_t",
                    np.array(
                        [
                            0,
                            0 if key == "RotationAngle" else 1,
                            0,
                            0,
                            1 if key == "RotationAngle" else 0,
                        ],
                        dtype=np.float32,
                    ),
                )
            )


def collect_surface_mesh():
    """Gather mesh polygons without any refitting or reconstructed geometry."""
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags, xyz = tags[order], np.asarray(coordinates).reshape(-1, 3)[order]
    faces = {}
    for name, tag in SURFACES.items():
        kinds, _, arrays = gmsh.model.mesh.getElements(2, tag)
        faces[name] = [
            np.searchsorted(tags, nodes).reshape(-1, 3 if kind == 2 else 4)
            for kind, nodes in zip(kinds, arrays)
        ]
    return xyz, faces
