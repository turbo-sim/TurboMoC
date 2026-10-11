"""Conforming reference meshing, spanwise deformation, volume sweep and export.

Reuses the 2D example for blade splines, smooth periodics, inflation layers,
and automatic core grading. Volume assembly and CGNS auditing follow the
ParaBlade mesh_examples/3D and 3D_v2 examples, without a ParaBlade dependency.
"""

import json
from pathlib import Path
from time import perf_counter

import gmsh
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import meshing_2d as msh2
from ..geometry import meridional_channel as meridional
from ..geometry import geometry_3d as geo
from .. import logging as log
from ..paths import GMSH_3D_DIR
from .quality import analyze_volume
from turbo_moc.io.cgns_adf import AdfNode, new_root, read_adf, write_adf

logger = log.get_logger("gmsh_3D.meshing")
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
# 1. Generate and extract a conforming 2D reference section
# =============================================================================


def generate_section(blade, curves, numerics, *, progress=None):
    """Build reference topology and statistics in the caller's Gmsh session.

    The staged workflow owns initialization, finalization and artifact files.
    Keep CAD, inflation, grading and extraction here so they have one owner.
    """
    def report(label, fraction=None):
        if progress:
            progress(label, fraction)

    report("Build reference CAD", 0.05)
    gmsh.model.add("conformal_reference_section")
    blade_curves = msh2.add_blade_boundary(blade, numerics["blade_size"])
    surface, boundaries = msh2.add_periodic_passage(
        curves, blade_curves, blade["pitch"], numerics["domain_size"]
    )
    profile = msh2.calculate_inflation_profile(numerics["inflation_layers"])
    msh2.set_mesh_sizes(blade_curves, numerics, profile, blade["pitch"])
    msh2.add_inflation_layers(blade_curves, numerics)
    msh2.configure_recombination(surface, numerics)
    report("Generate 1D mesh and match periodic nodes")
    gmsh.model.mesh.generate(1)
    msh2.check_periodic_conformity(boundaries)
    report("Prepare inflation/core grading", 0.25)
    transition = msh2.configure_core_transition(
        blade_curves, numerics, profile, blade["pitch"], use_mesh=True
    )
    report("Generate 2D reference mesh")
    gmsh.model.mesh.generate(2)
    report("Validate and extract reference topology", 0.85)
    msh2.check_periodic_conformity(boundaries)
    summary = msh2.mesh_statistics(surface, boundaries)
    xy, cells, edges, local = msh2.collect_fluid_mesh(surface, boundaries)
    slave, master, _ = msh2.collect_periodic_nodes(boundaries, local, xy)
    blocks = {}
    for width, kind in ((3, 2), (4, 3)):
        selected = [cell for cell in cells if len(cell) == width]
        if selected:
            blocks[kind] = np.asarray(selected, dtype=np.int32) - 1
    if not len(xy) or (numerics.get("recombine", True) and 3 not in blocks):
        raise ValueError(
            "The reference section must have fluid cells and requested quadrilaterals."
        )
    gmsh.model.mesh.removeSizeCallback()
    return {
        "xy": xy,
        "cells": blocks,
        "boundaries": {name: values[:, :2] - 1 for name, values in edges.items()},
        "slave": slave - 1,
        "master": master - 1,
        "profile": profile,
        "transition": transition,
        "summary": summary,
    }


# =============================================================================
# 2. Deform the core while prescribing the blade, collar and outer boundaries
# =============================================================================


class CoreDeformation:
    """Positive-weight harmonic core displacement with protected near-wall nodes.

    Every span station retains the reference connectivity. Prescribed collar
    coordinates are scaled with the blade, so smoothing cannot erase layers.
    This is a section-normal collar, not a 3D normal extrusion at a tapered wall.
    """

    def __init__(self, section, numerics):
        self.section = section
        xy = section["xy"]
        wall = np.unique(section["boundaries"]["blade"])
        # Distance to sampled wall nodes conservatively protects extra core rows.
        distance = cKDTree(xy[wall]).query(xy)[0]
        thickness = (
            section["profile"]["stack_thickness"]
            if section["profile"]["enabled"]
            else 0
        )
        protect = 1.5 * thickness + numerics["blade_size"]
        self.collar = np.flatnonzero(distance <= protect)
        outer = np.unique(
            np.concatenate(
                [
                    edges.ravel()
                    for name, edges in section["boundaries"].items()
                    if name != "blade"
                ]
            )
        )
        if np.intersect1d(self.collar, outer).size:
            raise ValueError(
                "Inflation/collar reaches a passage boundary; reduce layers/chord or blade count."
            )
        self.fixed = np.union1d(self.collar, outer)
        self.free = np.setdiff1d(np.arange(len(xy)), self.fixed)
        edges = np.unique(
            np.sort(
                np.concatenate(
                    [
                        np.column_stack(
                            (cells[:, j], cells[:, (j + 1) % cells.shape[1]])
                        )
                        for cells in section["cells"].values()
                        for j in range(cells.shape[1])
                    ]
                ),
                axis=1,
            ),
            axis=0,
        )
        weights = 1 / np.sum((xy[edges[:, 1]] - xy[edges[:, 0]]) ** 2, axis=1)
        adjacency = coo_matrix(
            (
                np.r_[weights, weights],
                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
            ),
            shape=(len(xy), len(xy)),
        ).tocsc()
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        self.factor = (
            splu(laplacian[self.free][:, self.free].tocsc()) if len(self.free) else None
        )
        self.coupling = laplacian[self.free][:, self.fixed]

    def deform(self, scale, reference, target, pitch):
        """Match a new section while keeping opposite periodic nodes paired."""
        warped = self.section["xy"] * scale
        result = warped.copy()
        for name, edges in self.section["boundaries"].items():
            if name != "blade":
                ids = np.unique(edges)
                result[ids] = meridional.corresponding_boundary(
                    self.section["xy"][ids], name, reference, target
                )
        prescribed = result[self.fixed] - warped[self.fixed]
        if self.factor is not None:
            result[self.free] += self.factor.solve(-self.coupling @ prescribed)
        if not np.allclose(
            result[self.section["slave"]] - result[self.section["master"]],
            [pitch, 0],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Span deformation lost periodic correspondence.")
        for cells in self.section["cells"].values():
            p = result[cells]
            # Each corner must remain convex, not merely positive in total area.
            a, b = np.roll(p, -1, axis=1) - p, np.roll(p, -2, axis=1) - np.roll(
                p, -1, axis=1
            )
            if np.any(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0] <= 0):
                raise ValueError(
                    "Section deformation inverted a cell; reduce taper or refine the core."
                )
        return result


# =============================================================================
# 3. Discrete volume assembly and rigid periodic correspondence
# =============================================================================


def assemble_volume(section, xyz_sections, meridional_channel):
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
        if meridional_channel.orientation < 0:
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
        if meridional_channel.orientation < 0:
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
        meridional_channel.periodic_transform().ravel(),
    )
    return counts


def validate_volume(section, xyz_sections, meridional_channel):
    """Check Jacobians, seven exterior patches and a complete periodic bijection."""
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
        raise ValueError("Incomplete swept periodic node map.")
    xyz = xyz_sections.reshape(-1, 3)
    expected_transform = meridional_channel.periodic_transform()
    transformed = xyz[master - 1] @ expected_transform[:3, :3].T + expected_transform[:3, 3]
    error = np.linalg.norm(xyz[slave - 1] - transformed, axis=1)
    if (
        not np.allclose(np.asarray(transform).reshape(4, 4), expected_transform)
        or max(error) > 1e-8 * meridional_channel.characteristic_length
    ):
        raise ValueError("Periodic surfaces do not match the one-pitch rigid transform.")
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


def export_cgns(path, settings, meridional_channel):
    """Write the current 3D Gmsh mesh as CGNS 3.4 ADF, and audit its BC tree.

    Requires exactly one fluid physical volume plus the seven named patches.
    Returns statistics suitable for the example's JSON/Markdown report. CGNS
    periodic patches remain named BCs; Fluent's periodic pairing is a solver
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
            "Periodic vertex maps and rigid transforms are included; pair the patches in the CFD solver.",
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
    add_cgns_periodicity(zone, node_tags, meridional_channel, settings["length_scale"])
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
        "periodic_pairing": "Conforming periodic_left/periodic_right patches; configure "
                            + ("translational" if meridional_channel.is_linear else "rotational")
                            + " pairing in Fluent.",
    }


def add_cgns_units(base):
    """CGNS coordinates are metres, while rotational angles are radians."""
    base.add(AdfNode("DataClass", "DataClass_t", "Dimensional"))
    units = np.full((32, 5), b" ", dtype="S1")
    for column, unit in enumerate(("Kilogram", "Meter", "Second", "Kelvin", "Radian")):
        units[: len(unit), column] = np.frombuffer(unit.encode("ascii"), dtype="S1")
    base.add(AdfNode("DimensionalUnits", "DimensionalUnits_t", units))


def add_cgns_periodicity(zone, node_tags, meridional_channel, length_scale=1.0):
    """Store reciprocal vertex maps and current-to-donor rigid transforms."""
    _, slave, master, _ = gmsh.model.mesh.getPeriodicNodes(
        2, SURFACES["periodic_right"]
    )
    current = (np.searchsorted(node_tags, slave) + 1).astype(np.int32)
    donor = (np.searchsorted(node_tags, master) + 1).astype(np.int32)
    interfaces = zone.add(AdfNode("ZoneGridConnectivity", "ZoneGridConnectivity_t"))
    # Gmsh stores master -> slave; CGNS specifies current -> connecting patch.
    forward = meridional_channel.periodic_transform()
    for name, nodes, donors, transform in (
        ("right_to_left", current, donor, np.linalg.inv(forward)),
        ("left_to_right", donor, current, forward),
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
            ("RotationAngle", [0, 0, np.arctan2(transform[1, 0], transform[0, 0])]),
            ("Translation", transform[:3, 3] * length_scale),
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


# =============================================================================
# 5. Python visualization of actual mesh faces; copies are display-only
# =============================================================================


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


def load_mesh_surfaces(mesh_file, surfaces=BOUNDARIES):
    """Read selected physical boundaries as compact points and polygon arrays."""
    surfaces = tuple(dict.fromkeys(surfaces))
    if not surfaces:
        raise ValueError("Select at least one surface to display.")
    unknown = set(surfaces) - set(BOUNDARIES)
    if unknown:
        raise ValueError(
            f"Unknown surfaces: {', '.join(sorted(unknown))}. "
            f"Available surfaces: {', '.join(BOUNDARIES)}."
        )
    mesh_file = Path(mesh_file)
    if not mesh_file.is_file():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_file}. Run run_mesh_generation.py first."
        )
    if gmsh.isInitialized():
        raise RuntimeError("Finish the current Gmsh session before loading a mesh.")

    gmsh.initialize(["blade_viewer", "-v", "0"], readConfigFiles=False)
    try:
        gmsh.open(str(mesh_file))
        xyz, faces = collect_surface_mesh()
        selected = {}
        for name in surfaces:
            blocks = [block for block in faces[name] if len(block)]
            if not blocks:
                raise ValueError(f"The mesh contains no '{name}' surface elements.")

            # Each surface retains only its own nodes, not the fluid-volume nodes.
            used = np.unique(np.concatenate([block.ravel() for block in blocks]))
            polygons = np.concatenate(
                [
                    np.column_stack(
                        (
                            np.full(len(block), block.shape[1]),
                            np.searchsorted(used, block),
                        )
                    ).ravel()
                    for block in blocks
                ]
            )
            selected[name] = (xyz[used], polygons)
        return selected
    finally:
        gmsh.finalize()


def pyvista_options(options):
    """Validate YAML surface selection and the mesh-line toggle."""
    if not isinstance(options, dict):
        raise ValueError("mesh.pyvista must be a mapping.")
    selected = options.get("surfaces", {"blade": True})
    if not isinstance(selected, dict) or set(selected) - set(BOUNDARIES):
        raise ValueError(
            "mesh.pyvista.surfaces must map physical boundary names to booleans."
        )
    if any(type(value) is not bool for value in selected.values()):
        raise ValueError("Every mesh.pyvista.surfaces value must be true or false.")
    surfaces = tuple(name for name, visible in selected.items() if visible)
    if not surfaces:
        raise ValueError("Select at least one mesh.pyvista surface to display.")
    show_edges = options.get("show_mesh_edges", False)
    if type(show_edges) is not bool:
        raise ValueError("mesh.pyvista.show_mesh_edges must be true or false.")
    return {"surfaces": surfaces, "show_mesh_edges": show_edges}


def view_mesh_pyvista(output_dir, options, *, show=True, save_figure=False):
    """View selected surfaces of one passage and its full row; save a safe preview."""
    options = pyvista_options(options)
    if not show and not save_figure:
        return
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Install the optional viewer from the repository root: "
            '.venv/Scripts/python.exe -m pip install ".[mesh-viz]"'
        ) from exc

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "mesh_summary.json").read_text(encoding="utf-8"))
    # New meshes record the general definition; retain support for older cases.
    definition = summary.get("geometry", summary.get("machine", {}))
    is_linear = definition.get("type") == "linear"
    num_blades = 3 if is_linear else int(definition["num_blades"])
    if num_blades < 1:
        raise ValueError("The saved case must have a positive blade count.")
    selected = load_mesh_surfaces(
        output_dir / "stator_mesh_3d.msh", options["surfaces"]
    )
    meshes = {
        name: pv.PolyData(points, polygons)
        for name, (points, polygons) in selected.items()
    }
    colors = {
        "blade": "orange",
        "hub": "silver",
        "shroud": "lightgrey",
        "inlet": "cornflowerblue",
        "outlet": "mediumseagreen",
        "periodic_left": "plum",
        "periodic_right": "khaki",
    }

    def populate(plotter):
        """Build the same two panels for image export and interactive inspection."""
        for panel, count in enumerate((1, num_blades)):
            plotter.subplot(0, panel)
            plotter.set_background("#f5f3ed")
            title = (
                ("Single blade" if tuple(meshes) == ("blade",) else "Single passage")
                if panel == 0
                else (f"Linear cascade ({count} displayed blades)" if is_linear
                      else f"Full turbine row ({count} blades)")
            )
            plotter.add_text(title, position="upper_left", color="black", font_size=16)
            for index in range(count):
                # Display copies follow the same one-pitch rigid transform as
                # the mesh, including a translation for a linear cascade.
                for name, surface in meshes.items():
                    if is_linear:
                        copy = surface.translate([0, index * definition["pitch_mm"], 0], inplace=False)
                    else:
                        copy = surface.rotate_z(-360.0 * index / num_blades, inplace=False)
                    plotter.add_mesh(
                        copy,
                        color=colors[name],
                        smooth_shading=True,
                        show_edges=options["show_mesh_edges"],
                        edge_color="black",
                        label=name if index == 0 else None,
                    )
            plotter.add_legend()
            plotter.show_axes()
            plotter.show_bounds(
                xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="black"
            )
            plotter.view_isometric()
            plotter.reset_camera()
            plotter.add_text(
                "Drag: rotate | Wheel: zoom | Shift + drag: pan",
                position="lower_left",
                color="black",
                font_size=10,
            )

    # Save the initial view using a separate offscreen window. Closing the
    # interactive window with its X button cannot destroy this screenshot.
    # Keep a distinct filename so Matplotlib's mesh_3d.png remains intact.
    modes = ([True] if save_figure else []) + ([False] if show else [])
    for off_screen in modes:
        plotter = pv.Plotter(
            shape=(1, 2), window_size=(1500, 800), off_screen=off_screen
        )
        try:
            populate(plotter)
            if off_screen:
                plotter.screenshot(str(output_dir / "pyvista_mesh.png"))
            else:
                # Independent cameras; mouse interaction uses the hovered panel.
                plotter.show(auto_close=True)
        finally:
            plotter.close()


# =============================================================================
# 6. Independent stage progress and reference-artifact serialization
# =============================================================================


def emit(progress, label, fraction=None):
    if progress:
        progress(label, fraction)


def save_section(directory, section, design):
    arrays = {"xy": section["xy"], "slave": section["slave"], "master": section["master"]}
    arrays.update({f"cells_{k}": v for k, v in section["cells"].items()})
    arrays.update({f"boundary_{k}": v for k, v in section["boundaries"].items()})
    np.savez_compressed(directory / "reference_mesh.npz", **arrays)
    metadata = {"profile": section["profile"], "transition": section["transition"],
                "design": geo.design_data(design)}
    (directory / "reference_definition.json").write_text(json.dumps(metadata), encoding="utf-8")


def load_section(directory):
    directory = Path(directory)
    with np.load(directory / "reference_mesh.npz", allow_pickle=False) as arrays:
        section = {name: arrays[name] for name in ("xy", "slave", "master")}
        section["cells"] = {int(name[6:]): arrays[name] for name in arrays.files if name.startswith("cells_")}
        section["boundaries"] = {name[9:]: arrays[name] for name in arrays.files if name.startswith("boundary_")}
    metadata = json.loads((directory / "reference_definition.json").read_text(encoding="utf-8"))
    section.update(profile=metadata["profile"], transition=metadata["transition"])
    return section, geo.load_design(metadata["design"])


# =============================================================================
# 7. Generate the reference mesh
# =============================================================================


def reference_mesh(design, config, output_dir, *, progress=None):
    """Generate and publish a 2D mesh only; no span sweep or volume export."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numerics = config["mesh"]["numerics"]
    msh2.validate_mesh_settings(config["mesh"] | {"cgns": {"enabled": False}, "report": {"enabled": False}})
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    start = perf_counter()
    gmsh.initialize(["turbomoc_reference", "-v", str(max(4, numerics["verbosity"]))], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(numerics["terminal_output"]))
        section = generate_section(design.blade, design.curves, numerics, progress=progress)
        summary = section.pop("summary")
        gmsh.write(str(directory / "reference_mesh.msh"))
        save_section(directory, section, design)
        summary.update(section_nodes=len(section["xy"]),
                       section_cells=sum(len(v) for v in section["cells"].values()),
                       elapsed_s=perf_counter() - start, shift_mm=design.curves["shift_mm"],
                       inflation_profile=section["profile"], core_transition=section["transition"])
        summary["coordinate_basis"] = "linear_physical" if design.meridional_channel.is_linear else "annular_conformal"
        if not design.meridional_channel.is_linear:
            summary["reference_radius_mm"] = design.meridional_channel.reference_radius
        (directory / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit(progress, "Reference mesh ready", 1.0)
        return summary
    finally:
        gmsh.model.mesh.removeSizeCallback()
        gmsh.finalize()


# =============================================================================
# 8. Sweep, validate and publish the physical volume
# =============================================================================


def collect_volume_arrays():
    """Retain real cell IDs/connectivity for histograms, slicing and worst cells."""
    nodes, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(nodes)
    nodes, xyz = nodes[order], xyz.reshape(-1, 3)[order]
    kinds, tag_arrays, node_arrays = gmsh.model.mesh.getElements(3, 1)
    blocks, tags = {}, {}
    for kind, ids, values in sorted(zip(kinds, tag_arrays, node_arrays), key=lambda item: item[0]):
        kind = int(kind)
        blocks[kind] = np.searchsorted(nodes, values).reshape(-1, 8 if kind == 5 else 6)
        tags[kind] = ids
    _, slave, master, rotation = gmsh.model.mesh.getPeriodicNodes(2, SURFACES["periodic_right"])
    _, _, right = gmsh.model.mesh.getElements(2, SURFACES["periodic_right"])
    return xyz, blocks, tags, {
        "periodic_faces": {"right": np.searchsorted(nodes, np.concatenate(right)).reshape(-1, 4)},
        "slave": np.searchsorted(nodes, slave), "master": np.searchsorted(nodes, master),
        "periodic_transform": np.asarray(rotation).reshape(4, 4),
    }


def volume_mesh(section, design, config, output_dir, *, progress=None):
    """Sweep the supplied reference artifact; never refit, recenter or remesh."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numerics = config["mesh"]["numerics"]
    eta = meridional.span_stations(numerics)
    start = perf_counter()
    deformation = None
    mapped = []
    for j, value in enumerate(eta):
        emit(progress, f"Map span section {j + 1}/{len(eta)}", 0.05 + 0.45 * j / len(eta))
        scale = design.meridional_channel.section_scale(float(value))
        if np.isclose(scale, 1, rtol=1e-12, atol=1e-12):
            q = section["xy"]
        else:
            if deformation is None:
                emit(progress, "Factor the sparse core deformation matrix")
                deformation = CoreDeformation(section, numerics)
            _, target = design.section(float(value))
            q = deformation.deform(scale, design.curves, target, design.meridional_channel.pitch)
        mapped.append(design.meridional_channel.map_points(q, float(value)))
    xyz_sections = np.asarray(mapped)
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    gmsh.initialize(["turbomoc_volume", "-v", "4"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(numerics["terminal_output"]))
        emit(progress, "Assemble volume topology")
        counts = assemble_volume(section, xyz_sections, design.meridional_channel)
        emit(progress, "Check Jacobians, boundary patches and periodicity", 0.65)
        validation = validate_volume(section, xyz_sections, design.meridional_channel)
        emit(progress, "Calculate volume skewness, orthogonality and edge aspect ratio")
        xyz, blocks, tags, periodic = collect_volume_arrays()
        report, metrics = analyze_volume(xyz, blocks, tags, **periodic)
        arrays = {"xyz": xyz, **{f"cells_{k}": v for k, v in blocks.items()},
                  **{f"tags_{k}": v for k, v in tags.items()}, **metrics}
        np.savez_compressed(directory / "volume_data.npz", **arrays)
        (directory / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        gmsh.write(str(directory / "stator_mesh_3d.msh"))
        emit(progress, "Extract browser boundary datasets", 0.9)
        # Extract before closing Gmsh: no reopening the volume solely to draw it.
        points, faces = collect_surface_mesh()
        surfaces = {}
        for name, arrays_for_surface in faces.items():
            used = np.unique(np.concatenate([a.ravel() for a in arrays_for_surface]))
            polys = np.concatenate([np.column_stack((np.full(len(a), a.shape[1]),
                                  np.searchsorted(used, a))).ravel() for a in arrays_for_surface])
            surfaces[name] = {"points": points[used].ravel().tolist(), "polys": polys.tolist()}
        (directory / "browser_surfaces.json").write_text(json.dumps(surfaces), encoding="utf-8")
        result = {
            "geometry": design.meridional_channel.definition.to_dict(), "cells": counts,
            "section_nodes": len(section["xy"]), "volume_nodes": len(xyz),
            "span_stations": eta.tolist(), "quality": validation,
            "shift_mm": design.curves["shift_mm"], "elapsed_s": perf_counter() - start,
            "inflation_profile": section["profile"], "core_transition": section["transition"],
            "periodic_transform_left_to_right": design.meridional_channel.periodic_transform().tolist(),
        }
        if not design.meridional_channel.is_linear:
            result["reference_radius_mm"] = design.meridional_channel.reference_radius
            result["periodic_rotation_left_to_right"] = design.meridional_channel.rotation().tolist()
        (directory / "mesh_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(progress, "Volume mesh ready", 1.0)
        return result
    finally:
        gmsh.finalize()


# =============================================================================
# 9. Export an existing mesh independently of remeshing
# =============================================================================


def export_reference_mesh(mesh_dir, config, output_dir, *, progress=None):
    """Export existing 2D cells, with physical coordinates for a linear cascade.

    Annular reference cells remain in the length-scaled conformal chart. Their
    explicit descriptor prevents confusing this design grid with a physical
    Cartesian two-dimensional CFD domain. Export never regenerates a mesh.
    """
    settings = dict(config["mesh"]["cgns"])
    configured_name = Path(settings["filename"])
    if configured_name.name != str(configured_name) or configured_name.suffix.lower() != ".cgns":
        raise ValueError("Choose a local .cgns filename.")
    if not np.isfinite(settings["length_scale"]) or settings["length_scale"] <= 0:
        raise ValueError("CGNS length scale must be positive.")
    names = {"blade", "inlet", "outlet", "periodic_left", "periodic_right"}
    settings["boundary_types"] = {name: settings["boundary_types"][name] for name in names}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"{configured_name.stem}_2d.cgns"
    native_file = filename.with_suffix(".msh")
    _, design = load_section(mesh_dir)
    mapping = design.meridional_channel
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    gmsh.initialize(["turbomoc_reference_export", "-v", "2"], readConfigFiles=False)
    try:
        gmsh.open(str(Path(mesh_dir) / "reference_mesh.msh"))
        groups = {
            gmsh.model.getPhysicalName(dim, tag): (dim, list(gmsh.model.getEntitiesForPhysicalGroup(dim, tag)))
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        if set(groups) != names | {"fluid"} or groups["fluid"][0] != 2 or len(groups["fluid"][1]) != 1:
            raise ValueError("Expected one fluid surface and the five reference boundary groups.")
        boundaries = {name: groups[name][1] for name in names}
        # Sort slave curves by their recorded master, avoiding assumptions about
        # physical-group/entity ordering after the MSH artifact is reopened.
        pairs = sorted((gmsh.model.mesh.getPeriodicNodes(1, right)[0], right)
                       for right in boundaries["periodic_right"])
        boundaries["periodic_left"] = [left for left, _ in pairs]
        boundaries["periodic_right"] = [right for _, right in pairs]
        if mapping.is_linear:
            emit(progress, "Transform existing 2D nodes to physical streamwise/pitchwise coordinates")
            axes = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            gmsh.model.mesh.affineTransform(axes.ravel())
            gmsh.model.mesh.setPeriodic(1, boundaries["periodic_right"], boundaries["periodic_left"],
                                        mapping.periodic_transform().ravel())
        emit(progress, "Audit 2D boundary connectivity and write CGNS/ADF")
        surface = groups["fluid"][1][0]
        xy, cells, edges, _ = msh2.collect_fluid_mesh(surface, boundaries)
        msh2.export_fluent_cgns(filename, surface, boundaries, settings)
        gmsh.write(str(native_file))
        root = read_adf(str(filename))
        base = root.get(label="CGNSBase_t")
        basis = "linear_physical" if mapping.is_linear else "annular_conformal_reference"
        description = (
            "Physical linear-cascade grid: X=streamwise distance, Y=pitchwise distance."
            if mapping.is_linear else
            "Reference conformal design grid: X=Rref*clockwise_theta, Y=-Rref*m_prime. "
            "These are not physical Cartesian coordinates of an annular passage."
        )
        base.add(AdfNode("CoordinateBasis", "Descriptor_t", description))
        write_adf(str(filename), root)
        emit(progress, "2D mesh export ready", 1.0)
        return {
            "file": str(filename), "native_mesh_file": str(native_file),
            "format": "CGNS 3.4 / ADF", "cell_dimension": 2, "physical_dimension": 2,
            "coordinate_basis": basis, "description": description,
            "nodes": len(xy), "cells": len(cells),
            "boundary_edges": {name: len(values) for name, values in edges.items()},
        }
    finally:
        gmsh.finalize()


def export_mesh(mesh_dir, config, output_dir, *, progress=None):
    """Export existing connectivity; filename/BC/unit edits do not remesh."""
    settings = config["mesh"]["cgns"]
    filename = settings["filename"]
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".cgns":
        raise ValueError("Choose a local .cgns filename.")
    if not np.isfinite(settings["length_scale"]) or settings["length_scale"] <= 0:
        raise ValueError("CGNS length scale must be positive.")
    boundary_types(settings["boundary_types"])
    summary = json.loads((Path(mesh_dir) / "mesh_summary.json").read_text(encoding="utf-8"))
    definition = summary["geometry"]
    meridional_channel = geo.mapping_from_data(definition)
    gmsh.initialize(["turbomoc_export", "-v", "2"], readConfigFiles=False)
    try:
        gmsh.open(str(Path(mesh_dir) / "stator_mesh_3d.msh"))
        emit(progress, "Audit boundary faces and write CGNS/ADF")
        return export_cgns(Path(output_dir) / filename, settings, meridional_channel)
    finally:
        gmsh.finalize()



# =============================================================================
# 10. Callable case workflow (used by the minimal example launcher)
# =============================================================================


def main(config_file=None, output_dir=None, *, row_id=None):
    """Solve MoC, construct one reference mesh, sweep it and optionally export.

    This is the same numerical sequence used by the app, with no browser or
    native viewer. Configuration imports are deferred to avoid a validation cycle.
    """
    from ..configuration import read_configuration
    from ..geometry.moc import solve_nozzle
    from turbo_moc.progress import make_progress_reporter

    config_file = Path(config_file or (GMSH_3D_DIR / "axial_stator.yaml"))
    if row_id is None:
        config = read_configuration(config_file)
    else:
        from ..configuration_project import load_project, row_configuration
        config = row_configuration(load_project(config_file), row_id)
    directory = Path(output_dir or GMSH_3D_DIR / "output" / config_file.stem)
    directory.mkdir(parents=True, exist_ok=True)
    timings = {}
    with log.workflow_logging(directory / "mesh_generation.log", config["workflow"]["log_level"]):
        runner_logger = log.get_logger("gmsh_3D.runner")

        def progress(label, fraction=None):
            runner_logger.info("%s%s", label,
                               "" if fraction is None else f" [{100 * fraction:.0f}%]")

        reporter = make_progress_reporter(
            lambda event: progress(event[1], float(event[0][:-1]) / 100)
        )
        start = perf_counter()
        nozzle = solve_nozzle(config["method_of_characteristics"], progress_callback=reporter)
        timings["moc_s"] = perf_counter() - start
        start = perf_counter()
        design = geo.build_design(nozzle, config)
        timings["geometry_s"] = perf_counter() - start
        reference = reference_mesh(design, config, directory, progress=progress)
        section, design = load_section(directory)
        volume = volume_mesh(section, design, config, directory, progress=progress)
        timings.update(reference_s=reference["elapsed_s"], volume_s=volume["elapsed_s"])
        if config["mesh"]["cgns"]["enabled"]:
            start = perf_counter()
            export_mesh(directory, config, directory, progress=progress)
            timings["export_s"] = perf_counter() - start
        (directory / "stage_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        runner_logger.info("Stage timings: %s", timings)
        return volume
