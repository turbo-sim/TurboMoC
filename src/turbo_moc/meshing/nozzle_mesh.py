"""Structured nozzle meshes: 2D half-domain, planar slab and axisymmetric O-grid."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator

from turbo_moc.geometry import nozzle as geometry
from turbo_moc.io.cgns_adf import AdfNode, new_root, read_adf, write_adf
from turbo_moc.meshing.mesh_3d import BC_TYPES, add_cgns_units
from turbo_moc.meshing.quality import analyze_volume, cosine

CGNS_TYPES = {2: (3, "BAR_2"), 4: (7, "QUAD_4"), 8: (17, "HEXA_8")}
DEFINITIONS_2D = {
    "skewness": "Maximum equiangular skewness per cell; 0 is ideal, 1 is degenerate.",
    "orthogonality": "Minimum absolute cosine between an edge normal and its centroid connector; 1 is ideal.",
    "aspect_ratio": "Maximum cell-edge length / minimum cell-edge length; 1 is ideal.",
    "centroids": "Arithmetic means of cell/edge vertices.",
}
O_GRID_CORE = 0.45
O_GRID_BULGE = 0.4


def emit(progress, label, fraction=None):
    if progress:
        progress(label, fraction)


def wall_distribution(cells, ratio):
    heights = ratio ** np.arange(cells, dtype=float)
    eta = 1.0 - np.concatenate([[0.0], np.cumsum(heights)]) / heights.sum()
    return eta[::-1]


def axial_stations(shape, cells):
    x, y = shape["x"], shape["y"]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    stations = np.interp(np.linspace(0, arc[-1], cells + 1), arc, x)
    stations[0], stations[-1] = x[0], x[-1]
    return stations, PchipInterpolator(x, y)(stations)


def section_grid(shape, settings):
    nx, nr = settings["axial_cells"], settings["radial_cells"]
    x, radius = axial_stations(shape, nx)
    eta = wall_distribution(nr, settings["wall_growth_ratio"])
    xy = np.stack([np.repeat(x[:, None], nr + 1, 1), radius[:, None] * eta[None, :]], -1).reshape(-1, 2)
    index = np.arange((nx + 1) * (nr + 1)).reshape(nx + 1, nr + 1)
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], -1).reshape(-1, 4)
    edges = {
        "inlet": np.stack([index[0, 1:], index[0, :-1]], -1),
        "outlet": np.stack([index[-1, :-1], index[-1, 1:]], -1),
        "wall": np.stack([index[1:, -1], index[:-1, -1]], -1),
        "axis": np.stack([index[:-1, 0], index[1:, 0]], -1),
    }
    return {"xy": xy, "quads": quads, "edges": edges, "x": x, "radius": radius, "eta": eta, "shape": (nx, nr)}


def analyze_section(xy, quads, edges):
    points = xy[quads]
    forward = np.roll(points, -1, axis=1) - points
    backward = np.roll(points, 1, axis=1) - points
    lengths = np.linalg.norm(forward, axis=2)
    if np.any(lengths <= 0):
        raise ValueError("Zero cell edge in quality analysis.")
    angles = np.degrees(np.arccos(np.clip(np.sum(forward * backward, axis=2)
                                          / (lengths * np.linalg.norm(backward, axis=2)), -1, 1)))
    skewness = np.clip(np.maximum((angles.max(axis=1) - 90) / 90, (90 - angles.min(axis=1)) / 90), 0, 1)
    aspect = lengths.max(axis=1) / lengths.min(axis=1)
    centers = points.mean(axis=1)
    owners = np.repeat(np.arange(len(quads)), 4)
    starts = quads.reshape(-1)
    ends = np.roll(quads, -1, axis=1).reshape(-1)
    tangent = xy[ends] - xy[starts]
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(tangent))])
    middle = 0.5 * (xy[ends] + xy[starts])
    keys = np.sort(np.column_stack([starts, ends]), axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    same = np.all(sorted_keys[:-1] == sorted_keys[1:], axis=1)
    connector = np.column_stack([middle - centers[owners], np.zeros(len(middle))])
    a, b = order[:-1][same], order[1:][same]
    link = np.column_stack([centers[owners[b]] - centers[owners[a]], np.zeros(len(a))])
    connector[a], connector[b] = link, link
    values = cosine(normals, connector)
    orthogonality = np.ones(len(quads))
    np.minimum.at(orthogonality, owners, values)
    metrics = {"skewness": skewness, "orthogonality": orthogonality, "aspect_ratio": aspect}
    report = {"definitions": DEFINITIONS_2D, "metrics": {}, "worst_cells": {}}
    for name, data in metrics.items():
        bins = np.linspace(0, 1, 121) if name != "aspect_ratio" else np.geomspace(1, max(1.01, float(data.max())), 121)
        hist, bins = np.histogram(data, bins=bins)
        report["metrics"][name] = {"min": float(data.min()), "mean": float(data.mean()), "max": float(data.max()),
                                   "p95": float(np.quantile(data, 0.95)), "counts": hist.tolist(),
                                   "bin_edges": bins.tolist(), "samples": len(data)}
        rank = np.argsort(data)
        if name != "orthogonality":
            rank = rank[::-1]
        report["worst_cells"][name] = [{"cell_id": int(i) + 1, "index": int(i), "value": float(data[i]),
                                        "center": centers[i].tolist()} for i in rank[:10]]
    return report, metrics


def core_map(points):
    level = np.abs(points).max(axis=-1, keepdims=True)
    distance = np.linalg.norm(points, axis=-1, keepdims=True)
    circle = np.divide(points * level, distance, out=np.zeros_like(points), where=distance > 0)
    return O_GRID_CORE * ((1 - O_GRID_BULGE) * points + O_GRID_BULGE * circle)


def o_grid_disk(core_cells, radial_cells, ratio):
    t = np.linspace(-1, 1, core_cells + 1)
    xi = wall_distribution(radial_cells, ratio)
    raw, quads, wall = [], [], []

    def block(points):
        start = sum(len(p) for p in raw)
        raw.append(points.reshape(-1, 2))
        return start + np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])

    a, b = np.meshgrid(t, t, indexing="ij")
    core = block(core_map(np.stack([a, b], -1)))
    quads.append(np.stack([core[:-1, :-1], core[1:, :-1], core[1:, 1:], core[:-1, 1:]], -1).reshape(-1, 4))
    for k in range(4):
        angle = 0.5 * np.pi * k
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        inner = core_map(np.column_stack([np.ones_like(t), t]))
        outer = np.column_stack([np.cos(0.25 * np.pi * t), np.sin(0.25 * np.pi * t)])
        points = (1 - xi)[:, None, None] * inner[None] + xi[:, None, None] * outer[None]
        ring = block(points @ rotation.T)
        quads.append(np.stack([ring[:-1, :-1], ring[1:, :-1], ring[1:, 1:], ring[:-1, 1:]], -1).reshape(-1, 4))
        wall.append(np.stack([ring[-1, :-1], ring[-1, 1:]], -1))
    raw = np.concatenate(raw)
    _, first, inverse = np.unique(np.round(raw, 12), axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(first)
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    points = raw[np.sort(first)]
    return points, renumber[inverse][np.concatenate(quads)], renumber[inverse][np.concatenate(wall)]


def sweep(stations, quads, count):
    return np.concatenate([np.concatenate([quads + i * count, quads + (i + 1) * count], axis=1)
                           for i in range(stations - 1)])


def side_faces(edges, stations, count):
    return np.concatenate([np.column_stack([edges[:, 0] + i * count, edges[:, 1] + i * count,
                                            edges[:, 1] + (i + 1) * count, edges[:, 0] + (i + 1) * count])
                           for i in range(stations - 1)])


def volume_grid(shape, settings, axisymmetric):
    nx = settings["axial_cells"]
    ratio = settings["wall_growth_ratio"]
    x, radius = axial_stations(shape, nx)
    if axisymmetric:
        disk, quads, wall = o_grid_disk(settings["circumferential_cells"] // 4, settings["radial_cells"], ratio)
        count = len(disk)
        xyz = np.concatenate([np.column_stack([np.full(count, xi), ri * disk]) for xi, ri in zip(x, radius)])
        hexes = sweep(len(x), quads, count)
        boundaries = {"inlet": quads[:, ::-1], "outlet": quads + nx * count,
                      "wall": side_faces(wall, len(x), count)}
        return xyz, hexes, boundaries
    grid = section_grid(shape, settings)
    nr = settings["radial_cells"]
    eta = np.concatenate([-grid["eta"][::-1], grid["eta"][1:]])
    rows = 2 * nr + 1
    plane = np.stack([np.repeat(x[:, None], rows, 1), radius[:, None] * eta[None, :]], -1).reshape(-1, 2)
    index = np.arange(len(plane)).reshape(nx + 1, rows)
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], -1).reshape(-1, 4)
    depth = float(settings["planar_depth"])
    z = np.linspace(-0.5 * depth, 0.5 * depth, settings["depth_cells"] + 1)
    count = len(plane)
    xyz = np.concatenate([np.column_stack([plane, np.full(count, zi)]) for zi in z])
    hexes = sweep(len(z), quads, count)
    inlet = np.stack([index[0, 1:], index[0, :-1]], -1)
    outlet = np.stack([index[-1, :-1], index[-1, 1:]], -1)
    walls = np.concatenate([np.stack([index[1:, -1], index[:-1, -1]], -1), np.stack([index[:-1, 0], index[1:, 0]], -1)])
    last = (len(z) - 1) * count
    boundaries = {"inlet": side_faces(inlet, len(z), count), "outlet": side_faces(outlet, len(z), count),
                  "wall": side_faces(walls, len(z), count),
                  "side": np.concatenate([quads[:, ::-1], quads + last])}
    return xyz, hexes, boundaries


def orient_hexes(xyz, hexes):
    p = xyz[hexes]
    volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 3] - p[:, 0], p[:, 4] - p[:, 0]))
    flipped = volume < 0
    hexes[flipped] = hexes[flipped][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return hexes, int(flipped.sum())


def write_cgns(path, xyz, cells, boundaries, types, description):
    dimension = xyz.shape[1]
    cell_dimension = 3 if cells.shape[1] == 8 else 2
    root = new_root()
    root.add(AdfNode("CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([3.4], dtype=np.float32)))
    base = root.add(AdfNode("TurboMoC", "CGNSBase_t", np.array([cell_dimension, dimension], dtype=np.int32)))
    base.add(AdfNode("ExportDescription", "Descriptor_t", description))
    zone = base.add(AdfNode("fluid_domain", "Zone_t", np.array([[len(xyz), len(cells), 0]], dtype=np.int32)))
    zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
    grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
    for i, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")[:dimension]):
        array = grid.add(AdfNode(name, "DataArray_t", xyz[:, i].astype(np.float64)))
        array.add(AdfNode("DimensionalExponents", "DimensionalExponents_t",
                          np.array([0, 1, 0, 0, 0], dtype=np.float32)))
    start = 1
    patches = {}

    def section(name, connectivity):
        nonlocal start
        kind = CGNS_TYPES[connectivity.shape[1]]
        end = start + len(connectivity) - 1
        node = zone.add(AdfNode(name, "Elements_t", np.array([kind[0], 0], dtype=np.int32)))
        node.add(AdfNode("ElementRange", "IndexRange_t", np.array([start, end], dtype=np.int32)))
        node.add(AdfNode("ElementConnectivity", "DataArray_t", (connectivity + 1).astype(np.int32).ravel()))
        result = {"first_element": start, "last_element": end, "elements": len(connectivity),
                  "element_type": kind[1]}
        start = end + 1
        return result

    section("fluid_domain", cells)
    zbc = AdfNode("ZoneBC", "ZoneBC_t")
    for name, faces in boundaries.items():
        patch = section(name, faces)
        bc = zbc.add(AdfNode(name, "BC_t", types[name]))
        bc.add(AdfNode("PointRange", "IndexRange_t",
                       np.array([[patch["first_element"], patch["last_element"]]], dtype=np.int32)))
        bc.add(AdfNode("GridLocation", "GridLocation_t", "FaceCenter"))
        patches[name] = {**patch, "bc_type": types[name]}
    zone.add(zbc)
    add_cgns_units(base)
    path = Path(path)
    write_adf(str(path), root)
    loaded = read_adf(str(path)).get(label="CGNSBase_t").get(label="Zone_t")
    if not np.array_equal(loaded.data, [[len(xyz), len(cells), 0]]):
        raise ValueError("CGNS zone dimensions did not survive serialization.")
    return {"file": str(path), "format": "CGNS 3.4 / ADF", "cell_dimension": cell_dimension,
            "physical_dimension": dimension, "nodes": len(xyz), "cells": len(cells), "boundaries": patches}


def boundary_types(settings, names):
    types = {name: settings["boundary_types"][name] for name in names}
    for name, value in types.items():
        if value not in BC_TYPES:
            raise ValueError(f"Unsupported CGNS BC type {value!r} for {name}.")
    return types


def check_boundaries(cells, boundaries, faces_per_cell):
    keys = np.concatenate([np.sort(f, axis=1) for f in faces_per_cell])
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    exterior = {tuple(row) for row in unique[counts == 1]}
    named = [tuple(row) for faces in boundaries.values() for row in np.sort(faces, axis=1)]
    if len(named) != len(set(named)) or set(named) != exterior:
        raise ValueError("Named boundary patches do not cover the exterior faces exactly once.")
    return len(exterior)


def mesh_2d(result, parameters, settings, output_dir, *, progress=None):
    geometry.validate_settings(settings)
    start = perf_counter()
    directory = Path(output_dir)
    emit(progress, "Nozzle contour and convergent inlet", 0.05)
    shape = geometry.contour(result, parameters, settings)
    emit(progress, "Structured 2D half-domain", 0.2)
    grid = section_grid(shape, settings)
    quads, edges = grid["quads"], grid["edges"]
    exterior = check_boundaries(quads, edges, [quads[:, [i, (i + 1) % 4]] for i in range(4)])
    emit(progress, "Cell skewness, orthogonality and aspect ratio", 0.5)
    report, metrics = analyze_section(grid["xy"], quads, edges)
    np.savez_compressed(directory / "nozzle_section.npz", xy=grid["xy"], quads=quads,
                        **{f"edge_{name}": value for name, value in edges.items()}, **metrics)
    emit(progress, "Write 2D CGNS", 0.8)
    axisymmetric = shape["axisymmetric"]
    export = write_cgns(directory / "nozzle_mesh_2d.cgns", grid["xy"], quads, edges,
                        boundary_types(settings, edges),
                        ("Axisymmetric" if axisymmetric else "Planar") + " nozzle half-domain in metres: "
                        "X = axial distance, Y = " + ("radius" if axisymmetric else "distance from the symmetry plane") + ".")
    summary = {"axisymmetric": axisymmetric, "cells": len(quads), "nodes": len(grid["xy"]),
               "axial_cells": grid["shape"][0], "radial_cells": grid["shape"][1],
               "first_wall_height": float(grid["radius"].min() * (1 - grid["eta"][-2])),
               "exterior_edges": exterior, "throat_radius": float(parameters["y_t"]),
               "exit_radius": float(shape["y"][-1]), "length": float(shape["x"][-1] - shape["x"][0]),
               "elapsed_s": perf_counter() - start, "export": export}
    (directory / "nozzle_mesh_2d.json").write_text(json.dumps({"summary": summary, "report": report}, indent=2),
                                                   encoding="utf-8")
    emit(progress, "Nozzle 2D mesh ready", 1.0)
    return summary


def mesh_3d(result, parameters, settings, output_dir, *, progress=None):
    geometry.validate_settings(settings)
    start = perf_counter()
    directory = Path(output_dir)
    shape = geometry.contour(result, parameters, settings)
    axisymmetric = shape["axisymmetric"]
    emit(progress, "Revolve the O-grid section" if axisymmetric else "Sweep the full-height section", 0.1)
    xyz, hexes, boundaries = volume_grid(shape, settings, axisymmetric)
    hexes, flipped = orient_hexes(xyz, hexes)
    emit(progress, "Check boundary patches", 0.3)
    faces = [hexes[:, list(face)] for face in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                                                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]
    exterior = check_boundaries(hexes, boundaries, faces)
    emit(progress, "Calculate volume skewness, orthogonality and edge aspect ratio", 0.45)
    ids = np.arange(1, len(hexes) + 1)
    report, metrics = analyze_volume(xyz, {5: hexes}, {5: ids})
    np.savez_compressed(directory / "volume_data.npz", xyz=xyz, cells_5=hexes, tags_5=ids, **metrics)
    (directory / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit(progress, "Extract browser boundary datasets", 0.7)
    surfaces = {}
    for name, faces_of_patch in boundaries.items():
        used = np.unique(faces_of_patch)
        polys = np.column_stack([np.full(len(faces_of_patch), 4), np.searchsorted(used, faces_of_patch)]).ravel()
        surfaces[name] = {"points": xyz[used].ravel().tolist(), "polys": polys.tolist()}
    (directory / "browser_surfaces.json").write_text(json.dumps(surfaces), encoding="utf-8")
    emit(progress, "Write 3D CGNS", 0.85)
    export = write_cgns(directory / "nozzle_mesh_3d.cgns", xyz, hexes, boundaries,
                        boundary_types(settings, boundaries),
                        ("Full 360 degree axisymmetric nozzle with an O-grid core" if axisymmetric else
                         "Planar nozzle slab, full height") + "; coordinates in metres, X = axial distance.")
    summary = {"axisymmetric": axisymmetric, "cells": len(hexes), "nodes": len(xyz), "exterior_faces": exterior,
               "reoriented_cells": flipped, "axial_cells": settings["axial_cells"],
               "radial_cells": settings["radial_cells"],
               "circumferential_cells" if axisymmetric else "depth_cells":
                   settings["circumferential_cells" if axisymmetric else "depth_cells"],
               "elapsed_s": perf_counter() - start, "export": export}
    (directory / "nozzle_mesh_3d.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit(progress, "Nozzle 3D mesh ready", 1.0)
    return summary
