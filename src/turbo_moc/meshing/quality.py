"""Shared planar diagnostics and first-order hex/prism volume quality.

Definitions are intentionally explicit, rather than claiming these are a
particular CFD solver's identically named quality measures.
"""
import numpy as np

FACES = {
    5: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    6: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
}
EDGES = {
    5: ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)),
    6: ((0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)),
}
DEFINITIONS = {
    "skewness": "Maximum equiangular face skewness per cell; 0 is ideal, 1 is degenerate.",
    "orthogonality": "Minimum absolute cosine between a face normal and its centroid connector; 1 is ideal. Periodic neighbors are transformed into the same frame.",
    "aspect_ratio": "Maximum cell-edge length / minimum cell-edge length; 1 is ideal. This is edge aspect ratio, not a solver-specific volume metric.",
    "centroids": "Arithmetic means of cell/face vertices.",
}


def face_keys(nodes):
    keys = np.full((len(nodes), 4), -1, dtype=np.int64)
    keys[:, :nodes.shape[1]] = nodes
    return np.sort(keys, axis=1)


def records(keys):
    dtype = np.dtype([(name, "<i8") for name in ("a", "b", "c", "d")])
    return np.ascontiguousarray(keys, dtype="<i8").view(dtype).ravel()


def cosine(normal, vector):
    denominator = np.linalg.norm(normal, axis=1) * np.linalg.norm(vector, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Degenerate face or centroid connector in quality analysis.")
    return np.clip(np.abs(np.einsum("ij,ij->i", normal, vector)) / denominator, 0, 1)


def analyze_volume(xyz, blocks, tags, *, periodic_faces=None, slave=None, master=None,
                   periodic_transform=None, rotation=None):
    """Return scalar arrays and a compact report; connectivity is zero-based."""
    n = sum(len(cells) for cells in blocks.values())
    centers = np.concatenate([xyz[cells].mean(axis=1) for cells in blocks.values()])
    ids = np.concatenate([tags[kind] for kind in blocks])
    skewness = np.zeros(n)
    aspect = np.empty(n)
    keys, owners, normals, face_centers = [], [], [], []
    offset = 0
    for kind, cells in blocks.items():
        p = xyz[cells]
        edges = np.asarray(EDGES[kind])
        lengths = np.linalg.norm(p[:, edges[:, 1]] - p[:, edges[:, 0]], axis=2)
        if np.any(lengths <= 0):
            raise ValueError("Zero cell edge in quality analysis.")
        aspect[offset:offset + len(cells)] = lengths.max(axis=1) / lengths.min(axis=1)
        for pattern in FACES[kind]:
            nodes = cells[:, pattern]
            vertices = xyz[nodes]
            local = vertices - vertices[:, :1]
            normals.append(np.cross(local, np.roll(local, -1, axis=1)).sum(axis=1))
            face_centers.append(vertices.mean(axis=1))
            keys.append(face_keys(nodes))
            owners.append(np.arange(offset, offset + len(cells)))
            forward = np.roll(vertices, -1, axis=1) - vertices
            backward = np.roll(vertices, 1, axis=1) - vertices
            den = np.linalg.norm(forward, axis=2) * np.linalg.norm(backward, axis=2)
            angles = np.degrees(np.arccos(np.clip(
                np.sum(forward * backward, axis=2) / den, -1, 1)))
            ideal = 60.0 if len(pattern) == 3 else 90.0
            value = np.maximum((angles.max(axis=1) - ideal) / (180 - ideal),
                               (ideal - angles.min(axis=1)) / ideal)
            skewness[offset:offset + len(cells)] = np.maximum(
                skewness[offset:offset + len(cells)], value)
        offset += len(cells)
    keys, owners = np.concatenate(keys), np.concatenate(owners)
    normals, face_centers = np.concatenate(normals), np.concatenate(face_centers)
    rec = records(keys)
    order = np.argsort(rec)
    sorted_rec = rec[order]
    _, counts = np.unique(sorted_rec, return_counts=True)
    if counts.max() > 2:
        raise ValueError("Nonmanifold face in volume quality analysis.")
    values = cosine(normals, face_centers - centers[owners])
    same = sorted_rec[:-1] == sorted_rec[1:]
    a, b = order[:-1][same], order[1:][same]
    connector = centers[owners[b]] - centers[owners[a]]
    values[a] = cosine(normals[a], connector)
    values[b] = cosine(normals[b], connector)

    if periodic_faces:
        def locate(nodes):
            wanted = records(face_keys(nodes))
            at = np.searchsorted(sorted_rec, wanted)
            if np.any(at >= len(sorted_rec)) or not np.array_equal(sorted_rec[at], wanted):
                raise ValueError("Periodic face missing from volume topology.")
            return order[at]
        right = periodic_faces["right"]
        mapping = np.arange(len(xyz))
        mapping[slave] = master
        ri, li = locate(right), locate(mapping[right])
        # The same rigid map covers annular rotations and linear translations.
        # Retain the former keyword for scientific callers using older artifacts.
        affine = periodic_transform if periodic_transform is not None else rotation
        transform, translation = affine[:3, :3], affine[:3, 3]
        right_neighbor = centers[owners[li]] @ transform.T + translation
        left_neighbor = (centers[owners[ri]] - translation) @ transform
        values[ri] = cosine(normals[ri], right_neighbor - centers[owners[ri]])
        values[li] = cosine(normals[li], left_neighbor - centers[owners[li]])
    orthogonality = np.ones(n)
    np.minimum.at(orthogonality, owners, values)
    metrics = {"skewness": np.clip(skewness, 0, 1), "orthogonality": orthogonality,
               "aspect_ratio": aspect}
    if any(not np.isfinite(v).all() for v in metrics.values()):
        raise ValueError("Nonfinite volume quality metric.")
    report = {"definitions": DEFINITIONS, "metrics": {}, "worst_cells": {}}
    for name, values in metrics.items():
        # Narrow bins show the distribution without interpolating cell samples.
        bins = np.linspace(0, 1, 121) if name != "aspect_ratio" else np.geomspace(1, max(1.01, float(values.max())), 121)
        hist, bins = np.histogram(values, bins=bins)
        report["metrics"][name] = {
            "min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "p95": float(np.quantile(values, 0.95)),
            "counts": hist.tolist(), "bin_edges": bins.tolist(), "samples": n,
        }
        rank = np.argsort(values)
        if name != "orthogonality":
            rank = rank[::-1]
        report["worst_cells"][name] = [
            {"cell_id": int(ids[i]), "index": int(i), "value": float(values[i]),
             "center_mm": centers[i].tolist()} for i in rank[:10]
        ]
    return report, metrics



# Planar diagnostics shared by canonical and preserved legacy meshes.
# These functions use arrays only; CAD, repair and workflow policy stay local.

def unique_plot_segments(segments):
    """Deduplicate coincident edges, including interfaces of pitchwise copies."""
    if not len(segments):
        return np.empty((0, 2, 2))
    points, indices = np.unique(
        np.round(np.asarray(segments).reshape(-1, 2), 9), axis=0, return_inverse=True
    )
    edges = np.unique(np.sort(indices.reshape(-1, 2), axis=1), axis=0)
    return points[edges]


def cell_geometry(xy, cells):
    """Compute polygon-centroid and angle/size measures for triangles/quads."""
    count = len(cells)
    centroids = np.zeros((count, 2))
    values = {
        key: np.zeros(count)
        for key in (
            "area_mm2",
            "equivalent_size_mm",
            "min_edge_mm",
            "max_edge_mm",
            "edge_aspect_ratio",
            "equiangle_skewness",
            "min_angle_deg",
            "max_angle_deg",
        )
    }
    clockwise = 0
    for size in (3, 4):
        indices = np.array(
            [i for i, cell in enumerate(cells) if len(cell) == size], dtype=int
        )
        if not len(indices):
            continue
        nodes = np.array([cells[i] for i in indices])
        points = xy[nodes]
        next_points = np.roll(points, -1, axis=1)
        cross = (
            points[:, :, 0] * next_points[:, :, 1]
            - next_points[:, :, 0] * points[:, :, 1]
        )
        signed_area = cross.sum(axis=1) / 2
        if np.any(np.abs(signed_area) < 1e-20):
            raise ValueError("Cannot analyze a zero-area cell.")
        reversed_cells = signed_area < 0
        clockwise += int(reversed_cells.sum())
        if np.any(reversed_cells):
            nodes[reversed_cells] = nodes[reversed_cells, ::-1]
            for index, cell in zip(indices, nodes):
                cells[index] = cell
            points = xy[nodes]
            next_points = np.roll(points, -1, axis=1)
            cross = (
                points[:, :, 0] * next_points[:, :, 1]
                - next_points[:, :, 0] * points[:, :, 1]
            )
        area = cross.sum(axis=1) / 2
        centroids[indices] = ((points + next_points) * cross[:, :, None]).sum(
            axis=1
        ) / (6 * area[:, None])
        edges = next_points - points
        lengths = np.linalg.norm(edges, axis=2)
        if np.any(lengths <= 0):
            raise ValueError("Cannot analyze a zero-length cell edge.")
        previous = -np.roll(edges, 1, axis=1)
        cosine = np.sum(previous * edges, axis=2) / (
            np.linalg.norm(previous, axis=2) * lengths
        )
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        turn = (
            np.roll(edges, 1, axis=1)[:, :, 0] * edges[:, :, 1]
            - np.roll(edges, 1, axis=1)[:, :, 1] * edges[:, :, 0]
        )
        angles = np.where(turn < 0, 360 - angles, angles)
        ideal = 180 * (size - 2) / size
        skew = np.maximum(
            (angles.max(axis=1) - ideal) / (180 - ideal),
            (ideal - angles.min(axis=1)) / ideal,
        )
        for key, array in {
            "area_mm2": area,
            "equivalent_size_mm": np.sqrt(area),
            "min_edge_mm": lengths.min(axis=1),
            "max_edge_mm": lengths.max(axis=1),
            "edge_aspect_ratio": lengths.max(axis=1) / lengths.min(axis=1),
            "equiangle_skewness": np.maximum(0, skew),
            "min_angle_deg": angles.min(axis=1),
            "max_angle_deg": angles.max(axis=1),
        }.items():
            values[key][indices] = array
    return centroids, values, clockwise


def edge_topology(cells):
    owners = {}
    for i, cell in enumerate(cells):
        for position, (a, b) in enumerate(zip(cell, np.roll(cell, -1))):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append((i, position))
    if any(len(items) > 2 for items in owners.values()):
        raise ValueError("Nonmanifold edge detected during quality analysis.")
    return owners


def face_geometry(xy, cells, centroids, owners, boundary_edges, periodic_map):
    """Include interior and translated periodic neighbors in FV diagnostics."""
    orthogonality = np.ones(len(cells))
    periodic_neighbors = {}
    if periodic_map:
        for edge in boundary_edges.get("periodic_right", []):
            right = tuple(sorted(map(int, edge)))
            left = tuple(sorted(periodic_map[node] for node in right))
            if (
                right not in owners
                or left not in owners
                or len(owners[right]) != 1
                or len(owners[left]) != 1
            ):
                raise ValueError("Periodic face matching is incomplete.")
            r, l = owners[right][0][0], owners[left][0][0]
            translation = xy[list(right)].mean(axis=0) - xy[list(left)].mean(axis=0)
            periodic_neighbors[right] = (l, translation)
            periodic_neighbors[left] = (r, -translation)

    face_angles, area_ratio_pairs = [], []
    for edge, adjacency in owners.items():
        points = xy[list(edge)]
        midpoint = points.mean(axis=0)
        scores = []
        for i, position in adjacency:
            to_face = midpoint - centroids[i]
            a, b = cells[i][position], cells[i][(position + 1) % len(cells[i])]
            oriented_edge = xy[b] - xy[a]
            outward = np.array([oriented_edge[1], -oriented_edge[0]]) / np.linalg.norm(
                oriented_edge
            )
            score = np.dot(outward, to_face) / np.linalg.norm(to_face)
            neighbor = None
            if len(adjacency) == 2:
                neighbor = adjacency[1][0] if adjacency[0][0] == i else adjacency[0][0]
                to_neighbor = centroids[neighbor] - centroids[i]
            elif edge in periodic_neighbors:
                neighbor, shift = periodic_neighbors[edge]
                to_neighbor = centroids[neighbor] + shift - centroids[i]
            if neighbor is not None:
                score = min(
                    score, np.dot(outward, to_neighbor) / np.linalg.norm(to_neighbor)
                )
            orthogonality[i] = min(orthogonality[i], score)
            scores.append(score)
        # Count the periodic seam once (on its right side), not twice.
        if edge not in periodic_neighbors or periodic_neighbors[edge][1][0] > 0:
            face_angles.append(
                float(np.rad2deg(np.arccos(np.clip(min(scores), -1, 1))))
            )
            if len(adjacency) == 2:
                area_ratio_pairs.append((adjacency[0][0], adjacency[1][0]))
            elif edge in periodic_neighbors:
                area_ratio_pairs.append((adjacency[0][0], periodic_neighbors[edge][0]))
    return np.clip(orthogonality, -1, 1), np.array(face_angles), area_ratio_pairs


