"""Vectorized geometric diagnostics for first-order hex/prism volume meshes.

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

