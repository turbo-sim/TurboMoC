"""Inspect saved volume cells without regenerating or changing their quality."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .quality import DEFINITIONS

METRIC_LABELS = {"skewness": "Skewness", "orthogonality": "Orthogonality",
                 "aspect_ratio": "Edge aspect ratio"}
RENDER_CELL_LIMIT = 50_000
DETAIL_LIMIT = 100


def _signature(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _integer(value, label, minimum=1):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    try:
        number = float(value)
    except (ValueError, TypeError):
        number = float("nan")
    if not np.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(number)


def _number(value, label, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if (isinstance(value, bool) or not np.isfinite(number)
            or (minimum is not None and number < minimum)
            or (maximum is not None and number > maximum)):
        bounds = (f" in [{minimum}, {maximum}]" if maximum is not None
                  else f" >= {minimum}" if minimum is not None else "")
        raise ValueError(f"{label} must be finite{bounds}.")
    return number


def _span_layers(summary, blocks, point_count):
    """Recover ownership from swept node sections, independent of cell order."""
    nodes = _integer(summary.get("section_nodes"), "Saved section_nodes")
    stations = np.asarray(summary.get("span_stations", []), dtype=float)
    if (stations.ndim != 1 or len(stations) < 2 or not np.isfinite(stations).all()
            or np.any(np.diff(stations) <= 0) or not np.isclose(stations[0], 0)
            or not np.isclose(stations[-1], 1)):
        raise ValueError("Saved span_stations must increase from hub (0) to shroud/tip (1).")
    if point_count != nodes * len(stations):
        raise ValueError("Saved section_nodes/span_stations do not match the volume nodes.")
    layers = []
    for cells in blocks:
        sections = cells // nodes
        lower, upper = sections.min(axis=1), sections.max(axis=1)
        half = cells.shape[1] // 2
        if (np.any(upper != lower + 1)
                or np.any((sections == lower[:, None]).sum(axis=1) != half)
                or np.any((sections == upper[:, None]).sum(axis=1) != half)
                or np.any(np.ptp(sections[:, :half], axis=1))
                or np.any(np.ptp(sections[:, half:], axis=1))):
            raise ValueError("Volume cells do not connect consecutive saved span sections.")
        if not np.array_equal(np.sort(cells[:, :half] % nodes, axis=1),
                              np.sort(cells[:, half:] % nodes, axis=1)):
            raise ValueError("Volume cells do not retain the saved spanwise sweep topology.")
        layers.append(lower + 1)
    return np.concatenate(layers), stations


@lru_cache(maxsize=2)
def _load_artifact(directory, volume_signature, summary_signature):
    """Keep at most two grids; signatures invalidate overwritten artifact paths."""
    import pyvista as pv

    directory = Path(directory)
    with np.load(directory / "volume_data.npz", allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("Saved volume coordinates must be finite XYZ points.")
        packed, kinds, tags, blocks = [], [], [], []
        for kind, width, vtk_type in ((5, 8, 12), (6, 6, 13)):
            if f"cells_{kind}" not in data:
                continue
            cells = data[f"cells_{kind}"]
            if (cells.ndim != 2 or cells.shape[1] != width
                    or not np.issubdtype(cells.dtype, np.integer)
                    or np.any(cells < 0) or np.any(cells >= len(xyz))):
                raise ValueError("Saved volume connectivity has invalid node indices.")
            ids = data[f"tags_{kind}"]
            if ids.shape != (len(cells),) or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("Saved volume cell IDs do not match connectivity.")
            blocks.append(cells)
            packed.append(np.column_stack((np.full(len(cells), width), cells)).ravel())
            kinds.append(np.full(len(cells), vtk_type, dtype=np.uint8))
            tags.append(ids)
        if not blocks or not sum(map(len, blocks)):
            raise ValueError("Saved volume has no supported hex/prism cells.")
        ids = np.concatenate(tags)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Saved volume cell IDs must be unique.")
        grid = pv.UnstructuredGrid(np.concatenate(packed), np.concatenate(kinds), xyz)
        grid.cell_data["cell_id"] = ids
        for metric in METRIC_LABELS:
            values = np.asarray(data[metric], dtype=float)
            minimum, maximum = (1, None) if metric == "aspect_ratio" else (0, 1)
            if (values.shape != (len(ids),) or not np.isfinite(values).all()
                    or np.any(values < minimum)
                    or (maximum is not None and np.any(values > maximum))):
                raise ValueError(f"Saved {metric} contains invalid cell values.")
            grid.cell_data[metric] = values
    layer_ids, stations, layer_error = None, np.array([]), None
    try:
        summary = json.loads((directory / "mesh_summary.json").read_text(encoding="utf-8"))
        layer_ids, stations = _span_layers(summary, blocks, len(xyz))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        # Legacy quality/slice inspections need no span metadata.
        layer_error = f"Span layers unavailable: {exc}"
    return {"grid": grid, "layer_ids": layer_ids, "stations": stations,
            "layer_error": layer_error}


def _artifact(directory):
    directory = Path(directory).resolve()
    return _load_artifact(str(directory), _signature(directory / "volume_data.npz"),
                          _signature(directory / "mesh_summary.json"))


def load_grid(directory):
    """Return an independent grid so callers cannot mutate the inspection cache."""
    return _artifact(directory)["grid"].copy(deep=True)


def inspection_info(directory):
    """Read available controls from the saved artifact, including outdated jobs."""
    artifact = _artifact(directory)
    grid = artifact["grid"]
    return {"total_count": grid.n_cells,
            "layer_count": max(0, len(artifact["stations"]) - 1),
            "span_stations": artifact["stations"].tolist(),
            "layer_error": artifact["layer_error"],
            "metric_ranges": {name: [float(grid.cell_data[name].min()),
                                     float(grid.cell_data[name].max())]
                              for name in METRIC_LABELS}}


def _aspect_color_ceiling(maximum):
    """Round strictly above the whole-mesh maximum on a 1, 2, 5 scale."""
    scale = 10.0 ** int(np.floor(np.log10(maximum)))
    for factor in (1, 2, 5, 10):
        ceiling = factor * scale
        if ceiling > maximum and np.isfinite(ceiling):
            return ceiling
    raise ValueError("Aspect ratio is too large for a finite rounded color-scale maximum.")


def _colors(values, metric, maximum):
    if metric == "orthogonality":
        return 1 - values
    if metric == "aspect_ratio":
        return np.clip(np.log(values) / np.log(maximum), 0, 1)
    return values


def inspect_mesh(directory, mode, axis="Z", fraction=0.5, metric="skewness", *,
                 n=10, threshold=None, layer=1, aspect_max=None):
    """Return selected cell surfaces and original, uninterpolated volume quality.

    Colors are badness: 0 is green and 1 red. Layers start at one at the hub.
    Thresholds include equality and select the poor side of the metric.
    The legacy ``slice``/``worst`` calls remain supported.
    """
    if mode not in ("slice", "worst", "threshold", "layer"):
        return None
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unknown mesh quality metric: {metric}.")
    # Normalize user input before hashing and before any cached result is reused.
    if mode == "worst":
        n = _integer(n, "Worst-cell count")
    elif mode == "threshold":
        threshold = _number(threshold, "Quality threshold",
                            minimum=1 if metric == "aspect_ratio" else 0,
                            maximum=None if metric == "aspect_ratio" else 1)
    elif mode == "layer":
        layer = _integer(layer, "Span layer")
    elif mode == "slice":
        if axis not in ("X", "Y", "Z"):
            raise ValueError("Slice axis must be X, Y or Z.")
        fraction = _number(fraction, "Slice fraction", minimum=0, maximum=1)
    if metric == "aspect_ratio" and aspect_max is not None:
        aspect_max = _number(aspect_max, "Aspect-ratio color maximum", minimum=1)
        if aspect_max <= 1:
            raise ValueError("Aspect-ratio color maximum must be greater than 1.")
    directory = Path(directory).resolve()
    return _inspect_cached(str(directory), _signature(directory / "volume_data.npz"),
                           _signature(directory / "mesh_summary.json"), mode, axis,
                           fraction, metric, n, threshold, layer, aspect_max)


@lru_cache(maxsize=2, typed=True)
def _inspect_cached(directory, volume_signature, summary_signature, mode, axis,
                    fraction, metric, n, threshold, layer, aspect_max):
    """Derived display data are read-only by convention, bounded to two views."""
    artifact = _artifact(directory)
    grid = artifact["grid"]
    values, ids = grid.cell_data[metric], grid.cell_data["cell_id"]
    maximum = 1.0
    if metric == "aspect_ratio":
        maximum = (_aspect_color_ceiling(float(values.max())) if aspect_max is None
                   else _number(aspect_max, "Aspect-ratio color maximum", minimum=1))
        if maximum <= 1:
            raise ValueError("Aspect-ratio color maximum must be greater than 1.")
    layer_count = max(0, len(artifact["stations"]) - 1)
    surface, selected_layer, span_range = None, None, None
    if mode == "slice":
        if axis not in ("X", "Y", "Z"):
            raise ValueError("Slice axis must be X, Y or Z.")
        fraction = _number(fraction, "Slice fraction", minimum=0, maximum=1)
        index = {"X": 0, "Y": 1, "Z": 2}[axis]
        origin = np.asarray(grid.center)
        lo, hi = grid.bounds[2 * index:2 * index + 2]
        origin[index] = lo + fraction * (hi - lo)
        surface = grid.slice(normal=np.eye(3)[index], origin=origin)
        chosen = np.flatnonzero(np.isin(ids, np.unique(surface.cell_data["cell_id"])))
        note = f"Physical {axis} = {origin[index]:.4g} mm plane; actual volume-cell intersections."
    elif mode == "worst":
        n = _integer(n, "Worst-cell count")
        rank = np.lexsort((ids, values if metric == "orthogonality" else -values))
        chosen = rank[:n]
        note = f"{len(chosen):,} worst cells by {METRIC_LABELS[metric].lower()}."
    elif mode == "threshold":
        threshold = _number(threshold, "Quality threshold",
                            minimum=1 if metric == "aspect_ratio" else 0,
                            maximum=None if metric == "aspect_ratio" else 1)
        chosen = np.flatnonzero(values <= threshold if metric == "orthogonality"
                                else values >= threshold)
        relation = "<=" if metric == "orthogonality" else ">="
        note = f"{len(chosen):,} cells with {METRIC_LABELS[metric].lower()} {relation} {threshold:.4g}."
    else:
        if artifact["layer_error"]:
            raise ValueError(artifact["layer_error"])
        selected_layer = _integer(layer, "Span layer")
        if selected_layer > layer_count:
            raise ValueError(f"Span layer must be between 1 and {layer_count}.")
        chosen = np.flatnonzero(artifact["layer_ids"] == selected_layer)
        span_range = artifact["stations"][selected_layer - 1:selected_layer + 1].tolist()
        note = (f"Layer {selected_layer} of {layer_count}; hub-to-shroud/tip span "
                f"{100 * span_range[0]:.3g}% to {100 * span_range[1]:.3g}%. "
                "Original 3D volume-cell quality; one complete cell layer.")
    selected_count = len(chosen)
    detail_order = np.lexsort((ids[chosen], values[chosen] if metric == "orthogonality"
                              else -values[chosen]))
    details = [{"cell_id": int(ids[i]), "value": float(values[i])}
               for i in chosen[detail_order[:DETAIL_LIMIT]]]
    limited = selected_count > RENDER_CELL_LIMIT
    payload = {"points": [], "polys": [], "cell_values": [], "cell_ids": [], "color_values": []}
    if limited:
        note += (f" Rendering paused: {selected_count:,} selected cells exceed the "
                 f"{RENDER_CELL_LIMIT:,}-cell display limit. Use a narrower quality filter; "
                 "no cells were sampled.")
    elif selected_count:
        if surface is None:
            surface = grid.extract_cells(chosen).extract_surface()
        face_values = np.asarray(surface.cell_data[metric])
        payload = {"points": surface.points.ravel().tolist(), "polys": surface.faces.tolist(),
                   "cell_values": face_values.tolist(),
                   "cell_ids": surface.cell_data["cell_id"].tolist(),
                   "color_values": _colors(face_values, metric, maximum).tolist()}
    else:
        note += " No cells match this selection."
    return {"surfaces": {"inspection": payload}, "mode": mode, "note": note,
            "metric": metric, "metric_label": METRIC_LABELS[metric],
            "definition": DEFINITIONS[metric],
            "color_range": [1.0 if metric == "aspect_ratio" else 0.0, maximum],
            "logarithmic": metric == "aspect_ratio", "selected_count": selected_count,
            "total_count": grid.n_cells, "layer_count": layer_count,
            "layer": selected_layer, "span_range": span_range,
            "details": details, "details_truncated": selected_count > DETAIL_LIMIT,
            "render_limited": limited, "render_limit": RENDER_CELL_LIMIT}

