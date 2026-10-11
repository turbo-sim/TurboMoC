"""Compact slices and worst-cell subsets extracted from the actual volume."""
import numpy as np


def load_grid(directory):
    import pyvista as pv
    with np.load(directory / "volume_data.npz", allow_pickle=False) as data:
        packed, kinds, tags = [], [], []
        for kind, width, vtk_type in ((5, 8, 12), (6, 6, 13)):
            name = f"cells_{kind}"
            if name not in data:
                continue
            cells = data[name]
            packed.append(np.column_stack((np.full(len(cells), width), cells)).ravel())
            kinds.append(np.full(len(cells), vtk_type, dtype=np.uint8))
            tags.append(data[f"tags_{kind}"])
        grid = pv.UnstructuredGrid(np.concatenate(packed), np.concatenate(kinds), data["xyz"])
        grid.cell_data["cell_id"] = np.concatenate(tags)
        for metric in ("skewness", "orthogonality", "aspect_ratio"):
            grid.cell_data[metric] = data[metric]
    return grid


def inspect_mesh(directory, mode, axis="Z", fraction=0.5, metric="skewness"):
    """Return drawing data plus an explanation of the exact inspection basis."""
    if mode not in ("slice", "worst"):
        return None
    grid = load_grid(directory)
    if mode == "slice":
        index = {"X": 0, "Y": 1, "Z": 2}[axis]
        origin = np.asarray(grid.center)
        lo, hi = grid.bounds[2 * index:2 * index + 2]
        origin[index] = lo + float(fraction) * (hi - lo)
        normal = np.eye(3)[index]
        surface = grid.slice(normal=normal, origin=origin)
        note = f"Physical {axis} = {origin[index]:.4g} mm plane; actual volume-cell intersections."
    else:
        values = grid.cell_data[metric]
        rank = np.argsort(values)
        if metric != "orthogonality":
            rank = rank[::-1]
        chosen = rank[:10]
        surface = grid.extract_cells(chosen).extract_surface()
        ids = grid.cell_data["cell_id"][chosen].tolist()
        note = f"Ten worst cells by {metric}; original Gmsh IDs: {ids}."
    return {"surfaces": {"inspection": {"points": surface.points.ravel().tolist(),
                                     "polys": surface.faces.tolist()}},
            "note": note, "mode": mode}

