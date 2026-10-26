"""Planar stage mesh: the rows' midspan reference sections assembled into one 2D CGNS for Fluent.

Each row's reference section lives in its own conformal frame (x along the pitch, y along the
meridional direction, millimetres), with periodic_right one pitch to +x of periodic_left. On an
axial channel at the reference radius this frame is the unwrapped midspan cylinder, so the rows
are joined by a translation that puts the rotor inlet on the stator outlet; the rotor passage is
repeated along the pitch until both rows span the same sector, as for the 3D frozen rotor.
"""

from pathlib import Path

import numpy as np

from turbo_moc.io.cgns_adf import AdfNode, new_root, write_adf

BOUNDARIES = ("inlet", "outlet", "blade", "periodic_left", "periodic_right")
BOUNDARY_TYPES = {"inlet": "BCInflowSubsonic", "outlet": "BCOutflowSubsonic", "blade": "BCWallViscous",
                  "periodic_left": "BCGeneral", "periodic_right": "BCGeneral"}


def load_section(path):
    data = np.load(path)
    xy = np.asarray(data["xy"], dtype=float)
    cells = [np.asarray(cell, dtype=np.int64) for key in ("cells_2", "cells_3") if key in data.files
             for cell in data[key]]
    boundaries = {name: np.asarray(data[f"boundary_{name}"], dtype=np.int64) for name in BOUNDARIES}
    master, slave = np.asarray(data["master"], dtype=np.int64), np.asarray(data["slave"], dtype=np.int64)
    pitch = (xy[slave] - xy[master]).mean(axis=0)
    if not np.allclose(xy[slave] - xy[master], pitch, atol=1e-6):
        raise ValueError(f"{path}: periodic nodes are not one constant translation apart.")
    return {"xy": xy, "cells": cells, "boundaries": boundaries, "master": master, "slave": slave, "pitch": pitch}


def replicate(section, copies):
    """``copies`` passages side by side along +x, the shared periodic seams merged into interior edges."""
    if copies == 1:
        return section
    count = len(section["xy"])
    xy = np.vstack([section["xy"] + k * section["pitch"] for k in range(copies)])
    index = np.arange(count * copies)
    for k in range(1, copies):
        index[k * count + section["master"]] = (k - 1) * count + section["slave"]
    used = np.unique(index)
    renumber = np.full(len(index), -1)
    renumber[used] = np.arange(len(used))
    remap = renumber[index]
    cells = [remap[cell + k * count] for k in range(copies) for cell in section["cells"]]
    boundaries = {}
    for name, edges in section["boundaries"].items():
        if name == "periodic_left":
            kept = [0]
        elif name == "periodic_right":
            kept = [copies - 1]
        else:
            kept = range(copies)
        boundaries[name] = np.vstack([remap[edges + k * count] for k in kept])
    return {"xy": xy[used], "cells": cells, "boundaries": boundaries, "master": remap[section["master"]],
            "slave": remap[section["slave"] + (copies - 1) * count], "pitch": section["pitch"] * copies}


def translated(section, offset):
    return dict(section, xy=section["xy"] + np.asarray(offset, dtype=float))


def edge_line(section, name):
    points = section["xy"][np.unique(section["boundaries"][name])]
    return points[:, 0].min(), points[:, 0].max(), float(np.median(points[:, 1]))


def assemble(stator, rotor, rotor_copies, match=True):
    """Rotor passages repeated to the stator pitch and moved so that their inlet lies on the stator outlet.

    With ``match=False`` (mixing plane) the rows keep their own pitches.
    """
    rotor = replicate(rotor, rotor_copies)
    if match and not np.isclose(rotor["pitch"][0], stator["pitch"][0], rtol=1e-6):
        raise ValueError(f"Sector mismatch: {rotor_copies} rotor passages span {rotor['pitch'][0]:.6f} mm, "
                         f"one stator passage {stator['pitch'][0]:.6f} mm.")
    outlet_start, _, outlet_y = edge_line(stator, "outlet")
    inlet_start, _, inlet_y = edge_line(rotor, "inlet")
    return stator, translated(rotor, (outlet_start - inlet_start, outlet_y - inlet_y))


def oriented(section):
    """Cells counterclockwise and each boundary edge with its owner cell and local edge position (1-based)."""
    xy = section["xy"]
    cells, owners = [], {}
    for number, cell in enumerate(section["cells"], 1):
        points = xy[cell]
        area = np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        if area == 0:
            raise ValueError("Degenerate cell in the planar section.")
        cell = cell if area > 0 else cell[::-1]
        cells.append(cell)
        for position, (a, b) in enumerate(zip(cell, np.roll(cell, -1)), 1):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b), number, position))
    exterior = {key: found[0] for key, found in owners.items() if len(found) == 1}
    edges = {}
    for name, pairs in section["boundaries"].items():
        rows = []
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            if key not in exterior:
                raise ValueError(f"Boundary {name} has an edge that is not on the mesh boundary.")
            rows.append(exterior.pop(key))
        edges[name] = np.array(rows, dtype=np.int64)
    if exterior:
        raise ValueError(f"{len(exterior)} boundary edges belong to no named boundary.")
    return cells, edges


def _units():
    units = np.full((32, 5), b" ", dtype="S1")
    for column, unit in enumerate(("Kilogram", "Meter", "Second", "Kelvin", "Radian")):
        units[: len(unit), column] = np.frombuffer(unit.encode("ascii"), dtype="S1")
    return units


def _array(parent, name, values):
    array = parent.add(AdfNode(name, "DataArray_t", values))
    array.add(AdfNode("DimensionalExponents", "DimensionalExponents_t",
                      np.array([0, 1, 0, 0, 0], dtype=np.float32)))


def write_cgns(filename, zones, length_scale=1e-3):
    """One 2D CGNS base with a zone per row; ``zones`` maps a row id to its section, boundaries named ``<row>_<name>``."""
    root = new_root()
    root.add(AdfNode("CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([3.4], dtype=np.float32)))
    base = root.add(AdfNode("Base", "CGNSBase_t", np.array([2, 2], dtype=np.int32)))
    base.add(AdfNode("DataClass", "DataClass_t", "Dimensional"))
    base.add(AdfNode("DimensionalUnits", "DimensionalUnits_t", _units()))
    for row, section in zones.items():
        cells, edges = oriented(section)
        name = f"{row}_fluid"
        base.add(AdfNode(f"{name}_family", "Family_t"))
        zone = base.add(AdfNode(name, "Zone_t", np.array([[len(section["xy"]), len(cells), 0]], dtype=np.int32)))
        zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
        zone.add(AdfNode("FamilyName", "FamilyName_t", f"{name}_family"))
        grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
        for axis, values in zip("XY", (section["xy"] * length_scale).T):
            _array(grid, f"Coordinate{axis}", np.ascontiguousarray(values))

        def add_section(label, kind, start, count, connectivity):
            node = zone.add(AdfNode(label, "Elements_t", np.array([kind, 0], dtype=np.int32)))
            node.add(AdfNode("ElementRange", "IndexRange_t", np.array([start, start + count - 1], dtype=np.int32)))
            node.add(AdfNode("ElementConnectivity", "DataArray_t", connectivity))
            return node

        mixed = np.concatenate([np.r_[5 if len(cell) == 3 else 7, cell + 1] for cell in cells]).astype(np.int32)
        add_section(f"{name}_cells", 20, 1, len(cells), mixed)
        zone_bc = zone.add(AdfNode("ZoneBC", "ZoneBC_t"))
        start = len(cells) + 1
        for boundary, rows in edges.items():
            label = f"{row}_{boundary}"
            node = add_section(label, 3, start, len(rows), (rows[:, :2] + 1).astype(np.int32).ravel())
            zeros = np.zeros(len(rows), dtype=np.int32)
            node.add(AdfNode("ParentElements", "DataArray_t", np.column_stack((rows[:, 2], zeros)).astype(np.int32)))
            node.add(AdfNode("ParentElementsPosition", "DataArray_t",
                             np.column_stack((rows[:, 3], zeros)).astype(np.int32)))
            bc = zone_bc.add(AdfNode(label, "BC_t", BOUNDARY_TYPES[boundary]))
            bc.add(AdfNode("GridLocation", "GridLocation_t", "EdgeCenter"))
            bc.add(AdfNode("PointRange", "IndexRange_t", np.array([[start, start + len(rows) - 1]], dtype=np.int32)))
            start += len(rows)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_adf(str(filename), root)
    return filename
