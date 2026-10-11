"""Swept 3D volume meshes of the app's stator and rotor rows (axial and radial).

The app's annular stator (turbo_moc.geometry.annular, flare "pitch_scale")
puts every blade point at theta = -x / r_hub and z = y at EVERY radius: the
blade surface is ruled along radial lines, and the section at radius r is the
hub section stretched pitchwise by r / r_hub. So the hub 2D mesh
(stator_mesh_2d.mesh_sized_stator) mapped onto the cylinder at each span
station (stator_mesh_2d.section_coordinates, "cylinder") is an exact sweep:

- swept blade faces lie on the exported solid (not only at the stations);
- each station is an affine image of the hub mesh (per cylinder), so no
  cell can invert and every station has the same topology;
- quadrilaterals become hexahedra, triangles prisms, and the periodic
  translation by one hub pitch becomes a rotation by 2 pi / N about Z.

Spanwise, the hub and shroud endwalls get the blade's inflation settings
(first height from the y+ estimate, number of layers, growth ratio), then
cells keep growing at the transition ratio up to a uniform core size.
The rotor's hub mesh (turbo_moc.meshing.rotor_mesh_2d) is built in the same
(X pitchwise, Y axial) frame, so it sweeps onto the app's rotor solid the
same way. Volume assembly, checks and CGNS export:
turbo_moc.meshing.passage_volume.
"""

import logging
from pathlib import Path

import gmsh
import numpy as np

from turbo_moc.meshing import passage_volume
from turbo_moc.meshing.stator_mesh_2d import section_coordinates

logger = logging.getLogger(__name__)

DEFAULT_MAX_CELLS = 20_000_000
DISPLAY_PATCHES = ("blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right")


class SweptAnnulus:
    """The `channel` interface of passage_volume for the swept stator.

    pitch_angle (+-2 pi / N) carries the sense of the periodic rotation:
    periodic_right is periodic_left rotated by -pitch_angle about Z
    (passage_volume's convention).
    """

    def __init__(self, num_blades, reference_radius, orientation, pitch_angle):
        self.num_blades = int(num_blades)
        self.pitch_angle = float(pitch_angle)
        self.reference_radius = float(reference_radius)
        self.orientation = int(orientation)

    def rotation(self):
        c, s = np.cos(-self.pitch_angle), np.sin(-self.pitch_angle)
        result = np.eye(4)
        result[:2, :2] = [[c, -s], [s, c]]
        return result


def span_distribution(height, first_height, growth_ratio, num_layers, transition_ratio, core_size):
    """Spanwise station heights [mm] from the hub (0) to the shroud (height).

    Symmetric about midspan: num_layers geometric layers (first_height *
    growth_ratio^k) on each endwall, then growth at transition_ratio until
    core_size, then a uniform core that fills the rest exactly.
    """
    values = (height, first_height, growth_ratio, num_layers, transition_ratio, core_size)
    if not all(np.isfinite(v) and v > 0 for v in values) or growth_ratio <= 1 or transition_ratio < 1:
        raise ValueError("Span distribution needs positive sizes, growth ratio > 1 and transition ratio >= 1.")
    graded, size = [], float(first_height)
    while size < core_size and 2.0 * (sum(graded) + size) <= height:
        graded.append(size)
        size *= growth_ratio if len(graded) < num_layers else max(transition_ratio, 1.0 + 1e-6)
    remaining = height - 2.0 * sum(graded)
    while graded and remaining < 0.5 * graded[-1]:  # no sliver core: merge the last layers into it
        remaining += 2.0 * graded.pop()
    n_core = max(1, int(np.ceil(remaining / core_size - 1e-9)))
    sizes = np.r_[graded, np.full(n_core, remaining / n_core), graded[::-1]]
    stations = np.r_[0.0, np.cumsum(sizes)]
    stations[-1] = height
    return stations


def swept_section(mesh_2d):
    """The 2D app mesh (1-based plain lists) as passage_volume's section dict
    (0-based arrays; slave = periodic_right nodes, master = periodic_left)."""
    cells = {}
    for width, kind in ((3, 2), (4, 3)):
        selected = [cell for cell in mesh_2d["cells"] if len(cell) == width]
        if selected:
            cells[kind] = np.asarray(selected, dtype=np.int64) - 1
    return {
        "xy": np.asarray(mesh_2d["xy"], dtype=float),
        "cells": cells,
        "boundaries": {name: np.asarray(edges, dtype=np.int64)[:, :2] - 1
                       for name, edges in mesh_2d["named_edges"].items()},
        "slave": np.asarray(mesh_2d["periodic"]["current"], dtype=np.int64) - 1,
        "master": np.asarray(mesh_2d["periodic"]["donor"], dtype=np.int64) - 1,
    }


def _cell_orientation(section, xyz_sections):
    """+1 if counter-clockwise section cells stacked hub -> shroud give
    positive volumes, else -1 (the cylinder map's handedness)."""
    cells = next(iter(section["cells"].values()))
    bottom, top = xyz_sections[0][cells[:, :3]], xyz_sections[1][cells[:, 0]]
    normal = np.cross(bottom[:, 1] - bottom[:, 0], bottom[:, 2] - bottom[:, 0])
    signs = np.sign(np.einsum("ij,ij->i", normal, top - bottom[:, 0]))
    if not np.all(signs == signs[0]) or signs[0] == 0:
        raise ValueError("Section cells are not consistently oriented.")
    return int(signs[0])


def _patch_surfaces(xyz, faces, names):
    """Per patch: its own points and flat VTK polys ([n, i0, i1, ...]...)."""
    surfaces = {}
    for name in names:
        blocks = [block for block in faces[name] if len(block)]
        if not blocks:
            continue
        used = np.unique(np.concatenate([block.ravel() for block in blocks]))
        local = np.full(len(xyz), -1, dtype=np.int64)
        local[used] = np.arange(len(used))
        polys = np.concatenate([
            np.column_stack((np.full(len(block), block.shape[1]), local[block])).ravel()
            for block in blocks
        ])
        surfaces[name] = {"points": xyz[used].astype(np.float32), "polys": polys.astype(np.int32)}
    return surfaces


def _sweep_and_export(mesh_2d, xyz_sections, heights, n_blades, reference_radius, output_dir,
                      boundary_types, name):
    """Assemble, check and export the volume from the 2D mesh's node
    coordinates on each station (xyz_sections: (n_stations, n_nodes, 3)).

    The periodic rotation is measured from the mapped nodes: it must be
    2 pi / n_blades about Z (its sense follows the mapping)."""
    if gmsh.isInitialized():
        raise RuntimeError("Finish the existing Gmsh session before sweeping a passage mesh.")
    section = swept_section(mesh_2d)
    right, left = xyz_sections[0][section["slave"]], xyz_sections[0][section["master"]]
    turn = np.angle(np.exp(1j * (np.arctan2(right[:, 1], right[:, 0]) - np.arctan2(left[:, 1], left[:, 0]))))
    expected = 2.0 * np.pi / n_blades
    if not np.allclose(np.abs(turn), expected, rtol=1e-8, atol=0) or not np.allclose(turn, turn[0]):
        raise ValueError(
            f"The periodic boundaries are {np.degrees(np.abs(turn).max()):.4f} deg apart, not "
            f"360/N = {np.degrees(expected):.4f} deg: the passage pitch does not match the blade count.")
    channel = SweptAnnulus(n_blades, reference_radius, _cell_orientation(section, xyz_sections), -turn[0])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"msh": f"{name}.msh", "cgns": f"{name}.cgns", "surfaces": f"{name}_surfaces.npz"}
    gmsh.initialize(["passage_mesh_3d", "-v", "0"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        logger.info("  Assemble %d span cells x %d section cells.", len(heights) - 1,
                    sum(len(cells) for cells in section["cells"].values()))
        counts = passage_volume.assemble_volume(section, xyz_sections, channel)
        quality = passage_volume.validate_volume(section, xyz_sections, channel)
        gmsh.option.setNumber("Mesh.Binary", 1)  # ~1/3 of the ASCII size
        gmsh.write(str(output_dir / files["msh"]))
        cgns = passage_volume.export_cgns(
            output_dir / files["cgns"],
            {"length_scale": 0.001, "boundary_types": boundary_types or {}}, channel)
        xyz, faces = passage_volume.collect_surface_mesh()
    finally:
        gmsh.finalize()

    surfaces = _patch_surfaces(xyz, faces, DISPLAY_PATCHES)
    np.savez_compressed(output_dir / files["surfaces"], **{
        f"{patch}__{key}": value for patch, surface in surfaces.items() for key, value in surface.items()})
    span_sizes = np.diff(heights)
    return {
        "output_dir": str(output_dir),
        "files": files,
        "cells": {key: int(value) for key, value in counts.items()},
        "nodes": int(xyz_sections.shape[0] * xyz_sections.shape[1]),
        "span_cells": int(len(heights) - 1),
        "span_first_height_mm": float(span_sizes[0]),
        "span_max_size_mm": float(span_sizes.max()),
        "quality": {key: float(value) if isinstance(value, float) else value for key, value in quality.items()},
        "boundary_faces": int(cgns["boundary_faces"]),
        "pitch_angle_deg": float(np.degrees(abs(channel.pitch_angle))),
    }


def _check_cell_budget(mesh_2d, n_span, max_cells):
    n_cells_2d = len(mesh_2d["cells"])
    if n_cells_2d * n_span > max_cells:
        raise ValueError(
            f"The swept mesh would have {n_cells_2d * n_span:,} cells ({n_cells_2d:,} per section x "
            f"{n_span} span cells), above the {max_cells:,} limit: coarsen the 2D mesh, "
            f"raise the span core size or the first-layer height.")


def mesh_swept_passage(mesh_2d, r_hub, r_shroud, n_blades, span, output_dir,
                       boundary_types=None, max_cells=DEFAULT_MAX_CELLS, name="stator_mesh_3d"):
    """Axial row: sweep the hub 2D mesh from r_hub to r_shroud and export it.

    `span` holds first_height, growth_ratio, num_layers, transition_ratio
    (endwall layers, normally the blade's) and core_size [mm]. Writes
    <name>.msh, <name>.cgns (metres) and <name>_surfaces.npz (compact
    boundary patches for display) to `output_dir`. Returns a JSON-friendly
    summary; its "files" maps msh/cgns/surfaces to those file names.
    """
    heights = span_distribution(float(r_shroud) - float(r_hub), span["first_height"], span["growth_ratio"],
                                int(span["num_layers"]), span["transition_ratio"], span["core_size"])
    _check_cell_budget(mesh_2d, len(heights) - 1, max_cells)
    xy = np.asarray(mesh_2d["xy"], dtype=float)
    xyz_sections = np.asarray([section_coordinates(xy, r_hub, float(r_hub) + h, "cylinder") for h in heights])
    result = _sweep_and_export(mesh_2d, xyz_sections, heights, n_blades, r_hub, output_dir, boundary_types, name)
    result["radii_mm"] = (float(r_hub) + heights).tolist()
    return result


def radial_map_reference(blade_data, r1, r2, source):
    """The log-spiral map of turbo_moc.geometry.radial for this blade, in the
    mesh frame (X pitchwise; Y = the stator's own chordwise y, or -x for the
    rotor, see rotor_mesh_2d): the same (x1, y1, c_axial) the radial wrap
    derives from the blade's main contour, so the mesh lands on its solid.
    The rotor's wrap mirrors its pitch (wrap_rotor_blade_radial: -y, for
    the same handedness as the stator), hence pitch_sign = -1 there."""
    from turbo_moc.geometry.radial import _reference_point_and_scale

    if source == "stator":
        curve = blade_data["blade_curve"]
        chord, pitch, chord_sign, pitch_sign = curve["y"], curve["x"], 1.0, 1.0
    elif source == "rotor":
        curve = blade_data["blade"]
        chord, pitch, chord_sign, pitch_sign = curve["x"], [-v for v in curve["y"]], -1.0, -1.0
    else:
        raise ValueError(f"source must be 'stator' or 'rotor', got {source!r}")
    x1, y1, c_axial = _reference_point_and_scale(chord, pitch)
    return {"x1": x1, "y1": y1, "c_axial": c_axial, "r1": float(r1), "r2": float(r2),
            "chord_sign": chord_sign, "pitch_sign": pitch_sign}


def radial_section_coordinates(xy, reference, z=0.0):
    """Mesh-frame nodes -> (X, Y, z) on the radial annulus (log-spiral map)."""
    from turbo_moc.geometry.radial import apply_conformal_mapping

    xy = np.asarray(xy, dtype=float)
    X, Y = apply_conformal_mapping(reference["chord_sign"] * xy[:, 1], reference["pitch_sign"] * xy[:, 0],
                                   reference["x1"],
                                   reference["y1"], reference["r1"], reference["r2"], reference["c_axial"])
    return np.column_stack((X, Y, np.full(len(xy), float(z))))


def radial_length_scale(reference, radius):
    """Physical length of one 2D-plane unit at `radius` (the map's exact
    local magnification |a| r, a = ln(r2/r1) / c_axial)."""
    return abs(np.log(reference["r2"] / reference["r1"]) / reference["c_axial"]) * float(radius)


def _split_nonconvex_quads(mesh_2d, mapped):
    """The log-spiral map bends straight 2D edges slightly, so a nearly flat
    quad (a corner close to 180 deg) can map to a non-convex one: split
    those into the two triangles (prisms after extrusion) whose orientation
    matches the section's. Boundary edges are unchanged. Returns (mesh_2d
    copy, number split)."""
    def turns(points):
        edge = np.roll(points, -1, axis=0) - points
        previous = np.roll(edge, 1, axis=0)
        return previous[:, 0] * edge[:, 1] - previous[:, 1] * edge[:, 0]

    def area(points):
        return 0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))

    sign = np.sign(np.median([area(mapped[np.asarray(c) - 1]) for c in mesh_2d["cells"][:2000]]))
    cells, split = [], 0
    for cell in mesh_2d["cells"]:
        points = mapped[np.asarray(cell) - 1]
        if np.all(sign * turns(points) > 0):
            cells.append(cell)
            continue
        if len(cell) == 4:
            a, b, c, d = cell
            for pair in (([a, b, c], [a, c, d]), ([a, b, d], [b, c, d])):
                if all(sign * area(mapped[np.asarray(t) - 1]) > 0 for t in pair):
                    cells.extend(pair)
                    split += 1
                    break
            else:
                raise ValueError("A section cell folds under the radial map; refine the 2D mesh.")
        else:
            raise ValueError("A section triangle folds under the radial map; refine the 2D mesh.")
    # The sweep reads only the named edges' end nodes, not their cell numbers.
    return ({**mesh_2d, "cells": cells}, split) if split else (mesh_2d, 0)


def mesh_radial_passage(mesh_2d, reference, n_blades, depth, span, output_dir,
                        boundary_types=None, max_cells=DEFAULT_MAX_CELLS, name="stator_mesh_3d"):
    """Radial row: map the 2D mesh with the log-spiral wrap and extrude it
    through `depth` (flat endwalls at z = 0 and z = depth, "hub"/"shroud").
    Same `span` settings and outputs as mesh_swept_passage."""
    heights = span_distribution(float(depth), span["first_height"], span["growth_ratio"],
                                int(span["num_layers"]), span["transition_ratio"], span["core_size"])
    base = radial_section_coordinates(mesh_2d["xy"], reference)
    mesh_2d, split = _split_nonconvex_quads(mesh_2d, base[:, :2])
    if split:
        logger.info("  Split %d quadrilaterals the radial map made non-convex.", split)
    _check_cell_budget(mesh_2d, len(heights) - 1, max_cells)
    xyz_sections = np.asarray([np.column_stack((base[:, :2], np.full(len(base), h))) for h in heights])
    reference_radius = min(reference["r1"], reference["r2"])
    result = _sweep_and_export(mesh_2d, xyz_sections, heights, n_blades, reference_radius, output_dir,
                               boundary_types, name)
    result["z_mm"] = heights.tolist()
    result["split_quads"] = int(split)
    return result


mesh_swept_stator = mesh_swept_passage  # the original (stator-only) name


def load_patch_surfaces(output_dir, names=DISPLAY_PATCHES, filename="stator_mesh_3d_surfaces.npz"):
    """Read the compact display patches written by mesh_swept_passage."""
    surfaces = {}
    with np.load(Path(output_dir) / filename) as data:
        for name in names:
            if f"{name}__points" in data:
                surfaces[name] = {"points": data[f"{name}__points"], "polys": data[f"{name}__polys"]}
    return surfaces
