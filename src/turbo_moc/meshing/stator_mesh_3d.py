"""Swept 3D volume mesh of the app's sized axial stator (and rotor).

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

    pitch_angle carries the sense of the periodic rotation: periodic_right is
    periodic_left rotated by -pitch_angle about Z (passage_volume's
    convention), so it is negative when periodic_right sits at lower x.
    """

    def __init__(self, num_blades, r_hub, orientation, sense=1.0):
        self.num_blades = int(num_blades)
        self.pitch_angle = float(sense) * 2.0 * np.pi / self.num_blades
        self.reference_radius = float(r_hub)
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
    n_core =max(1, int(np.ceil(remaining / core_size - 1e-9)))
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


def mesh_swept_passage(mesh_2d, r_hub, r_shroud, n_blades, span, output_dir,
                       boundary_types=None, max_cells=DEFAULT_MAX_CELLS, name="stator_mesh_3d"):
    """Sweep the hub 2D mesh from r_hub to r_shroud and export it.

    `span` holds first_height, growth_ratio, num_layers, transition_ratio
    (endwall layers, normally the blade's) and core_size [mm]. Writes
    <name>.msh, <name>.cgns (metres) and <name>_surfaces.npz (compact
    boundary patches for display) to `output_dir`. Returns a JSON-friendly
    summary; its "files" maps msh/cgns/surfaces to those file names.
    """
    if gmsh.isInitialized():
        raise RuntimeError("Finish the existing Gmsh session before sweeping a passage mesh.")
    section = swept_section(mesh_2d)
    heights = span_distribution(float(r_shroud) - float(r_hub), span["first_height"], span["growth_ratio"],
                                int(span["num_layers"]), span["transition_ratio"], span["core_size"])
    radii = float(r_hub) + heights
    n_cells_2d = sum(len(cells) for cells in section["cells"].values())
    n_cells = n_cells_2d * (len(radii) - 1)
    if n_cells > max_cells:
        raise ValueError(
            f"The swept mesh would have {n_cells:,} cells ({n_cells_2d:,} per section x "
            f"{len(radii) - 1} span cells), above the {max_cells:,} limit: coarsen the 2D mesh, "
            f"raise the span core size or the first-layer height.")

    xyz_sections = np.asarray([section_coordinates(section["xy"], r_hub, r, "cylinder") for r in radii])
    offset = section["xy"][section["slave"]] - section["xy"][section["master"]]
    sense = np.sign(offset[0, 0])
    if not np.allclose(offset, offset[0], atol=1e-8) or sense == 0:
        raise ValueError("The 2D mesh's periodic nodes are not one constant pitchwise translation.")
    channel = SweptAnnulus(n_blades, r_hub, _cell_orientation(section, xyz_sections), sense)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"msh": f"{name}.msh", "cgns": f"{name}.cgns", "surfaces": f"{name}_surfaces.npz"}
    gmsh.initialize(["stator_mesh_3d", "-v", "0"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        logger.info("  Assemble %d span cells x %d section cells.", len(radii) - 1, n_cells_2d)
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
        f"{name}__{key}": value for name, surface in surfaces.items() for key, value in surface.items()})
    span_sizes = np.diff(heights)
    return {
        "output_dir": str(output_dir),
        "files": files,
        "cells": {key: int(value) for key, value in counts.items()},
        "nodes": int(xyz_sections.shape[0] * xyz_sections.shape[1]),
        "span_cells": int(len(radii) - 1),
        "span_first_height_mm": float(span_sizes[0]),
        "span_max_size_mm": float(span_sizes.max()),
        "radii_mm": radii.tolist(),
        "quality": {key: float(value) if isinstance(value, float) else value for key, value in quality.items()},
        "boundary_faces": int(cgns["boundary_faces"]),
        "pitch_angle_deg": float(np.degrees(abs(channel.pitch_angle))),
    }


mesh_swept_stator = mesh_swept_passage  # the original (stator-only) name


def load_patch_surfaces(output_dir, names=DISPLAY_PATCHES, filename="stator_mesh_3d_surfaces.npz"):
    """Read the compact display patches written by mesh_swept_passage."""
    surfaces = {}
    with np.load(Path(output_dir) / filename) as data:
        for name in names:
            if f"{name}__points" in data:
                surfaces[name] = {"points": data[f"{name}__points"], "polys": data[f"{name}__polys"]}
    return surfaces
