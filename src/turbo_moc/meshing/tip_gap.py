"""Rotor tip clearance for the swept 3D passage (TurboExpo 3D, 2026-11-15).

Opt-in: TURBOMOC_TIP_GAP_MM (radial clearance [mm], rotor rows only) and TURBOMOC_TIP_GAP_LAYERS (span layers inside
the clearance, default 6, uniform). The blade occupies the span fraction [0, eta_tip], eta_tip = 1 - gap / h, swept with the
row's own clustered span stations scaled to eta_tip. Above the tip the passage section continues (tip blade profile at
the local radius) and the blade interior is filled with a triangle mesh that shares the blade-wall nodes of the
reference section, swept as prisms through the clearance layers (uniform).
Boundary patches are unchanged: the tip cap joins "blade", the shroud gains the interior triangles. Interior nodes
exist only at the clearance stations (node tags are not contiguous; the exporters work from sorted tags).
"""
import os

import gmsh
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from ..geometry import meridional_channel as meridional


def settings(config):
    gap = float(os.environ.get("TURBOMOC_TIP_GAP_MM", 0.0)) if "rotor_design" in config else 0.0
    if gap <= 0:
        return None
    height = float(config["meridional_channel"]["blade_height"])
    layers = int(os.environ.get("TURBOMOC_TIP_GAP_LAYERS", 6))
    return {"gap_mm": gap, "height_mm": height, "eta_tip": 1.0 - gap / height, "layers": layers}


def stations(numerics, gap):
    """Blade stations (row distribution scaled to eta_tip) + clearance stations; index of the tip station."""
    blade = meridional.span_stations(numerics) * gap["eta_tip"]
    t = np.linspace(0, 1, gap["layers"] + 1)
    clearance = gap["eta_tip"] + (1 - gap["eta_tip"]) * t  # uniform: the blade tip layer is ~0.2 mm
    eta = np.concatenate((blade, clearance[1:]))
    return eta, len(blade) - 1


def _wall_loop(section):
    edges = np.asarray(section["boundaries"]["blade"])
    nxt = {int(a): int(b) for a, b in edges}
    start = int(edges[0, 0])
    loop, node = [start], nxt[start]
    while node != start:
        loop.append(node)
        node = nxt[node]
    if len(loop) != len(edges):
        raise ValueError("The blade wall is not a single closed loop.")
    return np.asarray(loop)


class BladeInterior:
    """Triangle mesh of the blade interior on the reference wall nodes, and its harmonic deformation."""

    def __init__(self, section, max_size):
        xy, n = section["xy"], len(section["xy"])
        loop = _wall_loop(section)
        pts = xy[loop]
        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        size = 0.5 * (seg + np.roll(seg, 1))
        if gmsh.isInitialized():
            raise RuntimeError("A Gmsh session is already active.")
        gmsh.initialize(["blade_interior", "-v", "2"], readConfigFiles=False)
        try:
            gmsh.model.add("blade_interior")
            tags = [gmsh.model.geo.addPoint(x, y, 0, h) for (x, y), h in zip(pts, size)]
            lines = [gmsh.model.geo.addLine(a, b) for a, b in zip(tags, np.roll(tags, -1))]
            surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
            gmsh.model.geo.synchronize()
            for line in lines:
                gmsh.model.mesh.setTransfiniteCurve(line, 2)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(max_size))
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            # Wall spacing at the wall, growing to the core size 1 mm inside (wall spacing everywhere: 56k triangles)
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "CurvesList", lines)
            gmsh.model.mesh.field.setNumber(dist, "Sampling", 4)
            ramp = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ramp, "InField", dist)
            gmsh.model.mesh.field.setNumber(ramp, "SizeMin", float(np.median(seg)))
            gmsh.model.mesh.field.setNumber(ramp, "SizeMax", float(max_size))
            gmsh.model.mesh.field.setNumber(ramp, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(ramp, "DistMax", 1.0)
            gmsh.model.mesh.field.setAsBackgroundMesh(ramp)
            gmsh.model.mesh.generate(2)
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            coords = np.asarray(coords).reshape(-1, 3)[:, :2]
            _, _, conn = gmsh.model.mesh.getElements(2, surface)
            tris = np.asarray(conn[0]).reshape(-1, 3)
        finally:
            gmsh.finalize()
        distance, nearest = cKDTree(pts).query(coords)
        on_wall = distance < 1e-9
        index = np.empty(len(node_tags), dtype=np.int64)
        index[on_wall] = loop[nearest[on_wall]]
        interior = np.flatnonzero(~on_wall)
        index[interior] = n + np.arange(len(interior))
        lookup = dict(zip(node_tags.tolist(), index.tolist()))
        tris = np.vectorize(lookup.get)(tris)
        self.n, self.xy = n, coords[interior]
        full = np.vstack((xy, self.xy))
        p = full[tris]
        area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        tris[area < 0] = tris[area < 0][:, ::-1]
        self.tris = tris.astype(np.int64)
        # Harmonic extension of the wall displacement (unknowns: interior nodes; Dirichlet: wall nodes)
        e = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
        w = 1 / np.sum((full[e[:, 1]] - full[e[:, 0]]) ** 2, axis=1)
        m = len(full)
        adj = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(m, m)).tocsc()
        lap = (diags(np.asarray(adj.sum(axis=1)).ravel()) - adj).tocsc()
        self.free = np.arange(n, m)
        self.wall = loop
        self.factor = splu(lap[self.free][:, self.free].tocsc())
        self.coupling = lap[self.free][:, self.wall]
        self.reference = xy

    def positions(self, q):
        """Interior node positions for a deformed passage section q (n x 2)."""
        move = q[self.wall] - self.reference[self.wall]
        out = self.xy + self.factor.solve(-self.coupling @ move)
        p = np.vstack((q, out))[self.tris]
        area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        if np.any(area <= 0):
            raise ValueError(f"Blade-interior deformation inverted {int(np.sum(area <= 0))} triangle(s).")
        return out


def assemble_volume(section, interior, xyz_sections, meridional_channel, j_tip, surfaces):
    """assemble_volume with a tip clearance: passage hexes/prisms at every layer, interior prisms above the tip."""
    gmsh.model.add("moc_3d_passage")
    for name, tag in surfaces.items():
        gmsh.model.addDiscreteEntity(2, tag)
        gmsh.model.setEntityName(2, tag, name)
    gmsh.model.addDiscreteEntity(3, 1, list(surfaces.values()))
    n_span, n_full, _ = xyz_sections.shape
    n = interior.n
    used = [np.arange(n) + j * n_full for j in range(n_span)]
    used += [np.arange(n, n_full) + j * n_full for j in range(j_tip, n_span)]
    used = np.sort(np.concatenate(used))
    xyz = xyz_sections.reshape(-1, 3)
    gmsh.model.mesh.addNodes(3, 1, used + 1, xyz[used].ravel())
    flip = meridional_channel.orientation < 0
    counts = {"hexahedra": 0, "prisms": 0}

    def sweep(cells, layers):
        cells = cells[:, ::-1] if flip else cells
        bottom = cells[None] + (np.asarray(layers)[:, None, None] * n_full) + 1
        return cells, np.concatenate((bottom, bottom + n_full), axis=2)

    tris = interior.tris
    for kind, cells in section["cells"].items():
        oriented, volumes = sweep(cells, range(n_span - 1))
        gmsh.model.mesh.addElementsByType(1, 5 if kind == 3 else 6, [], volumes.ravel())
        counts["hexahedra" if kind == 3 else "prisms"] += int(np.prod(volumes.shape[:2]))
        gmsh.model.mesh.addElementsByType(surfaces["hub"], kind, [], (oriented[:, ::-1] + 1).ravel())
        gmsh.model.mesh.addElementsByType(surfaces["shroud"], kind, [], (oriented + (n_span - 1) * n_full + 1).ravel())
    oriented, volumes = sweep(tris, range(j_tip, n_span - 1))
    gmsh.model.mesh.addElementsByType(1, 6, [], volumes.ravel())
    counts["prisms"] += int(np.prod(volumes.shape[:2]))
    counts["clearance_prisms"] = int(np.prod(volumes.shape[:2]))
    gmsh.model.mesh.addElementsByType(surfaces["shroud"], 2, [], (oriented + (n_span - 1) * n_full + 1).ravel())
    gmsh.model.mesh.addElementsByType(surfaces["blade"], 2, [], (oriented[:, ::-1] + j_tip * n_full + 1).ravel())
    for name, edges in section["boundaries"].items():
        edges = edges[:, ::-1] if flip else edges
        layers = np.arange(j_tip) if name == "blade" else np.arange(n_span - 1)
        bottom = edges[None] + layers[:, None, None] * n_full + 1
        faces = np.concatenate((bottom, bottom[:, :, ::-1] + n_full), axis=2)
        gmsh.model.mesh.addElementsByType(surfaces[name], 3, [], faces.ravel())
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.mesh.createTopology(makeSimplyConnected=False)
    for name, tag in surfaces.items():
        gmsh.model.addPhysicalGroup(2, [tag], tag, name)
    gmsh.model.addPhysicalGroup(3, [1], 1, "fluid_domain")
    gmsh.model.mesh.setPeriodic(2, [surfaces["periodic_right"]], [surfaces["periodic_left"]],
                                meridional_channel.periodic_transform().ravel())
    return counts
