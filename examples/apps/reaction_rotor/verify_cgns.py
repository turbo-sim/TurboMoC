"""Reopen the exported reaction-rotor CGNS files and audit them independently of the mesher.

    python examples/apps/reaction_rotor/verify_cgns.py

For every output/<case>/jobs/*/*/reaction_rotor_3d.cgns: element counts, minimum corner Jacobian determinant and
minimum scaled corner Jacobian of the hexahedra, boundary-patch coverage (every exposed hexahedron face must belong
to exactly one named patch, and no patch face may be interior), the seven required patches, and the periodic
pairing error (periodic_left nodes rotated by one pitch onto periodic_right nodes). Writes output/cgns_audit.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from turbo_moc.io.cgns_adf import read_adf

HERE = Path(__file__).resolve().parent
REQUIRED = ("blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right")
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
CORNERS = ((0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7), (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2),
           (7, 6, 4, 3))  # corner, then its three edge neighbours (right-handed)


def child(node, name):
    return next(c for c in node.children if c.name == name)


def audit(path, num_blades):
    root = read_adf(path)
    base = next(c for c in root.children if c.label == "CGNSBase_t")
    zone = next(c for c in base.children if c.label == "Zone_t")
    xyz = np.column_stack([child(child(zone, "GridCoordinates"), f"Coordinate{a}").data for a in "XYZ"])
    xyz = 1e3 * xyz  # The exported CGNS is in metres (Fluent-ready); audit in mm
    sections = {c.name: c for c in zone.children if c.label == "Elements_t"}
    types = {name: int(np.asarray(s.data)[0]) for name, s in sections.items()}
    conn = {name: np.asarray(child(s, "ElementConnectivity").data, dtype=np.int64) - 1 for name, s in sections.items()}
    volume_name = next(n for n, t in types.items() if t == 17)  # HEXA_8
    hexes = conn[volume_name].reshape(-1, 8)
    report = {"file": str(path), "nodes": len(xyz), "hexahedra": len(hexes),
              "element_types": types, "patches_present": sorted(set(sections) - {volume_name})}
    # Corner Jacobians: det[e1 e2 e3] at each of the eight corners (mm^3), and the scaled value.
    p = xyz[hexes]
    determinants, scaled = [], []
    for c, a, b, d in CORNERS:
        e1, e2, e3 = p[:, a] - p[:, c], p[:, b] - p[:, c], p[:, d] - p[:, c]
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        determinants.append(det)
        scaled.append(det / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)))
    determinants, scaled = np.array(determinants), np.array(scaled)
    sign = np.sign(np.median(determinants))  # Orientation convention of the writer
    report["min_corner_jacobian_det_mm3"] = float((sign * determinants).min())
    report["min_scaled_jacobian"] = float((sign * scaled).min())
    report["inverted_hexahedra"] = int(np.sum((sign * determinants).min(axis=0) <= 0))
    worst = int(np.argmin((sign * scaled).min(axis=0)))
    centre = p[worst].mean(axis=0)
    report["worst_scaled_jacobian_cell"] = {"r_mm": float(np.hypot(*centre[:2])), "z_mm": float(centre[2]),
                                            "theta_deg": float(np.degrees(np.arctan2(centre[1], centre[0])))}
    report["hexahedra_scaled_jacobian_below_0.3"] = int(np.sum((sign * scaled).min(axis=0) < 0.3))
    # Exposed faces of the volume mesh versus the patch faces
    faces = np.sort(hexes[:, HEX_FACES].reshape(-1, 4), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    exposed = {tuple(f) for f in unique[counts == 1]}
    patch_faces, per_patch = set(), {}
    for name in report["patches_present"]:
        quads = np.sort(conn[name].reshape(-1, 4), axis=1)
        per_patch[name] = len(quads)
        patch_faces |= {tuple(f) for f in quads}
    report["patch_faces"] = per_patch
    report["exposed_faces"] = len(exposed)
    report["exposed_faces_without_patch"] = len(exposed - patch_faces)
    report["patch_faces_not_exposed"] = len(patch_faces - exposed)
    report["patch_faces_listed_twice"] = sum(per_patch.values()) - len(patch_faces)
    report["coverage"] = len(exposed & patch_faces) / len(exposed)
    report["all_required_patches"] = all(name in sections for name in REQUIRED)
    # Periodic pairing: rotate left nodes by one pitch about z, nearest right node
    left = np.unique(conn["periodic_left"])
    right = np.unique(conn["periodic_right"])
    tree = cKDTree(xyz[right])
    best = np.inf
    for angle in (2 * np.pi / num_blades, -2 * np.pi / num_blades):
        c, s = np.cos(angle), np.sin(angle)
        rotated = xyz[left] @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        error = tree.query(rotated)[0].max()
        best = min(best, error)
    report["periodic_nodes_left_right"] = [len(left), len(right)]
    report["max_periodic_error_mm"] = float(best)
    return report


if __name__ == "__main__":
    results = []
    for summary in sorted((HERE / "output").glob("*/summary.json")):
        data = json.loads(summary.read_text())
        jobs = summary.parent / "jobs" / data["jobs"]["volume"]["session"] / data["jobs"]["volume"]["job"]
        result = audit(next(jobs.glob("*.cgns")), 24)
        result["case"] = data["case"]
        result["mesher_quality"] = data["volume_result"]["summary"]["quality"]
        results.append(result)
        print(json.dumps(result, indent=1))
    (HERE / "output" / "cgns_audit.json").write_text(json.dumps(results, indent=1))
