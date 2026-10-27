"""ANSYS BladeGen export (.inf + profile/hub/shroud .curve) for TurboGrid meshing.

Format as the legacy exporter (matched against turbodash): profile points are
Cartesian (X axial, Y, Z) with the leading and trailing edges marked, the
endwalls are (axial, radius, 0) meridional curves, lengths in millimetres and
X the axis of rotation. The geometry comes from the app's design object: the
same mapped span sections as the meshes and STEP solids, with dense hub and
shroud curves so flared, radial and curved endwalls are represented.
"""
import json
from pathlib import Path

import numpy as np

from turbo_moc.geometry import passage_2d as planar

MIN_SEGMENT = 1e-6


def _machine(points, transform):
    if transform is not None:
        points = points @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
    return np.column_stack((points[:, 2], points[:, 0], points[:, 1]))


def _section(design, eta, count, transform):
    blade, _ = design.section(float(eta))
    q = planar.resample_curve(planar.blade_contour(blade), count + 1)[:count]
    leading, trailing = int(np.argmax(q[:, 1])), int(np.argmin(q[:, 1]))
    q = np.roll(q, -leading, axis=0)
    trailing = (trailing - leading) % len(q)
    xyz = _machine(design.meridional_channel.map_points(q, float(eta)), transform)
    keep = [0] + [i for i in range(1, len(xyz)) if np.linalg.norm(xyz[i] - xyz[i - 1]) > MIN_SEGMENT]
    trailing = int(np.argmin(np.abs(np.asarray(keep) - trailing)))
    xyz = xyz[keep]
    if np.linalg.norm(xyz[0] - xyz[-1]) <= MIN_SEGMENT:
        xyz = xyz[:-1]
    return xyz, trailing


def _endwall(design, eta, transform, samples=80):
    mapping = design.meridional_channel
    _, curves = design.section(float(eta))
    stations = []
    for name in ("inlet", "outlet"):
        q = np.asarray(curves[name])
        prime = -q[:, 1] / mapping.reference_radius
        stations.append(np.asarray(mapping.parameter_from_prime(prime, float(eta))).mean())
    u = np.linspace(min(stations), max(stations), samples)
    radius, axial = mapping.meridional_points(u, float(eta)).T
    if transform is not None:
        axial = axial + float(np.asarray(transform)[2, 3])
    return np.column_stack((axial, radius, np.zeros_like(axial)))


def export_bladegen(design, output_dir, *, row_id, role="stator", span_sections=11, inset=0.01,
                    profile_points=200, transform=None):
    mapping = design.meridional_channel
    if mapping.is_linear:
        raise ValueError("BladeGen export requires an annular (axial or radial) row.")
    if span_sections < 2 or not 0 <= inset < 0.5:
        raise ValueError("BladeGen export requires at least two span sections and an inset in [0, 0.5).")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    names = {"inf": f"{row_id}_BladeGen.inf", "profile": f"{row_id}_profile.curve",
             "hub": f"{row_id}_hub.curve", "shroud": f"{row_id}_shroud.curve"}
    label = "beta" if role == "rotor" else "alpha"
    angle_in = float(design.blade.get("metal_angle_in", np.nan))
    angle_out = float(design.blade.get("metal_angle_out", np.nan))
    lines = []
    for index, eta in enumerate(np.linspace(inset, 1 - inset, span_sections)):
        xyz, trailing = _section(design, eta, profile_points, transform)
        lines.append(f"# Span {index:03d} s={eta:.8f} {label}_in={angle_in:.6f} {label}_out={angle_out:.6f}")
        for point, (x, y, z) in enumerate(xyz):
            marker = " le" if point == 0 else " te" if point == trailing else ""
            lines.append(f"{x:.12e} {y:.12e} {z:.12e}{marker}")
    (directory / names["profile"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    hub, shroud = _endwall(design, 0.0, transform), _endwall(design, 1.0, transform)
    np.savetxt(directory / names["hub"], hub, fmt="%.12e")
    np.savetxt(directory / names["shroud"], shroud, fmt="%.12e")
    (directory / names["inf"]).write_text("\n".join([
        "!======  CFX-BladeGen Export  ========",
        "Axis of Rotation: X",
        f"Number of Blade Sets: {mapping.num_blades}",
        "Number of Blades Per Set: 1",
        "Geometry Units: MM",
        "Blade 0 LE: EllipseEnd",
        "Blade 0 TE: EllipseEnd",  # Stator and rotor trailing edges are rounded (arc or ellipse), not cut off
        f"Hub Data File: {names['hub']}",
        f"Shroud Data File: {names['shroud']}",
        f"Profile Data File: {names['profile']}",
        "",
    ]), encoding="utf-8")
    report = {"units": "mm", "frame": "machine" if transform is not None else "row-local",
              "axis": "X (machine axis)", "files": names, "checks": check_bladegen(directory, names)}
    (directory / "bladegen_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_profile(path):
    sections, current = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            current = {"header": line, "points": [], "le": [], "te": []}
            sections.append(current)
        elif line.strip():
            parts = line.split()
            if len(parts) > 3:
                current[parts[3]].append(len(current["points"]))
            current["points"].append([float(value) for value in parts[:3]])
    for section in sections:
        section["points"] = np.asarray(section["points"])
    return sections


def check_bladegen(directory, names):
    from matplotlib.path import Path as Outline
    directory = Path(directory)
    sections = read_profile(directory / names["profile"])
    hub, shroud = np.loadtxt(directory / names["hub"]), np.loadtxt(directory / names["shroud"])
    channel = Outline(np.vstack((hub[:, :2], shroud[::-1, :2])))
    meridional = np.vstack([np.column_stack((s["points"][:, 0], np.hypot(s["points"][:, 1], s["points"][:, 2])))
                            for s in sections])
    segments = [float(np.linalg.norm(np.diff(np.vstack((s["points"], s["points"][:1])), axis=0), axis=1).min())
                for s in sections]
    return {
        "span_sections": len(sections),
        "points_per_section": sorted({len(section["points"]) for section in sections}),
        "one_le_and_te_per_section": all(len(s["le"]) == 1 and len(s["te"]) == 1 for s in sections),
        "minimum_segment_mm": min(segments),
        "profiles_inside_meridional_channel": bool(channel.contains_points(meridional).all()),
        "hub_points": len(hub), "shroud_points": len(shroud),
    }
