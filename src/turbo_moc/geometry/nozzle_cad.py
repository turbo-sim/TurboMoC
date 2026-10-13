"""STEP export of a nozzle: fluid domain and printable nozzle body, in millimetres."""
import json
from pathlib import Path

import numpy as np

from turbo_moc.geometry import nozzle as geometry
from turbo_moc.geometry.cad_export import _check, _reimport, _render, volume

MM = 1e3


def _offset(x, y, thickness):
    tangent = np.gradient(np.column_stack((x, y)), axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal *= np.sign(normal[:, 1:2])
    return x + thickness * normal[:, 0], y + thickness * normal[:, 1]


def _profile(cq, plane, outer, inner):
    work = cq.Workplane(plane).moveTo(*outer[0]).spline([tuple(p) for p in outer[1:]], includeCurrent=True)
    work = work.lineTo(*inner[-1]).spline([tuple(p) for p in inner[-2::-1]], includeCurrent=True).close()
    return work


def export_nozzle(result, parameters, settings, output_dir, *, progress=None):
    import cadquery as cq

    def emit(message, fraction):
        if progress:
            progress(message, fraction)

    geometry.validate_settings(settings)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shape = geometry.contour(result, parameters, settings)
    x, y = shape["x"] * MM, shape["y"] * MM
    thickness = float(settings["wall_thickness"]) * MM
    depth = float(settings["planar_depth"]) * MM
    ox, oy = _offset(x, y, thickness)
    axis = np.column_stack((x, np.zeros_like(x)))
    wall, outer = np.column_stack((x, y)), np.column_stack((ox, oy))

    emit("Build the fluid domain and the nozzle body", 0.2)
    if shape["axisymmetric"]:
        fluid = _profile(cq, "XY", wall, axis).revolve(360, (0, 0, 0), (1, 0, 0)).val()
        body = _profile(cq, "XY", outer, wall).revolve(360, (0, 0, 0), (1, 0, 0)).val()
    else:
        fluid = _profile(cq, "XY", wall, wall * [1, -1]).extrude(depth).val().translate(cq.Vector(0, 0, -depth / 2))
        upper = _profile(cq, "XY", outer, wall).extrude(depth).val()
        lower = _profile(cq, "XY", wall * [1, -1], outer * [1, -1]).extrude(depth).val()
        height = float(oy.max())
        side = cq.Solid.makeBox(float(ox.max() - ox.min()), 2 * height, thickness,
                                cq.Vector(float(ox.min()), -height, depth))
        body = upper.fuse(lower, side, side.translate(cq.Vector(0, 0, -depth - thickness))).clean()
        body = body.translate(cq.Vector(0, 0, -depth / 2))
        body = body.Solids()[0] if len(body.Solids()) == 1 else body

    files = {"fluid_domain": "nozzle_fluid_domain.step", "nozzle_body": "nozzle_body.step"}
    emit("Write STEP files", 0.5)
    for name, solid in (("fluid_domain", fluid), ("nozzle_body", body)):
        cq.exporters.export(solid, str(directory / files[name]))
    emit("Validate solids and volumes", 0.7)
    if shape["axisymmetric"]:
        expected = np.pi * np.trapezoid(y ** 2, x)
    else:
        expected = 2 * np.trapezoid(y, x) * depth
    report = {
        "units": "mm", "frame": "x along the nozzle axis, throat at x = 0",
        "type": "axisymmetric" if shape["axisymmetric"] else "planar",
        "wall_thickness_mm": thickness, "files": files,
        "shapes": {"fluid_domain": _check(cq, fluid), "nozzle_body": _check(cq, body)},
        "volume_checks": {"fluid_over_contour_integral": volume(fluid) / expected},
        "reimport": {name: _reimport(cq, directory / path) for name, path in files.items()},
        "printability": [
            f"Nozzle wall thickness {thickness:.3g} mm, offset normal to the contour; no flanges or ports are modelled.",
            "Check overhangs and support strategy, and the slicer's own watertightness test, before printing.",
        ],
    }
    emit("Render previews", 0.85)
    tolerance = 2e-3 * float(y.max())
    for name, solid in (("fluid_domain", fluid), ("nozzle_body", body)):
        _render(solid, directory / f"nozzle_{name}.png", f"nozzle: {name.replace('_', ' ')}", tolerance)
    (directory / "cad_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit("CAD export ready", 1.0)
    return report
