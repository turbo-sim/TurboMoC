"""STEP export of a blade row: fluid domain, blade, and disk-and-blade assembly.

All shapes are built from the same design object as the app's 3D view and
meshes: section curves at span stations mapped with the row's meridional
mapping. Lengths are in millimetres, in the row-local frame, or in the machine
frame when an assembly placement is supplied.
"""
import json
from pathlib import Path

import numpy as np

from turbo_moc.geometry import passage_2d as planar
from turbo_moc.geometry.stator_blade import _cap_wire, _closed_wire_3d

CAD_DEFAULTS = {
    "span_sections": 9,
    "disk_thickness": 5.0,
    "axial_margin": 0.25,
    "root_overlap": 0.03,
    "linear_blades": 5,
    "fluid_annulus": False,
}


def validate_settings(settings):
    count = settings["span_sections"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("mesh.cad.span_sections must be an integer of at least 3.")
    count = settings["linear_blades"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("mesh.cad.linear_blades must be a positive integer.")
    for name in ("disk_thickness", "axial_margin", "root_overlap"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"mesh.cad.{name} must be a positive number.")
    if not isinstance(settings["fluid_annulus"], bool):
        raise ValueError("mesh.cad.fluid_annulus must be true or false.")


def _resample(points, count, closed=False):
    return planar.resample_curve(np.asarray(points, float), count + int(closed))[:count]


def _span_direction(mapping, q, eta):
    low, high = max(0.0, eta - 1e-3), min(1.0, eta + 1e-3)
    return (mapping.map_points(q, high) - mapping.map_points(q, low)) / (high - low)


def _mapped(mapping, q, eta, extension=0.0):
    xyz = mapping.map_points(q, eta)
    if extension:
        xyz = xyz + extension * _span_direction(mapping, q, eta)
    return xyz


def _segment_edge(cq, points):
    points = [cq.Vector(*p) for p in points]
    return cq.Edge.makeLine(points[0], points[-1]) if len(points) == 2 else cq.Edge.makeSpline(points)


def _sections(design, count, blade_points=180, wall_points=90):
    result = []
    for eta in np.linspace(0, 1, count):
        blade, curves = design.section(float(eta))
        result.append({"eta": float(eta),
                       "blade": _resample(planar.blade_contour(blade), blade_points, True),
                       "periodic_left": _resample(curves["periodic_left"], wall_points),
                       "periodic_right": _resample(curves["periodic_right"], wall_points)})
    return result


def _end_extension(eta, hub, tip):
    return -hub if eta == 0 else tip if eta == 1 else 0.0


def _closed_loft(cq, wires, length, ruled=False):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    loft = cq.Solid.makeLoft(wires, ruled)
    if loft.ShapeType() == "Solid" and _free_edges(loft) == 0 and loft.isValid():
        return _oriented(cq, loft)
    pieces = (loft, _cap_wire(cq, wires[0]), _cap_wire(cq, wires[-1]))
    for tolerance in (1e-6, 1e-5 * length, 1e-4 * length, 1e-3 * length):
        sewing = BRepBuilderAPI_Sewing(tolerance)
        for shape in pieces:
            sewing.Add(shape.wrapped)
        sewing.Perform()
        sewed = sewing.SewedShape()
        if sewed.ShapeType() == TopAbs_SHELL and sewing.NbFreeEdges() == 0:
            return _oriented(cq, cq.Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewed)).Solid()))
    raise ValueError("The lofted surfaces could not be closed into a solid.")


def blade_solid(cq, design, sections, hub_extension=0.0, tip_extension=0.0):
    mapping = design.meridional_channel
    wires = [_closed_wire_3d(cq, [tuple(p) for p in _mapped(
        mapping, row["blade"], row["eta"], _end_extension(row["eta"], hub_extension, tip_extension))])
        for row in sections]
    return _closed_loft(cq, wires, _length(wires))


def _length(wires):
    box = wires[0].BoundingBox()
    return max(box.xlen, box.ylen, box.zlen, 1e-9)


def _oriented(cq, solid):
    from OCP.BRepLib import BRepLib
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopoDS import TopoDS
    fixer = ShapeFix_Solid(TopoDS.Solid_s(solid.wrapped))
    fixer.Perform()
    fixed = TopoDS.Solid_s(fixer.Solid())
    BRepLib.OrientClosedSolid_s(fixed)
    return cq.Solid(fixed)


def passage_solid(cq, design, sections):
    mapping = design.meridional_channel
    wires = []
    for row in sections:
        left = _mapped(mapping, row["periodic_left"], row["eta"])
        right = _mapped(mapping, row["periodic_right"], row["eta"])
        edges = [_segment_edge(cq, left), _segment_edge(cq, [left[-1], right[-1]]),
                 _segment_edge(cq, right[::-1]), _segment_edge(cq, [right[0], left[0]])]
        wires.append(cq.Wire.assembleEdges(edges))
    return _closed_loft(cq, wires, _length(wires), ruled=True)


def disk_solid(cq, design, settings, sections, blades=()):
    mapping = design.meridional_channel
    thickness = float(settings["disk_thickness"])
    if mapping.is_linear:
        boxes = [blade.BoundingBox() for blade in blades]
        margin = settings["axial_margin"] * mapping.reference_chord
        low_x, high_x = min(b.xmin for b in boxes) - margin, max(b.xmax for b in boxes) + margin
        low_y, high_y = min(b.ymin for b in boxes) - margin, max(b.ymax for b in boxes) + margin
        return cq.Solid.makeBox(high_x - low_x, high_y - low_y, thickness, cq.Vector(low_x, low_y, -thickness))
    u = np.linspace(-settings["axial_margin"], 1 + settings["axial_margin"], 80)
    hub = mapping.meridional_points(u, 0.0)
    span = mapping.meridional_points(u, 1.0) - hub
    span /= np.linalg.norm(span, axis=1, keepdims=True)
    inner = hub - thickness * span
    if np.any(inner[:, 0] <= 0):
        raise ValueError("The disk thickness reaches the machine axis; reduce mesh.cad.disk_thickness.")
    profile = (cq.Workplane("XZ").moveTo(*hub[0]).spline([tuple(p) for p in hub[1:]], includeCurrent=True)
               .lineTo(*inner[-1]).spline([tuple(p) for p in inner[-2::-1]], includeCurrent=True).close())
    return profile.revolve(360, (0, 0, 0), (0, 1, 0)).val()


def _copies(cq, design, shape, settings):
    mapping = design.meridional_channel
    if mapping.is_linear:
        return [shape.translate(cq.Vector(0, k * mapping.pitch, 0)) for k in range(int(settings["linear_blades"]))]
    return [shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), np.degrees(k * mapping.pitch_angle))
            for k in range(mapping.num_blades)]


def _placed(cq, shape, transform):
    if transform is None:
        return shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    trsf = gp_Trsf()
    trsf.SetValues(*[float(value) for value in np.asarray(transform)[:3].ravel()])
    return type(shape)(BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape())


def _free_edges(shape):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, ancestors)
    count = 0
    for index in range(1, ancestors.Extent() + 1):
        edge = TopoDS.Edge_s(ancestors.FindKey(index))
        if not BRep_Tool.Degenerated_s(edge) and ancestors.FindFromIndex(index).Size() < 2:
            count += 1
    return count


def volume(shape, tolerance=1e-3):
    total = 0.0
    for solid in shape.Solids():
        vertices, triangles = solid.tessellate(tolerance, 0.1)
        points = np.array([v.toTuple() for v in vertices])
        a, b, c = (points[np.asarray(triangles)[:, i]] for i in range(3))
        total += abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)
    return total


def _check(cq, shape):
    box = shape.BoundingBox()
    free = _free_edges(shape)
    return {"solids": len(shape.Solids()), "valid": bool(shape.isValid()), "volume_mm3": volume(shape),
            "free_edges": free, "watertight": free == 0,
            "bounding_box_mm": [[box.xmin, box.ymin, box.zmin], [box.xmax, box.ymax, box.zmax]]}


def _reimport(cq, path):
    shape = cq.importers.importStep(str(path))
    solids = shape.solids().vals()
    return {"solids": len(solids), "valid": all(s.isValid() for s in solids),
            "volume_mm3": float(sum(volume(s) for s in solids))}


def _render(shape, path, title, tolerance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices, triangles = shape.tessellate(tolerance)
    points = np.array([v.toTuple() for v in vertices])
    figure = plt.figure(figsize=(6, 5))
    axes = figure.add_subplot(projection="3d")
    axes.add_collection3d(Poly3DCollection(points[np.asarray(triangles)], facecolor="#9fb7c2",
                                           edgecolor="#526d79", linewidths=0.05, alpha=0.9))
    low, high = points.min(axis=0), points.max(axis=0)
    centre, radius = 0.5 * (low + high), 0.5 * float((high - low).max())
    for setter, value in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), centre):
        setter(value - radius, value + radius)
    axes.set_box_aspect((1, 1, 1))
    axes.set_xlabel("x [mm]"), axes.set_ylabel("y [mm]"), axes.set_zlabel("z [mm]")
    axes.set_title(title, fontsize=10)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def export_row(design, settings, output_dir, *, row_id="row", role="stator", transform=None, progress=None):
    import cadquery as cq

    def emit(message, fraction):
        if progress:
            progress(message, fraction)

    validate_settings(settings)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    mapping = design.meridional_channel
    sections = _sections(design, int(settings["span_sections"]))
    height = float(np.linalg.norm(_span_direction(mapping, sections[0]["blade"][:1], 0.5)))
    overlap = float(settings["root_overlap"]) * height
    if overlap >= float(settings["disk_thickness"]):
        raise ValueError("The blade root overlap must be smaller than the disk thickness.")
    tolerance = 2e-3 * height

    emit("Loft the blade through the span sections", 0.1)
    blade = blade_solid(cq, design, sections)
    cutter = blade_solid(cq, design, sections, overlap, overlap)
    rooted = blade_solid(cq, design, sections, overlap, 0.0)
    emit("Loft the periodic passage and subtract the blade", 0.3)
    passage = passage_solid(cq, design, sections)
    fluid = passage.cut(cutter).Solids()[0]
    if settings["fluid_annulus"] and not mapping.is_linear:
        fluid_export = cq.Compound.makeCompound(_copies(cq, design, fluid, settings))
    else:
        fluid_export = fluid
    emit("Build the disk and pattern the blades", 0.5)
    blades = _copies(cq, design, rooted, settings)
    disk = disk_solid(cq, design, settings, sections, blades)
    fused = disk.fuse(*blades).clean()
    fused_solids = fused.Solids()
    fused = fused_solids[0] if len(fused_solids) == 1 else fused

    files = {"fluid_domain": f"{row_id}_fluid_domain.step", "blade": f"{row_id}_blade.step",
             "assembly_parts": f"{row_id}_disk_blades_parts.step", "assembly_fused": f"{row_id}_disk_blades_fused.step"}
    emit("Write STEP files", 0.65)
    placed = {name: _placed(cq, shape, transform) for name, shape in
              (("fluid_domain", fluid_export), ("blade", blade), ("assembly_fused", fused))}
    for name, shape in placed.items():
        cq.exporters.export(shape, str(directory / files[name]))
    assembly = cq.Assembly(name=f"{row_id}_disk_blades")
    assembly.add(_placed(cq, disk, transform), name="disk", color=cq.Color(0.65, 0.7, 0.72))
    for index, shape in enumerate(blades, 1):
        assembly.add(_placed(cq, shape, transform), name=f"blade_{index}", color=cq.Color(0.28, 0.49, 0.7))
    assembly.save(str(directory / files["assembly_parts"]), exportType="STEP")

    emit("Validate solids, volumes and watertightness", 0.8)
    one_overlap = volume(rooted.intersect(disk))
    expected_fused = volume(disk) + len(blades) * (volume(rooted) - one_overlap)
    passage_volume, fluid_volume, blade_volume = volume(passage), volume(fluid), volume(blade)
    report = {
        "units": "mm", "frame": "machine" if transform is not None else "row-local",
        "axis": "linear cascade: x streamwise, y pitchwise, z span" if mapping.is_linear else "z (machine axis)",
        "span_sections": len(sections), "blade_count": len(blades), "root_overlap_mm": overlap,
        "disk_thickness_mm": float(settings["disk_thickness"]), "files": files,
        "shapes": {"fluid_domain": _check(cq, _placed(cq, fluid, transform)),
                   "blade": _check(cq, placed["blade"]), "disk": _check(cq, _placed(cq, disk, transform)),
                   "assembly_fused": _check(cq, placed["assembly_fused"])},
        "volume_checks": {
            "passage_mm3": passage_volume,
            "fluid_plus_blade_over_passage": (fluid_volume + blade_volume) / passage_volume,
            "fused_over_expected": volume(fused) / expected_fused,
        },
        "reimport": {name: _reimport(cq, directory / path) for name, path in files.items()},
        "printability": [
            "The fused disk-and-blade solid is a single closed body; blade roots overlap the disk by "
            f"{overlap:.3g} mm.",
            "No root fillet is modelled; add one in CAD if required by the printing process.",
            f"Thinnest feature: trailing edge, about {2 * float(design.blade.get('r_trailing', np.nan)):.3g} mm thick "
            "at the reference section; check it against the printer's minimum wall thickness.",
            "Check overhangs, support strategy and the slicer's own watertightness test before printing.",
        ],
    }
    emit("Render previews", 0.9)
    for name, shape in (("fluid_domain", fluid), ("blade", blade), ("assembly_fused", fused)):
        _render(shape, directory / f"{row_id}_{name}.png", f"{row_id}: {name.replace('_', ' ')}",
                5 * tolerance if name == "assembly_fused" else tolerance)
    (directory / "cad_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit("CAD export ready", 1.0)
    return report
