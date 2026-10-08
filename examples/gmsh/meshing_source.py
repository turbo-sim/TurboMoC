"""Meshing source: Gmsh CAD, inflation, periodic mesh, export, and diagnostics.

Sections: settings; CAD; sizing; conformity; CGNS export; visualization;
quality metrics; report writing; generation workflow. All lengths are mm.

Transfer the existing fitted B-spline exactly to OpenCASCADE, close the trailing
edge with circular arcs, and mesh the surrounding fluid with inflation layers.
"""

import csv
from html import escape
from io import StringIO
import json
from pathlib import Path

import gmsh
import numpy as np
from scipy.spatial import cKDTree

from turbo_moc.io.cgns_adf import AdfNode, new_root, write_adf
from examples.workflow_logging import display_path, get_logger, summary

logger = get_logger("gmsh.meshing")

# =============================================================================
# 1. Meshing defaults and input validation
# =============================================================================


# Internal target for area growth through the approximately isotropic core.
# This is a sizing target, not a bound on the generated cells' area ratios.
CORE_AREA_GROWTH = 1.2
MESH_ORANGE = (255, 140, 0)


def validate_mesh_settings(settings):
    """Check mesh sizes before entering Gmsh."""
    for name in ("blade_size", "domain_size"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"mesh.{name} must be finite and positive.")
    if settings["domain_size"] < settings["blade_size"]:
        raise ValueError("mesh.domain_size must be at least mesh.blade_size.")
    if settings["algorithm"] not in (1, 2, 5, 6, 8, 9):
        raise ValueError("Unsupported 2D mesh algorithm.")
    verbosity = settings.get("verbosity", 2)
    if type(verbosity) is not int or not 0 <= verbosity <= 99:
        raise ValueError("mesh.verbosity must be an integer from 0 to 99.")
    if not isinstance(settings.get("recombine", True), bool):
        raise ValueError("mesh.recombine must be true or false.")
    recombination_algorithm = settings.get("recombination_algorithm", 1)
    if type(recombination_algorithm) is not int or recombination_algorithm not in (
        0,
        1,
        2,
        3,
    ):
        raise ValueError("mesh.recombination_algorithm must be 0, 1, 2, or 3.")
    for name in ("smoothing_steps", "curvature_elements"):
        if not isinstance(settings[name], int) or settings[name] < 0:
            raise ValueError(f"mesh.{name} must be a nonnegative integer.")
    image = settings.get("image", {})
    image_name = image.get("filename", "stator_mesh.png")
    if Path(image_name).suffix.lower() != ".png" or Path(image_name).name != image_name:
        raise ValueError(
            "mesh.image.filename must be a PNG filename within the output directory."
        )
    dpi = image.get("dpi", 200)
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("mesh.image.dpi must be a positive integer.")
    cgns = settings.get("cgns", {})
    if cgns.get("enabled", False):
        if Path(cgns["filename"]).suffix.lower() != ".cgns":
            raise ValueError("mesh.cgns.filename must end in .cgns.")
        if Path(cgns["filename"]).name != cgns["filename"]:
            raise ValueError("mesh.cgns.filename must be within the output directory.")
        scale = cgns["length_scale"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("mesh.cgns.length_scale must be finite and positive.")
        supported = {
            "BCWallViscous",
            "BCWall",
            "BCInflowSubsonic",
            "BCInflowSupersonic",
            "BCOutflowSubsonic",
            "BCOutflowSupersonic",
            "BCGeneral",
            "BCSymmetryPlane",
        }
        names = {"blade", "inlet", "outlet", "periodic_left", "periodic_right"}
        if set(cgns["boundary_types"]) != names:
            raise ValueError("CGNS boundary_types must specify all five boundaries.")
        if not set(cgns["boundary_types"].values()) <= supported:
            raise ValueError("Unsupported CGNS boundary type.")
    inflation = settings["inflation"]
    if inflation["enabled"]:
        for name in ("first_height", "thickness", "growth_ratio"):
            if not np.isfinite(inflation[name]) or inflation[name] <= 0:
                raise ValueError(f"mesh.inflation.{name} must be finite and positive.")
        if inflation["growth_ratio"] <= 1:
            raise ValueError("mesh.inflation.growth_ratio must exceed 1.")
        if inflation["thickness"] < inflation["first_height"]:
            raise ValueError(
                "Inflation thickness must be at least the first-layer height."
            )
    reporting = settings.get("report", {})
    if reporting.get("enabled", True):
        for key, default in (("histogram_bins", 30), ("worst_cells", 10)):
            value = reporting.get(key, default)
            if type(value) is not int or value < (2 if key == "histogram_bins" else 1):
                raise ValueError(
                    f"mesh.report.{key} must be a positive integer (histogram_bins >= 2)."
                )
        tolerance = reporting.get("inflation_tolerance", 0.35)
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("mesh.report.inflation_tolerance must be in (0, 1).")


# =============================================================================
# 2. Transfer fitted blade and periodic passage to Gmsh CAD
# Preserve spline knots and control points rather than refitting samples.
# =============================================================================


def add_blade_boundary(blade, mesh_size):
    """Reuse the fitted spline and close its ends with two quarter-circle arcs."""
    control_points = np.asarray(blade["control_points"], dtype=float)
    point_tags = [
        gmsh.model.occ.addPoint(x, y, 0, mesh_size) for x, y in control_points
    ]
    # The package stores expanded knots; Gmsh wants distinct knots and counts.
    knots, multiplicities = np.unique(blade["knots"], return_counts=True)
    spline = gmsh.model.occ.addBSpline(
        point_tags,
        degree=3,
        knots=knots.tolist(),
        multiplicities=multiplicities.tolist(),
    )
    trailing_edge = np.column_stack(
        (blade["trailing_edge"]["x"], blade["trailing_edge"]["y"])
    )
    if not (
        np.allclose(control_points[-1], trailing_edge[0], atol=1e-7, rtol=0)
        and np.allclose(control_points[0], trailing_edge[-1], atol=1e-7, rtol=0)
    ):
        raise ValueError(
            "The fitted blade endpoints do not meet the trailing-edge arc."
        )
    centre = (control_points[0] + control_points[-1]) / 2
    # The full stator's sampler includes the semicircle midpoint.
    midpoint = trailing_edge[len(trailing_edge) // 2]
    centre_tag = gmsh.model.occ.addPoint(*centre, 0)
    midpoint_tag = gmsh.model.occ.addPoint(*midpoint, 0, mesh_size)
    arc_upper = gmsh.model.occ.addCircleArc(point_tags[-1], centre_tag, midpoint_tag)
    arc_lower = gmsh.model.occ.addCircleArc(midpoint_tag, centre_tag, point_tags[0])
    return [spline, arc_upper, arc_lower]


def add_periodic_passage(curves, blade_curves, pitch, mesh_size):
    """Transfer the selected smooth domain and constrain translated 1D meshes."""
    left = curves["periodic_left"]
    right = curves["periodic_right"]
    translation = np.tile([pitch, 0], (len(left), 1))
    if left.shape != right.shape or not np.allclose(right - left, translation):
        raise ValueError("Periodic boundaries must be exact pitch-translated copies.")
    cad_reference = np.vstack(
        (
            curves["inlet_spline"](np.linspace(0, 1, len(curves["inlet_extension"]))),
            curves["convergent_spline"](np.linspace(0, 1, len(curves["convergent"])))[
                1:
            ],
            curves["divergent"][1:],
            curves["outlet_spline"](np.linspace(0, 1, len(curves["outlet_extension"])))[
                1:
            ],
        )
    )
    if not np.allclose(cad_reference - [pitch / 2, 0.0], left, rtol=0, atol=1e-8):
        raise ValueError(
            "Periodic CAD splines must match the selected plotted flow domain."
        )

    def add_side(offset):
        # Preserve the analytic cubic extensions and the convergent cubic
        # interpolant. Shared endpoint tags join the four CAD sections.
        point_tags, side_curves = [], []
        previous = None
        for name in ("inlet_spline", "convergent_spline", "divergent", "outlet_spline"):
            is_line = name == "divergent"
            points = (curves[name][[0, -1]] if is_line else curves[name].c) + [
                offset,
                0.0,
            ]
            if previous is not None and not np.allclose(
                previous, points[0], atol=1e-9, rtol=0
            ):
                raise ValueError("Flow-domain CAD sections do not share endpoints.")
            tags = [
                (
                    point_tags[-1]
                    if point_tags
                    else gmsh.model.occ.addPoint(*points[0], 0, mesh_size)
                )
            ]
            tags.extend(
                gmsh.model.occ.addPoint(*point, 0, mesh_size) for point in points[1:]
            )
            if is_line:
                curve = gmsh.model.occ.addLine(tags[0], tags[-1])
            else:
                spline = curves[name]
                knots, multiplicities = np.unique(spline.t, return_counts=True)
                curve = gmsh.model.occ.addBSpline(
                    tags,
                    degree=spline.k,
                    knots=knots.tolist(),
                    multiplicities=multiplicities.tolist(),
                )
            if not point_tags:
                point_tags.append(tags[0])
            point_tags.append(tags[-1])
            previous = points[-1]
            side_curves.append(curve)
        return point_tags, side_curves

    left_points, left_lines = add_side(-pitch / 2)
    right_points, right_lines = add_side(pitch / 2)
    inlet = gmsh.model.occ.addLine(left_points[0], right_points[0])
    outlet = gmsh.model.occ.addLine(right_points[-1], left_points[-1])
    outer_loop = gmsh.model.occ.addCurveLoop(
        [inlet, *right_lines, outlet, *[-tag for tag in reversed(left_lines)]]
    )
    blade_loop = gmsh.model.occ.addCurveLoop(blade_curves)
    fluid_surface = gmsh.model.occ.addPlaneSurface([outer_loop, blade_loop])
    gmsh.model.occ.synchronize()

    # x is pitchwise. Right is the slave of left, translated by one pitch.
    transform = [1, 0, 0, pitch, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, right_lines, left_lines, transform)
    boundaries = {
        "blade": blade_curves,
        "periodic_left": left_lines,
        "periodic_right": right_lines,
        "inlet": [inlet],
        "outlet": [outlet],
    }
    for name, tags in boundaries.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, name)
    group = gmsh.model.addPhysicalGroup(2, [fluid_surface])
    gmsh.model.setPhysicalName(2, group, "fluid")
    return fluid_surface, boundaries


# =============================================================================
# 3. Inflation profile, core sizing, and quadrilateral recombination
# Match the core transition to the actual tangential spacing on the blade.
# =============================================================================


def calculate_inflation_profile(inflation):
    """Resolve the complete geometric layers that fit the thickness limit.

    These are nominal values: Gmsh may shorten or merge columns near curvature
    and neighbouring boundaries. The three user inputs are never adjusted.
    """
    if not inflation["enabled"]:
        return {
            "enabled": False,
            "layers": 0,
            "heights": [],
            "stack_thickness": 0.0,
            "last_height": 0.0,
        }
    first, ratio, limit = (
        inflation[name] for name in ("first_height", "growth_ratio", "thickness")
    )
    if (
        not np.isfinite([first, ratio, limit]).all()
        or first <= 0
        or ratio <= 1
        or limit < first
    ):
        raise ValueError(
            "Inflation requires first_height > 0, growth_ratio > 1, and thickness >= first_height."
        )
    count = int(np.floor(np.log1p(limit * (ratio - 1) / first) / np.log(ratio) + 1e-10))
    heights = first * ratio ** np.arange(count)
    return {
        "enabled": True,
        "layers": count,
        "heights": heights.tolist(),
        "stack_thickness": float(heights.sum()),
        "last_height": float(heights[-1]),
    }


def core_mesh_size(distance, tangent_size, profile, domain_size, recombine=True):
    """Match the nominal last-layer area, then grade towards the far field.

    Last-layer area is estimated as tangential spacing times normal height.
    Convert it to a square/equilateral-triangle size; this keeps the two wall
    directions independent. A small linear size gradient gives approximately
    CORE_AREA_GROWTH between successive regular core cells.
    """
    if profile["enabled"]:
        shape_factor = 1.0 if recombine else np.sqrt(3) / 4
        start_size = np.sqrt(
            tangent_size * profile["last_height"] * CORE_AREA_GROWTH / shape_factor
        )
        # The first core cells still share long edges with inflation quads.
        # Limit their normal step as well as their equivalent area size.
        start_size = np.minimum(start_size, profile["last_height"] * CORE_AREA_GROWTH)
    else:
        start_size = tangent_size
    gradient = np.sqrt(CORE_AREA_GROWTH) - 1
    outside = np.maximum(0.0, np.asarray(distance) - profile["stack_thickness"])
    return np.minimum(domain_size, start_size + gradient * outside)


def configure_core_transition(
    blade_curves, settings, profile, pitch=None, use_mesh=False
):
    """Install local, periodic sizing; refresh from actual wall edges after 1D.

    Before the 1D mesh exists, CAD samples and curvature estimate tangential
    spacing. Afterwards, actual blade-edge lengths replace that estimate.
    The blade's 1D sizing remains separate from the core's area-based sizing.
    """
    starts, ends, tangents = [], [], []
    if use_mesh:
        tags, values, _ = gmsh.model.mesh.getNodes()
        xy = dict(zip(tags, np.asarray(values).reshape(-1, 3)[:, :2]))
    for curve in blade_curves:
        if use_mesh:
            kinds, _, nodes = gmsh.model.mesh.getElements(1, curve)
            if any(kind != 1 for kind in kinds):
                raise ValueError("Core sizing requires first-order blade edges.")
            edges = np.concatenate(nodes).reshape(-1, 2)
            points = np.array([[xy[a], xy[b]] for a, b in edges])
            start, end = points[:, 0], points[:, 1]
            tangent = np.linalg.norm(end - start, axis=1)
        else:
            lo, hi = gmsh.model.getParametrizationBounds(1, curve)
            parameters = np.linspace(lo[0], hi[0], 1001)
            points = np.asarray(gmsh.model.getValue(1, curve, parameters)).reshape(
                -1, 3
            )[:, :2]
            start, end = points[:-1], points[1:]
            tangent = np.full(len(start), settings["blade_size"])
            if settings["curvature_elements"]:
                curvature = np.abs(
                    gmsh.model.getCurvature(
                        1, curve, (parameters[:-1] + parameters[1:]) / 2
                    )
                )
                tangent = np.minimum(
                    tangent,
                    2
                    * np.pi
                    / (settings["curvature_elements"] * np.maximum(curvature, 1e-12)),
                )
        starts.append(start)
        ends.append(end)
        tangents.append(tangent)
    starts, ends, tangents = (
        np.vstack(starts),
        np.vstack(ends),
        np.concatenate(tangents),
    )
    centre_x = float(np.mean((starts[:, 0] + ends[:, 0]) / 2))
    if pitch is not None:
        # Wrap query x into one pitch, and include both neighbouring blades.
        # The field is then identical at translated periodic points.
        shifts = np.array([[-pitch, 0.0], [0.0, 0.0], [pitch, 0.0]])
        starts = np.vstack([starts + shift for shift in shifts])
        ends = np.vstack([ends + shift for shift in shifts])
        tangents = np.tile(tangents, 3)
    vectors = ends - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    if np.any(squared_lengths <= 0):
        raise ValueError("Zero-length blade edge in core sizing.")
    tree = cKDTree((starts + ends) / 2)
    blade_tags = set(blade_curves)

    def size_callback(dim, tag, x, y, z, current_size):
        if dim == 0:
            return current_size
        if dim == 1 and tag in blade_tags:
            return min(current_size, settings["blade_size"])
        if pitch is not None:
            x = centre_x + (x - centre_x + pitch / 2) % pitch - pitch / 2
        point = np.array([x, y])
        _, candidates = tree.query(point, k=min(8, len(starts)))
        candidates = np.atleast_1d(candidates)
        delta = point - starts[candidates]
        fractions = np.clip(
            np.einsum("ij,ij->i", delta, vectors[candidates])
            / squared_lengths[candidates],
            0.0,
            1.0,
        )
        distances = np.linalg.norm(
            delta - fractions[:, None] * vectors[candidates], axis=1
        )
        nearest = np.argmin(distances)
        size = core_mesh_size(
            distances[nearest],
            tangents[candidates[nearest]],
            profile,
            settings["domain_size"],
            settings.get("recombine", True),
        )
        # Keep Gmsh's curvature and point-size constraints when they are finer.
        return float(min(current_size, size))

    gmsh.model.mesh.setSizeCallback(size_callback)
    start_sizes = core_mesh_size(
        0.0, tangents, profile, settings["domain_size"], settings.get("recombine", True)
    )
    distances = profile["stack_thickness"] + (settings["domain_size"] - start_sizes) / (
        np.sqrt(CORE_AREA_GROWTH) - 1
    )
    return {
        "area_growth_target": CORE_AREA_GROWTH,
        "tangential_spacing_source": (
            "measured blade edges" if use_mesh else "CAD curvature estimate"
        ),
        "tangential_spacing_min_mm": float(tangents.min()),
        "tangential_spacing_max_mm": float(tangents.max()),
        "core_start_size_min_mm": float(start_sizes.min()),
        "core_start_size_max_mm": float(start_sizes.max()),
        "far_field_distance_min_mm": float(distances.min()),
        "far_field_distance_max_mm": float(distances.max()),
    }


def set_mesh_sizes(blade_curves, settings, profile=None, pitch=None):
    """Set blade resolution and automatically grade the surrounding core."""
    if profile is None:
        profile = calculate_inflation_profile(settings["inflation"])
    gmsh.option.setNumber("Mesh.Algorithm", settings["algorithm"])
    gmsh.option.setNumber("Mesh.Smoothing", settings["smoothing_steps"])
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", settings["curvature_elements"])
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings["domain_size"])
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    background = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(background, "F", str(settings["domain_size"]))
    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    return configure_core_transition(blade_curves, settings, profile, pitch)


def add_inflation_layers(blade_curves, settings):
    """Create quadrilateral layers normal to every blade-boundary curve."""
    inflation = settings["inflation"]
    if not inflation["enabled"]:
        return
    field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(field, "CurvesList", blade_curves)
    gmsh.model.mesh.field.setNumber(field, "Size", inflation["first_height"])
    gmsh.model.mesh.field.setNumber(field, "SizeFar", settings["domain_size"])
    gmsh.model.mesh.field.setNumber(field, "Ratio", inflation["growth_ratio"])
    gmsh.model.mesh.field.setNumber(field, "Thickness", inflation["thickness"])
    gmsh.model.mesh.field.setNumber(field, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(field)


def configure_recombination(fluid_surface, settings):
    """Recombine the fluid surface, as in ParaBlade's 2D mesh builder.

    Algorithm 8 prepares triangles for recombination; setRecombine actually
    enables their conversion to quadrilaterals outside the inflation layers.
    """
    gmsh.option.setNumber(
        "Mesh.RecombinationAlgorithm", settings.get("recombination_algorithm", 1)
    )
    if settings.get("recombine", True):
        gmsh.model.mesh.setRecombine(2, fluid_surface)


# =============================================================================
# 4. Mesh conformity and summary checks
# Validate periodic correspondence before exporting the generated mesh.
# =============================================================================


def check_periodic_conformity(boundaries):
    """Verify complete node bijections, translations, and matching 1D edges."""
    if not boundaries["periodic_left"] or len(boundaries["periodic_left"]) != len(
        boundaries["periodic_right"]
    ):
        raise RuntimeError(
            "Periodic sides must contain matching nonempty lists of CAD curves."
        )
    pairs = {}
    xyz_tags, xyz_values, _ = gmsh.model.mesh.getNodes()
    xyz = dict(zip(xyz_tags, np.asarray(xyz_values).reshape(-1, 3)))
    for master, slave in zip(boundaries["periodic_left"], boundaries["periodic_right"]):
        found_master, slave_nodes, master_nodes, affine = (
            gmsh.model.mesh.getPeriodicNodes(1, slave)
        )
        if (
            found_master != master
            or not len(slave_nodes)
            or len(slave_nodes) != len(master_nodes)
        ):
            raise RuntimeError(
                "Gmsh did not generate the requested periodic node mapping."
            )
        if len(set(slave_nodes)) != len(slave_nodes) or len(set(master_nodes)) != len(
            master_nodes
        ):
            raise RuntimeError("Periodic node correspondence is not bijective.")
        for curve, nodes in ((slave, slave_nodes), (master, master_nodes)):
            actual, _, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            if set(actual) != set(nodes):
                raise RuntimeError(
                    "Periodic mapping does not cover all boundary nodes."
                )
        mapped = dict(zip(slave_nodes, master_nodes))
        for a, b in mapped.items():
            if a in pairs and pairs[a] != b:
                raise RuntimeError(
                    "Conflicting periodic mapping at a shared curve endpoint."
                )
            pairs[a] = b
        matrix = np.asarray(affine).reshape(4, 4)
        source = np.array([xyz[tag] for tag in master_nodes])
        target = np.array([xyz[tag] for tag in slave_nodes])
        expected = source @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.allclose(target, expected, atol=1e-8, rtol=0):
            raise RuntimeError(
                "Periodic node coordinates do not satisfy the prescribed translation."
            )

        def edge_set(curve, mapping=None):
            kinds, _, arrays = gmsh.model.mesh.getElements(1, curve)
            if any(kind != 1 for kind in kinds):
                raise RuntimeError(
                    "Periodic conformity checks require first-order line elements."
                )
            return {
                tuple(
                    sorted(mapping[tag] if mapping is not None else tag for tag in edge)
                )
                for array in arrays
                for edge in array.reshape(-1, 2)
            }

        if edge_set(slave, mapped) != edge_set(master):
            raise RuntimeError(
                "Periodic boundaries have different 1D element connectivity."
            )
    if len(set(pairs.values())) != len(pairs):
        raise RuntimeError("Combined periodic mapping is not bijective.")
    return len(pairs)


def mesh_statistics(fluid_surface, boundaries):
    """Check cell quality and periodic node correspondence before saving."""
    element_types, element_tags, _ = gmsh.model.mesh.getElements(2, fluid_surface)
    counts = {
        gmsh.model.mesh.getElementProperties(kind)[0]: len(tags)
        for kind, tags in zip(element_types, element_tags)
    }
    tags = np.concatenate(element_tags) if element_tags else np.array([], dtype=int)
    if not len(tags):
        raise RuntimeError("Gmsh generated no fluid cells.")
    quality = gmsh.model.mesh.getElementQualities(tags, "minSJ")
    if not np.isfinite(quality).all() or quality.min() <= 0:
        raise RuntimeError("The mesh contains invalid or inverted fluid cells.")
    periodic_nodes = check_periodic_conformity(boundaries)
    quadrilaterals = sum(
        count for name, count in counts.items() if name.startswith("Quadrilateral")
    )
    return {
        "cells": counts,
        "quadrilateral_fraction": quadrilaterals / len(tags),
        "minimum_scaled_jacobian": float(quality.min()),
        "periodic_node_pairs": periodic_nodes,
    }


# =============================================================================
# 5. Fluent CGNS export
# Collect first-order cells and named boundaries with periodic connectivity.
# =============================================================================


def collect_fluid_mesh(fluid_surface, boundaries):
    """Renumber used nodes and orient cells/edges with fluid on their left.

    CAD control points and other unused Gmsh nodes are excluded. Each exterior
    cell edge must belong to exactly one named boundary. This follows the
    section extraction used by ParaBlade's annular_mesh.py.
    """
    kinds, _, connectivity = gmsh.model.mesh.getElements(2, fluid_surface)
    if not len(kinds) or any(kind not in (2, 3) for kind in kinds):
        raise ValueError("CGNS export requires first-order triangles/quadrilaterals.")
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    coordinates = dict(zip(node_tags, np.asarray(xyz).reshape(-1, 3)))
    used = np.unique(np.concatenate(connectivity))
    local = {tag: index + 1 for index, tag in enumerate(used)}
    xy = np.array([coordinates[tag][:2] for tag in used])
    cells = []
    edge_owners = {}
    for kind, nodes in zip(kinds, connectivity):
        for tags in nodes.reshape(-1, 3 if kind == 2 else 4):
            cell = np.array([local[tag] for tag in tags], dtype=np.int32)
            points = xy[cell - 1]
            area = (
                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
            ) / 2
            if not np.isfinite(area) or area == 0:
                raise ValueError("Degenerate fluid cell during CGNS export.")
            if area < 0:
                cell = cell[::-1]
            cells.append(cell)
            for position, (a, b) in enumerate(zip(cell, np.roll(cell, -1)), 1):
                edge_owners.setdefault(tuple(sorted((a, b))), []).append(
                    (int(a), int(b), len(cells), position)
                )
    for owners in edge_owners.values():
        if len(owners) > 2 or (
            len(owners) == 2 and owners[0][:2] != owners[1][:2][::-1]
        ):
            raise ValueError("Nonmanifold or inconsistently oriented fluid mesh.")
    exterior = {
        key: owners[0] for key, owners in edge_owners.items() if len(owners) == 1
    }
    named_edges = {}
    assigned = set()
    for name, curves in boundaries.items():
        edges = []
        for curve in curves:
            types, _, nodes = gmsh.model.mesh.getElements(1, curve)
            if any(kind != 1 for kind in types):
                raise ValueError("CGNS boundaries require first-order line elements.")
            for array in nodes:
                for a, b in array.reshape(-1, 2):
                    key = tuple(sorted((local[a], local[b])))
                    if key not in exterior or key in assigned:
                        raise ValueError(f"Invalid or overlapping boundary: {name}.")
                    assigned.add(key)
                    edges.append(exterior[key])
        if not edges:
            raise ValueError(f"Empty boundary: {name}.")
        named_edges[name] = np.array(edges, dtype=np.int32)
    if assigned != set(exterior):
        raise ValueError("Physical groups do not cover the entire fluid boundary.")
    return xy, cells, named_edges, local


def collect_periodic_nodes(boundaries, local, xy):
    """Merge per-curve maps into one checked right-to-left node bijection."""
    pairs = {}
    for left, right in zip(boundaries["periodic_left"], boundaries["periodic_right"]):
        master, slaves, masters, _ = gmsh.model.mesh.getPeriodicNodes(1, right)
        if master != left or not len(slaves):
            raise ValueError("Missing Gmsh periodic node mapping.")
        for slave, donor in zip(slaves, masters):
            a, b = local[slave], local[donor]
            if a in pairs and pairs[a] != b:
                raise ValueError("Conflicting periodic node mapping.")
            pairs[a] = b
    current = np.array(sorted(pairs), dtype=np.int32)
    donor = np.array([pairs[tag] for tag in current], dtype=np.int32)
    if len(np.unique(donor)) != len(current):
        raise ValueError("Periodic node mapping is not bijective.")
    translation = xy[donor - 1] - xy[current - 1]
    if not np.allclose(translation, translation[0], rtol=0, atol=1e-8):
        raise ValueError("Periodic coordinates do not match a constant translation.")
    return current, donor, translation[0]


def export_fluent_cgns(filename, fluid_surface, boundaries, settings):
    """Write a true 2D CGNS/ADF mesh with named BCs and periodic connectivity.

    A CGNS mesh describes boundary *types*, not pressures, temperatures, or a
    Fluent case setup. Coordinate scaling affects only this export. Mixed
    cells use the CGNS 3.4 layout supported by the ParaBlade ADF converter.
    """
    xy, cells, edges, local = collect_fluid_mesh(fluid_surface, boundaries)
    current, donor, translation = collect_periodic_nodes(boundaries, local, xy)
    for name, nodes in (("periodic_right", current), ("periodic_left", donor)):
        if set(nodes) != set(edges[name][:, :2].ravel()):
            raise ValueError(f"Periodic map does not cover {name} completely.")
    scale = settings["length_scale"]
    root = new_root()
    root.add(
        AdfNode(
            "CGNSLibraryVersion",
            "CGNSLibraryVersion_t",
            np.array([3.4], dtype=np.float32),
        )
    )
    base = root.add(AdfNode("Base", "CGNSBase_t", np.array([2, 2], dtype=np.int32)))
    base.add(AdfNode("DataClass", "DataClass_t", "Dimensional"))
    units = np.full((32, 5), b" ", dtype="S1")
    for column, unit in enumerate(("Kilogram", "Meter", "Second", "Kelvin", "Radian")):
        units[: len(unit), column] = np.frombuffer(unit.encode("ascii"), dtype="S1")
    base.add(AdfNode("DimensionalUnits", "DimensionalUnits_t", units))
    base.add(AdfNode("fluid_family", "Family_t"))
    zone = base.add(
        AdfNode("fluid", "Zone_t", np.array([[len(xy), len(cells), 0]], dtype=np.int32))
    )
    zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
    zone.add(AdfNode("FamilyName", "FamilyName_t", "fluid_family"))

    def add_dimensional_array(parent, name, values, angle=False):
        array = parent.add(AdfNode(name, "DataArray_t", values))
        exponents = np.array(
            [0, 0 if angle else 1, 0, 0, 1 if angle else 0], dtype=np.float32
        )
        array.add(AdfNode("DimensionalExponents", "DimensionalExponents_t", exponents))

    grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
    for axis, values in zip("XY", (xy * scale).T):
        add_dimensional_array(grid, f"Coordinate{axis}", values)

    def add_section(name, kind, start, count, connectivity):
        section = zone.add(
            AdfNode(name, "Elements_t", np.array([kind, 0], dtype=np.int32))
        )
        section.add(
            AdfNode(
                "ElementRange",
                "IndexRange_t",
                np.array([start, start + count - 1], dtype=np.int32),
            )
        )
        section.add(AdfNode("ElementConnectivity", "DataArray_t", connectivity))
        return section

    mixed = np.concatenate(
        [np.r_[5 if len(cell) == 3 else 7, cell] for cell in cells]
    ).astype(np.int32)
    add_section("fluid_cells", 20, 1, len(cells), mixed)
    zone_bc = zone.add(AdfNode("ZoneBC", "ZoneBC_t"))
    start = len(cells) + 1
    for name, boundary_edges in edges.items():
        section = add_section(
            name, 3, start, len(boundary_edges), boundary_edges[:, :2].ravel()
        )
        section.add(
            AdfNode(
                "ParentElements",
                "DataArray_t",
                np.column_stack(
                    (
                        boundary_edges[:, 2],
                        np.zeros(len(boundary_edges), dtype=np.int32),
                    )
                ),
            )
        )
        section.add(
            AdfNode(
                "ParentElementsPosition",
                "DataArray_t",
                np.column_stack(
                    (
                        boundary_edges[:, 3],
                        np.zeros(len(boundary_edges), dtype=np.int32),
                    )
                ),
            )
        )
        bc = zone_bc.add(AdfNode(name, "BC_t", settings["boundary_types"][name]))
        bc.add(AdfNode("GridLocation", "GridLocation_t", "EdgeCenter"))
        bc.add(
            AdfNode(
                "PointRange",
                "IndexRange_t",
                np.array([[start, start + len(boundary_edges) - 1]], dtype=np.int32),
            )
        )
        start += len(boundary_edges)
    interfaces = zone.add(AdfNode("ZoneGridConnectivity", "ZoneGridConnectivity_t"))
    for name, nodes, donors, offset in (
        ("right_to_left", current, donor, translation),
        ("left_to_right", donor, current, -translation),
    ):
        interface = interfaces.add(AdfNode(name, "GridConnectivity_t", "fluid"))
        interface.add(
            AdfNode("GridConnectivityType", "GridConnectivityType_t", "Abutting1to1")
        )
        interface.add(AdfNode("GridLocation", "GridLocation_t", "Vertex"))
        interface.add(AdfNode("PointList", "IndexArray_t", nodes.reshape(1, -1)))
        interface.add(AdfNode("PointListDonor", "IndexArray_t", donors.reshape(1, -1)))
        properties = interface.add(
            AdfNode("GridConnectivityProperty", "GridConnectivityProperty_t")
        )
        periodic = properties.add(AdfNode("Periodic", "Periodic_t"))
        # CGNS 3.4 requires R4 arrays of length PhysicalDimension here.
        for key, values in (
            ("RotationCenter", np.zeros(2)),
            ("RotationAngle", np.zeros(2)),
            ("Translation", offset * scale),
        ):
            add_dimensional_array(
                periodic,
                key,
                np.asarray(values, dtype=np.float32),
                angle=key == "RotationAngle",
            )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_adf(str(filename), root)
    logger.info("  CGNS mesh saved: %s", display_path(filename))
    logger.debug(
        "CGNS export: %d nodes, %d cells; coordinates in metres.", len(xy), len(cells)
    )
    return filename


# =============================================================================
# 6. Mesh visualization and PNG export
# =============================================================================


def configure_mesh_display():
    """Use orange for both 2D element types, independent of saved GUI colors."""
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)
    for name in ("Triangles", "Quadrangles", "Lines"):
        gmsh.option.setColor(f"Mesh.Color.{name}", *MESH_ORANGE)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)


def save_mesh_image(filename, fluid_surface, *, dpi=200):
    """Render the actual Gmsh cells in orange without opening a GUI window.

    Agg keeps PNG output available in unattended CFD workflows. Shared edges
    are drawn once; unused CAD/control-point nodes are excluded.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    kinds, _, arrays = gmsh.model.mesh.getElements(2, fluid_surface)
    if not len(kinds) or any(kind not in (2, 3) for kind in kinds):
        raise ValueError("Mesh images require first-order triangles/quadrilaterals.")
    used = np.unique(np.concatenate(arrays))
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    xyz = dict(zip(tags, np.asarray(coordinates).reshape(-1, 3)))
    local = {tag: index for index, tag in enumerate(used)}
    xy = np.array([xyz[tag][:2] for tag in used])
    edges = []
    for kind, nodes in zip(kinds, arrays):
        cells = np.array([local[tag] for tag in nodes], dtype=int).reshape(
            -1, 3 if kind == 2 else 4
        )
        edges.append(
            np.stack((cells, np.roll(cells, -1, axis=1)), axis=-1).reshape(-1, 2)
        )
    edges = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
    fig = Figure(figsize=(10, 10), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.add_collection(
        LineCollection(xy[edges], colors=np.array(MESH_ORANGE) / 255, linewidths=0.35)
    )
    ax.autoscale_view()
    ax.margins(0.025)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pitchwise x (mm)")
    ax.set_ylabel("Axial y (mm)")
    ax.set_title("Stator passage mesh")
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi)
    fig.clear()
    logger.info("  Mesh image saved: %s", display_path(filename))
    return filename


# =============================================================================
# 7. Geometric quality metrics and inflation reconstruction
# Quantiles are unweighted; reconstructed layers are estimates, not Gmsh labels.
# =============================================================================


QUANTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.5,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}
DEFINITIONS = {
    "equiangle_skewness": "max((theta_max-theta_ideal)/(180-theta_ideal), (theta_ideal-theta_min)/theta_ideal); ideal 60 deg for triangles, 90 deg for quads. Zero is best.",
    "orthogonality": "Minimum outward edge-normal cosine with cell-centroid-to-edge-midpoint and, where available, cell-centroid-to-neighbor-centroid vectors. Periodic neighbors are translated into the local frame. One is best. This geometric metric is not a solver-specific composite orthogonal quality.",
    "nonorthogonality_deg": "acos(orthogonality), in degrees; zero is best. Reported per cell using its worst face.",
    "scaled_jacobian": "Gmsh minSJ: sampled minimum scaled Jacobian per cell; positive is required. Different element types can have different ideal normalization.",
    "signed_inverse_condition": "Gmsh minSICN: sampled minimum signed inverse condition number. Thin intentional inflation cells can have low values.",
    "edge_aspect_ratio": "Longest straight cell edge divided by shortest straight cell edge. High values can be intentional in inflation layers.",
    "area_mm2": "Signed polygon area after orienting connectivity counterclockwise.",
    "equivalent_size_mm": "sqrt(cell area); a size indicator, not wall-normal height.",
    "min_edge_mm": "Shortest cell edge length.",
    "max_edge_mm": "Longest cell edge length.",
    "min_angle_deg": "Minimum interior cell angle.",
    "max_angle_deg": "Maximum interior cell angle; concave corners exceed 180 degrees.",
}


def distribution(values):
    """JSON-safe count, extrema, mean, standard deviation, and quantiles."""
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("Mesh statistics contain non-finite values.")
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        **{
            key: float(np.quantile(values, fraction))
            for key, fraction in QUANTILES.items()
        },
    }


def cell_geometry(xy, cells):
    """Compute polygon-centroid and angle/size measures for triangles/quads."""
    count = len(cells)
    centroids = np.zeros((count, 2))
    values = {
        key: np.zeros(count)
        for key in (
            "area_mm2",
            "equivalent_size_mm",
            "min_edge_mm",
            "max_edge_mm",
            "edge_aspect_ratio",
            "equiangle_skewness",
            "min_angle_deg",
            "max_angle_deg",
        )
    }
    clockwise = 0
    for size in (3, 4):
        indices = np.array(
            [i for i, cell in enumerate(cells) if len(cell) == size], dtype=int
        )
        if not len(indices):
            continue
        nodes = np.array([cells[i] for i in indices])
        points = xy[nodes]
        next_points = np.roll(points, -1, axis=1)
        cross = (
            points[:, :, 0] * next_points[:, :, 1]
            - next_points[:, :, 0] * points[:, :, 1]
        )
        signed_area = cross.sum(axis=1) / 2
        if np.any(np.abs(signed_area) < 1e-20):
            raise ValueError("Cannot analyze a zero-area cell.")
        reversed_cells = signed_area < 0
        clockwise += int(reversed_cells.sum())
        if np.any(reversed_cells):
            nodes[reversed_cells] = nodes[reversed_cells, ::-1]
            for index, cell in zip(indices, nodes):
                cells[index] = cell
            points = xy[nodes]
            next_points = np.roll(points, -1, axis=1)
            cross = (
                points[:, :, 0] * next_points[:, :, 1]
                - next_points[:, :, 0] * points[:, :, 1]
            )
        area = cross.sum(axis=1) / 2
        centroids[indices] = ((points + next_points) * cross[:, :, None]).sum(
            axis=1
        ) / (6 * area[:, None])
        edges = next_points - points
        lengths = np.linalg.norm(edges, axis=2)
        if np.any(lengths <= 0):
            raise ValueError("Cannot analyze a zero-length cell edge.")
        previous = -np.roll(edges, 1, axis=1)
        cosine = np.sum(previous * edges, axis=2) / (
            np.linalg.norm(previous, axis=2) * lengths
        )
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        turn = (
            np.roll(edges, 1, axis=1)[:, :, 0] * edges[:, :, 1]
            - np.roll(edges, 1, axis=1)[:, :, 1] * edges[:, :, 0]
        )
        angles = np.where(turn < 0, 360 - angles, angles)
        ideal = 180 * (size - 2) / size
        skew = np.maximum(
            (angles.max(axis=1) - ideal) / (180 - ideal),
            (ideal - angles.min(axis=1)) / ideal,
        )
        for key, array in {
            "area_mm2": area,
            "equivalent_size_mm": np.sqrt(area),
            "min_edge_mm": lengths.min(axis=1),
            "max_edge_mm": lengths.max(axis=1),
            "edge_aspect_ratio": lengths.max(axis=1) / lengths.min(axis=1),
            "equiangle_skewness": np.maximum(0, skew),
            "min_angle_deg": angles.min(axis=1),
            "max_angle_deg": angles.max(axis=1),
        }.items():
            values[key][indices] = array
    return centroids, values, clockwise


def edge_topology(cells):
    owners = {}
    for i, cell in enumerate(cells):
        for position, (a, b) in enumerate(zip(cell, np.roll(cell, -1))):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append((i, position))
    if any(len(items) > 2 for items in owners.values()):
        raise ValueError("Nonmanifold edge detected during quality analysis.")
    return owners


def face_geometry(xy, cells, centroids, owners, boundary_edges, periodic_map):
    """Include interior and translated periodic neighbors in FV diagnostics."""
    orthogonality = np.ones(len(cells))
    periodic_neighbors = {}
    if periodic_map:
        for edge in boundary_edges.get("periodic_right", []):
            right = tuple(sorted(map(int, edge)))
            left = tuple(sorted(periodic_map[node] for node in right))
            if (
                right not in owners
                or left not in owners
                or len(owners[right]) != 1
                or len(owners[left]) != 1
            ):
                raise ValueError("Periodic face matching is incomplete.")
            r, l = owners[right][0][0], owners[left][0][0]
            translation = xy[list(right)].mean(axis=0) - xy[list(left)].mean(axis=0)
            periodic_neighbors[right] = (l, translation)
            periodic_neighbors[left] = (r, -translation)

    face_angles, area_ratio_pairs = [], []
    for edge, adjacency in owners.items():
        points = xy[list(edge)]
        midpoint = points.mean(axis=0)
        scores = []
        for i, position in adjacency:
            to_face = midpoint - centroids[i]
            a, b = cells[i][position], cells[i][(position + 1) % len(cells[i])]
            oriented_edge = xy[b] - xy[a]
            outward = np.array([oriented_edge[1], -oriented_edge[0]]) / np.linalg.norm(
                oriented_edge
            )
            score = np.dot(outward, to_face) / np.linalg.norm(to_face)
            neighbor = None
            if len(adjacency) == 2:
                neighbor = adjacency[1][0] if adjacency[0][0] == i else adjacency[0][0]
                to_neighbor = centroids[neighbor] - centroids[i]
            elif edge in periodic_neighbors:
                neighbor, shift = periodic_neighbors[edge]
                to_neighbor = centroids[neighbor] + shift - centroids[i]
            if neighbor is not None:
                score = min(
                    score, np.dot(outward, to_neighbor) / np.linalg.norm(to_neighbor)
                )
            orthogonality[i] = min(orthogonality[i], score)
            scores.append(score)
        # Count the periodic seam once (on its right side), not twice.
        if edge not in periodic_neighbors or periodic_neighbors[edge][1][0] > 0:
            face_angles.append(
                float(np.rad2deg(np.arccos(np.clip(min(scores), -1, 1))))
            )
            if len(adjacency) == 2:
                area_ratio_pairs.append((adjacency[0][0], adjacency[1][0]))
            elif edge in periodic_neighbors:
                area_ratio_pairs.append((adjacency[0][0], periodic_neighbors[edge][0]))
    return np.clip(orthogonality, -1, 1), np.array(face_angles), area_ratio_pairs


def reconstruct_inflation(xy, cells, owners, wall_edges, settings, tolerance=0.35):
    """Trace opposite-edge quad columns consistent with the configured layers.

    Membership is an estimate: recombined core quads do not carry a public
    BoundaryLayer provenance tag. Reject heights inconsistent with the
    geometric series, stop at the configured thickness allowance, and expose
    column depth/coverage rather than asserting a uniform layer count.
    """
    mask = np.zeros(len(cells), dtype=bool)
    layer_index = np.zeros(len(cells), dtype=int)
    depths, first_heights, thicknesses, ratios = [], [], [], []
    shared = set()
    selected = set()
    reason_counts = {}
    enabled = bool(settings.get("enabled", False))
    theoretical = 0
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Inflation reconstruction tolerance must be in (0, 1).")
    if enabled:
        h0, growth, limit = (
            settings["first_height"],
            settings["growth_ratio"],
            settings["thickness"],
        )
        if (
            not np.isfinite([h0, growth, limit]).all()
            or h0 <= 0
            or growth <= 1
            or limit < h0
        ):
            raise ValueError(
                "Inflation reconstruction requires positive first height, growth > 1, and thickness >= first height."
            )
        theoretical = int(
            np.floor(np.log1p(limit * (growth - 1) / h0) / np.log(growth) + 1e-10)
        )
        max_layers = theoretical + 1
        for wall in wall_edges:
            current = tuple(sorted(map(int, wall)))
            previous = None
            heights, visited = [], set()
            reason = "layer_limit"
            for layer in range(max_layers):
                adjacent = [i for i, _ in owners.get(current, []) if i != previous]
                if len(adjacent) != 1:
                    reason = "no_unique_next_cell"
                    break
                i = adjacent[0]
                cell = cells[i]
                if i in visited or len(cell) != 4:
                    reason = "non_quad_or_cycle"
                    break
                position = next(
                    j
                    for j in range(4)
                    if tuple(sorted((int(cell[j]), int(cell[(j + 1) % 4])))) == current
                )
                opposite = tuple(
                    sorted(
                        (int(cell[(position + 2) % 4]), int(cell[(position + 3) % 4]))
                    )
                )
                inner, outer = xy[list(current)], xy[list(opposite)]
                tangent = inner[1] - inner[0]
                normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
                height = abs(
                    float(np.dot(outer.mean(axis=0) - inner.mean(axis=0), normal))
                )
                remaining = limit - sum(heights)
                if remaining <= h0 * 1e-6:
                    reason = "thickness_limit"
                    break
                expected = min(h0 * growth**layer, remaining)
                if abs(height / expected - 1) > tolerance:
                    reason = "height_mismatch"
                    break
                if sum(heights) + height > limit * (1 + 1e-6):
                    reason = "thickness_limit"
                    break
                if i in selected:
                    shared.add(i)
                selected.add(i)
                visited.add(i)
                heights.append(height)
                mask[i] = True
                layer_index[i] = min(layer_index[i] or layer + 1, layer + 1)
                previous, current = i, opposite
            depths.append(len(heights))
            if heights:
                first_heights.append(heights[0])
                thicknesses.append(sum(heights))
                ratios.extend(np.asarray(heights[1:]) / np.asarray(heights[:-1]))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    histogram = {
        str(value): int(np.sum(np.asarray(depths) == value))
        for value in sorted(set(depths))
    }
    covered = int(np.count_nonzero(depths))
    return (
        mask,
        layer_index,
        {
            "enabled": enabled,
            "membership": "reconstructed estimate" if enabled else "disabled",
            "configured": settings,
            "relative_height_tolerance": tolerance,
            "nominal_full_layers_from_geometric_series": theoretical,
            "reconstructed_elements": int(mask.sum()),
            "wall_edge_columns": len(wall_edges),
            "columns_with_layers": covered,
            "wall_edge_coverage_fraction": (
                covered / len(wall_edges) if len(wall_edges) else 0.0
            ),
            "layer_depth": distribution(depths),
            "layer_depth_histogram": histogram,
            "first_height_mm": distribution(first_heights),
            "column_thickness_mm": distribution(thicknesses),
            "measured_growth_ratio": distribution(ratios),
            "shared_cells_between_columns": len(shared),
            "column_stop_reasons": reason_counts,
            "note": "Quad stacks are inferred from wall adjacency and configured height/growth/thickness; fan/merged regions and matching core quads can make membership ambiguous. This is not an exact Gmsh layer label.",
        },
    )


def analyze_mesh(
    xy,
    cells,
    element_tags,
    boundary_edges,
    *,
    periodic_map=None,
    inflation=None,
    gmsh_quality=None,
    worst_cells=10,
    inflation_tolerance=0.35,
):
    """Pure-Python geometric analysis; connectivity uses zero-based node indices."""
    xy = np.asarray(xy, float)
    cells = [np.asarray(cell, int).copy() for cell in cells]
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Mesh coordinates must be finite 2D points.")
    if type(worst_cells) is not int or worst_cells < 1:
        raise ValueError("worst_cells must be a positive integer.")
    if not cells or any(len(cell) not in (3, 4) for cell in cells):
        raise ValueError(
            "Mesh reporting supports nonempty first-order triangle/quad meshes."
        )
    if len(element_tags) != len(cells):
        raise ValueError("Element tags must correspond to the cell connectivity.")
    if any(np.min(cell) < 0 or np.max(cell) >= len(xy) for cell in cells):
        raise ValueError("Cell connectivity references an invalid node index.")
    centroids, metrics, clockwise = cell_geometry(xy, cells)
    owners = edge_topology(cells)
    orthogonality, face_angles, adjacent = face_geometry(
        xy, cells, centroids, owners, boundary_edges, periodic_map or {}
    )
    metrics.update(
        orthogonality=orthogonality,
        nonorthogonality_deg=np.rad2deg(np.arccos(orthogonality)),
    )
    metrics.update(gmsh_quality or {})
    inflation_mask, layer_index, inflation_report = reconstruct_inflation(
        xy,
        cells,
        owners,
        boundary_edges.get("blade", []),
        inflation or {},
        inflation_tolerance,
    )
    sizes = np.array([len(cell) for cell in cells])
    groups = {
        "all_cells": np.ones(len(cells), bool),
        "triangles": sizes == 3,
        "quadrilaterals": sizes == 4,
    }
    if inflation_report["enabled"]:
        groups.update(
            reconstructed_inflation=inflation_mask, remaining_cells=~inflation_mask
        )
    quality = {
        group: {
            metric: distribution(values[mask]) for metric, values in metrics.items()
        }
        for group, mask in groups.items()
    }
    area = metrics["area_mm2"]
    jumps = [max(area[i], area[j]) / min(area[i], area[j]) for i, j in adjacent]
    transition_jumps = [
        jump
        for jump, (i, j) in zip(jumps, adjacent)
        if inflation_mask[i] != inflation_mask[j]
    ]
    neighbors = [set() for _ in cells]
    for i, j in adjacent:
        neighbors[i].add(j)
        neighbors[j].add(i)
    unseen = set(range(len(cells)))
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            available = neighbors[pending.pop()] & unseen
            unseen.difference_update(available)
            pending.extend(available)
    boundary_report = {}
    for name, edges in boundary_edges.items():
        edges = np.asarray(edges, int).reshape(-1, 2)
        lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
        boundary_report[name] = {
            "edges": len(edges),
            "nodes": int(len(np.unique(edges))),
            "length_mm": float(lengths.sum()),
            "edge_length_mm": distribution(lengths),
        }
    tails = {
        "skewness_gt_0.5": int(np.sum(metrics["equiangle_skewness"] > 0.5)),
        "skewness_gt_0.8": int(np.sum(metrics["equiangle_skewness"] > 0.8)),
        "skewness_gt_0.95": int(np.sum(metrics["equiangle_skewness"] > 0.95)),
        "orthogonality_lt_0.2": int(np.sum(orthogonality < 0.2)),
        "orthogonality_lt_0.1": int(np.sum(orthogonality < 0.1)),
        "nonorthogonality_gt_75_deg": int(np.sum(metrics["nonorthogonality_deg"] > 75)),
        "edge_aspect_ratio_gt_100": int(np.sum(metrics["edge_aspect_ratio"] > 100)),
    }
    worst = {}
    rankings = [("equiangle_skewness", True), ("orthogonality", False)]
    if "scaled_jacobian" in metrics:
        rankings.append(("scaled_jacobian", False))
    for metric, reverse in rankings:
        order = np.argsort(metrics[metric])
        order = order[::-1] if reverse else order
        worst[metric] = [
            {
                "element_tag": int(element_tags[i]),
                "value": float(metrics[metric][i]),
                "centroid_x_mm": float(centroids[i, 0]),
                "centroid_y_mm": float(centroids[i, 1]),
                "reconstructed_inflation": bool(inflation_mask[i]),
            }
            for i in order[:worst_cells]
        ]
    connected_nodes = set(int(node) for cell in cells for node in cell)
    return {
        "summary": {
            "fluid_nodes": len(connected_nodes),
            "fluid_cells": len(cells),
            "triangles": int(np.sum(sizes == 3)),
            "quadrilaterals": int(np.sum(sizes == 4)),
            "quadrilateral_fraction": float(np.mean(sizes == 4)),
            "fluid_area_mm2": float(area.sum()),
            "clockwise_cells_reoriented_for_analysis": clockwise,
            "concave_cells": int(np.sum(metrics["max_angle_deg"] > 180)),
            "connected_cell_regions": components,
            "interior_edges": sum(len(items) == 2 for items in owners.values()),
            "exterior_edges": sum(len(items) == 1 for items in owners.values()),
        },
        "quality": quality,
        "inflation": inflation_report,
        "boundaries": boundary_report,
        "face_nonorthogonality_deg": distribution(face_angles),
        "adjacent_area_ratio": distribution(jumps),
        "inflation_core_area_ratio": distribution(transition_jumps),
        "tail_counts": tails,
        "worst_cells": worst,
        "definitions": DEFINITIONS,
        "cell_data": {
            "element_tag": np.asarray(element_tags, int),
            "cell_type": np.where(sizes == 3, "triangle", "quadrilateral"),
            "gmsh_element_type": np.where(sizes == 3, 2, 3),
            "cell_vertices": sizes,
            "centroid_x_mm": centroids[:, 0],
            "centroid_y_mm": centroids[:, 1],
            "reconstructed_inflation": inflation_mask,
            "reconstructed_layer": layer_index,
            **metrics,
        },
    }


# =============================================================================
# 8. Collect diagnostics from an active Gmsh mesh
# =============================================================================


def compute_mesh_report(
    fluid_surface,
    boundaries,
    *,
    inflation=None,
    worst_cells=10,
    inflation_tolerance=0.35,
):
    """Analyze an active Gmsh 2D surface before gmsh.finalize()."""
    kinds, tag_arrays, node_arrays = gmsh.model.mesh.getElements(2, fluid_surface)
    if any(kind not in (2, 3) for kind in kinds):
        raise ValueError("Reporting requires first-order triangles and quadrilaterals.")
    used = np.unique(np.concatenate(node_arrays))
    all_tags, all_xyz, _ = gmsh.model.mesh.getNodes()
    coordinates = dict(zip(all_tags, np.asarray(all_xyz).reshape(-1, 3)))
    local = {tag: i for i, tag in enumerate(used)}
    xy = np.array([coordinates[tag][:2] for tag in used])
    cells, tags = [], []
    for kind, cell_tags, connectivity in zip(kinds, tag_arrays, node_arrays):
        tags.extend(cell_tags)
        cells.extend(
            np.array([local[tag] for tag in cell], int)
            for cell in connectivity.reshape(-1, 3 if kind == 2 else 4)
        )
    boundary_edges = {}
    for name, curves in boundaries.items():
        edges = []
        for curve in curves:
            types, _, arrays = gmsh.model.mesh.getElements(1, curve)
            if any(kind != 1 for kind in types):
                raise ValueError("Reporting requires first-order boundary edges.")
            edges.extend(
                [local[tag] for tag in edge]
                for array in arrays
                for edge in array.reshape(-1, 2)
            )
        boundary_edges[name] = edges
    periodic = {}
    for left, right in zip(boundaries["periodic_left"], boundaries["periodic_right"]):
        master, slaves, masters, _ = gmsh.model.mesh.getPeriodicNodes(1, right)
        if master != left or not len(slaves):
            raise ValueError(
                "Quality report requires the registered periodic correspondence."
            )
        periodic.update(
            (local[slave], local[donor]) for slave, donor in zip(slaves, masters)
        )
    qualities = {
        name: np.asarray(gmsh.model.mesh.getElementQualities(tags, measure))
        for name, measure in (
            ("scaled_jacobian", "minSJ"),
            ("signed_inverse_condition", "minSICN"),
        )
    }
    report = analyze_mesh(
        xy,
        cells,
        tags,
        boundary_edges,
        periodic_map=periodic,
        inflation=inflation,
        gmsh_quality=qualities,
        worst_cells=worst_cells,
        inflation_tolerance=inflation_tolerance,
    )
    report["periodic_node_pairs"] = len(periodic)
    report["gmsh_version"] = gmsh.__version__
    return report


# =============================================================================
# 9. Offline HTML, JSON, CSV, and histogram reports
# =============================================================================


def _table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def _format(value):
    if isinstance(value, bool):
        return str(value)
    return f"{value:.5g}" if isinstance(value, (int, float)) else str(value)


def write_mesh_report(
    report, output_dir, *, histogram_bins=30, source_mesh="Active Gmsh mesh"
):
    """Write a self-contained HTML report, JSON summary, CSV cells, and plots."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = report["cell_data"]
    summary = {key: value for key, value in report.items() if key != "cell_data"}
    summary["source_mesh"] = str(source_mesh)
    (output_dir / "mesh_report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    keys = list(data)
    with (output_dir / "mesh_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(data[key] for key in keys)))

    # Reports never create GUI windows or alter the application's backend.
    fig = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for ax, metric, label in zip(
        axes.flat,
        (
            "equiangle_skewness",
            "orthogonality",
            "scaled_jacobian",
            "edge_aspect_ratio",
            "area_mm2",
        ),
        (
            "Equiangle skewness (lower is better)",
            "Geometric orthogonality (higher is better)",
            "Gmsh scaled Jacobian",
            "Edge aspect ratio",
            "Cell area [mm²]",
        ),
    ):
        if metric not in data:
            ax.set_visible(False)
            continue
        values = data[metric]
        if (
            metric in ("edge_aspect_ratio", "area_mm2")
            and values.max() > values.min() * 5
        ):
            bins = np.geomspace(values.min(), values.max(), histogram_bins + 1)
            ax.set_xscale("log")
        else:
            low, high = float(values.min()), float(values.max())
            if high - low < 1e-12 * max(1.0, abs(low), abs(high)):
                padding = max(abs(low) * 0.01, 0.005)
                low, high = low - padding, high + padding
            bins = np.linspace(low, high, histogram_bins + 1)
        ax.hist(values, bins=bins, color="#2563eb", alpha=0.8, label="All cells")
        mask = data["reconstructed_inflation"]
        if mask.any():
            ax.hist(
                values[mask],
                bins=bins,
                color="#f59e0b",
                alpha=0.7,
                label="Reconstructed inflation",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Cells")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    histogram = report["inflation"]["layer_depth_histogram"]
    if histogram:
        layer_ticks = [int(key) for key in histogram]
        axes.flat[-1].bar(
            layer_ticks, list(histogram.values()), color="#f59e0b", width=0.7
        )
        axes.flat[-1].set_xticks(layer_ticks)
        axes.flat[-1].set_xlim(min(layer_ticks) - 1, max(layer_ticks) + 1)
        axes.flat[-1].set_xlabel("Reconstructed layers per wall-edge column")
        axes.flat[-1].set_ylabel("Columns")
    else:
        axes.flat[-1].text(
            0.5,
            0.5,
            "Inflation disabled",
            ha="center",
            va="center",
            transform=axes.flat[-1].transAxes,
        )
        axes.flat[-1].set_axis_off()
    fig.savefig(output_dir / "mesh_quality.png", dpi=160)
    svg = StringIO()
    fig.savefig(svg, format="svg")
    svg_text = svg.getvalue()[svg.getvalue().index("<svg") :]

    columns = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean")
    tables = []
    for group, measures in report["quality"].items():
        rows = [
            [
                metric.replace("_", " "),
                *[_format(stats.get(key, "—")) for key in columns],
            ]
            for metric, stats in measures.items()
        ]
        content = _table(["Metric", *[key.upper() for key in columns]], rows)
        tables.append(
            f'<details {"open" if group == "all_cells" else ""}><summary>{escape(group.replace("_", " ").title())}</summary><div class="scroll">{content}</div></details>'
        )
    mesh = report["summary"]
    inflation = report["inflation"]
    cards = [
        ("Fluid cells", f"{mesh['fluid_cells']:,}"),
        ("Fluid nodes", f"{mesh['fluid_nodes']:,}"),
        ("Quadrilaterals", f"{100 * mesh['quadrilateral_fraction']:.1f}%"),
        ("Periodic pairs", str(report.get("periodic_node_pairs", "—"))),
        ("Inflation cells · reconstructed", f"{inflation['reconstructed_elements']:,}"),
        ("Fluid area", f"{mesh['fluid_area_mm2']:.5g} mm²"),
    ]
    inflation_rows = [
        [key.replace("_", " "), _format(value)]
        for key, value in inflation.items()
        if isinstance(value, (str, int, float)) and key != "note"
    ]
    inflation_rows += [
        [
            key.replace("_", " "),
            ", ".join(
                f"{quantile}: {_format(value.get(quantile, '—'))}"
                for quantile in ("min", "p05", "p50", "p95", "max")
            ),
        ]
        for key, value in inflation.items()
        if key
        in (
            "layer_depth",
            "first_height_mm",
            "column_thickness_mm",
            "measured_growth_ratio",
        )
    ]
    boundary_rows = [
        [
            name,
            value["edges"],
            value["nodes"],
            _format(value["length_mm"]),
            _format(value["edge_length_mm"].get("p50", 0)),
        ]
        for name, value in report["boundaries"].items()
    ]
    face_rows = [
        [
            key.replace("_", " "),
            *[_format(report[key].get(column, "—")) for column in columns],
        ]
        for key in (
            "face_nonorthogonality_deg",
            "adjacent_area_ratio",
            "inflation_core_area_ratio",
        )
    ]
    transition_table = _table(
        ["Calculated parameter", "Value"],
        [
            [key.replace("_", " "), _format(value)]
            for key, value in report.get("core_transition", {}).items()
        ],
    )
    worst_tables = []
    for metric, cells in report["worst_cells"].items():
        worst_tables.append(
            f"<h3>{escape(metric.replace('_', ' '))}</h3>"
            + _table(
                [
                    "Gmsh cell tag",
                    "Value",
                    "Centroid x [mm]",
                    "Centroid y [mm]",
                    "Inflation estimate",
                ],
                [
                    [
                        cell["element_tag"],
                        _format(cell["value"]),
                        _format(cell["centroid_x_mm"]),
                        _format(cell["centroid_y_mm"]),
                        cell["reconstructed_inflation"],
                    ]
                    for cell in cells
                ],
            )
        )
    notes = "".join(
        f"<dt>{escape(key.replace('_', ' '))}</dt><dd>{escape(value)}</dd>"
        for key, value in DEFINITIONS.items()
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2D mesh quality report</title><style>
body{{font:15px/1.6 'Segoe UI',Arial,sans-serif;background:#f1f5f9;color:#172033;margin:0}}main{{max-width:1250px;margin:auto;padding:36px}}
h1{{font-size:32px;line-height:1.2;margin-bottom:8px}}h2{{font-size:22px;margin-top:32px}}h3{{font-size:16px}}.muted{{color:#64748b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}}.card,section,details{{background:white;border:1px solid #dce3ec;border-radius:10px;padding:18px}}
.card strong{{display:block;font-size:27px;color:#1d4ed8}}.card span{{font-size:12px;color:#64748b}}section{{margin-bottom:18px}}details{{margin:10px 0}}summary{{cursor:pointer;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:9px 8px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#f8fafc}}.scroll{{overflow-x:auto}}
svg{{width:100%;height:auto}}dt{{font-weight:600;margin-top:12px}}dd{{margin-left:0;color:#475569}}.note{{padding:14px;border-left:4px solid #f59e0b;background:#fffbeb}}a{{color:#1d4ed8}}
</style><main><div class="muted">GEOMETRY · TOPOLOGY · QUALITY</div><h1>2D mesh quality report</h1>
<p class="muted">{escape(str(source_mesh))} · lengths in mm · unweighted cell quantiles · first-order triangles and quads</p>
<div class="cards">{"".join(f'<div class="card"><strong>{value}</strong><span>{label}</span></div>' for label, value in cards)}</div>
<section><h2>Mesh inventory</h2>{_table(['Quantity', 'Value'], [[key.replace('_', ' '), _format(value)] for key, value in mesh.items()])}</section>
<section><h2>Quality distributions</h2>{svg_text}<p class="muted">Orange overlays show reconstructed inflation cells. Quantiles below are unweighted; P50 is the median.</p>{''.join(tables)}</section>
<section><h2>Inflation reconstruction</h2><p class="note">{escape(inflation['note'])}</p>{_table(['Quantity', 'Value'], inflation_rows)}
<h3>Configured layer inputs</h3>{_table(['Parameter', 'Value'], [[key, _format(value)] for key, value in inflation['configured'].items()])}
<h3>Layer-depth counts</h3>{_table(['Layers', 'Wall-edge columns'], list(inflation['layer_depth_histogram'].items()))}
<h3>Column stopping reasons</h3>{_table(['Reason', 'Columns'], list(inflation['column_stop_reasons'].items()))}</section>
<section><h2>Faces and size transitions</h2><div class="scroll">{_table(['Metric', *columns], face_rows)}</div>
{transition_table}
<p class="muted">Adjacent area ratio is max(area1,area2)/min(area1,area2), including periodic neighbors. Face statistics count each periodic pair once.</p></section>
<section><h2>Boundary inventory</h2>{_table(['Boundary', 'Edges', 'Nodes', 'Length [mm]', 'Median edge [mm]'], boundary_rows)}</section>
<section><h2>Distribution tails</h2>{_table(['Diagnostic threshold', 'Cells', 'Fraction'], [[key.replace('_gt_', ' > ').replace('_lt_', ' < ').replace('_', ' '), value, f"{100 * value / mesh['fluid_cells']:.3f}%"] for key, value in report['tail_counts'].items()])}
<p class="muted">These are diagnostic counts, not a universal CFD pass/fail criterion. High aspect ratios are often intentional in inflation cells.</p></section>
<section><h2>Cells to inspect</h2>{''.join(worst_tables)}</section>
<section><h2>Definitions and interpretation</h2><dl>{notes}</dl><p>y+ is not computed: it depends on the flow solution and wall shear stress.</p>
<p><a href="https://gmsh.info/doc/texinfo/gmsh.html#gmsh_002fmodel_002fmesh_002fgetElementQualities">Gmsh quality definitions</a> ·
<a href="https://innovationspace.ansys.com/knowledge/forums/topic/skewness-in-ansys-meshing-2/">Equiangle skewness</a></p></section>
<p class="muted">Companion files: mesh_report.json (summary), mesh_cells.csv (per-cell metrics), mesh_quality.png (histograms).</p></main></html>"""
    path = output_dir / "mesh_report.html"
    path.write_text(html, encoding="utf-8")
    return path


# =============================================================================
# 10. Complete mesh-generation workflow
# Own the Gmsh session and run CAD, sizing, validation, export, and reports.
# =============================================================================


def create_mesh(blade, curves, settings, output_dir, show_gui=True):
    """Generate, check, export and report a mesh using the caller's logging setup."""
    validate_mesh_settings(settings)
    profile = calculate_inflation_profile(settings["inflation"])
    if gmsh.isInitialized():
        raise RuntimeError(
            "Finish the existing Gmsh session before creating this mesh."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Initialize Gmsh.")
    gmsh.initialize(["stator_mesh", "-v", str(settings.get("verbosity", 2))])
    try:
        gmsh.option.setNumber("General.Terminal", int(settings["terminal_output"]))
        # Level 2 keeps warnings/errors and hides progress and the GUI banner.
        gmsh.option.setNumber("General.Verbosity", settings.get("verbosity", 2))
        gmsh.option.setNumber("Geometry.NumSubEdges", 500)
        gmsh.model.add("stator_passage")
        logger.info("  Build blade CAD, fluid surface and periodic boundaries.")
        blade_curves = add_blade_boundary(blade, settings["blade_size"])
        surface, boundaries = add_periodic_passage(
            curves, blade_curves, blade["pitch"], settings["domain_size"]
        )
        logger.info("  Configure mesh sizes, recombination and inflation layers.")
        set_mesh_sizes(blade_curves, settings, profile, blade["pitch"])
        add_inflation_layers(blade_curves, settings)
        configure_recombination(surface, settings)
        logger.info("  Generate the 1D mesh and check periodic conformity.")
        gmsh.model.mesh.generate(1)
        check_periodic_conformity(boundaries)
        logger.info("  Match core sizing to the measured blade spacing.")
        transition = configure_core_transition(
            blade_curves, settings, profile, blade["pitch"], use_mesh=True
        )
        if profile["enabled"]:
            summary(
                logger,
                "Inflation layers",
                {
                    "Number of layers": profile["layers"],
                    "Stack thickness": f"{profile['stack_thickness']:.4f} mm",
                    "Last layer height": f"{profile['last_height']:.4f} mm",
                },
            )
        summary(
            logger,
            "Automatic core transition",
            {
                "Starting cell size": f"{transition['core_start_size_min_mm']:.4f} to {transition['core_start_size_max_mm']:.4f} mm",
                "Far-field distance": f"{transition['far_field_distance_min_mm']:.2f} to {transition['far_field_distance_max_mm']:.2f} mm",
            },
        )
        logger.info("  Generate the 2D mesh.")
        gmsh.model.mesh.generate(2)
        logger.info("  Check mesh quality and periodic node mapping.")
        statistics = mesh_statistics(surface, boundaries)
        statistics["inflation_profile"] = profile
        statistics["core_transition"] = transition
        if settings["inflation"]["enabled"] and not any(
            name.startswith("Quadrilateral") for name in statistics["cells"]
        ):
            raise RuntimeError("Gmsh generated no inflation-layer quadrilaterals.")
        if (
            settings.get("recombine", True)
            and statistics["quadrilateral_fraction"] == 0
        ):
            raise RuntimeError("Gmsh recombination generated no quadrilaterals.")
        summary(
            logger,
            "Mesh quality",
            {
                "Fluid cells": f"{sum(statistics['cells'].values()):,}",
                "Quadrilateral fraction": f"{statistics['quadrilateral_fraction']:.1%}",
                "Minimum scaled Jacobian": f"{statistics['minimum_scaled_jacobian']:.3f}",
                "Periodic node pairs": statistics["periodic_node_pairs"],
            },
        )
        logger.debug("Element counts by type: %s", statistics["cells"])
        logger.info("  Export the mesh image and named-boundary CGNS.")
        configure_mesh_display()
        image = settings.get("image", {})
        image_path = save_mesh_image(
            output_dir / image.get("filename", "stator_mesh.png"),
            surface,
            dpi=image.get("dpi", 200),
        )
        statistics["image_file"] = str(image_path)
        if settings.get("cgns", {}).get("enabled", False):
            mesh_file = export_fluent_cgns(
                output_dir / settings["cgns"]["filename"],
                surface,
                boundaries,
                settings["cgns"],
            )
            statistics["mesh_file"] = str(mesh_file)
        reporting = settings.get("report", {})
        if reporting.get("enabled", True):
            logger.info(
                "  Analyze cell quality and inflation stacks; write the mesh report."
            )
            report = compute_mesh_report(
                surface,
                boundaries,
                inflation=settings["inflation"],
                worst_cells=reporting.get("worst_cells", 10),
                inflation_tolerance=reporting.get("inflation_tolerance", 0.35),
            )
            report["core_transition"] = transition
            report["nominal_inflation_profile"] = profile
            report_path = write_mesh_report(
                report,
                output_dir,
                histogram_bins=reporting.get("histogram_bins", 30),
                source_mesh=(
                    settings["cgns"]["filename"]
                    if "mesh_file" in statistics
                    else "Active Gmsh mesh"
                ),
            )
            statistics["report"] = {
                key: value for key, value in report.items() if key != "cell_data"
            }
            statistics["report_file"] = str(report_path)
            details = {
                "Reconstructed inflation cells": f"{report['inflation']['reconstructed_elements']:,}",
                "Median layer depth": report["inflation"]["layer_depth"].get(
                    "p50", "disabled"
                ),
            }
            jumps = report["inflation_core_area_ratio"]
            if jumps["count"]:
                details["Inflation/core area ratio"] = (
                    f"median {jumps['p50']:.2f}; P95 {jumps['p95']:.2f}; max {jumps['max']:.2f} (estimated interface)"
                )
            summary(logger, "Inflation diagnostics", details)
            logger.info("  Mesh quality report saved: %s", display_path(report_path))
        if show_gui and settings["show_gui"]:
            logger.info("  Opening the Gmsh viewer; close it to complete the workflow.")
            gmsh.fltk.run()
        return statistics
    finally:
        logger.debug("Finalize Gmsh and remove the mesh-size callback.")
        gmsh.model.mesh.removeSizeCallback()
        gmsh.finalize()
