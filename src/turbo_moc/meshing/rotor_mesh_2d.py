"""2D blade-to-blade mesh of the sized axial rotor (hub section).

Not exposed on its own in the app: it is the section the swept 3D rotor mesh
(turbo_moc.meshing.stator_mesh_3d.mesh_swept_passage) is built from.

Frame. The rotor blade (design_rotor_vortex_blade / rescale_rotor_blade)
has x = chordwise (flow toward +x) and y = pitchwise. The app's 3D rotor is
the mirrored blade wrapped with flare "pitch_scale" (theta = -y / r_hub,
z = -x), so the mesh is built in X = y, Y = -x: the stator mesh's frame (X
pitchwise, Y axial, flow toward -Y). section_coordinates then maps it onto
exactly the exported rotor solid, and the 3D sweep is shared with the stator.

Passage. Periodic boundaries run along the middle of the channels on either
side of the blade, (suction side of this blade + pressure side of the next) / 2,
continued straight at the inlet/outlet relative flow angles, so the blade
sits in the middle of a smooth periodic channel. Only the CAD differs from
the stator: sizing, inflation, recombination, fold repair and checks are
stator_mesh_2d's (_generate_passage_mesh).
"""

import logging

import gmsh
import numpy as np

from turbo_moc.meshing import stator_mesh_2d as base

logger = logging.getLogger(__name__)

ROTOR_PASSAGE = {"inlet_chords": 1.0, "outlet_chords": 1.5}  # axial chords up/downstream
BLADE_SAMPLES = 600  # per side, uniform in arc length, for the CAD splines
EDGE_MARGIN = 0.05  # axial-chord fraction near LE/TE where the channel midline is bridged


def rotor_contour(scaled_blade):
    """Closed rotor contour in the mesh frame (X = y, Y = -x), no repeated end point."""
    x = np.asarray(scaled_blade["blade"]["x"], dtype=float)
    y = np.asarray(scaled_blade["blade"]["y"], dtype=float)
    points = np.column_stack((y, -x))
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    return points[keep]


def _resample(polyline, count):
    """`count` points uniformly spaced in arc length along an open polyline."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], count)
    return np.column_stack([np.interp(t, s, polyline[:, i]) for i in (0, 1)])


def blade_sides(contour):
    """The two open sides between the leading edge (max Y) and trailing edge
    (min Y), each ordered LE -> TE, as (left, right) by mean X."""
    le, te = int(np.argmax(contour[:, 1])), int(np.argmin(contour[:, 1]))
    rolled = np.roll(contour, -le, axis=0)
    k = (te - le) % len(contour)
    first = rolled[: k + 1]
    second = np.vstack((rolled[k:], rolled[:1]))[::-1]
    sides = sorted((first, second), key=lambda side: side[:, 0].mean())
    return sides[0], sides[1]


def rotor_flow_domain(scaled_blade, passage=None):
    """Periodic boundaries (left = right - pitch), inlet/outlet stations and
    the blade sides, all in the mesh frame. Raises if the blade does not fit
    one pitch or the channel midline cannot be formed."""
    passage = {**ROTOR_PASSAGE, **(passage or {})}
    pitch = float(scaled_blade["pitch"])
    contour = rotor_contour(scaled_blade)
    left, right = blade_sides(contour)
    y_le, y_te = left[0, 1], left[-1, 1]
    axial_chord = y_le - y_te
    # Each side as X(Y); valid while the surfaces never turn past axial-normal.
    for side in (left, right):
        if np.any(np.diff(side[:, 1]) > 1e-9 * axial_chord):
            raise ValueError("Rotor blade sides are not single-valued in the axial direction; "
                             "the channel midline cannot be formed.")
    def x_left(y):
        return np.interp(y, left[::-1, 1], left[::-1, 0])

    def x_right(y):
        return np.interp(y, right[::-1, 1], right[::-1, 0])

    margin = EDGE_MARGIN * axial_chord
    y_mid = np.linspace(y_le - margin, y_te + margin, 80)
    gap = x_left(y_mid) + pitch - x_right(y_mid)
    if np.any(gap <= 0):
        raise ValueError("Adjacent rotor blades overlap at this pitch: the passage is closed.")
    middle = np.column_stack((0.5 * (x_right(y_mid) + x_left(y_mid) + pitch), y_mid))

    # Straight continuations at the relative flow angles (raw frame slope
    # dy/dx = tan(beta); in X = y, Y = -x: dX/dY = -tan(beta)).
    le_point = contour[np.argmax(contour[:, 1])]
    te_point = contour[np.argmin(contour[:, 1])]
    y_in = y_le + passage["inlet_chords"] * axial_chord
    y_out = y_te - passage["outlet_chords"] * axial_chord
    up = np.linspace(y_in, y_le, 25)
    down = np.linspace(y_te, y_out, 30)
    upstream = np.column_stack((le_point[0] + 0.5 * pitch - np.tan(np.radians(scaled_blade["beta_inlet"]))
                                * (up - le_point[1]), up))
    downstream = np.column_stack((te_point[0] + 0.5 * pitch - np.tan(np.radians(scaled_blade["beta_outlet"]))
                                  * (down - te_point[1]), down))
    periodic_right = np.vstack((upstream, middle, downstream))
    periodic_left = periodic_right - [pitch, 0.0]

    clearance = min(_min_distance(periodic_right, contour), _min_distance(periodic_left, contour))
    return {
        "pitch": pitch,
        "contour": contour,
        "blade_left": left,
        "blade_right": right,
        "periodic_left": periodic_left,
        "periodic_right": periodic_right,
        "axial_chord": float(axial_chord),
        "clearance_mm": float(clearance),
        "shift_mm": 0.0,
    }


def _min_distance(points, polyline):
    """Smallest distance from sampled points to a closed polyline's segments."""
    starts, ends = polyline, np.roll(polyline, -1, axis=0)
    edge = ends - starts
    t = np.clip(np.einsum("pij,ij->pi", points[:, None] - starts, edge)
                / np.einsum("ij,ij->i", edge, edge), 0.0, 1.0)
    return float(np.min(np.linalg.norm(points[:, None] - (starts + t[..., None] * edge), axis=2)))


def _spline(points, mesh_size, first_tag=None, last_tag=None):
    """Gmsh OCC interpolating spline through `points`, optionally reusing end-point tags."""
    tags = [first_tag or gmsh.model.occ.addPoint(*points[0], 0, mesh_size)]
    tags += [gmsh.model.occ.addPoint(*p, 0, mesh_size) for p in points[1:-1]]
    tags.append(last_tag or gmsh.model.occ.addPoint(*points[-1], 0, mesh_size))
    return gmsh.model.occ.addSpline(tags), tags[0], tags[-1]


def _rotor_cad(domain):
    """build_cad for stator_mesh_2d._generate_passage_mesh: blade (two
    splines LE -> TE), periodic splines, inlet/outlet lines, periodic
    translation and physical groups."""
    def build(numerics):
        blade_size, domain_size = numerics["blade_size"], numerics["domain_size"]
        left = _resample(domain["blade_left"], BLADE_SAMPLES)
        right = _resample(domain["blade_right"], BLADE_SAMPLES)
        left_curve, le, te = _spline(left, blade_size)
        right_curve, _, _ = _spline(right, blade_size, le, te)
        blade_curves = [left_curve, right_curve]

        periodic = _resample(domain["periodic_left"], 200)
        left_side, left_in, left_out = _spline(periodic, domain_size)
        right_side, right_in, right_out = _spline(periodic + [domain["pitch"], 0.0], domain_size)
        inlet = gmsh.model.occ.addLine(left_in, right_in)
        outlet = gmsh.model.occ.addLine(right_out, left_out)
        outer = gmsh.model.occ.addCurveLoop([inlet, right_side, outlet, -left_side])
        blade_loop = gmsh.model.occ.addCurveLoop([left_curve, -right_curve])
        surface = gmsh.model.occ.addPlaneSurface([outer, blade_loop])
        gmsh.model.occ.synchronize()
        transform = [1, 0, 0, domain["pitch"], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, [right_side], [left_side], transform)
        boundaries = {"blade": blade_curves, "periodic_left": [left_side], "periodic_right": [right_side],
                      "inlet": [inlet], "outlet": [outlet]}
        for name, tags in boundaries.items():
            gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), name)
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface]), "fluid")
        return blade_curves, surface, boundaries
    return build


def mesh_sized_rotor(scaled_blade, sizes, passage=None):
    """Unstructured (quad-dominant, inflation-layered) hub mesh of the sized
    rotor, in the same JSON-friendly layout as stator_mesh_2d.mesh_sized_stator
    (so it sweeps with stator_mesh_3d.mesh_swept_passage). `sizes`: blade_size,
    domain_size, first_height [mm], num_layers, growth_ratio, transition_ratio."""
    numerics = base.app_numerics(sizes)
    domain = rotor_flow_domain(scaled_blade, passage)
    stack = numerics["inflation_layers"]["thickness"]
    if domain["clearance_mm"] <= stack:
        raise ValueError(
            f"The inflation stack ({stack:.3f} mm) does not fit between the blade and the periodic "
            f"boundary ({domain['clearance_mm']:.3f} mm, half the narrowest channel): reduce the layers "
            f"or the growth ratio.")
    blade = {"pitch": domain["pitch"]}
    with base._gmsh_session(numerics):
        surface, boundaries, statistics, _, profile = base._generate_passage_mesh(
            blade, None, numerics, build_cad=_rotor_cad(domain))
        xy, cells, named_edges, local = base.collect_fluid_mesh(surface, boundaries)
        current, donor, _ = base.collect_periodic_nodes(boundaries, local, xy)
    return {
        "topology": "unstructured",
        "fallback_reason": None,
        "xy": xy.tolist(),
        "cells": [cell.tolist() for cell in cells],
        "named_edges": {name: edges.tolist() for name, edges in named_edges.items()},
        "periodic": {"current": current.tolist(), "donor": donor.tolist()},
        "pitch": domain["pitch"],
        "clearance_mm": domain["clearance_mm"],
        "statistics": {
            "cells": {name: int(count) for name, count in statistics["cells"].items()},
            "quadrilateral_fraction": float(statistics["quadrilateral_fraction"]),
            "minimum_scaled_jacobian": float(statistics["minimum_scaled_jacobian"]),
            "periodic_node_pairs": int(statistics["periodic_node_pairs"]),
            "inflation_layers": int(profile["layers"]),
            "stack_thickness_mm": float(profile["stack_thickness"]),
        },
    }
