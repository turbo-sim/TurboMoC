"""General meridional geometry and conformal blade mapping; lengths are mm.

The Gmsh plane uses q = (Rref*theta, -Rref*m'), so existing stator meshing
functions retain their pitchwise-x, downstream-negative-y convention.
Theta increases clockwise; physical azimuth = -q_x/Rref about global Z.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import BSpline

import examples.gmsh.geometry_source as geo2
from turbo_moc.geometry import parametrize_stator_blade

# =============================================================================
# 1. Thin input adapters: axial/radial dimensions -> common geometry definition
# =============================================================================


def validate_machine(machine):
    """Validate independent dimensions; radii/chords refer to blade LE and TE."""
    if machine.get("type") not in ("axial", "radial"):
        raise ValueError("machine.type must be axial or radial.")
    if type(machine.get("num_blades")) is not int or machine["num_blades"] < 2:
        raise ValueError("machine.num_blades must be an integer >= 2.")
    names = (
        ("hub_radius", "blade_height", "hub_chord", "tip_chord")
        if machine["type"] == "axial"
        else ("inlet_radius", "outlet_radius", "blade_height")
    )
    for name in names:
        if not np.isfinite(machine[name]) or machine[name] <= 0:
            raise ValueError(f"machine.{name} must be finite and positive [mm].")
    if (
        machine["type"] == "radial"
        and machine["inlet_radius"] == machine["outlet_radius"]
    ):
        raise ValueError("Radial inlet and outlet radii must differ.")


@dataclass(frozen=True)
class GeometryDefinition:
    """Hub/shroud B-splines in (r, z), with blade LE at u=0 and TE at u=1.

    Corresponding curve stations are interpolated from hub (eta=0) to shroud
    (eta=1). This defines the stream surfaces and blade meridional placement.
    Rotation is about global Z. Radius and axial coordinates are in mm.
    """

    hub: BSpline
    shroud: BSpline
    num_blades: int

    def to_dict(self):
        """Record the common geometry definition alongside the exported mesh."""
        return {
            "num_blades": self.num_blades,
            "hub": {
                "knots": self.hub.t.tolist(),
                "control_points_rz_mm": self.hub.c.tolist(),
                "degree": self.hub.k,
            },
            "shroud": {
                "knots": self.shroud.t.tolist(),
                "control_points_rz_mm": self.shroud.c.tolist(),
                "degree": self.shroud.k,
            },
        }


def meridional_line(start, end):
    """Represent a straight meridional wall segment as a normalized B-spline."""
    return BSpline([0, 0, 1, 1], np.asarray([start, end], dtype=float), 1)


def axial_geometry(machine):
    """Convert cylindrical walls and hub/tip chords to meridional curves."""
    hub, tip = machine["hub_radius"], machine["hub_radius"] + machine["blade_height"]
    return GeometryDefinition(
        hub=meridional_line((hub, 0), (hub, machine["hub_chord"])),
        shroud=meridional_line((tip, 0), (tip, machine["tip_chord"])),
        num_blades=machine["num_blades"],
    )


def radial_geometry(machine):
    """Convert inlet/outlet radii and parallel endwalls to meridional curves."""
    inlet, outlet, height = (
        machine["inlet_radius"],
        machine["outlet_radius"],
        machine["blade_height"],
    )
    return GeometryDefinition(
        hub=meridional_line((inlet, 0), (outlet, 0)),
        shroud=meridional_line((inlet, height), (outlet, height)),
        num_blades=machine["num_blades"],
    )


def geometry_from_machine(machine):
    """Preprocess convenience YAML dimensions; the core receives only curves."""
    validate_machine(machine)
    return {"axial": axial_geometry, "radial": radial_geometry}[machine["type"]](
        machine
    )


# =============================================================================
# 2. General stream surfaces: meridional curves -> m' -> physical XYZ
# =============================================================================


class Channel:
    """Map a common hub/shroud definition without axial/radial specialization.

    P(u, eta) = (1-eta)*hub(u) + eta*shroud(u), with P = (r, z).
    m'(u, eta) = integral_0^u |dP/du|/r du. Blade sections share the MoC
    template and are scaled by their conformal LE-to-TE length. Extensions
    follow wall endpoint tangents beyond u=0 and u=1.
    """

    def __init__(self, definition):
        if not isinstance(definition, GeometryDefinition):
            raise TypeError(
                "Channel requires a GeometryDefinition, not machine inputs."
            )
        if type(definition.num_blades) is not int or definition.num_blades < 2:
            raise ValueError("The geometry must specify an integer blade count >= 2.")
        self.definition = definition
        self.num_blades = definition.num_blades
        self.pitch_angle = 2 * np.pi / self.num_blades
        self._curves = (definition.hub, definition.shroud)
        for curve in self._curves:
            if (
                not isinstance(curve, BSpline)
                or curve.c.ndim != 2
                or curve.c.shape[1] != 2
                or not np.isfinite(curve.c).all()
                or curve.t[curve.k] != 0
                or curve.t[-curve.k - 1] != 1
            ):
                raise ValueError(
                    "Hub/shroud must be finite (r, z) B-splines on [0, 1]."
                )
        self._derivatives = tuple(curve.derivative() for curve in self._curves)
        self._straight = all(
            curve.k == 1 and len(curve.c) == 2 for curve in self._curves
        )
        self._chords = {}

        # Curve orientation, rather than a machine label, sets cell winding.
        stations = np.linspace(0, 1, 33)
        orientations = []
        for eta in np.linspace(0, 1, 5):
            points = self.meridional_points(stations, eta)
            tangent = self.meridional_points(stations, eta, derivative=True)
            across = definition.shroud(stations) - definition.hub(stations)
            determinant = tangent[:, 0] * across[:, 1] - tangent[:, 1] * across[:, 0]
            if np.any(points[:, 0] <= 0) or np.any(
                np.linalg.norm(tangent, axis=1) == 0
            ):
                raise ValueError(
                    "Meridional curves require positive radius and nonzero tangents."
                )
            if np.any(
                np.abs(determinant)
                <= 1e-12
                * np.linalg.norm(tangent, axis=1)
                * np.linalg.norm(across, axis=1)
            ):
                raise ValueError("Hub/shroud curves define a degenerate channel.")
            orientations.extend(np.sign(determinant))
        if not np.all(np.asarray(orientations) == orientations[0]):
            raise ValueError("Hub/shroud station correspondence folds the channel.")
        self.orientation = -int(orientations[0])

        # The midspan radius at half the conformal chord reproduces the previous
        # axial midspan radius and radial geometric-mean radius automatically.
        prime = self.prime_chord(0.5)
        midpoint = self.parameter_from_prime(np.asarray(prime / 2), 0.5)
        self.reference_radius = float(self.meridional_points(midpoint, 0.5)[0])
        self.pitch = self.reference_radius * self.pitch_angle
        self.reference_chord = self.reference_radius * prime

    def meridional_points(self, u, eta, *, derivative=False):
        """Interpolate wall curves; extend them linearly along endpoint tangents."""
        u = np.asarray(u, dtype=float)
        clipped = np.clip(u, 0, 1)
        values = []
        for curve, tangent in zip(self._curves, self._derivatives):
            values.append(
                tangent(clipped)
                if derivative
                else curve(clipped) + (u - clipped)[..., None] * tangent(clipped)
            )
        return (1 - eta) * values[0] + eta * values[1]

    def prime_chord(self, eta):
        """Integrate dimensionless meridional distance between blade LE and TE."""
        if eta not in self._chords:
            if self._straight:
                start, end = self.meridional_points(np.array([0, 1]), eta)
                dr = end[0] - start[0]
                speed = np.linalg.norm(end - start)
                value = (
                    speed / start[0]
                    if dr == 0
                    else speed * np.log1p(dr / start[0]) / dr
                )
            else:

                def integrand(u):
                    return (
                        np.linalg.norm(self.meridional_points(u, eta, derivative=True))
                        / self.meridional_points(u, eta)[0]
                    )

                value = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-11)[0]
            self._chords[eta] = float(value)
        return self._chords[eta]

    def parameter_from_prime(self, prime, eta):
        """Invert the conformal integral using the same formula for every channel."""
        prime = np.asarray(prime, dtype=float)
        if self._straight:
            start, end = self.meridional_points(np.array([0, 1]), eta)
            dr = end[0] - start[0]
            speed = np.linalg.norm(end - start)
            return (
                prime * start[0] / speed
                if dr == 0
                else start[0] * np.expm1(prime * dr / speed) / dr
            )

        # du/dm' = r/|dP/du|. Dense ODE output evaluates every mesh node without
        # per-node root solves, including upstream and downstream extensions.
        def derivative(_, u):
            point = self.meridional_points(u[0], eta)
            tangent = self.meridional_points(u[0], eta, derivative=True)
            return [point[0] / np.linalg.norm(tangent)]

        result = np.zeros_like(prime)
        for mask in (prime < 0, prime > 0):
            if not np.any(mask):
                continue
            end = float(
                prime[mask].min() if np.any(prime[mask] < 0) else prime[mask].max()
            )
            solution = solve_ivp(
                derivative,
                (0, end),
                [0],
                method="DOP853",
                dense_output=True,
                rtol=1e-12,
                atol=1e-13,
            )
            if not solution.success:
                raise ValueError("Could not invert the meridional conformal mapping.")
            result[mask] = solution.sol(prime[mask])[0]
        return result

    def section_scale(self, eta):
        """Isotropic scale in the conformal plane, preserving section angles."""
        return self.prime_chord(eta) / self.prime_chord(0.5)

    def map_points(self, q, eta):
        """Map conformal nodes onto an interpolated surface of revolution."""
        q = np.asarray(q, dtype=float)
        theta = -q[..., 0] / self.reference_radius
        prime = -q[..., 1] / self.reference_radius
        u = self.parameter_from_prime(prime, eta)
        points = self.meridional_points(u, eta)
        radius, axial = points[..., 0], points[..., 1]
        if not np.isfinite(points).all() or np.any(radius <= 0):
            raise ValueError(
                "The mapped domain requires finite coordinates and positive radii."
            )
        return np.stack(
            (radius * np.cos(theta), radius * np.sin(theta), axial), axis=-1
        )

    def rotation(self):
        """Homogeneous transform from periodic_left to periodic_right."""
        c, s = np.cos(-self.pitch_angle), np.sin(-self.pitch_angle)
        result = np.eye(4)
        result[:2, :2] = [[c, -s], [s, c]]
        return result


# =============================================================================
# 3. MoC section fitting and pitch-controlled, smoothly extended passage
# =============================================================================


def scale_blade(blade, scale, origin):
    """Apply one similarity transform to the fitted profile and its metadata."""
    result = deepcopy(blade)
    for value in result.values():
        if isinstance(value, dict) and {"x", "y"} <= value.keys():
            points = (np.column_stack((value["x"], value["y"])) - origin) * scale
            value["x"], value["y"] = points.T.tolist()
    result["control_points"] = (np.asarray(result["control_points"]) - origin) * scale
    for name in (
        "pitch",
        "throat_opening",
        "inlet_opening",
        "axial_chord_convergent",
        "axial_chord_divergent",
        "r_trailing",
    ):
        result[name] *= scale
    return result


def build_section(nozzle, config, channel, eta=0.5, *, template=None):
    """Fit the MoC blade and center a blade-count-based passage.

    Both coordinates share a scale; independent 0–1 normalization is avoided.
    Conformal chord fraction xi=-q_y/conformal_chord is 0–1 over the blade.
    The wall-curve station u is obtained separately by the inverse map. Pitch comes
    from N and Rref, independent of the nozzle's original design pitch.
    """
    if not nozzle.get("converged"):
        raise ValueError("A converged MoC nozzle solution is required.")
    wall = nozzle["wall_final"]
    raw = (
        parametrize_stator_blade(wall["x"], wall["y"], **config["blade"])
        if template is None
        else deepcopy(template)
    )
    contour = geo2.blade_contour(raw)
    origin = contour[np.argmax(contour[:, 1])]
    chord = np.ptp(contour[:, 1])
    scale = channel.reference_radius * channel.prime_chord(eta) / chord
    # Alter passage pitch, not the already fitted blade coordinates.
    raw["pitch"] = channel.pitch / scale
    passage = config["passage"]
    domain = geo2.generate_flow_domain(
        raw,
        wall,
        inlet_chords=passage["inlet_chords"],
        outlet_chords=passage["outlet_chords"],
        smooth=passage.get("smooth", {}),
        domain_shift_mode=passage.get("domain_shift_mode", "auto"),
        domain_shift=passage.get("domain_shift", 0),
    )
    curves = {}
    for name, value in domain.items():
        if isinstance(value, BSpline):
            curves[name] = BSpline(value.t, (value.c - origin) * scale, value.k)
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2:
            curves[name] = (value - origin) * scale
        else:
            curves[name] = value
    curves["shift_mm"] = domain["shift_mm"] * scale
    blade = scale_blade(raw, scale, origin)
    clearance = geo2.assess_clearance(blade, curves)
    if not clearance["contained"]:
        raise ValueError(
            "Blade does not fit the selected pitch. Reduce num_blades or chord."
        )
    curves["blade_curve"] = geo2.xy(blade["blade_curve"])
    curves["trailing_edge"] = geo2.xy(blade["trailing_edge"])
    return blade, curves


# =============================================================================
# 4. Display tessellation of the geometry (no Gmsh or computational mesh)
# =============================================================================


def preview_geometry(nozzle, config, *, span_sections=7, curve_points=180):
    """Sample the same fitted blade and passage as the mesher for browser viewing.

    Polygon tessellation represents curved surfaces for display only; no
    inflation, core mesh or volume cells are generated. Coordinates are not
    cached on disk. The MoC template is fitted once per geometry update.
    """
    from vtkmodules.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    channel = Channel(geometry_from_machine(config["machine"]))
    wall = nozzle["wall_final"]
    template = parametrize_stator_blade(wall["x"], wall["y"], **config["blade"])
    blade, reference = build_section(nozzle, config, channel, template=template)

    def sample(points, count, *, closed=False):
        """Sample a polyline at equal arc length, retaining its endpoints."""
        points = np.asarray(points)
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        stations = np.linspace(0, distance[-1], count, endpoint=not closed)
        return np.column_stack(
            [np.interp(stations, distance, points[:, axis]) for axis in (0, 1)]
        )

    def section_loops(section_blade, curves):
        loops = {
            "blade": sample(
                geo2.blade_contour(section_blade), curve_points, closed=True
            ),
            "periodic_left": sample(curves["periodic_left"], curve_points),
            "periodic_right": sample(curves["periodic_right"], curve_points),
            "inlet": sample(curves["inlet"], 32),
            "outlet": sample(curves["outlet"], 32),
        }
        loops["outer"] = np.vstack(
            (
                loops["periodic_left"],
                loops["outlet"][1:],
                loops["periodic_right"][-2::-1],
                loops["inlet"][-2:0:-1],
            )
        )
        return loops

    stations = np.linspace(0, 1, span_sections)
    sections = []
    reference_loops = section_loops(blade, reference)
    for eta in stations:
        if np.isclose(channel.section_scale(eta), 1, rtol=1e-12, atol=1e-12):
            # Untapered sections share the same conformal outline. Only their
            # mapping into physical space changes across the span.
            sections.append(reference_loops)
        else:
            section_blade, curves = build_section(
                nozzle, config, channel, eta, template=template
            )
            sections.append(section_loops(section_blade, curves))
    surfaces = {}
    for name in ("blade", "periodic_left", "periodic_right", "inlet", "outlet"):
        strips = np.array(
            [
                channel.map_points(section[name], eta)
                for section, eta in zip(sections, stations)
            ]
        )
        width = strips.shape[1]
        edge = np.arange(width if name == "blade" else width - 1)
        following = (edge + 1) % width
        cells = np.array(
            [
                np.column_stack(
                    (
                        edge + layer * width,
                        following + layer * width,
                        following + (layer + 1) * width,
                        edge + (layer + 1) * width,
                    )
                )
                for layer in range(span_sections - 1)
            ]
        ).reshape(-1, 4)
        surfaces[name] = {
            "points": strips.reshape(-1).tolist(),
            "polys": np.column_stack((np.full(len(cells), 4), cells)).ravel().tolist(),
        }

    for name, section, eta in (("hub", sections[0], 0), ("shroud", sections[-1], 1)):
        outer, inner = section["outer"], section["blade"]

        def signed_area(points):
            next_point = np.roll(points, -1, axis=0)
            return np.sum(
                points[:, 0] * next_point[:, 1] - points[:, 1] * next_point[:, 0]
            )

        if signed_area(outer) * signed_area(inner) > 0:
            inner = inner[::-1]
        points = np.vstack((outer, inner))
        loops = (np.arange(len(outer)), np.arange(len(inner)) + len(outer))
        lines = np.concatenate(
            [np.r_[len(loop) + 1, loop, loop[0]] for loop in loops]
        ).astype(np.int64)
        vtk_points = vtkPoints()
        vtk_points.SetData(
            numpy_to_vtk(np.column_stack((points, np.zeros(len(points)))), deep=True)
        )
        vtk_lines = vtkCellArray()
        vtk_lines.SetCells(2, numpy_to_vtkIdTypeArray(lines, deep=True))
        polygon = vtkPolyData()
        polygon.SetPoints(vtk_points)
        polygon.SetLines(vtk_lines)
        triangulator = vtkContourTriangulator()
        triangulator.SetInputData(polygon)
        triangulator.Update()
        if triangulator.GetTriangulationError():
            raise ValueError("Could not tessellate the endwall geometry for display.")
        triangles = triangulator.GetOutput()
        q = vtk_to_numpy(triangles.GetPoints().GetData())[:, :2]
        surfaces[name] = {
            "points": channel.map_points(q, eta).ravel().tolist(),
            "polys": vtk_to_numpy(triangles.GetPolys().GetData()).tolist(),
        }

    section = {
        name: (reference[name] / channel.reference_radius).tolist()
        for name in (
            "periodic_left",
            "periodic_right",
            "inlet",
            "outlet",
            "passage_reference",
            "inlet_extension",
            "outlet_extension",
        )
    }
    section["blade"] = (geo2.blade_contour(blade) / channel.reference_radius).tolist()
    return {
        "section": section,
        "surfaces": surfaces,
        "num_blades": channel.num_blades,
        "pitch_angle": channel.pitch_angle,
        "reference_radius_mm": channel.reference_radius,
    }


# =============================================================================
# 5. Span stations and corresponding outer-boundary positions
# =============================================================================


def span_stations(numerics):
    """Use uniform or symmetric tanh clustering at the two endwalls."""
    count, stretch = numerics["span_layers"], numerics["span_stretch"]
    if type(count) is not int or count < 1:
        raise ValueError("mesh.numerics.span_layers must be an integer >= 1.")
    if not np.isfinite(stretch) or not 0 <= stretch <= 8:
        raise ValueError("mesh.numerics.span_stretch must be between 0 and 8.")
    t = np.linspace(0, 1, count + 1)
    return (
        t
        if stretch < 1e-8
        else (1 + np.tanh(stretch * (2 * t - 1)) / np.tanh(stretch)) / 2
    )


def corresponding_boundary(points, name, reference, target):
    """Preserve meridional station and cap pitch fraction during core deformation."""
    points = np.asarray(points)
    ref, new = reference[name], target[name]
    if name in ("inlet", "outlet"):
        fraction = (points[:, 0] - ref[0, 0]) / (ref[-1, 0] - ref[0, 0])
        return new[0] + fraction[:, None] * (new[-1] - new[0])
    lo, hi = reference["outlet"][0, 1], reference["inlet"][0, 1]
    fraction = (points[:, 1] - lo) / (hi - lo)
    y = target["outlet"][0, 1] + fraction * (
        target["inlet"][0, 1] - target["outlet"][0, 1]
    )
    x = np.interp(y, new[::-1, 1], new[::-1, 0])
    return np.column_stack((x, y))
