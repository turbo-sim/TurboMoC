"""Stator passage geometry: blade construction, periodic passage placement, plotting.

Sections: input validation; spline construction; clearance and placement;
blade assembly; Matplotlib views. Gmsh operations live in stator_mesh_2d.py.

Coordinates are millimetres: x is pitchwise, y is axial, flow toward -y.
The reference is the integrated convergent metal-angle law followed by the
straight planar MoC symmetry-axis direction, translated to the fitted LE.
It is a construction camberline, not a recovered mean-thickness blade line.
The joins are tangent-continuous (G1); curvature continuity is not imposed.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import minimize_scalar

import turbo_moc
from turbo_moc.geometry import parametrize_stator_blade


# =============================================================================
# 1. Geometry defaults and input validation
# All sampled curves and spline control points use millimetres.
# =============================================================================


DEFAULTS = {
    "inlet_axial_tangent": 0.35,
    "inlet_blade_tangent": 0.35,
    "outlet_blade_tangent": 0.35,
    "outlet_axial_tangent": 0.35,
    "num_points": 200,
}


def validate_flow_domain_settings(settings):
    """Validate dimensionless handle lengths and sampling resolution."""
    options = DEFAULTS | settings
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown passage.smooth settings: {sorted(unknown)}")
    for key in DEFAULTS.keys() - {"num_points"}:
        value = options[key]
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"passage.smooth.{key} must be in (0, 1).")
    count = options["num_points"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("passage.smooth.num_points must be an integer >= 3.")
    return options


def metal_direction(angle):
    """Unit downstream tangent for a signed metal angle in degrees."""
    beta = np.deg2rad(angle)
    if not np.isfinite(beta) or abs(beta) >= np.pi / 2:
        raise ValueError("Metal angles must be finite and strictly between -90 and 90 degrees.")
    return np.array([np.sin(beta), -np.cos(beta)])


def validate_domain_shift(mode, shift):
    """Manual shift is a signed fraction of pitch; auto ignores its value."""
    if mode not in ("manual", "auto"):
        raise ValueError("passage.domain_shift_mode must be 'manual' or 'auto'.")
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not np.isfinite(shift):
        raise ValueError("passage.domain_shift must be a finite number (fraction of pitch).")


# =============================================================================
# 2. Passage splines and rigid translation
# Build the reference curve first; periodic sides are separated by one pitch.
# =============================================================================


def apply_domain_shift(blade, domain, domain_shift):
    """Set an absolute manual shift without accumulating slider movements.

    Returns a new dictionary; the input arrays and splines remain unchanged.
    Use this function on a cached domain for inexpensive GUI slider updates.
    """
    validate_domain_shift("manual", domain_shift)
    previous = domain.get("domain_shift", 0.)
    delta = (domain_shift - previous) * blade["pitch"]
    result = dict(domain)
    for name in ("passage_reference", "periodic_left", "periodic_right", "inlet", "outlet",
                 "convergent", "divergent", "inlet_extension", "outlet_extension",
                 "inlet_control_points", "outlet_control_points"):
        if name in domain:
            result[name] = domain[name] + [delta, 0.]
    for name in ("inlet_spline", "convergent_spline", "outlet_spline"):
        if name in domain:
            spline = domain[name]
            result[name] = BSpline(spline.t.copy(), spline.c + [delta, 0.], spline.k)
    result.pop("optimizer_success", None)
    result.update(domain_shift_mode="manual", domain_shift=float(domain_shift),
                  shift_mm=float(domain_shift * blade["pitch"]))
    return result


def cubic_blend(start, end, tangent_start, tangent_end,
                length_start, length_end, num_points=200):
    """Return samples, control points, and a clamped degree-3 B-spline.

    The four-control-point spline is also a cubic Bezier curve. Endpoint
    derivatives are 3*(P1-P0) and 3*(P3-P2), so handle lengths control the
    rate of turning independently at each end. Lengths are in mm.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    directions = np.asarray([tangent_start, tangent_end], float)
    norms = np.linalg.norm(directions, axis=1)
    if (start.shape != (2,) or end.shape != (2,) or directions.shape != (2, 2)
            or not np.isfinite([start, end, *directions]).all()
            or np.any(norms == 0)):
        raise ValueError("Blend endpoints and nonzero tangents must be finite 2D vectors.")
    if not np.isfinite([length_start, length_end]).all() or min(length_start, length_end) <= 0:
        raise ValueError("Blend handle lengths must be finite and positive.")
    directions /= norms[:, None]
    controls = np.array([start, start + length_start * directions[0],
                         end - length_end * directions[1], end])
    # Descending control-point y guarantees a downstream, non-folding curve.
    if np.any(np.diff(controls[:, 1]) > 0):
        raise ValueError("Blend handles overlap axially; reduce the tangent proportions.")
    spline = BSpline([0, 0, 0, 0, 1, 1, 1, 1], controls, 3)
    return spline(np.linspace(0, 1, num_points)), controls, spline


def blade_reference_camberline(blade, wall=None, num_points=200):
    """Return convergent and divergent references anchored at the fitted LE.

    Integrate d(eta)/d(xi)=tan(beta(xi)), with beta linear from inlet to
    throat, exactly rather than reusing the sparsely sampled suction side.
    The MoC symmetry axis is straight and rotated by the outlet metal angle.
    After anchoring at the fitted LE, continue this direction to the blade's
    downstream axial extremum, so the exit blend begins beyond the blade.
    `wall` (the MoC nozzle wall, metres) is only sanity-checked, so a blade
    dict alone (e.g. an already sized blade) is enough when it is None.
    """
    contour = np.column_stack((blade["blade_curve"]["x"], blade["blade_curve"]["y"]))
    leading_edge = contour[np.argmax(contour[:, 1])].copy()
    beta_in, beta_out = np.deg2rad([blade["metal_angle_in"], blade["metal_angle_out"]])
    metal_direction(blade["metal_angle_in"])
    direction_out = metal_direction(blade["metal_angle_out"])
    length = blade["axial_chord_convergent"]
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Convergent axial chord must be finite and positive.")
    if wall is not None:
        wall_length = np.ptp(np.asarray(wall["x"], float)) * 1000
        if not np.isfinite(wall_length) or wall_length <= 0:
            raise ValueError("MoC divergent length must be finite and positive.")
    xi = np.linspace(0, length, num_points)
    rate = (beta_out - beta_in) / length
    if abs(beta_out - beta_in) < 1e-10:
        eta = xi * np.tan(beta_in)
    else:
        eta = (np.log(np.cos(beta_in)) - np.log(np.cos(beta_in + rate * xi))) / rate
    convergent = leading_edge + np.column_stack((eta, -xi))
    trailing = np.column_stack((blade["trailing_edge"]["x"], blade["trailing_edge"]["y"]))
    trailing_y = min(contour[:, 1].min(), trailing[:, 1].min())
    divergent_length = (convergent[-1, 1] - trailing_y) / (-direction_out[1])
    if divergent_length <= 0:
        raise ValueError("Blade trailing edge must be downstream of the convergent reference.")
    divergent = convergent[-1] + np.linspace(0, divergent_length, num_points)[:, None] * direction_out
    return convergent, divergent


def generate_flow_domain(blade, wall=None, *, inlet_chords, outlet_chords, smooth=None,
                         domain_shift_mode="manual", domain_shift=0.):
    """Build the extended reference, +/- half-pitch periodics, and flat caps.

    Cap axial positions use the contour extrema and nominal axial chord.
    Far-field pitchwise positions assume half of each
    extension's axial travel occurs at the blade angle. Handle lengths are
    proportions of the respective extension's axial length, not its arc length.
    """
    validate_domain_shift(domain_shift_mode, domain_shift)
    options = validate_flow_domain_settings(smooth or {})
    if not np.isfinite([inlet_chords, outlet_chords, blade["pitch"]]).all() or min(
            inlet_chords, outlet_chords, blade["pitch"]) <= 0:
        raise ValueError("Passage extents and pitch must be finite and positive.")
    count = options["num_points"]
    convergent, divergent = blade_reference_camberline(blade, wall, count)
    # A cubic interpolant represents the integrated reference in CAD, with
    # exact metal-angle endpoint tangents. The plotting samples lie on it.
    length = blade["axial_chord_convergent"]
    convergent_spline = make_interp_spline(
        np.linspace(0, 1, count), convergent, k=3,
        bc_type=([(1, [length * np.tan(np.deg2rad(blade["metal_angle_in"])), -length])],
                 [(1, [length * np.tan(np.deg2rad(blade["metal_angle_out"])), -length])]),
    )
    contour = np.concatenate([np.column_stack((blade[key]["x"], blade[key]["y"]))
                              for key in ("blade_curve", "trailing_edge")])
    chord = blade["axial_chord_convergent"] + blade["axial_chord_divergent"]
    inlet_y = contour[:, 1].max() + inlet_chords * chord
    outlet_y = contour[:, 1].min() - outlet_chords * chord
    upstream = inlet_y - convergent[0, 1]
    downstream = divergent[-1, 1] - outlet_y
    if min(upstream, downstream) <= 0:
        raise ValueError("Inlet/outlet caps must lie beyond the reference camberline.")
    direction_in = metal_direction(blade["metal_angle_in"])
    direction_out = metal_direction(blade["metal_angle_out"])
    axial = np.array([0., -1.])
    inlet_point = convergent[0] + [-0.5 * upstream * np.tan(np.deg2rad(blade["metal_angle_in"])), upstream]
    outlet_point = divergent[-1] + [0.5 * downstream * np.tan(np.deg2rad(blade["metal_angle_out"])), -downstream]
    inlet, inlet_cp, inlet_spline = cubic_blend(
        inlet_point, convergent[0], axial, direction_in,
        upstream * options["inlet_axial_tangent"],
        upstream * options["inlet_blade_tangent"], count)
    outlet, outlet_cp, outlet_spline = cubic_blend(
        divergent[-1], outlet_point, direction_out, axial,
        downstream * options["outlet_blade_tangent"],
        downstream * options["outlet_axial_tangent"], count)
    reference = np.vstack((inlet, convergent[1:], divergent[1:], outlet[1:]))
    domain = _domain(reference, blade["pitch"]) | {
        "convergent": convergent, "divergent": divergent,
        "inlet_extension": inlet, "outlet_extension": outlet,
        "inlet_control_points": inlet_cp, "outlet_control_points": outlet_cp,
        "inlet_spline": inlet_spline, "outlet_spline": outlet_spline,
        "convergent_spline": convergent_spline,
    }
    if domain_shift_mode == "auto":
        return auto_center_flow_domain(blade, domain)
    return apply_domain_shift(blade, domain, domain_shift)


# =============================================================================
# 3. Geometric distances, containment, and automatic placement
# These checks operate on sampled geometry independently of Gmsh.
# =============================================================================


def blade_contour(blade):
    """Return the closed fitted blade, including its trailing-edge arc."""
    body = np.column_stack((blade["blade_curve"]["x"], blade["blade_curve"]["y"]))
    arc = np.column_stack((blade["trailing_edge"]["x"], blade["trailing_edge"]["y"]))
    # Match the arc orientation to the end of the fitted body.
    if np.linalg.norm(body[-1] - arc[-1]) < np.linalg.norm(body[-1] - arc[0]):
        arc = arc[::-1]
    points = np.vstack((body, arc, body[:1]))
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
    if len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("A finite closed blade contour is required.")
    return points


def _positive(value, name, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")


def _count(value, name, minimum=20):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def resample_curve(points, count):
    """Uniform arclength samples of a polyline."""
    points = np.asarray(points, float)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-12]
    stations = np.linspace(0, lengths[-1], count)
    return np.column_stack([np.interp(stations, lengths[keep], points[keep, axis])
                            for axis in (0, 1)])


def point_segment_distances(points, polyline):
    """Shortest distance and nearest point on polyline for every input point."""
    starts, edges = polyline[:-1], np.diff(polyline, axis=0)
    length2 = np.sum(edges * edges, axis=1)
    valid = length2 > 1e-20
    starts, edges, length2 = starts[valid], edges[valid], length2[valid]
    delta = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(np.einsum("ijk,jk->ij", delta, edges) / length2, 0, 1)
    nearest = starts[None, :, :] + fraction[:, :, None] * edges[None, :, :]
    distance2 = np.sum((points[:, None, :] - nearest) ** 2, axis=2)
    index = np.argmin(distance2, axis=1)
    return np.sqrt(distance2[np.arange(len(points)), index]), nearest[np.arange(len(points)), index]


def _pair_clearance(blade_points, boundary):
    """Check both sampling directions to resolve endpoint/interior minima."""
    first, first_nearest = point_segment_distances(blade_points, boundary)
    second, second_nearest = point_segment_distances(boundary, blade_points)
    i, j = np.argmin(first), np.argmin(second)
    if first[i] <= second[j]:
        return float(first[i]), np.array([blade_points[i], first_nearest[i]])
    return float(second[j]), np.array([second_nearest[j], boundary[j]])


def _domain(reference, pitch):
    left = reference - [pitch / 2, 0.]
    right = reference + [pitch / 2, 0.]
    return {"passage_reference": reference, "periodic_left": left, "periodic_right": right,
            "inlet": np.array([left[0], right[0]]), "outlet": np.array([left[-1], right[-1]])}


def assess_clearance(blade, domain, *, clearance_samples=600,
                     inflation_thickness=0., transition_clearance=0.):
    """Report Euclidean gaps, containment, and protected-envelope feasibility.

    The envelope is the contour dilated by inflation + transition distance.
    Comparing boundary-to-contour distances avoids unreliable normal offsets
    around the rounded LE. Neighbor overlap is checked against blade + pitch.
    """
    _count(clearance_samples, "clearance_samples")
    _positive(inflation_thickness, "inflation_thickness", zero=True)
    _positive(transition_clearance, "transition_clearance", zero=True)
    contour = resample_curve(blade_contour(blade), clearance_samples)
    reference = domain["passage_reference"][::-1]
    center_x = np.interp(contour[:, 1], reference[:, 1], reference[:, 0])
    horizontal_margin = blade["pitch"] / 2 - np.abs(contour[:, 0] - center_x)
    result = {}
    for side in ("left", "right"):
        boundary = resample_curve(domain[f"periodic_{side}"], clearance_samples)
        gap, pair = _pair_clearance(contour, boundary)
        result[f"{side}_clearance"] = gap
        result[f"{side}_closest_pair"] = pair
    required = inflation_thickness + transition_clearance
    neighbor_gap, _ = _pair_clearance(contour, contour + [blade["pitch"], 0.])
    result.update(
        minimum_clearance=min(result["left_clearance"], result["right_clearance"]),
        contained=bool(np.min(horizontal_margin) > 0),
        required_clearance=required, neighbor_gap=neighbor_gap,
        neighbor_envelopes_overlap=bool(neighbor_gap < 2 * required),
    )
    result["target_met"] = result["contained"] and result["minimum_clearance"] >= required - 1e-6
    return result


def auto_center_flow_domain(blade, baseline, *, max_shift_pitch=0.5, shift_tolerance=0.001,
                            clearance_samples=200):
    """Maximize the smaller Euclidean gap with one rigid x translation.

    max_shift_pitch bounds |shift| / pitch. shift_tolerance is in mm.
    A coarse search brackets a local bounded solve; it is not a global proof.
    """
    _positive(max_shift_pitch, "max_shift_pitch")
    _positive(shift_tolerance, "shift_tolerance")
    _count(clearance_samples, "clearance_samples")
    baseline = apply_domain_shift(blade, baseline, 0.)
    contour = blade_contour(blade)
    reference = np.asarray(baseline["passage_reference"])
    pitch = blade["pitch"]
    ref_x = np.interp(contour[:, 1], reference[::-1, 1], reference[::-1, 0])
    lower = max(np.max(contour[:, 0] - ref_x - pitch / 2) + 1e-6, -max_shift_pitch * pitch)
    upper = min(np.min(contour[:, 0] - ref_x + pitch / 2) - 1e-6, max_shift_pitch * pitch)
    if lower >= upper:
        raise ValueError("No rigid shift within the configured range contains the blade.")
    boundary = resample_curve(reference, clearance_samples)

    def score(shift):
        return min(_pair_clearance(contour, boundary + [shift - pitch / 2, 0.])[0],
                   _pair_clearance(contour, boundary + [shift + pitch / 2, 0.])[0])

    grid = np.linspace(lower, upper, 31)
    best = int(np.argmax([score(shift) for shift in grid]))
    fit = minimize_scalar(lambda shift: -score(shift), method="bounded",
                          bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]),
                          options={"xatol": shift_tolerance})
    candidates = [grid[best], fit.x]
    if lower <= 0 <= upper:
        candidates.append(0.)
    shift = max(candidates, key=score)
    result = apply_domain_shift(blade, baseline, shift / pitch)
    result.update(domain_shift_mode="auto", optimizer_success=bool(fit.success))
    return result
# =============================================================================
# 4. Blade and passage assembly
# Regenerate the fitted blade and construction curves from a nozzle solution.
# =============================================================================


def xy(surface):
    """Convert a package surface dictionary to an (N, 2) array in mm."""
    return np.column_stack((surface["x"], surface["y"]))


def build_geometry(nozzle, config):
    """Rebuild the blade and periodic passage from the current parameters."""
    wall = nozzle["wall_final"]
    if not nozzle["converged"] or len(wall["x"]) < 2:
        raise RuntimeError("The nozzle design did not converge to a usable wall.")
    blade = parametrize_stator_blade(wall["x"], wall["y"], **config["blade"])
    passage = config["passage"]
    flow_domain = generate_flow_domain(
        blade, wall, inlet_chords=passage["inlet_chords"],
        outlet_chords=passage["outlet_chords"], smooth=passage.get("smooth", {}),
        domain_shift_mode=passage.get("domain_shift_mode", "manual"),
        domain_shift=passage.get("domain_shift", 0.),
    )

    # The full parametrization concatenates the integrated convergent curve,
    # the MoC wall, and a three-point trailing-edge connector into suction.
    # Extract that actual construction curve rather than integrating it again.
    # It is called 'camberline' internally, but forms the convergent suction
    # surface, not the mean-thickness line of the final fitted solid blade.
    n_convergent = len(blade["suction"]["x"]) - len(wall["x"]) - 3
    if n_convergent < 2:
        raise RuntimeError("Unexpected stator suction-surface layout.")
    curves = flow_domain | {
        "blade_curve": xy(blade["blade_curve"]),
        "trailing_edge": xy(blade["trailing_edge"]),
        "suction": xy(blade["suction"]),
        "pressure": xy(blade["pressure"]),
        "control_points": np.asarray(blade["control_points"]),
        "construction_camberline": xy(blade["suction"])[:n_convergent],
    }
    if not all(np.isfinite(points).all() for points in curves.values() if isinstance(points, np.ndarray)):
        raise RuntimeError("The stator geometry contains non-finite coordinates.")
    return blade, curves


# =============================================================================
# 5. Matplotlib construction views
# Reuse the package blade plot and overlay passage and construction features.
# =============================================================================


def plot_flow_domain(ax, domain):
    """Plot boundaries, reference sections, and extension control polygons."""
    for key in ("periodic_left", "periodic_right"):
        ax.plot(*domain[key].T, color="tab:purple", lw=1.5,
                label="Smooth periodic boundaries" if key.endswith("left") else None)
    for key, color in (("inlet", "tab:green"), ("outlet", "tab:orange")):
        ax.plot(*domain[key].T, color=color, lw=2, label=key.capitalize())
    for key, color, label in (
        ("convergent", "tab:red", "Integrated convergent reference"),
        ("divergent", "tab:blue", "MoC axis direction"),
        ("inlet_extension", "0.4", "Cubic B-spline extensions"),
        ("outlet_extension", "0.4", None),
    ):
        ax.plot(*domain[key].T, color=color, ls="--", lw=1.5, label=label)
    for key in ("inlet_control_points", "outlet_control_points"):
        ax.plot(*domain[key].T, ":o", color="0.6", ms=4,
                label="Extension control polygon" if key.startswith("inlet") else None)
    ax.plot(*domain["convergent"][0], "o", color="tab:red", ms=5, label="Reference start")


def plot_construction(blade, curves, *, figure_size=(14, 8)):
    """Overlay construction features on the package's Matplotlib blade plots."""
    fig, axes = plt.subplots(1, 2, figsize=figure_size, layout="constrained")
    detail, domain = axes
    for ax in axes:
        turbo_moc.mpl.plot_blade(
            blade, ax=ax, n_blades=1,
            show_control_points=ax is detail, show_cp_labels=False,
        )
        ax.set_xlabel("Pitchwise x (mm)")
        ax.set_ylabel("Axial y (mm); flow toward decreasing y")
        for spine in ax.spines.values():
            spine.set_visible(True)

    for name, color in (("suction", "tab:green"), ("pressure", "tab:orange")):
        detail.plot(*curves[name].T, color=color, lw=1, label=f"Raw {name} surface")
    detail.plot([], [], "--o", color=turbo_moc.mpl.COLOR_CP_LINE,
                mfc=turbo_moc.mpl.COLOR_CP_FACE, mec="black", ms=6,
                label="B-spline control polygon")
    camber = curves["construction_camberline"]
    detail.plot(*camber.T, color="tab:red", lw=3, ls="--",
                label="Construction camberline (convergent suction)")
    detail.plot(*camber[[0, -1]].T, "o", color="tab:red", ms=5)
    detail.set_title("Blade construction")

    draw_domain(blade, curves, domain)
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
        ax.margins(x=0.22, y=0.12)
    return fig, axes


def draw_domain(blade, curves, ax):
    """Draw the selected domain and report numerical side clearances."""
    plot_flow_domain(ax, curves)
    report = assess_clearance(blade, curves, clearance_samples=400)
    status = "" if report["contained"] else "\nBlade crosses a periodic boundary"
    ax.set_title(
        f"Smooth passage ({curves['domain_shift_mode']}); shift = {curves['shift_mm']:+.2f} mm\n"
        f"Clearance left/right: {report['left_clearance']:.2f} / {report['right_clearance']:.2f} mm{status}",
        color="black" if report["contained"] else "tab:red", fontsize=10,
    )
