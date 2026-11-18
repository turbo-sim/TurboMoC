from typing import Optional

import numpy as np

from turbo_moc.geometry.stator_blade import find_side_intersections, correct_suction_intersections


class SigmaProfile:
    """Smooth standard-deviation profile along the blade."""

    def __init__(self, start_te_pside, start_te_sside, sigma0, sigma1):
        if not 0.0 <= start_te_pside < start_te_sside <= 1.0:
            raise ValueError(
                "Sigma transition endpoints must satisfy 0 <= start_te_pside "
                "< start_te_sside <= 1."
            )
        self.start_te_pside = start_te_pside
        self.start_te_sside = start_te_sside
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)

    def sigma(self, s):
        s = float(s)
        if not 0.0 <= s <= 1.0:
            raise ValueError("s must be between 0 and 1.")
        if s <= self.start_te_pside or s >= self.start_te_sside:
            return self.sigma1
        u = (s - self.start_te_pside) / (self.start_te_sside - self.start_te_pside)
        transition = 2.0 * u if u <= 0.5 else 2.0 * (1.0 - u)
        smoothstep = transition * transition * (3.0 - 2.0 * transition)
        return self.sigma1 + (self.sigma0 - self.sigma1) * smoothstep


def _to_array(x):
    return np.asarray(x, dtype=np.float64)


def _squared_distance(s, t=None):
    """Compute squared Euclidean distances for points in [batch..., N, D]."""
    s = _to_array(s)
    if s.ndim < 2:
        raise ValueError("s must have shape [..., N, D]")
    s_flat = s.reshape(-1, s.shape[-2], s.shape[-1])
    if t is None:
        diff = s_flat[:, :, None, :] - s_flat[:, None, :, :]
        result = np.sum(diff * diff, axis=-1)
        return result[0] if result.shape[0] == 1 and s.ndim == 2 else result
    t = _to_array(t)
    t_flat = t.reshape(-1, t.shape[-2], t.shape[-1])
    diff = s_flat[:, :, None, :] - t_flat[:, None, :, :]
    result = np.sum(diff * diff, axis=-1)
    return result[0] if result.shape[0] == 1 and s.ndim == 2 and t.ndim == 2 else result


def _length_scale(s, L0=0.1, L_le=1e-2, omega=0.1):
    s_array = _to_array(s)
    center = np.full((1, 1), 0.5)
    return L0 + (L_le - L0) * np.exp(
        -_squared_distance(s_array, center) / omega**2
    )


def _sigma_values(points, sigma0):
    values = points.reshape(-1, points.shape[-1])[:, 0]
    if isinstance(sigma0, SigmaProfile):
        return np.asarray([sigma0.sigma(value) for value in values])
    return np.full(values.shape, float(sigma0), dtype=np.float64)


def autocovariance(
    s: np.ndarray,
    t: Optional[np.ndarray] = None,
    sigma0: float | SigmaProfile = 6e-5,
    L0=0.1,
    L_le=1e-2,
    omega=0.1,
):
    """Compute the covariance kernel C(s, t) for the blade random field."""
    s = _to_array(s)
    l_s = _length_scale(s, L0=L0, L_le=L_le, omega=omega)
    sigma_s = _sigma_values(s, sigma0)
    if t is None:
        sq_dist = _squared_distance(s)
        l_s_flat = l_s.reshape(-1, s.shape[-2])
        denominator = l_s_flat[:, :, None] * l_s_flat[:, None, :]
        if denominator.shape[0] == 1 and sq_dist.ndim == 2:
            denominator = denominator[0]
        return sigma_s[:, None] * sigma_s[None, :] * np.exp(-sq_dist / denominator)

    t = _to_array(t)
    l_t = _length_scale(t, L0=L0, L_le=L_le, omega=omega)
    sigma_t = _sigma_values(t, sigma0)
    sq_dist = _squared_distance(s, t)
    l_s_flat = l_s.reshape(-1, s.shape[-2])
    l_t_flat = l_t.reshape(-1, t.shape[-2])
    denominator = l_s_flat[:, :, None] * l_t_flat[:, None, :]
    if denominator.shape[0] == 1 and sq_dist.ndim == 2:
        denominator = denominator[0]
    return sigma_s[:, None] * sigma_t[None, :] * np.exp(-sq_dist / denominator)


def gauss_legendre_grid(n_points=500, lower=0.0, upper=1.0):
    """Return Gauss-Legendre nodes and weights on the interval [lower, upper]."""
    nodes, weights = np.polynomial.legendre.leggauss(int(n_points))
    return (
        0.5 * (upper - lower) * (nodes + 1.0) + lower,
        0.5 * (upper - lower) * weights,
    )


def compute_contour_length(points, closed=True):
    """Compute the length of an ordered polygonal curve."""
    points = _to_array(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if points.shape[0] < 2:
        return 0.0
    if closed:
        points = np.concatenate((points, points[:1]), axis=0)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def compute_blade_half_arclength(points, closed=True):
    """Return the half-arclength of the blade profile from the ordered contour."""
    return 0.5 * compute_contour_length(points, closed=closed)


def build_kl_expansion(
    n_quad=500,
    sigma0: float | SigmaProfile = 6e-5,
    L0=0.1,
    L_le=1e-2,
    omega=0.1,
    energy_threshold=0.99,
    mean=None,
):
    """Build a KL expansion for the non-stationary Gaussian field."""
    s_nodes, weights = gauss_legendre_grid(n_points=n_quad)
    covariance = autocovariance(
        s_nodes[:, None], sigma0=sigma0, L0=L0, L_le=L_le, omega=omega
    )
    sqrt_weights = np.sqrt(weights)
    weighted_covariance = sqrt_weights[:, None] * covariance * sqrt_weights[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total_energy = np.sum(eigenvalues)
    if total_energy <= 0.0:
        raise ValueError("KL covariance has zero total energy.")
    cumulative_energy = np.cumsum(eigenvalues) / total_energy
    n_modes = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
    mean_values = (
        np.zeros_like(s_nodes)
        if mean is None
        else _to_array(mean(s_nodes) if callable(mean) else mean)
    )
    eigenfunctions = eigenvectors[:, :n_modes] / sqrt_weights[:, None]
    return {
        "s_nodes": s_nodes,
        "weights": weights,
        "eigenvalues": eigenvalues[:n_modes],
        "eigenfunctions": eigenfunctions,
        "cumulative_energy": cumulative_energy,
        "mean": mean_values,
        "n_modes": n_modes,
        "sigma0": sigma0,
        "L0": L0,
        "L_le": L_le,
        "omega": omega,
    }


def sample_random_field(expansion, coefficients=None, n_samples=0, random_state=None):
    """Sample realizations of the KL expansion on the evaluation grid."""
    if (coefficients is None) == (n_samples == 0):
        raise ValueError("Either provide coefficients or specify n_samples, but not both.")
    eigenvalues = np.asarray(expansion["eigenvalues"])
    if coefficients is None:
        rng = np.random.default_rng(random_state)
        coefficients = rng.standard_normal((n_samples, eigenvalues.size))
    else:
        coefficients = _to_array(coefficients)
    if coefficients.ndim != 2 or coefficients.shape[1] != eigenvalues.size:
        raise ValueError(
            f"The coefficients must have shape (n_samples, {eigenvalues.size})."
        )
    weighted_modes = np.sqrt(eigenvalues)[:, None] * np.asarray(
        expansion["eigenfunctions"]
    ).T
    samples = coefficients @ weighted_modes
    samples += np.asarray(expansion.get("mean", 0.0))[None, :]
    return samples, coefficients

def _curve_sides(blade, include_trailing_edge=True):
    """Recover the two perturbed walls with the trailing-edge arch attached."""
    curve = np.column_stack((
        blade["blade_curve"]["x"],
        blade["blade_curve"]["y"],
    ))
    suction_count = len(blade["suction"]["x"])
    pressure_count = len(blade["pressure"]["x"])
    split = round(curve.shape[0] * suction_count / (suction_count + pressure_count))
    suction = curve[:split][::-1]
    pressure = curve[split:]
    trailing = blade.get("trailing_edge")
    if include_trailing_edge and trailing is not None:
        trailing_points = np.column_stack((trailing["x"], trailing["y"]))
        if len(trailing_points) >= 2:
            if np.linalg.norm(suction[-1] - trailing_points[0]) > np.linalg.norm(
                suction[-1] - trailing_points[-1]
            ):
                trailing_points = trailing_points[::-1]
            suction = np.vstack((suction, trailing_points[1:]))
    return suction, pressure


def _cut_side_at_x(side, x_limit):
    """Keep a wall prefix and insert the point crossing x_limit."""
    side = np.asarray(side, dtype=float)
    if side.ndim != 2 or side.shape[1] != 2:
        raise ValueError("Blade side must be a two-column point array.")

    if len(side) < 2:
        return side.copy()
    increasing = side[-1, 0] >= side[0, 0]
    retained = []
    previous = side[0].copy()
    for point in side:
        if increasing:
            stays = point[0] <= x_limit
        else:
            stays = point[0] >= x_limit
        if stays:
            retained.append(point.copy())
            previous = point
            continue
        if previous is not None:
            denominator = point[0] - previous[0]
            if abs(denominator) > 1e-15:
                fraction = (x_limit - previous[0]) / denominator
                retained.append(previous + fraction * (point - previous))
        break
    if not retained:
        return np.asarray(side[:1], dtype=float)
    return np.asarray(retained, dtype=float)


def _trim_side_before_crossing(side, crossing):
    """Return the wall prefix ending at the point before the crossing segment."""
    side = np.asarray(side, dtype=float)
    crossing = np.asarray(crossing, dtype=float)
    if len(side) < 2:
        return side.copy()

    closest_segment = 0
    closest_distance = np.inf
    for index in range(len(side) - 1):
        start = side[index]
        direction = side[index + 1] - start
        denominator = float(direction @ direction)
        fraction = 0.0 if denominator <= 1e-18 else float(
            np.clip((crossing - start) @ direction / denominator, 0.0, 1.0)
        )
        projection = start + fraction * direction
        distance = float(np.linalg.norm(crossing - projection))
        if distance < closest_distance:
            closest_distance = distance
            closest_segment = index

    return side[:closest_segment + 1].copy()


def _blade_range_distance(blade):
    """Return the Euclidean range of all points in the blade perimeter."""
    curve = np.column_stack((
        blade["blade_curve"]["x"],
        blade["blade_curve"]["y"],
    ))
    return float(np.linalg.norm(np.ptp(curve, axis=0)))


def _correct_trailing_edge_intersection(blade, suction, pressure, intersections):
    """Cut the main walls at the first crossing and reconnect them to the arch."""
    intersections = np.asarray(intersections, dtype=float)
    if not len(intersections):
        return blade

    x_limit = float(np.min(intersections[:, 0]))
    trailing = np.column_stack((
        blade["trailing_edge"]["x"],
        blade["trailing_edge"]["y"],
    ))
    if len(trailing) < 2:
        return blade

    suction_main = np.asarray(suction, dtype=float)
    pressure_main = np.asarray(pressure, dtype=float)
    first_crossing = intersections[0]
    corrected_suction = _trim_side_before_crossing(suction_main, first_crossing)
    corrected_pressure = _trim_side_before_crossing(pressure_main, first_crossing)
    if len(corrected_suction) < 2 or len(corrected_pressure) < 2:
        return blade

    suction_te_index = int(np.argmin(
        np.linalg.norm(trailing - suction_main[-1], axis=1)
    ))
    pressure_te_index = int(np.argmin(
        np.linalg.norm(trailing - pressure_main[-1], axis=1)
    ))
    suction_te = trailing[suction_te_index]
    pressure_te = trailing[pressure_te_index]
    if suction_te_index > pressure_te_index:
        trailing = trailing[::-1]
        suction_te = trailing[0]
        pressure_te = trailing[-1]
    else:
        suction_te = trailing[0]
        pressure_te = trailing[-1]

    corrected_suction = np.vstack((corrected_suction, suction_te))
    corrected_pressure = np.vstack((corrected_pressure, pressure_te))

    result = dict(blade)
    result["suction"] = {
        "x": corrected_suction[:, 0].tolist(),
        "y": corrected_suction[:, 1].tolist(),
    }
    result["pressure"] = {
        "x": corrected_pressure[:, 0].tolist(),
        "y": corrected_pressure[:, 1].tolist(),
    }
    closed_curve = np.vstack((corrected_suction[::-1], corrected_pressure))
    result["blade_curve"] = {
        "x": closed_curve[:, 0].tolist(),
        "y": closed_curve[:, 1].tolist(),
    }
    return result


def _profile_points(blade):
    curve = np.column_stack((blade["blade_curve"]["x"], blade["blade_curve"]["y"]))
    trailing = np.column_stack((blade["trailing_edge"]["x"], blade["trailing_edge"]["y"]))
    return np.vstack((curve, trailing))


def plot_geometric_uncertainty(
    result,
    output=None,
    show=True,
    figsize=(12, 7),
):
    """Plot KL blade samples, nominal geometry, and residual intersections.

    ``result`` must be the dictionary returned by
    :func:`define_perturbated_blade`. If ``output`` is supplied, PNG and SVG
    files are written using it as the filename stem.
    """
    import matplotlib.pyplot as plt

    if not isinstance(result, dict) or not {"blades", "nominal", "chord"} <= result.keys():
        raise ValueError("result must be returned by define_perturbated_blade.")
    blades = result["blades"]
    nominal = result["nominal"]
    figure, axis = plt.subplots(figsize=figsize)
    sample_colors = plt.get_cmap("turbo", len(blades))
    intersection_points = []
    for index, blade in enumerate(blades, start=1):
        points = _profile_points(blade)
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=sample_colors(index - 1),
            alpha=0.28,
            linewidth=1.6,
        )
        suction, pressure = _curve_sides(blade)
        intersection_points.extend(find_side_intersections(pressure, suction))

    if intersection_points:
        intersections = np.asarray(intersection_points)
        axis.scatter(
            intersections[:, 0],
            intersections[:, 1],
            color="yellow",
            edgecolor="black",
            linewidth=0.5,
            s=24,
            zorder=10,
            label="Profile intersections",
        )
    axis.plot(
        *_profile_points(nominal).T,
        color="purple",
        linewidth=1.8,
        label="Nominal blade",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("Pitchwise coordinate [mm]")
    axis.set_ylabel("Axial coordinate [mm]")
    axis.set_title(
        f"{len(blades)} KL geometry samples "
        f"($\\sigma_0={1.5:g}$, normalized by chord={result['chord']:.3f} mm)"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    if output is not None:
        output = str(output)
        figure.savefig(output + ".png", dpi=240)
        figure.savefig(output + ".svg")
    if show:
        plt.show()
    return figure, axis


def define_perturbed_blade(
    expansion,
    nominal,
    n_samples=1,
    xi=None,
    random_state=None,
    *,
    stop_intersections=True,
    uncertainties_only=False,
):
    """Build blade contours from KL realizations on an arbitrary blade shape.

    ``nominal`` supplies the blade to perturb and may be a nominal or already
    control-point-perturbed blade returned by the stator geometry helpers.
    Supply either ``xi`` coefficient rows or ``n_samples`` to generate
    coefficients through :func:`sample_random_field`.

    Returns a dictionary containing ``points`` (one closed contour per sample),
    ``blades``, the nominal blade, sampled fields, coefficients, chord, and
    ``uncertainties``. Set ``uncertainties_only=True`` to evaluate the KL fields
    and chord-normalized profile uncertainties without applying them to the
    blade geometry.
    """
    from turbo_moc.geometry.stator_blade import apply_geometry_uncertainty, blade_chord

    if xi is not None and n_samples != 1:
        raise ValueError("Provide xi or n_samples, not both.")

    profile = np.column_stack((
        nominal["blade_curve"]["x"],
        nominal["blade_curve"]["y"],
    ))
    distances = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    profile_s = np.concatenate(([0.0], np.cumsum(distances)))
    if profile_s[-1] <= 0.0:
        raise ValueError("The nominal blade curve must have positive arclength.")
    profile_s /= profile_s[-1]
    chord = float(blade_chord(nominal))

    fields, coefficients = sample_random_field(
        expansion,
        coefficients=xi,
        n_samples=0 if xi is not None else n_samples,
        random_state=random_state,
    )
    uncertainties = np.asarray([
        np.interp(profile_s, expansion["s_nodes"], field) / chord
        for field in fields
    ])

    blades = []
    points = []
    if not uncertainties_only:
        for uncertainty in uncertainties:
            blade = apply_geometry_uncertainty(nominal, uncertainty.tolist())
            suction, pressure = _curve_sides(blade)
            intersections = find_side_intersections(suction, pressure)
            if stop_intersections:
                if len(intersections):
                    suction_main, pressure_main = _curve_sides(
                        blade,
                        include_trailing_edge=False,
                    )
                    blade = _correct_trailing_edge_intersection(
                        blade,
                        suction_main,
                        pressure_main,
                        intersections,
                    )
                    suction, pressure = _curve_sides(blade)
                    residual = find_side_intersections(suction, pressure)
                    if len(residual):
                        blade = correct_suction_intersections(
                            blade,
                            residual,
                        )
            blades.append(blade)
            curve = np.column_stack((
                blade["blade_curve"]["x"],
                blade["blade_curve"]["y"],
            ))
            trailing = np.column_stack((
                blade["trailing_edge"]["x"],
                blade["trailing_edge"]["y"],
            ))
            points.append(np.vstack((curve, trailing)))
    return {
        "points": points,
        "blades": blades,
        "nominal": nominal,
        "fields": fields,
        "coefficients": coefficients,
        "uncertainties": uncertainties,
        "chord": chord,
    }
