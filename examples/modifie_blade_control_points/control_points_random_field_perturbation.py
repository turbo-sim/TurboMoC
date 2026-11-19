"""Generate deterministic control-point and KL perturbation samples.

The control-point samples are normalized values in ``[0, 1]``. They are
mapped to signed normal displacements bounded by plus/minus five percent of
the blade pitch, then applied to the YAML-defined baseline blade. A KL profile
perturbation is applied to every control-point sample for the second plot.
"""
from copy import deepcopy
from pathlib import Path
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from turbo_moc.configuration.planar import from_project
from turbo_moc.configuration.planar import build_geometry
from turbo_moc.geometry import karhunen_loeve
from turbo_moc.geometry.karhunen_loeve import (
    build_kl_expansion, define_perturbed_blade,
)
from turbo_moc.geometry.stator_blade import (
    apply_control_point_displacements,
    edit_blade_geometry,
    find_side_intersections,
    parametrize_stator,
)
from turbo_moc.workflows.nozzle import solve_nozzle

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "blade_design_baseline.yaml"
FIGURES = HERE / "figures"
GENERATED_BLADES = HERE / "generated_blades"
CP_SAMPLES = 5
CP_RANDOM_SEED = 42
KL_RANDOM_SEED = 90
CP_PERTURBED_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 25, 26, 27]
KL_PARAMETERS = {
    "sigma0": 0.8,
    "sigma1": 0.2,
    "cp_te_pressure": 1e-3,
    "cp_te_suction": 0.999,
    "L0": 0.1,
    "L_le": 1e-2,
    "omega": 0.1,
    "energy_threshold": 0.95,
    "n_quad": 500,
}

def _baseline_config(config):
    return {
        key: value
        for key, value in config["blade"].items()
        if key not in {"control_point_displacements", "control_point_offsets", "geometry_uncertainty"}
    }

def _normalized_cp_samples(count, size, seed):
    if count < 1:
        raise ValueError("count must be positive.")
    if count == 1:
        return np.full((1, size), 0.5)
    generator = np.random.default_rng(seed)
    samples = np.empty((count, size), dtype=float)
    samples[0] = 0.5
    samples[1:] = generator.random((count - 1, size))
    return samples


def  _apply_cp_displacements(blade, normalized_displacements):
    displacements = {
        index: values.tolist()
        for index, values in enumerate(np.asarray(normalized_displacements, dtype=float))
        if np.any(values)
    }
    return apply_control_point_displacements(blade, displacements)


def _save_generated_blades(project, cp_displacements, kl_samples, output):
    """Save complete blade-design project YAML files for all CP+KL samples."""
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, (cp_values, kl_result) in enumerate(
        zip(cp_displacements, kl_samples),
        start=1,
    ):
        coefficients = np.asarray(kl_result["coefficients"])
        if coefficients.shape[0] != 1:
            raise ValueError("Each project YAML must contain exactly one KL realization.")

        sample_project = deepcopy(project)
        row = next(
            row for row in sample_project["rows"]
            if row["id"] == sample_project["row_id"]
        )
        blade_geometry = row["geometry"]["blade_geometry"]
        blade_geometry.pop("geometry_uncertainty", None)

        cp_values = np.asarray(cp_values, dtype=float)
        if cp_values.ndim != 2 or cp_values.shape[1] != 2:
            raise ValueError("Control-point displacements must contain [dx, dy] rows.")
        if not np.isfinite(cp_values).all():
            raise ValueError("Control-point displacements must contain finite values.")
        cp_displacement_entries = {
            str(control_point_index): value.tolist()
            for control_point_index, value in enumerate(cp_values)
            if np.any(value)
        }
        if cp_displacement_entries:
            blade_geometry["control_point_displacements"] = cp_displacement_entries
        blade_geometry["geometry_uncertainty"] = {
            "Xi": [float(value) for value in coefficients[0]],
            **KL_PARAMETERS,
        }

        path = output / f"case_{index:02d}.yaml"
        path.write_text(
            yaml.safe_dump(
                sample_project,
                sort_keys=False,
                allow_unicode=True
            ),
            encoding="utf-8",
        )
        saved.append(path)
    return saved


def _plot_cp_samples(baseline, samples, path):
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(*karhunen_loeve._profile_points(baseline).T, "k--", linewidth=1.5, label="YAML baseline blade")
    colors = plt.get_cmap("viridis", len(samples))
    for index, blade in enumerate(samples, start=1):
        axis.plot(
            *karhunen_loeve._profile_points(blade).T,
            color=colors(index - 1),
            linewidth=1.0,
            linestyle=":",
            label=f"CP sample {index}",
        )
    axis.set_aspect("equal")
    axis.set_xlabel("Pitchwise coordinate [mm]")
    axis.set_ylabel("Axial coordinate [mm]")
    axis.set_title("YAML baseline blade versus control-point perturbation samples")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def _plot_kl_samples(baseline, cp_samples, kl_samples, path):
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(*karhunen_loeve._profile_points(baseline).T, "k--", linewidth=1.6, label="YAML baseline blade")
    cp_colors = plt.get_cmap("viridis", len(cp_samples))
    kl_colors = plt.get_cmap("plasma", len(kl_samples))
    intersection_points = []
    for index, blade in enumerate(cp_samples, start=1):
        axis.plot(
            *karhunen_loeve._profile_points(blade).T,
            color=cp_colors(index - 1),
            linewidth=1.0,
            alpha=0.85,
            linestyle=":",
            label=f"CP sample {index}",
        )
    for index, result in enumerate(kl_samples, start=1):
        for blade in result["blades"]:
            axis.plot(
                *karhunen_loeve._profile_points(blade).T,
                color=kl_colors(index - 1),
                linewidth=0.9,
                label=f"CP + KL sample {index}",
            )
            intersection_points.extend(find_side_intersections(
                np.column_stack((blade["suction"]["x"], blade["suction"]["y"])),
                np.column_stack((blade["pressure"]["x"], blade["pressure"]["y"])),
            ))
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
            label="KL side intersections",
        )
    axis.set_aspect("equal")
    axis.set_xlabel("Pitchwise coordinate [mm]")
    axis.set_ylabel("Axial coordinate [mm]")
    axis.set_title("Control-point samples and KL-perturbed geometries")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def _plot_sample_comparisons(baseline, cp_samples, kl_samples, output):
    """Write one baseline/CP/KL comparison figure for every CP sample."""
    for index, (cp_blade, kl_results) in enumerate(zip(cp_samples, kl_samples), start=1):
        figure, axis = plt.subplots(figsize=(12, 7))
        axis.plot(
            *karhunen_loeve._profile_points(baseline).T,
            "k--",
            linewidth=1.5,
            label="YAML baseline blade",
        )
        axis.plot(
            *karhunen_loeve._profile_points(cp_blade).T,
            color="tab:blue",
            linewidth=1.2,
            linestyle=":",
            label="CP-defined geometry",
        )
        for blade in kl_results["blades"]:
            axis.plot(
                *karhunen_loeve._profile_points(blade).T,
                color="tab:red",
                linewidth=1.0,
                label="CP + KL geometry",
            )
        axis.set_aspect("equal")
        axis.set_xlabel("Pitchwise coordinate [mm]")
        axis.set_ylabel("Axial coordinate [mm]")
        axis.set_title(f"Baseline, CP-defined, and KL-perturbed geometry {index}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = output / f"sample_{index:02d}_comparison"
        figure.savefig(path.with_suffix(".png"), dpi=240)
        figure.savefig(path.with_suffix(".svg"))
        #plt.close(figure)
        plt.show()


def main():
    print(f"Loading blade configuration from {CONFIG.name}")
    with CONFIG.open(encoding="utf-8") as stream:
        project = yaml.safe_load(stream)
    machine_project = {key: project[key] for key in ("machine", "rows", "workflow")}
    config = from_project(machine_project, row_id=project["row_id"])

    print("Solving nozzle")
    nozzle = solve_nozzle(config["nozzle"])
    print("Building YAML-defined baseline blade")
    nominal_config = _baseline_config(config)
    nominal = parametrize_stator(
        nozzle["wall_final"]["x"],
        nozzle["wall_final"]["y"],
        **nominal_config,
    )
    baseline_parameters = {
        key: value
        for key, value in config["blade"].items()
        if key != "geometry_uncertainty"
    }
    baseline = parametrize_stator(
        nozzle["wall_final"]["x"],
        nozzle["wall_final"]["y"],
        **baseline_parameters,
    )

    normalized_samples = _normalized_cp_samples(
        CP_SAMPLES, len(CP_PERTURBED_IDS), CP_RANDOM_SEED
    )
    print(f"Generating {CP_SAMPLES} control-point samples")
    pitch = float(baseline["pitch"])
    bounds = np.tile((-0.015 * pitch, 0.015 * pitch), (len(CP_PERTURBED_IDS), 1))
    cp_samples = []
    cp_displacements = []
    for normalized_values in normalized_samples:
        displacements = edit_blade_geometry(
            nozzle,
            CP_PERTURBED_IDS,
            normalized_values,
            bounds,
            baseline_parameters,
        )
        cp_displacements.append(displacements)
        cp_samples.append(_apply_cp_displacements(nominal, displacements))

    sigma0 = KL_PARAMETERS["sigma0"]
    if "sigma1" in KL_PARAMETERS:
        sigma0 = karhunen_loeve.SigmaProfile(
            start_te_sside=KL_PARAMETERS["cp_te_suction"],
            start_te_pside=KL_PARAMETERS["cp_te_pressure"],
            sigma0 = KL_PARAMETERS["sigma0"],
            sigma1 = KL_PARAMETERS["sigma1"],
        )

    expansion = build_kl_expansion(
        sigma0 = sigma0,
        L0 = KL_PARAMETERS["L0"],
        L_le = KL_PARAMETERS["L_le"],
        omega = KL_PARAMETERS["omega"],
        energy_threshold = KL_PARAMETERS["energy_threshold"],
        n_quad = KL_PARAMETERS["n_quad"],
    )
    xi_coefficients = np.random.default_rng(KL_RANDOM_SEED).standard_normal(
        (CP_SAMPLES, expansion["n_modes"])
    )

    print("Generating KL perturbations")
    kl_samples = [
        define_perturbed_blade(
            expansion,
            blade,
            xi=xi_coefficients[index],
            stop_intersections=True,
        )
        for index, blade in enumerate(cp_samples)
    ]

    plot_baseline = build_geometry(nozzle, config)[0]
    plot_cp_samples = [
        build_geometry(nozzle, config, perturbed_blade=blade)[0]
        for blade in cp_samples
    ]
    for result in kl_samples:
        result["blades"] = [
            build_geometry(nozzle, config, perturbed_blade=blade)[0]
            for blade in result["blades"]
        ]

    print(f"Saving figures to {FIGURES}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_cp_samples(plot_baseline, plot_cp_samples, FIGURES / "baseline_vs_cp_samples")
    _plot_kl_samples(plot_baseline, plot_cp_samples, kl_samples, FIGURES / "cp_and_kl_samples")
    _plot_sample_comparisons(plot_baseline, plot_cp_samples, kl_samples, FIGURES)

    print(f"Saving generated blades to {GENERATED_BLADES}")
    blade_paths = _save_generated_blades(
        project,
        cp_displacements,
        kl_samples,
        GENERATED_BLADES,
    )
    for path in blade_paths:
        print(f"  {path}")
    print("Finished generating all figures and blade YAML files")


if __name__ == "__main__":
    main()
