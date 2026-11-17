"""Generate and plot Karhunen-Loeve manufacturing perturbation samples.

The KL module returns dimensional perturbations on its quadrature grid. They
are interpolated onto the sampled blade curve, divided by the nominal blade
chord, and passed to the shared geometry pipeline as signed chord fractions.
"""
from pathlib import Path

import yaml
from turbo_moc.configuration.planar import from_project
from turbo_moc.geometry.karhunen_loeve import (
    build_kl_expansion,
    define_perturbed_blade,
    plot_geometric_uncertainty,
)
from turbo_moc.geometry.stator_blade import parametrize_stator
from turbo_moc.workflows.nozzle import solve_nozzle

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "blade_design_baseline.yaml"
N_SAMPLES = 10
SIGMA0 = 2
RANDOM_SEED = 42


def main():
    with CONFIG.open(encoding="utf-8") as stream:
        project = yaml.safe_load(stream)
    machine_project = {key: project[key] for key in ("machine", "rows", "workflow")}
    config = from_project(machine_project, row_id=project["row_id"])

    nozzle = solve_nozzle(config["nozzle"])
    wall = nozzle["wall_final"]
    blade_config = {
        key: value
        for key, value in config["blade"].items()
        if key != "geometry_uncertainty"
    }
    nominal = parametrize_stator(wall["x"], wall["y"], **blade_config)

    expansion = build_kl_expansion(
        n_quad=500,
        sigma0=SIGMA0,
        energy_threshold=0.99,
    )
    result = define_perturbed_blade(
        expansion,
        nominal,
        blade_config,
        n_samples=N_SAMPLES,
        random_state=RANDOM_SEED,
    )
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    plot_geometric_uncertainty(result, output=output / "geometric_uncertainty")


if __name__ == "__main__":
    main()
