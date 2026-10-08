"""Design and mesh a YAML-driven stator with periodic boundaries and inflation.

From the repository root::

    poetry run python examples/gmsh/run_mesh_generation.py

Edit the settings below and the accompanying YAML. The nozzle solution is
cached between runs; the blade, passage, and mesh are rebuilt each time.
See readme.md for the construction details.
"""

import hashlib
from importlib.metadata import version
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import turbo_moc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.gmsh.geometry_source import (
    build_geometry as construct_geometry,
    plot_construction as create_geometry_figure,
    validate_flow_domain_settings,
    validate_domain_shift,
    assess_clearance,
)
from examples.gmsh.meshing_source import create_mesh, validate_mesh_settings
from examples.workflow_logging import (
    display_path,
    get_logger,
    heading,
    step,
    summary,
    workflow_logging,
)

logger = get_logger("gmsh.runner")


# =============================================================================
# 1. User settings
# Paths are relative to this script, not the working directory.
# =============================================================================


EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EXAMPLE_DIR / "stator_mesh.yaml"
OUTPUT_DIR = EXAMPLE_DIR / "output"
MOC_CACHE_FILE = OUTPUT_DIR / "nozzle_solution.json"
USE_MOC_CACHE = True
FORCE_MOC_SOLVE = False  # Set True to replace the cached nozzle solution.
SAVE_FIGURES = True
GENERATE_MESH = True  # Mesh the same smooth CAD domain shown in the figure.
SHOW_FIGURES = False
SHOW_GMSH = True
FIGURE_SIZE = (14, 8)
FIGURE_DPI = 180
FIGURE_FORMATS = ("svg", "png")
LOG_LEVEL = "INFO"  # DEBUG also reports detailed export diagnostics.
LOG_FILENAME = "mesh_generation.log"  # Replaced on every run, like mesh outputs.


# =============================================================================
# 2. MoC cache: signature, validation, and solve/load policy
# Cache nozzle results only. Blade geometry and meshes are regenerated each run.
# =============================================================================


def nozzle_signature(parameters):
    """Invalidate saved results when inputs, solver code, or dependencies change."""
    package_dir = Path(turbo_moc.__file__).resolve().parent
    solver_files = [package_dir / "api.py"]
    solver_files += sorted((package_dir / "core").rglob("*.py"))
    solver_files += sorted((package_dir / "solvers").rglob("*.py"))
    source_hash = hashlib.sha256()
    for source_file in solver_files:
        source_hash.update(source_file.relative_to(package_dir).as_posix().encode())
        source_hash.update(source_file.read_bytes())
    return {
        "cache_format": 1,
        "nozzle_parameters": parameters,
        "solver_source": source_hash.hexdigest(),
        "versions": {
            name: version(name)
            for name in (
                "turbo_moc",
                "jax",
                "jaxlib",
                "jaxprop",
                "CoolProp",
                "numpy",
                "scipy",
            )
        },
    }


def usable_nozzle(result):
    """Only reuse converged results with a finite, nonempty nozzle wall."""
    try:
        wall = result["wall_final"]
        points = np.column_stack((wall["x"], wall["y"]))
        return bool(
            result["converged"] and len(points) >= 2 and np.isfinite(points).all()
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_or_solve_nozzle(parameters, cache_file, use_cache=True, force_solve=False):
    """Load matching nozzle data; otherwise solve and replace the JSON cache."""
    signature = nozzle_signature(parameters) if use_cache else None
    if use_cache and not force_solve and cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(saved, dict)
                and saved.get("signature") == signature
                and usable_nozzle(saved.get("result"))
            ):
                logger.info(
                    "  Reusing cached MoC solution: %s", display_path(cache_file)
                )
                return saved["result"]
            logger.info(
                "  Cached solution does not match the current design; solving again."
            )
        except (OSError, ValueError) as exc:
            logger.warning("Could not read the MoC cache; solving again: %s", exc)

    logger.info("  Calculating a new MoC solution (solver output follows).")
    result = turbo_moc.design_nozzle(**parameters)
    if not usable_nozzle(result):
        raise RuntimeError("The nozzle design did not converge to a usable wall.")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = cache_file.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps({"signature": signature, "result": result}), encoding="utf-8"
        )
        temporary_file.replace(cache_file)
        logger.info("  MoC solution saved: %s", display_path(cache_file))
    return result


# =============================================================================
# 3. Configuration and geometry helpers
# =============================================================================


def read_configuration(config_file=CONFIG_FILE):
    """Read the nozzle, blade, and passage parameters from YAML."""
    with Path(config_file).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {
        "nozzle",
        "blade",
        "passage",
        "mesh",
    }:
        raise ValueError(
            "Configuration must contain nozzle, blade, passage, and mesh mappings."
        )
    if not all(isinstance(value, dict) for value in config.values()):
        raise ValueError("Each configuration section must be a mapping.")
    passage = config["passage"]
    for key in ("inlet_chords", "outlet_chords"):
        if not np.isfinite(passage[key]) or passage[key] <= 0:
            raise ValueError(f"passage.{key} must be finite and positive.")
    validate_domain_shift(
        passage.get("domain_shift_mode", "manual"), passage.get("domain_shift", 0.0)
    )
    validate_flow_domain_settings(passage.get("smooth", {}))

    validate_mesh_settings(config["mesh"])
    return config


def get_nozzle(nozzle_parameters):
    """Use the current cache settings to obtain the nozzle solution."""
    return load_or_solve_nozzle(
        nozzle_parameters,
        MOC_CACHE_FILE,
        use_cache=USE_MOC_CACHE,
        force_solve=FORCE_MOC_SOLVE,
    )


def build_geometry(nozzle, config):
    """Build the blade and passage using the shared geometry module."""
    return construct_geometry(nozzle, config)


def plot_construction(blade, curves):
    """Draw the construction using the figure settings above."""
    return create_geometry_figure(blade, curves, figure_size=FIGURE_SIZE)


def save_figures(figure):
    """Render and export each configured figure format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        figure.savefig(
            OUTPUT_DIR / f"stator_geometry.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        logger.info(
            "  Geometry figure saved: %s",
            display_path(OUTPUT_DIR / f"stator_geometry.{extension}"),
        )


# =============================================================================
# 4. Run mesh generation
# =============================================================================


def main():
    """Report the mesh workflow to the terminal and one persistent log."""
    log_file = OUTPUT_DIR / LOG_FILENAME
    total_steps = 4 if GENERATE_MESH else 3
    with workflow_logging(log_file, LOG_LEVEL):
        heading(logger, "Gmsh stator mesh workflow")
        summary(
            logger,
            "Run settings",
            {
                "Configuration": display_path(CONFIG_FILE),
                "Output directory": display_path(OUTPUT_DIR),
                "Execution log": display_path(log_file),
                "Mode": "Geometry and mesh" if GENERATE_MESH else "Geometry only",
            },
        )
        step(logger, 1, total_steps, "Read and validate configuration")
        config = read_configuration()
        logger.info("  Configuration validated.")

        step(logger, 2, total_steps, "Load or calculate the MoC design")
        nozzle = get_nozzle(config["nozzle"])

        step(logger, 3, total_steps, "Build blade geometry and periodic passage")
        blade, curves = build_geometry(nozzle, config)
        summary(
            logger,
            "Geometry",
            {
                "Stator pitch": f"{blade['pitch']:.4f} mm",
                "Domain shift method": curves["domain_shift_mode"],
                "Domain shift": f"{curves['shift_mm']:+.4f} mm ({curves['domain_shift']:+.5f} pitch)",
            },
        )
        figure, _ = plot_construction(blade, curves)
        try:
            if SAVE_FIGURES:
                save_figures(figure)
            if SHOW_FIGURES:
                logger.info("  Opening the geometry viewer; close it to continue.")
                plt.show()
        finally:
            plt.close(figure)

        if GENERATE_MESH:
            step(logger, 4, total_steps, "Generate, validate and export the mesh")
            if not assess_clearance(blade, curves)["contained"]:
                raise ValueError(
                    "Cannot mesh: the selected domain shift places a periodic boundary inside the blade."
                )
            create_mesh(blade, curves, config["mesh"], OUTPUT_DIR, show_gui=SHOW_GMSH)

        heading(logger, "Results")
        summary(
            logger,
            "Mesh workflow completed",
            {
                "Status": "Completed",
                "Output directory": display_path(OUTPUT_DIR),
                "Execution log": display_path(log_file),
            },
        )


if __name__ == "__main__":
    main()
