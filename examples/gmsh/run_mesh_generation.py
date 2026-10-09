"""Design and mesh a YAML-driven stator with periodic boundaries and inflation.

From the repository root::

    poetry run python examples/gmsh/run_mesh_generation.py

Edit the settings below and the accompanying YAML. The nozzle solution is
cached between runs; the blade, passage, and mesh are rebuilt each time.
See readme.md for the construction details.
"""

import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import barotropy
import matplotlib.pyplot as plt
import numpy as np
import yaml

import turbo_moc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import examples.gmsh.geometry_source as geo
import examples.gmsh.meshing_source as msh
import examples.workflow_logging as log

logger = log.get_logger("gmsh.runner")


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
FIGURE_DPI = 300
FIGURE_FORMATS = (".png", ".svg")
LOG_LEVEL = "INFO"  # DEBUG also reports detailed export diagnostics.
LOG_FILENAME = "mesh_generation.log"  # Replaced on every run, like mesh outputs.


# =============================================================================
# 2. MoC cache: signature, validation, and solve/load policy
# Cache nozzle results only. Blade geometry and meshes are regenerated each run.
# =============================================================================


def load_or_solve_nozzle(
    parameters, cache_file, use_cache=True, force_solve=False, *, cache_only=False
):
    """Load or solve a validated MoC design; cache_only returns None on a cache miss."""

    def cache_signature():
        # Solver inputs, source files and dependency versions identify a result.
        # Blade, passage and plotting settings deliberately stay out of the key.
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
                name: metadata.version(name)
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

    def usable_result(result):
        # Never reuse or save an unsuccessful design or a non-finite nozzle wall.
        try:
            wall = result["wall_final"]
            points = np.column_stack((wall["x"], wall["y"]))
            return bool(
                result["converged"] and len(points) >= 2 and np.isfinite(points).all()
            )
        except (KeyError, TypeError, ValueError):
            return False

    def read_cached_result(signature):
        if not cache_file.exists():
            return None
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(saved, dict)
                and saved.get("signature") == signature
                and usable_result(saved.get("result"))
            ):
                logger.info(
                    "  Reusing cached MoC solution: %s", log.display_path(cache_file)
                )
                return saved["result"]
            logger.info(
                "  Cached solution does not match the current design; solving again."
            )
        except (OSError, ValueError) as exc:
            logger.warning("Could not read the MoC cache; solving again: %s", exc)
        return None

    def write_cached_result(signature, result):
        # Replace the cache only after a complete write, preserving a previous
        # good result if solving or writing the temporary file fails.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = cache_file.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps({"signature": signature, "result": result}), encoding="utf-8"
        )
        temporary_file.replace(cache_file)
        logger.info("  MoC solution saved: %s", log.display_path(cache_file))

    cache_file = Path(cache_file)
    signature = cache_signature() if use_cache else None
    if use_cache and not force_solve:
        result = read_cached_result(signature)
        if result is not None:
            return result

    if cache_only:
        return None

    logger.info("  Calculating a new MoC solution (solver output follows).")
    result = turbo_moc.design_nozzle(**parameters)
    if not usable_result(result):
        raise RuntimeError("The nozzle design did not converge to a usable wall.")
    if use_cache:
        write_cached_result(signature, result)
    return result


# =============================================================================
# 3. YAML configuration
# =============================================================================


def read_configuration(config_file=CONFIG_FILE):
    """Read and validate the nozzle, blade, passage and mesh sections of the YAML."""
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
    geo.validate_domain_shift(
        passage.get("domain_shift_mode", "manual"), passage.get("domain_shift", 0.0)
    )
    geo.validate_flow_domain_settings(passage.get("smooth", {}))

    msh.validate_mesh_settings(config["mesh"])
    return config


# =============================================================================
# 4. Run mesh generation
# =============================================================================


def main():
    """Read the design, obtain the nozzle wall, then rebuild and mesh one stator passage."""
    log_file = OUTPUT_DIR / LOG_FILENAME
    total_steps = 4 if GENERATE_MESH else 3

    # The logging context sends progress to both the console and a fresh log.
    # A separate plotting context restores Matplotlib settings when this run ends.
    with log.workflow_logging(log_file, LOG_LEVEL), plt.rc_context():
        # Apply barotropy's publication style before creating any figure.
        # Keep a complete axes box, and use PNG/SVG for the construction exports.
        barotropy.set_plot_options()
        plt.rcParams.update(
            {
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.spines.bottom": True,
                "axes.spines.left": True,
                "savefig.dpi": FIGURE_DPI,
                "savefig.bbox": "tight",
                "svg.fonttype": "none",
            }
        )

        log.heading(logger, "Gmsh stator mesh workflow")
        log.summary(
            logger,
            "Run settings",
            {
                "Configuration": log.display_path(CONFIG_FILE),
                "Output directory": log.display_path(OUTPUT_DIR),
                "Execution log": log.display_path(log_file),
                "Mode": "Geometry and mesh" if GENERATE_MESH else "Geometry only",
            },
        )

        # 1. Read the four YAML sections and reject invalid inputs before solving.
        # Nozzle inputs are in metres; blade, passage and mesh lengths are in mm.
        log.step(logger, 1, total_steps, "Read and validate configuration")
        config = read_configuration()
        logger.info("  Configuration validated.")

        # 2. The MoC calculation supplies a nozzle wall for the blade construction.
        # Only this expensive nozzle solution is cached. Changing blade or mesh
        # settings therefore reuses the nozzle while rebuilding all later stages.
        log.step(logger, 2, total_steps, "Load or calculate the MoC design")
        nozzle = load_or_solve_nozzle(
            config["nozzle"],
            MOC_CACHE_FILE,
            use_cache=USE_MOC_CACHE,
            force_solve=FORCE_MOC_SOLVE,
        )

        # 3. Fit the blade and construct the surrounding periodic flow passage.
        # The figure shows these construction curves and the domain we will mesh.
        log.step(logger, 3, total_steps, "Build blade geometry and periodic passage")
        blade, curves = geo.build_geometry(nozzle, config)
        log.summary(
            logger,
            "Geometry",
            {
                "Stator pitch": f"{blade['pitch']:.4f} mm",
                "Domain shift method": curves["domain_shift_mode"],
                "Domain shift": f"{curves['shift_mm']:+.4f} mm ({curves['domain_shift']:+.5f} pitch)",
            },
        )
        # Avoid constructing a figure when neither export nor viewing is wanted.
        if SAVE_FIGURES or SHOW_FIGURES:
            figure, _ = geo.plot_construction(blade, curves, figure_size=FIGURE_SIZE)
            try:
                if SAVE_FIGURES:
                    barotropy.savefig_in_formats(
                        figure,
                        OUTPUT_DIR / "stator_geometry",
                        formats=FIGURE_FORMATS,
                    )
                if SHOW_FIGURES:
                    logger.info("  Opening the geometry viewer; close it to continue.")
                    plt.show()
            finally:
                plt.close(figure)

        # 4. Check that the blade fits inside the passage before transferring the
        # fitted splines to Gmsh. The meshing module handles inflation, periodic
        # constraints, mesh figures, CGNS export and the optional quality report.
        # Set GENERATE_MESH=False to stop after inspecting the geometry instead.
        if GENERATE_MESH:
            log.step(logger, 4, total_steps, "Generate, validate and export the mesh")
            if not geo.assess_clearance(blade, curves)["contained"]:
                raise ValueError(
                    "Cannot mesh: the selected domain shift places a periodic boundary inside the blade."
                )
            msh.create_mesh(
                blade, curves, config["mesh"], OUTPUT_DIR, show_gui=SHOW_GMSH
            )

        log.heading(logger, "Results")
        log.summary(
            logger,
            "Mesh workflow completed",
            {
                "Status": "Completed",
                "Output directory": log.display_path(OUTPUT_DIR),
                "Execution log": log.display_path(log_file),
            },
        )


if __name__ == "__main__":
    main()
