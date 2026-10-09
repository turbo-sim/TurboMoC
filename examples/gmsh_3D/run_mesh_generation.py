"""Build a MoC blade passage; choose the case here and its controls in YAML."""

import sys
from pathlib import Path
from time import perf_counter

import barotropy
import matplotlib.pyplot as plt
import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import examples.gmsh.run_mesh_generation as nozzle_workflow
import examples.gmsh.meshing_source as msh2
import examples.gmsh_3D.geometry_source as geo
import examples.gmsh_3D.meshing_source as msh
import examples.workflow_logging as log

# =============================================================================
# 1. User controls: no CLI arguments; outputs stay inside this example directory
# =============================================================================

EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EXAMPLE_DIR / "axial_stator.yaml"  # Or radial_stator.yaml
USE_MOC_CACHE = True
FORCE_MOC_SOLVE = False

logger = log.get_logger("gmsh_3D.runner")


def read_configuration(config_file=CONFIG_FILE):
    """Validate a complete MoC design, no-flare channel and meshing configuration."""
    with Path(config_file).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_configuration(config)


def validate_configuration(config):
    """Validate a YAML-shaped mapping, including browser-edited configurations."""
    if not isinstance(config, dict) or set(config) != {
        "nozzle",
        "blade",
        "passage",
        "machine",
        "mesh",
        "workflow",
    }:
        raise ValueError(
            "YAML requires nozzle, blade, passage, machine, mesh, and workflow mappings."
        )
    if not all(isinstance(value, dict) for value in config.values()):
        raise ValueError("Each top-level section must be a mapping.")
    workflow = config["workflow"]
    for name in ("save_figures", "show_gmsh", "show_pyvista"):
        if type(workflow.get(name)) is not bool:
            raise ValueError(f"workflow.{name} must be true or false.")
    if workflow.get("log_level") not in (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ):
        raise ValueError(
            "workflow.log_level must be DEBUG, INFO, WARNING, ERROR or CRITICAL."
        )
    msh.pyvista_options(config["mesh"].get("pyvista", {}))
    geo.validate_machine(config["machine"])
    geo.span_stations(config["mesh"]["numerics"])
    geo.geo2.validate_flow_domain_settings(config["passage"].get("smooth", {}))
    geo.geo2.validate_domain_shift(
        config["passage"]["domain_shift_mode"], config["passage"]["domain_shift"]
    )
    for name in ("inlet_chords", "outlet_chords"):
        if not np.isfinite(config["passage"][name]) or config["passage"][name] <= 0:
            raise ValueError(f"passage.{name} must be positive.")
    # The common 2D numerical/plot validation applies, but 3D has seven BCs.
    msh2.validate_mesh_settings(
        config["mesh"] | {"cgns": {"enabled": False}, "report": {"enabled": False}}
    )
    cgns = config["mesh"]["cgns"]
    if cgns["enabled"]:
        name = cgns["filename"]
        if Path(name).name != name or Path(name).suffix.lower() != ".cgns":
            raise ValueError("mesh.cgns.filename must be a local .cgns filename.")
        if not np.isfinite(cgns["length_scale"]) or cgns["length_scale"] <= 0:
            raise ValueError("mesh.cgns.length_scale must be positive.")
        if set(cgns["boundary_types"]) != set(msh.BOUNDARIES):
            raise ValueError("CGNS boundary_types must name all seven 3D boundaries.")
        msh.boundary_types(cgns["boundary_types"])
    return config


def main():
    """Obtain the MoC design, regenerate geometry, mesh and export one passage."""
    config = read_configuration(CONFIG_FILE)
    workflow = config["workflow"]
    output = EXAMPLE_DIR / "output" / CONFIG_FILE.stem
    with log.workflow_logging(
        output / "mesh_generation.log", workflow["log_level"]
    ), plt.rc_context():
        barotropy.set_plot_options()
        plt.rcParams.update(
            {f"axes.spines.{side}": True for side in ("left", "right", "top", "bottom")}
        )
        log.heading(logger, "MoC 3D mesh workflow")
        log.step(logger, 1, 3, "Read and validate the case")
        # Translate the convenient axial/radial inputs once. Everything below
        # uses the same hub/shroud geometry definition and mesh generation.
        geometry = geo.geometry_from_machine(config["machine"])

        # Cache only thermodynamic MoC results, never fitted blade coordinates.
        # Reuse the existing cache helper; its signature covers nozzle inputs.
        log.step(logger, 2, 3, "Load or solve the MoC nozzle")
        start = perf_counter()
        nozzle = nozzle_workflow.load_or_solve_nozzle(
            config["nozzle"],
            EXAMPLE_DIR / "output" / "nozzle_solution.json",
            use_cache=USE_MOC_CACHE,
            force_solve=FORCE_MOC_SOLVE,
        )
        logger.info("  MoC design ready in %.2f s.", perf_counter() - start)

        # One 2D topology is mapped to every section; copied blades are visual
        # only. Physical exports always contain exactly one periodic passage.
        log.step(logger, 3, 3, "Generate and export the conforming 3D passage")
        start = perf_counter()
        result = msh.create_mesh(
            nozzle,
            config,
            output,
            geometry=geometry,
            save_figures=workflow["save_figures"],
            show_gmsh=workflow["show_gmsh"],
        )
        logger.info("  3D mesh workflow completed in %.2f s.", perf_counter() - start)
        # Gmsh has released the mesh file before PyVista opens the two-panel
        # viewer. Rendering and interaction use the case's mesh.pyvista controls.
        if workflow["show_pyvista"]:
            logger.info("  Open the one-passage and full-row PyVista views.")
            msh.view_mesh_pyvista(
                output,
                config["mesh"].get("pyvista", {}),
                save_figure=workflow["save_figures"],
            )
        return result


if __name__ == "__main__":
    main()
