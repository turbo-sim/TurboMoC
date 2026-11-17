"""Run the bounded adaptive-size 2D Gmsh example."""

from pathlib import Path

from turbo_moc.configuration.planar import build_geometry, read_configuration
from turbo_moc.geometry import passage_2d as geometry
from turbo_moc.meshing.mesh_2d import create_mesh
from turbo_moc.progress import make_progress_reporter
from turbo_moc.workflows.nozzle import solve_nozzle


HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_gmsh_stator_2D.yaml"
OUTPUT_DIR = HERE / "output"


def main():
    """Build the configured geometry and mesh it with bounded sizing."""
    config = read_configuration(CONFIG_FILE)
    nozzle = solve_nozzle(
        config["nozzle"],
        progress_callback=make_progress_reporter(lambda *_: None),
    )
    blade, curves = build_geometry(nozzle, config)
    if not geometry.assess_clearance(blade, curves)["contained"]:
        raise ValueError("The configured passage does not contain the blade.")
    statistics = create_mesh(
        blade,
        curves,
        config["mesh"],
        OUTPUT_DIR,
        show_gui=False,
    )
    element_count = sum(statistics["cells"].values())
    print(f"Generated {element_count:,} 2D elements.")
    return statistics


if __name__ == "__main__":
    main()
