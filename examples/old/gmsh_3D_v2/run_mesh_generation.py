"""Run the same V2 stages without the browser (no native viewer windows)."""
import argparse
from pathlib import Path
import shutil
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
from examples import workflow_logging as log
from examples.gmsh_3D_v2.app import workflow as flow, geometry_source as geo, meshing_source as mesh
from turbo_moc.progress import make_progress_reporter


def main(config_file=None, output_dir=None):
    config_file = Path(config_file or flow.CASES["axial_stator"])
    config = flow.read_configuration(config_file)
    directory = Path(output_dir or flow.HERE / "output" / config_file.stem)
    directory.mkdir(parents=True, exist_ok=True)
    timings = {}
    with log.workflow_logging(directory / "mesh_generation.log", config["workflow"]["log_level"]):
        logger = log.get_logger("gmsh_3D_v2.runner")
        def progress(label, fraction=None):
            logger.info("%s%s", label, "" if fraction is None else f" [{100 * fraction:.0f}%]")
        shared = flow.ROOT / "examples/gmsh_3D/output/nozzle_solution.json"
        cache = directory / "nozzle_solution.json"
        if not cache.exists() and shared.exists():
            shutil.copy2(shared, cache)  # The loader still validates every input/signature.
        reporter = make_progress_reporter(lambda event: progress(event[1], float(event[0][:-1]) / 100))
        nozzle = flow.load_or_solve_nozzle(config["method_of_characteristics"], cache, progress_callback=reporter)
        start = perf_counter()
        design = geo.build_design(nozzle, config)
        timings["geometry_s"] = perf_counter() - start
        reference = mesh.reference_mesh(design, config, directory, progress=progress)
        section, design = mesh.load_section(directory)
        volume = mesh.volume_mesh(section, design, config, directory, progress=progress)
        timings.update(reference_s=reference["elapsed_s"], volume_s=volume["elapsed_s"])
        if config["mesh"]["cgns"]["enabled"]:
            start = perf_counter()
            mesh.export_mesh(directory, config, directory, progress=progress)
            timings["export_s"] = perf_counter() - start
        flow.write_json(directory / "stage_timings.json", timings)
        logger.info("Stage timings: %s", timings)
        return volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=flow.CASES["axial_stator"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.config, args.output)

