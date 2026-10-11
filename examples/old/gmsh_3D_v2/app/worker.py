"""One isolated numerical job; status is atomic, stdout is a live log."""
import json
from pathlib import Path
import sys
import time
import traceback
import logging

import matplotlib
matplotlib.use("Agg")

from . import workflow as flow
from . import meshing_source as mesh
from .inspection_source import inspect_mesh
from turbo_moc.progress import make_progress_reporter


def run(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    folder, config, refs = request_path.parent, request["config"], request["references"]
    handle = request["handle"]
    # Imported numerical helpers deliberately do not install handlers. The
    # worker owns reporting, including their structured INFO diagnostics.
    logger = logging.getLogger("stator_workflow")
    logger.setLevel(config["workflow"]["log_level"])
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.propagate = False

    def progress(message, fraction=None):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
        flow.write_json(folder / "status.json", {"state": "running", "message": message,
                        "fraction": fraction, "started": handle["started"]})

    try:
        if request.get("source_revision", flow.SOURCE_REVISION) != flow.SOURCE_REVISION:
            raise ValueError("Application source changed. Restart the studio before running jobs.")
        action = handle["action"]
        progress(f"Run {action} from configuration snapshot")
        if action == "nozzle":
            # Reuse the established cache validator, not an unchecked wall file.
            reporter = make_progress_reporter(lambda event:
                progress(f"MoC: {event[1]}", float(event[0].rstrip("%")) / 100))
            shared = flow.ROOT / "examples/gmsh_3D/output/nozzle_solution.json"
            nozzle = flow.load_or_solve_nozzle(config["method_of_characteristics"], shared, cache_only=True)
            if nozzle is None:
                nozzle = flow.load_or_solve_nozzle(config["method_of_characteristics"], folder / "nozzle_solution.json",
                                                   progress_callback=reporter)
            else:
                import shutil
                shutil.copy2(shared, folder / "nozzle_solution.json")
                progress("Reused validated MoC cache", 1)
            result = {"reference": handle}
        elif action == "reference":
            progress("Fit/reference geometry (one placement search)", 0.02)
            design = flow.get_design(config, refs["nozzle"])
            summary = mesh.reference_mesh(design, config, folder, progress=progress)
            result = {"reference": handle, "summary": summary}
        elif action == "volume":
            section, design = mesh.load_section(flow.directory(refs["reference"]))
            summary = mesh.volume_mesh(section, design, config, folder, progress=progress)
            result = {"reference": handle, "summary": summary, "reference_mesh": refs["reference"]}
        elif action == "inspect":
            progress("Extract actual volume cells")
            result = {"inspection": inspect_mesh(flow.directory(refs["volume"]),
                                                   **request["inspection"]),
                      "volume_job": refs["volume"]["job"]}
        elif action == "export":
            result = {"export": mesh.export_mesh(flow.directory(refs["volume"]), config,
                                                  folder, progress=progress), "reference": handle}
        else:
            raise ValueError(f"Unknown action {action}")
        flow.write_json(folder / "result.json", result)
        flow.write_json(folder / "status.json", {"state": "complete", "message": f"{action.title()} ready",
                        "fraction": 1.0, "started": handle["started"]})
        print("Job completed successfully.", flush=True)
    except Exception as exc:
        traceback.print_exc()
        flow.write_json(folder / "status.json", {"state": "failed", "message": str(exc),
                        "fraction": None, "started": handle["started"]})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))

