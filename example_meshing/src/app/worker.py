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
from ..meshing import meshing_3d as mesh
from ..meshing.inspection import inspect_mesh
from ..geometry.moc import solve_nozzle
from turbo_moc.progress import make_progress_reporter


def run(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    folder, config, refs = request_path.parent, request["config"], request["references"]
    handle = request["handle"]
    row_id = handle.get("row_id", "stator_1")
    # Imported numerical helpers deliberately do not install handlers. The
    # worker owns reporting, including their structured INFO diagnostics.
    logger = logging.getLogger("stator_workflow")
    logger.setLevel(config["workflow"]["log_level"])
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(f"[{row_id}] %(message)s"))
    logger.addHandler(stream_handler)
    logger.propagate = False

    def progress(message, fraction=None):
        message = f"[{row_id}] {message}"
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
        flow.write_json(folder / "status.json", {"state": "running", "message": message,
                        "fraction": fraction, "started": handle["started"], "row_id": row_id})

    try:
        if request.get("source_revision", flow.SOURCE_REVISION) != flow.SOURCE_REVISION:
            raise ValueError("Application source changed. Restart the studio before running jobs.")
        if handle.get("row_role", "stator") != "stator":
            raise ValueError("Rotor geometry and meshing integration is pending; rotor jobs are unavailable.")
        for reference in refs.values():
            if reference and reference.get("row_id", "stator_1") != row_id:
                raise ValueError("A blade-row job cannot use artifacts from another row.")
        action = handle["action"]
        progress(f"Run {action} from configuration snapshot")
        if action == "nozzle":
            reporter = make_progress_reporter(lambda event:
                progress(f"MoC: {event[1]}", float(event[0].rstrip("%")) / 100))
            nozzle = solve_nozzle(config["method_of_characteristics"], progress_callback=reporter)
            # This is the output of this explicit computation, passed to later
            # workers. It is never searched for or reused by a new MoC job.
            flow.write_json(folder / "moc_result.json", nozzle)
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
        elif action == "export_2d":
            result = {"export": mesh.export_reference_mesh(flow.directory(refs["reference"]), config,
                                                            folder, progress=progress), "reference": handle}
        else:
            raise ValueError(f"Unknown action {action}")
        flow.write_json(folder / "result.json", result)
        flow.write_json(folder / "status.json", {"state": "complete", "message": f"[{row_id}] {action.title()} ready",
                        "fraction": 1.0, "started": handle["started"], "row_id": row_id})
        print(f"[{row_id}] Job completed successfully.", flush=True)
    except Exception as exc:
        traceback.print_exc()
        flow.write_json(folder / "status.json", {"state": "failed", "message": f"[{row_id}] {exc}",
                        "fraction": None, "started": handle["started"], "row_id": row_id})
        return 1
    finally:
        logger.removeHandler(stream_handler)
        stream_handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))

