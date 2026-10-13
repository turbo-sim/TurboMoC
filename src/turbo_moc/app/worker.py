"""One isolated numerical job; status is atomic, stdout is a live log."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")

from turbo_moc.app import workflow as flow
from turbo_moc.meshing import mesh_3d as mesh
from turbo_moc.meshing.inspection import inspect_mesh
from turbo_moc.workflows.nozzle import solve_nozzle
from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.progress import make_progress_reporter
from turbo_moc.reporting import logging_context


def run(request_path):
    request_path = Path(request_path).resolve()
    # Requests live at <output>/<session>/<job>/request.json. Workers run
    # independently of Flask and recover the output root from this path.
    flow.OUTPUT = request_path.parents[2]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # Reuse the package's MoC reporting, including its throttled stage details
    # and handler cleanup. Popen captures both these logs and native Gmsh stdout.
    with logging_context(level=request["config"]["workflow"]["log_level"]):
        return _run_job(request_path, request)


def _run_job(request_path, request):
    folder, config, refs = request_path.parent, deepcopy(request["config"]), request["references"]
    # App meshing always produces its Fluent-ready CGNS deliverable.
    config["mesh"]["cgns"]["enabled"] = True
    handle = request["handle"]
    row_id = handle.get("row_id", "stator_1")

    def progress(message, fraction=None):
        message = f"[{row_id}] {message}"
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
        flow.write_json(folder / "status.json", {"state": "running", "message": message,
                        "fraction": fraction, "started": handle["started"], "row_id": row_id})

    def meshing_progress(message, fraction=None):
        progress(message, None if fraction is None else 0.85 * fraction)

    def cgns_progress(message, fraction=None):
        progress(message, None if fraction is None else 0.85 + 0.15 * fraction)

    try:
        if request.get("source_revision", flow.SOURCE_REVISION) != flow.SOURCE_REVISION:
            raise ValueError("Application source changed. Restart the studio before running jobs.")
        role = handle.get("row_role", "stator")
        if role not in ("stator", "rotor") or (role == "rotor") != ("rotor_design" in config):
            raise ValueError("Blade-row role does not match its design configuration.")
        for reference in refs.values():
            if reference and reference.get("row_id", "stator_1") != row_id:
                raise ValueError("A blade-row job cannot use artifacts from another row.")
        action = handle["action"]
        nozzle_only = (request.get("project") or {}).get("machine", {}).get("design_mode") == "nozzle"
        action_name = "vortex blade design" if role == "rotor" and action == "nozzle" else action
        progress(f"Run {action_name} from configuration snapshot")
        if action == "nozzle":
            if role == "rotor":
                from turbo_moc.core.vortex_blade import design_rotor_vortex_blade
                from turbo_moc.app.rotor_inlet import require_supersonic_inlet
                require_supersonic_inlet(config)
                # The protected vortex kernel has no progress callback. Keep
                # an indeterminate indicator while it builds its fluid table
                # and marches the transition arcs, rather than inventing %s.
                progress("Rotor: thermodynamic table and vortex transition arcs")
                from turbo_moc.configuration.rotor import vortex_parameters
                inputs = vortex_parameters(solver_parameters(config["rotor_design"], rotor=True))
                rotor = design_rotor_vortex_blade(**inputs)
                rotor.update(P0_rel=inputs["P0_rel"], T0_rel=inputs["T0_rel"],
                             Q0_rel=inputs.get("Q0_rel", float("nan")))
                flow.write_json(folder / "rotor_result.json", rotor)
                from turbo_moc.app.blade_counts import native_passage, sizing
                natural = sizing(config, native_passage(config, rotor))
                # Publish a successful 2D solve before mapping its passage.
                # Geometry/clearance errors remain editable in the preview;
                # they must not discard the solver's natural pitch.
                progress("Rotor: natural passage sizing", 0.9)
                result = {"reference": handle, "rotor_summary": {
                    name: rotor[name] for name in ("beta_inlet", "beta_outlet", "pitch")}}
                result["blade_sizing"] = natural
            else:
                reporter = make_progress_reporter(lambda event:
                    progress(f"MoC: {event[1]}", float(event[0].rstrip("%")) / 100))
                nozzle = solve_nozzle(config["method_of_characteristics"], progress_callback=reporter)
                # Output of this explicit computation; never searched for or
                # reused by a new MoC job.
                flow.write_json(folder / "moc_result.json", nozzle)
                result = {"reference": handle}
                if nozzle.get("converged") and not nozzle_only:
                    from turbo_moc.app.blade_counts import native_passage, sizing
                    template = flow.get_stator_template(config, handle, nozzle)
                    result["blade_sizing"] = sizing(config, native_passage(config, nozzle, template=template))
        elif action == "cad":
            nozzle = flow.load_nozzle(refs["nozzle"], config)
            if nozzle_only:
                from turbo_moc.geometry.nozzle_cad import export_nozzle
                report = export_nozzle(nozzle, config["method_of_characteristics"], config["mesh"]["nozzle"], folder,
                                       progress=progress)
            else:
                from turbo_moc.geometry.cad_export import export_row
                design = flow.get_design(config, refs["nozzle"])
                transform = None
                if config.get("assembly"):
                    transform = mesh._assembly_transform(config["assembly"], design.meridional_channel.is_linear)
                report = export_row(design, config["mesh"]["cad"], folder, row_id=row_id, transform=transform,
                                    progress=progress)
            result = {"reference": handle, "files": report["files"]}
        elif action in ("reference", "volume") and nozzle_only:
            from turbo_moc.meshing import nozzle_mesh
            nozzle = flow.load_nozzle(refs["nozzle"], config)
            build = nozzle_mesh.mesh_2d if action == "reference" else nozzle_mesh.mesh_3d
            summary = build(nozzle, config["method_of_characteristics"], config["mesh"]["nozzle"], folder,
                            progress=meshing_progress)
            result = {"reference": handle, "summary": summary, "export": summary["export"]}
        elif action == "reference":
            progress("Fit/reference geometry (one placement search)", 0.02)
            design = flow.get_design(config, refs["nozzle"])
            summary = mesh.reference_mesh(design, config, folder, progress=meshing_progress)
            output = mesh.export_reference_mesh(folder, config, folder, progress=cgns_progress)
            output.pop("native_mesh_file", None)
            result = {"reference": handle, "summary": summary, "export": output}
        elif action == "volume":
            section, design = mesh.load_section(flow.directory(refs["reference"]))
            summary = mesh.volume_mesh(section, design, config, folder, progress=meshing_progress)
            output = mesh.export_mesh(folder, config, folder, progress=cgns_progress)
            output.pop("native_mesh_file", None)
            result = {"reference": handle, "summary": summary, "export": output,
                      "reference_mesh": refs["reference"]}
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
        label = "Rotor blade" if action == "nozzle" and role == "rotor" else action.title()
        flow.write_json(folder / "status.json", {"state": "complete", "message": f"[{row_id}] {label} ready",
                        "fraction": 1.0, "started": handle["started"], "row_id": row_id})
        print(f"[{row_id}] Job completed successfully.", flush=True)
    except Exception as exc:
        traceback.print_exc()
        flow.write_json(folder / "status.json", {"state": "failed", "message": f"[{row_id}] {exc}",
                        "fraction": None, "started": handle["started"], "row_id": row_id})
        return 1
    finally:
        # Gmsh topology is only needed while converting this job to CGNS.
        # Section/volume arrays and browser surfaces retain everything needed
        # for subsequent span mapping, inspection and visualization.
        for native_file in folder.glob("*.msh"):
            native_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))

