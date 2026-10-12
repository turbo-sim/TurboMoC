"""Stage dependencies, durable job snapshots, and local worker orchestration."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from copy import deepcopy
from uuid import uuid4

import diskcache
import yaml
from flask import current_app, has_app_context

from ...configuration import read_configuration
from ...geometry import geometry_3d as geo

APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR.parents[1]
OUTPUT = Path("output").resolve()
INTEGER_KEYS = {"n", "n_cp", "num_baseline_points", "num_curve_points", "num_points",
                "num_blades", "curvature_elements", "algorithm", "recombination_algorithm",
                "smoothing_steps", "span_layers", "verbosity"}


def digest(value):
    """Hash numerical inputs consistently across Python/browser round trips.

    JavaScript has one number type, so a computed ``65.0`` can return from a
    Dash store as ``65``. These are the same input, especially in the derived
    rotor inlet state. Raw JSON hashing would mark the completed job stale and
    hide its geometry immediately. Canonicalize only integral floats: retain
    every other numeric value exactly, and keep booleans distinct from numbers.
    """
    def canonical(item):
        if isinstance(item, dict):
            return {key: canonical(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [canonical(value) for value in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


# Include the numerical implementation in artifact identities. Compute once at
# startup, not on every slider event. View settings never enter these identities.
SOURCE_REVISION = hashlib.sha256(b"".join(
    p.read_bytes() for p in sorted(SOURCE_DIR.rglob("*.py"))
    if "deprecated" not in p.relative_to(SOURCE_DIR).parts
)).hexdigest()


def keys(config, nozzle_reference=None):
    rotor = "rotor_design" in config
    solver_inputs = ([config["rotor_design"], config.get("rotor_source")]
                     if rotor else config["method_of_characteristics"])
    design = {n: config[n] for n in (
        "method_of_characteristics", "blade_geometry", "rotor_design", "rotor_source",
        "flow_domain", "meridional_channel", "assembly") if n in config}
    numerics = {k: v for k, v in config["mesh"]["numerics"].items()
                if k not in ("span_layers", "span_stretch", "verbosity", "terminal_output")}
    # A fresh Compute MoC action must invalidate its downstream geometry even
    # when the input settings match a previous solve. Identify the explicitly
    # selected job, rather than searching for a saved nozzle with those inputs.
    nozzle_job = [nozzle_reference.get(n) for n in ("session", "job")] if nozzle_reference else None
    geometry = digest([SOURCE_REVISION, design, nozzle_job])
    # A generated mesh now includes CGNS; BC/unit/filename edits invalidate
    # the deliverable instead of silently offering an older CGNS snapshot.
    cgns = {k: v for k, v in config["mesh"]["cgns"].items() if k != "enabled"}
    reference = digest([geometry, numerics, cgns])
    volume = digest([reference, config["mesh"]["numerics"]["span_layers"],
                     config["mesh"]["numerics"]["span_stretch"]])
    # Retain the original stage name in saved references; for a rotor this
    # stage contains its vortex solution, not a fictitious nozzle computation.
    return {"nozzle": digest([SOURCE_REVISION, "rotor" if rotor else "stator", solver_inputs]), "geometry": geometry,
            "reference": reference, "volume": volume}


def resolved_configuration(project, row_id, registry):
    from .rotor_inlet import resolve_configuration
    from ...configuration_project import row_assembly
    config = resolve_configuration(project, row_id, registry)
    assembly = row_assembly(project, row_id)
    if assembly is not None:
        config["assembly"] = assembly
    return config


def output_directory():
    """Use the owning app's output directory, or the standalone default."""
    if has_app_context():
        return current_app.config.get("TURBO_MOC_OUTPUT_DIR", OUTPUT)
    return OUTPUT


def directory(reference):
    """Browser data contains opaque identifiers, never arbitrary file paths."""
    if not reference or any(not re.fullmatch(r"[0-9a-f]{32}", str(reference.get(n, "")))
                            for n in ("session", "job")):
        raise ValueError("Invalid job reference.")
    return output_directory() / reference["session"] / reference["job"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    # Windows may briefly refuse replacement while Dash/OneDrive reads the
    # previous status. Preserve atomic publication and retry that short lock.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def read_json(path):
    """Read an atomically published snapshot despite a brief Windows lock."""
    # Replacement can momentarily overlap a reader on Windows/OneDrive.
    # Retry only sharing failures; missing files or invalid JSON remain errors.
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def current(reference, key):
    return bool(reference and reference.get("key") == key)


def load_nozzle(reference, config):
    """Read the selected completed job's output for subsequent design stages."""
    rotor = "rotor_design" in config
    action = "Compute rotor blade" if rotor else "Compute MoC"
    if not current(reference, keys(config)["nozzle"]):
        raise ValueError(f"{action} for the current inputs first.")
    try:
        name = "rotor_result.json" if rotor else "moc_result.json"
        return json.loads((directory(reference) / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"The selected design result is missing or unreadable. {action} again.") from exc


def get_design(config, nozzle_reference):
    """Cache template separately: passage/machine edits do not refit the blade."""
    cache = diskcache.Cache(str(output_directory() / "geometry_cache"))
    try:
        nozzle = load_nozzle(nozzle_reference, config)
        geometry_key = keys(config, nozzle_reference)["geometry"]
        data = cache.get(("design", geometry_key))
        if data is not None:
            return geo.load_design(data)
        if "rotor_design" in config:
            from ...geometry.rotor_geometry import build_rotor_design
            design = build_rotor_design(nozzle, config)
            cache.set(("design", geometry_key), geo.design_data(design))
            return design
        template_key = digest([nozzle_reference["session"], nozzle_reference["job"], config["blade_geometry"]])
        template = cache.get(("template", template_key))
        if template is None:
            from turbo_moc.geometry import parametrize_stator_blade
            wall = nozzle["wall_final"]
            template = parametrize_stator_blade(wall["x"], wall["y"], **config["blade_geometry"])
            cache.set(("template", template_key), geo.encode(template))
        else:
            template = geo.decode(template)
        design = geo.build_design(nozzle, config, template=template)
        cache.set(("design", geometry_key), geo.design_data(design))
        return design
    finally:
        cache.close()


def start_job(session, action, config, references, *, inspection=None,
              row_id="stator_1", row_role="stator", project=None, registry=None):
    """Launch a dedicated Gmsh/VTK process and immediately return to Dash."""
    if row_role not in ("stator", "rotor") or (row_role == "rotor") != ("rotor_design" in config):
        raise ValueError("Blade-row role does not match its design configuration.")
    if project is not None:
        from ...configuration_project import get_row, active_rows, row_configuration
        row = get_row(project, row_id)
        if row["role"] != row_role or row not in active_rows(project):
            raise ValueError("The requested blade row is not active in this machine project.")
        if config != resolved_configuration(project, row_id, registry):
            raise ValueError("The submitted row configuration does not match the machine-project snapshot.")
    from .rotor_inlet import require_supersonic_inlet
    require_supersonic_inlet(config)
    for reference in references.values():
        # Historical handles belong to the former single stator. They must not
        # be silently interpreted as an artifact of an arbitrary new row.
        if reference and reference.get("row_id", "stator_1") != row_id:
            raise ValueError("A blade-row job cannot use artifacts from another row.")
    stage = keys(config, references.get("nozzle"))
    if action != "nozzle":
        load_nozzle(references.get("nozzle"), config)
    if action == "volume" and not current(references.get("reference"), stage["reference"]):
        raise ValueError("Generate an up-to-date 2D reference mesh first.")
    if action == "export_2d" and not current(references.get("reference"), stage["reference"]):
        raise ValueError("Generate an up-to-date 2D reference mesh before exporting it.")
    if action in ("export", "inspect") and not current(references.get("volume"), stage["volume"]):
        raise ValueError("Generate an up-to-date 3D mesh first.")
    handle = {"session": session, "job": uuid4().hex, "action": action,
              "key": stage["reference"] if action == "export_2d" else stage.get(action, stage["volume"]),
              "row_id": row_id, "row_role": row_role, "started": time.time()}
    folder = directory(handle)
    folder.mkdir(parents=True)
    write_json(folder / "request.json", {"handle": handle, "config": config,
               "source_revision": SOURCE_REVISION,
               "references": references, "inspection": inspection,
               "project": project})
    (folder / "configuration.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(folder / "status.json", {"state": "running", "message": f"[{row_id}] Starting worker",
                                      "fraction": None, "started": handle["started"], "row_id": row_id})
    if project is not None:
        (folder / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    with (folder / "activity.log").open("ab", buffering=0) as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m",
            "turbo_moc.apps.app.worker", str(folder / "request.json")],
            cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    handle["pid"] = process.pid
    # Keep and reap child handles in the server; no zombie workers on Unix.
    PROCESSES[(session, handle["job"])] = process
    return handle


PROCESSES = {}


def poll_job(handle):
    folder = directory(handle)
    state = read_json(folder / "status.json")
    process = PROCESSES.get((handle["session"], handle["job"]))
    if process is not None and process.poll() is not None:
        PROCESSES.pop((handle["session"], handle["job"]), None)
        if state["state"] == "running":
            row_id = handle.get("row_id", "stator_1")
            state.update(state="failed", message=f"[{row_id}] Worker exited with code {process.returncode}.",
                         row_id=row_id)
            write_json(folder / "status.json", state)
    # Read a bounded tail. Gmsh's native stdout is redirected by Popen as well.
    with (folder / "activity.log").open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 16000))
        log = stream.read().decode("utf-8", errors="replace")
    return state, log


def cancel_job(handle):
    process = PROCESSES.pop((handle["session"], handle["job"]), None)
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    row_id = handle.get("row_id", "stator_1")
    state = {"state": "cancelled", "message": f"[{row_id}] Cancelled; previous artifacts retained.",
             "fraction": None, "started": handle["started"], "row_id": row_id}
    write_json(directory(handle) / "status.json", state)


def edited_configuration(base, values, identifiers):
    config = deepcopy(base)
    for value, identifier in zip(values, identifiers):
        path = identifier["path"].split(".")
        target = config
        for name in path[:-1]:
            target = target[name]
        # A meridional channel family change replaces just its controls. Ignore obsolete
        # fields during the brief browser transition to the new family.
        if path[0] == "meridional_channel" and path[-1] not in target:
            continue
        if value is None:
            raise ValueError(f"{identifier['path']} requires a value.")
        if path[-1] in INTEGER_KEYS:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{identifier['path']} requires an integer.")
            value = int(value)
        target[path[-1]] = value
    return config


def edited_project(base, values, identifiers):
    """Apply sidebar changes by stable row ID, independent of selected-row order."""
    from ...configuration_project import get_row, GEOMETRY_SECTIONS

    project = deepcopy(base)
    for value, identifier in zip(values, identifiers):
        row = get_row(project, identifier["row"])
        path = identifier["path"].split(".")
        if path[0] == "axial_interspace_mm" and row["geometry"]["meridional_channel"]["type"] != "axial":
            # The previous family's controls can arrive during the sidebar
            # replacement. An axial-gap field must not reappear in a radial or
            # linear row after the machine-family switch has removed it.
            continue
        target = row["geometry"] if path[0] in GEOMETRY_SECTIONS or path[0] in (
            "parametrization_type", "axial_interspace_mm", "rotor_design", "inlet_coupling") else row
        for name in path[:-1]:
            target = target[name]
        if path[0] == "meridional_channel" and path[-1] not in target:
            # Ignore the old family's controls during a selector transition.
            continue
        if value is None:
            raise ValueError(f"{row['id']}.{identifier['path']} requires a value.")
        if path[-1] in INTEGER_KEYS:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{row['id']}.{identifier['path']} requires an integer.")
            value = int(value)
        target[path[-1]] = value
    # The visible pair controls replace the historical PT/PQ fields together.
    # Keep untouched historical rows unchanged, including inactive rows.
    from ...fluid_inputs import INLET_FIELDS
    for row in project["rows"]:
        for section, legacy in (("method_of_characteristics", ("P0", "T0", "Q0")),
                                ("rotor_design", ("P0_rel", "T0_rel", "Q0_rel"))):
            inputs = row["geometry"].get(section, {})
            if any(name in inputs for name in INLET_FIELDS):
                for name in legacy:
                    inputs.pop(name, None)
    return project

