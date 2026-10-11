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

from ..configuration import read_configuration
from ..geometry import geometry_3d as geo

APP_DIR = Path(__file__).resolve().parent
from ..paths import GMSH_3D_DIR as HERE, PROJECT_DIR as ROOT, SOURCE_DIR
OUTPUT = HERE / "output" / "app"
CASES = {n: HERE / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}
INTEGER_KEYS = {"n", "n_cp", "num_baseline_points", "num_curve_points", "num_points",
                "num_blades", "curvature_elements", "algorithm", "recombination_algorithm",
                "smoothing_steps", "span_layers", "verbosity"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Include the numerical implementation in artifact identities. Compute once at
# startup, not on every slider event. View settings never enter these identities.
SOURCE_REVISION = hashlib.sha256(b"".join(
    p.read_bytes() for folder in (SOURCE_DIR, ROOT / "src/turbo_moc")
    for p in sorted(folder.rglob("*.py")) if "output" not in p.parts
)).hexdigest()


def keys(config, nozzle_reference=None):
    design = {n: config[n] for n in (
        "method_of_characteristics", "blade_geometry", "flow_domain", "meridional_channel")}
    numerics = {k: v for k, v in config["mesh"]["numerics"].items()
                if k not in ("span_layers", "span_stretch", "verbosity", "terminal_output")}
    # A fresh Compute MoC action must invalidate its downstream geometry even
    # when the input settings match a previous solve. Identify the explicitly
    # selected job, rather than searching for a saved nozzle with those inputs.
    nozzle_job = [nozzle_reference.get(n) for n in ("session", "job")] if nozzle_reference else None
    geometry = digest([SOURCE_REVISION, design, nozzle_job])
    reference = digest([geometry, numerics])
    volume = digest([reference, config["mesh"]["numerics"]["span_layers"],
                     config["mesh"]["numerics"]["span_stretch"]])
    return {"nozzle": digest([SOURCE_REVISION, config["method_of_characteristics"]]), "geometry": geometry,
            "reference": reference, "volume": volume}


def directory(reference):
    """Browser data contains opaque identifiers, never arbitrary file paths."""
    if not reference or any(not re.fullmatch(r"[0-9a-f]{32}", str(reference.get(n, "")))
                            for n in ("session", "job")):
        raise ValueError("Invalid job reference.")
    return OUTPUT / reference["session"] / reference["job"]


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


def current(reference, key):
    return bool(reference and reference.get("key") == key)


def load_nozzle(reference, config):
    """Read the selected completed job's output for subsequent design stages."""
    if not current(reference, keys(config)["nozzle"]):
        raise ValueError("Compute MoC for the current nozzle inputs first.")
    try:
        return json.loads((directory(reference) / "moc_result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("The selected MoC job result is missing or unreadable. Compute MoC again.") from exc


def get_design(config, nozzle_reference):
    """Cache template separately: passage/machine edits do not refit the blade."""
    cache = diskcache.Cache(str(OUTPUT / "geometry_cache"))
    try:
        nozzle = load_nozzle(nozzle_reference, config)
        geometry_key = keys(config, nozzle_reference)["geometry"]
        data = cache.get(("design", geometry_key))
        if data is not None:
            return geo.load_design(data)
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
              row_id="stator_1", row_role="stator", project=None):
    """Launch a dedicated Gmsh/VTK process and immediately return to Dash."""
    # Reject placeholders before looking up artifacts or starting any process.
    # Row snapshots remain flat so the numerical stages need no assembly branch.
    if row_role != "stator":
        raise ValueError("Rotor geometry and meshing integration is pending; rotor jobs are unavailable.")
    if project is not None:
        from ..configuration_project import get_row, active_rows, row_configuration
        row = get_row(project, row_id)
        if row["role"] != row_role or row not in active_rows(project):
            raise ValueError("The requested blade row is not active in this machine project.")
        if config != row_configuration(project, row_id):
            raise ValueError("The submitted row configuration does not match the machine-project snapshot.")
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
            "example_meshing.src.app.worker", str(folder / "request.json")],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    handle["pid"] = process.pid
    # Keep and reap child handles in the server; no zombie workers on Unix.
    PROCESSES[(session, handle["job"])] = process
    return handle


PROCESSES = {}


def poll_job(handle):
    folder = directory(handle)
    state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
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
    from ..configuration_project import get_row, GEOMETRY_SECTIONS

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
            "parametrization_type", "axial_interspace_mm") else row
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
    return project

