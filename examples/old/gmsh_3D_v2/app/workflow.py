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

from examples.gmsh.run_mesh_generation import load_or_solve_nozzle
from .configuration import read_configuration
from . import geometry_source as geo

APP_DIR = Path(__file__).resolve().parent
HERE = APP_DIR.parent
ROOT = HERE.parents[1]
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
    p.read_bytes() for folder in (HERE, ROOT / "examples/gmsh", ROOT / "examples/gmsh_3D",
                                  ROOT / "src/turbo_moc")
    for p in sorted(folder.rglob("*.py")) if "output" not in p.parts
)).hexdigest()


def keys(config):
    design = {n: config[n] for n in (
        "method_of_characteristics", "blade_geometry", "flow_domain", "meridional_channel")}
    numerics = {k: v for k, v in config["mesh"]["numerics"].items()
                if k not in ("span_layers", "span_stretch", "verbosity", "terminal_output")}
    geometry = digest([SOURCE_REVISION, design])
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
    if not current(reference, keys(config)["nozzle"]):
        raise ValueError("Compute MoC for the current nozzle inputs first.")
    result = load_or_solve_nozzle(config["method_of_characteristics"],
                                 directory(reference) / "nozzle_solution.json", cache_only=True)
    if result is None:
        raise ValueError("The validated nozzle artifact is missing. Compute MoC again.")
    return result


def get_design(config, nozzle_reference):
    """Cache template separately: passage/machine edits do not refit the blade."""
    cache = diskcache.Cache(str(OUTPUT / "geometry_cache"))
    try:
        geometry_key = keys(config)["geometry"]
        data = cache.get(("design", geometry_key))
        if data is not None:
            return geo.load_design(data)
        nozzle = load_nozzle(nozzle_reference, config)
        template_key = digest([nozzle_reference["key"], config["blade_geometry"]])
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


def start_job(session, action, config, references, *, inspection=None):
    """Launch a dedicated Gmsh/VTK process and immediately return to Dash."""
    stage = keys(config)
    if action != "nozzle":
        load_nozzle(references.get("nozzle"), config)
    if action == "volume" and not current(references.get("reference"), stage["reference"]):
        raise ValueError("Generate an up-to-date 2D reference mesh first.")
    if action in ("export", "inspect") and not current(references.get("volume"), stage["volume"]):
        raise ValueError("Generate an up-to-date 3D mesh first.")
    handle = {"session": session, "job": uuid4().hex, "action": action,
              "key": stage.get(action, stage["volume"]), "started": time.time()}
    folder = directory(handle)
    folder.mkdir(parents=True)
    write_json(folder / "request.json", {"handle": handle, "config": config,
               "source_revision": SOURCE_REVISION,
               "references": references, "inspection": inspection})
    (folder / "configuration.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(folder / "status.json", {"state": "running", "message": "Starting worker",
                                      "fraction": None, "started": handle["started"]})
    with (folder / "activity.log").open("ab", buffering=0) as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m",
            "examples.gmsh_3D_v2.app.worker", str(folder / "request.json")],
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
            state.update(state="failed", message=f"Worker exited with code {process.returncode}.")
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
    state = {"state": "cancelled", "message": "Cancelled; previous artifacts retained.",
             "fraction": None, "started": handle["started"]}
    write_json(directory(handle) / "status.json", state)


def edited_configuration(base, values, identifiers):
    config = deepcopy(base)
    for value, identifier in zip(values, identifiers):
        path = identifier["path"].split(".")
        target = config
        for name in path[:-1]:
            target = target[name]
        # A channel-family change replaces just its controls. Ignore obsolete
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

