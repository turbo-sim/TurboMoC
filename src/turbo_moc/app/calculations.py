"""Sequential row calculations using the existing isolated numerical workers.

The calculation scope never changes the machine design mode. Each queue keeps
an immutable project snapshot; completed row results are published before the
next row resolves its inlet state. No scientific solver is duplicated here.
"""
from copy import deepcopy
import time
from threading import RLock

from turbo_moc.configuration import project as projects
from turbo_moc.app import workflow as flow

ALL_ROWS = "all"
STAGE_LABELS = {"nozzle": "blade design", "reference": "2D mesh", "volume": "3D mesh"}
_ADVANCE_LOCK = RLock()


def scope_options(project):
    rows = projects.active_rows(project)
    options = [{"label": "Nozzle" if project["machine"]["design_mode"] == "nozzle" else
        "Stator" if row["id"] == "stator_1" else
        "Rotor" if row["id"] == "rotor_1" else row["id"].replace("_", " ").title(),
        "value": row["id"]} for row in rows]
    return options


def scoped_rows(project, scope):
    """Checkbox selections execute in machine order, not click order.

    Retain string scopes for callers of the previous workflow API. An empty
    checkbox selection deliberately starts no calculation.
    """
    rows = projects.active_rows(project)
    if scope == ALL_ROWS:
        return rows
    ids = {scope} if isinstance(scope, str) else set(scope or [])
    if not ids or not ids.issubset({row["id"] for row in rows}):
        raise ValueError("Check at least one active row to calculate.")
    selected = [row for row in rows if row["id"] in ids]
    return selected


def check_prerequisites(project, scope, action, registry):
    """Validate all selected rows before launching any mesh job.

    A coupled rotor's new inlet cannot be resolved before a queued upstream
    stator solve finishes. Defer only that dependency for a blade-design queue;
    a rotor-only action still requires an already current stator solution.
    """
    if action not in STAGE_LABELS:
        raise ValueError(f"Unknown calculation action {action!r}.")
    projects.validate_project(project)
    projects.shared_interfaces(project)
    rows = scoped_rows(project, scope)
    preceding = set()
    from turbo_moc.app.rotor_inlet import require_supersonic_inlet
    for row in rows:
        upstream = projects.upstream_stator(project, row["id"])
        coupled = row["role"] == "rotor" and row["geometry"]["inlet_coupling"]["mode"] == "stator_exit"
        deferred = action == "nozzle" and coupled and upstream and upstream["id"] in preceding
        if not deferred:
            config = flow.resolved_configuration(project, row["id"], registry)
            require_supersonic_inlet(config)
            refs = (registry or {}).get(row["id"], {})
            keys = flow.keys(config, refs.get("nozzle"))
            if action != "nozzle" and not flow.current(refs.get("nozzle"), keys["nozzle"]):
                raise ValueError(f"{row['id']}: compute the current blade design first.")
            if action == "volume" and not flow.current(refs.get("reference"), keys["reference"]):
                raise ValueError(f"{row['id']}: generate an up-to-date 2D mesh first.")
        preceding.add(row["id"])
    return rows


def _launch_next(session, queue, registry):
    """Skip current meshes, but honour explicit blade-design computations."""
    while queue["index"] < len(queue["rows"]):
        row = projects.get_row(queue["project"], queue["rows"][queue["index"]])
        config = flow.resolved_configuration(queue["project"], row["id"], registry)
        refs = (registry or {}).get(row["id"], {})
        if queue["action"] != "nozzle" and flow.current(refs.get(queue["action"]),
                flow.keys(config, refs.get("nozzle"))[queue["action"]]):
            queue["index"] += 1
            continue
        handle = flow.start_job(session, queue["action"], config, refs,
            row_id=row["id"], row_role=row["role"], project=queue["project"], registry=registry)
        queue.setdefault("root_job", handle["job"])
        # Queue metadata belongs to the browser job handle, not the immutable
        # per-row result reference that the worker publishes.
        return {**handle, "queue": deepcopy(queue)}
    return None


def start(session, project, scope, action, registry):
    rows = check_prerequisites(project, scope, action, registry)
    queue = {"project": deepcopy(project), "rows": [row["id"] for row in rows],
        "action": action, "index": 0, "started": time.time(), "log": ""}
    return _launch_next(session, queue, registry)


def cancellation_requested(handle):
    queue = handle.get("queue")
    return bool(queue and (flow.directory({"session": handle["session"],
        "job": queue["root_job"]}) / "cancel_queue").exists())


def cancel(handle):
    """Stop the active worker and the remaining queue, preserving successes."""
    with _ADVANCE_LOCK:
        if handle.get("queue"):
            root = flow.directory({"session": handle["session"], "job": handle["queue"]["root_job"]})
            (root / "cancel_queue").touch()
        status, _ = flow.poll_job(handle)
        if status["state"] == "running":
            flow.cancel_job(handle)


def advance(handle, registry, log):
    if "queue" not in handle:
        return None
    with _ADVANCE_LOCK:
        if cancellation_requested(handle):
            return None
        # Dash interval requests can overlap. Publish the successor once so
        # polling the same completed child cannot launch a second rotor solve.
        publication = flow.directory(handle) / "queue_next.json"
        if publication.exists():
            return flow.read_json(publication)["handle"]
        queue = deepcopy(handle["queue"])
        if handle["action"] == "nozzle":
            result = flow.read_json(flow.directory(handle) / "result.json")
            count = result.get("blade_sizing", {}).get("suggested_num_blades")
            if count is not None:
                projects.get_row(queue["project"], handle["row_id"])["geometry"]["meridional_channel"]["num_blades"] = count
        queue["index"] += 1
        queue["log"] = (queue["log"] + log + "\n")[-60000:]
        successor = _launch_next(handle["session"], queue, registry)
        flow.write_json(publication, {"handle": successor})
        return successor
