"""Package the current project's completed meshes and readable results."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from ... import configuration_project as projects
from . import workflow as flow


NO_MESH = "Generate a current 3D mesh, or a 2D mesh for a linear cascade, before downloading."


def _identity(reference):
    return (reference.get("session"), reference.get("job")) if reference else None


def _mesh_inputs(row, stage):
    """Compare scientific inputs for either row role, excluding display settings."""
    geometry = {k: v for k, v in row["geometry"].items() if k != "axial_interspace_mm"}
    ignored = {"verbosity", "terminal_output"}
    if stage == "reference":
        ignored.update(("span_layers", "span_stretch"))
    numerics = {k: v for k, v in row["mesh"]["numerics"].items() if k not in ignored}
    cgns = {k: v for k, v in row["mesh"]["cgns"].items() if k != "enabled"}
    return geometry, numerics, cgns


def _completed_mesh(project, row, stage, references, registry=None):
    """Resolve only selected, completed jobs whose inputs still match the row."""
    reference = references.get(stage)
    if not reference:
        raise ValueError("not generated")
    folder = flow.directory(reference)
    if flow.read_json(folder / "status.json").get("state") != "complete":
        raise ValueError("not completed successfully")
    request = flow.read_json(folder / "request.json")
    handle = request["handle"]
    if (_identity(handle) != _identity(reference) or handle.get("row_id") != row["id"]
            or handle.get("row_role") != row["role"] or handle.get("action") != stage):
        raise ValueError("job does not belong to this row and stage")
    snapshot = request["project"]
    previous_row = projects.get_row(snapshot, row["id"])
    if (previous_row["role"] != row["role"]
            or snapshot["machine"]["geometry_type"] != project["machine"]["geometry_type"]
            or _mesh_inputs(previous_row, stage) != _mesh_inputs(row, stage)
            or request.get("source_revision", flow.SOURCE_REVISION) != flow.SOURCE_REVISION):
        raise ValueError("out of date; regenerate for the current settings")
    # Every row depends on its resolved interfaces and assembly placement;
    # the stator outlet can change when only the rotor's gap changes. A coupled
    # rotor additionally depends on the selected upstream MoC result. Compare
    # effective stage keys rather than accepting matching row-local YAML alone.
    effective = flow.resolved_configuration(project, row["id"], registry)
    expected = flow.keys(effective, references.get("nozzle"))[stage]
    if not flow.current(reference, expected):
        raise ValueError("out of date; resolved geometry, interfaces or inlet conditions have changed")
    # Recomputing a predecessor invalidates its descendants even when the
    # numerical inputs are unchanged. Keep the selected computation chain.
    for dependency in (("nozzle", "reference") if stage == "volume" else ("nozzle",)):
        if _identity(request["references"].get(dependency)) != _identity(references.get(dependency)):
            raise ValueError("out of date; a preceding stage has been recomputed")
    filename = row["mesh"]["cgns"]["filename"]
    name = Path(filename)
    if name.name != filename or name.suffix.lower() != ".cgns":
        raise ValueError("invalid CGNS filename")
    if stage == "reference":
        name = name.with_name(f"{name.stem}_2d.cgns")
    path = folder / name
    required = [path, folder / "activity.log", folder / (
        "mesh_summary.json" if stage == "volume" else "reference_summary.json")]
    if stage == "volume":
        required.append(folder / "mesh_report.json")
    if any(not item.is_file() for item in required) or path.stat().st_size == 0:
        raise ValueError("mesh, report or activity log is missing")
    return {"row": row, "stage": stage, "folder": folder, "path": path,
            "file": f"meshes/{row['id']}/{name.name}", "request": request}


def collect_meshes(project, registry):
    """List available deliverables and explain omissions without reading mesh bytes."""
    projects.validate_project(project)
    stages = ("reference", "volume") if project["machine"]["geometry_type"] == "linear" else ("volume",)
    meshes, omitted = [], []
    for row in projects.active_rows(project):
        references = (registry or {}).get(row["id"], {})
        for stage in stages:
            try:
                meshes.append(_completed_mesh(project, row, stage, references, registry))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else "saved job files are missing or unreadable"
                dimension = "3D" if stage == "volume" else "2D"
                omitted.append(f"{row['id']} ({row['role']}), {dimension}: {reason}.")
    return meshes, omitted


def _timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


def _logs(meshes):
    """Combine each included row's computation chain once, in execution order."""
    jobs = {}
    for mesh in meshes:
        request, row = mesh["request"], mesh["row"]
        references = [request["references"].get(stage) for stage in ("nozzle", "reference")]
        references.append(request["handle"])
        for reference in references:
            if not reference or _identity(reference) in jobs:
                continue
            folder = flow.directory(reference)
            saved = flow.read_json(folder / "request.json")["handle"]
            if _identity(saved) != _identity(reference) or saved.get("row_id") != row["id"]:
                raise ValueError("A computation log does not belong to its mesh row.")
            jobs[_identity(reference)] = (saved, folder / "activity.log")
    chunks = []
    for handle, path in sorted(jobs.values(), key=lambda item: item[0].get("started", 0)):
        title = f"{handle['row_id']} | {handle['action']} | {_timestamp(handle['started'])}"
        chunks.append(f"{'=' * 72}\n{title}\n{'=' * 72}\n{path.read_text(encoding='utf-8', errors='replace').rstrip()}\n")
    return "\n".join(chunks)


def build_archive(project, registry):
    """Return a ZIP containing exactly the user's selected deliverable types."""
    meshes, omitted = collect_meshes(project, registry)
    if not meshes:
        raise ValueError(NO_MESH)
    now = datetime.now(timezone.utc)
    generated = now.isoformat(timespec="seconds")
    report = {"created_utc": generated, "machine": project["machine"], "rows": {}, "interfaces": []}
    interfaces = {}
    readme = ["TurboMoC project", f"Packaged: {generated}",
              f"Machine: {project['machine']['geometry_type']}; design mode: {project['machine']['design_mode']}", "",
              "CONTENTS", "project.yaml - final settings; load this file in TurboMoC to reproduce the design.",
              "mesh_report.json - consolidated mesh statistics and quality results, grouped by row and dimension.",
              "activity.log - computation logs for the included meshes and their preceding stages.", "", "MESHES"]
    for mesh in meshes:
        row, stage, folder = mesh["row"], mesh["stage"], mesh["folder"]
        dimension = "3d" if stage == "volume" else "2d"
        summary = flow.read_json(folder / ("mesh_summary.json" if stage == "volume" else "reference_summary.json"))
        entry = {"file": mesh["file"], "generated_utc": _timestamp(mesh["request"]["handle"]["started"]),
                 "coordinate_basis": "physical_cartesian" if stage == "volume" else "linear_physical",
                 "cgns_length_scale": row["mesh"]["cgns"]["length_scale"], "summary": summary}
        if stage == "volume":
            entry["quality"] = flow.read_json(folder / "mesh_report.json")
        report["rows"].setdefault(row["id"], {"role": row["role"], "meshes": {}})["meshes"][dimension] = entry
        effective = mesh["request"].get("config", {})
        assembly = effective.get("assembly")
        if stage == "volume" and assembly:
            entry["coordinate_basis"] = "assembly_cartesian"
            entry["assembly"] = assembly
            for boundary, interface in assembly.get("interfaces", {}).items():
                pair = interfaces.setdefault(interface["id"], {**interface, "exported_boundaries": []})
                pair["exported_boundaries"].append({"row_id": row["id"], "boundary": boundary,
                                                     "file": mesh["file"]})
        if row["role"] == "rotor" and "rotor_design" in effective:
            # Project YAML retains manual fallback values. Record the actual
            # effective inlet used for this mesh alongside its diagnostics.
            report["rows"][row["id"]]["rotor_design_inputs"] = effective["rotor_design"]
            report["rows"][row["id"]]["inlet_coupling"] = effective.get("rotor_source")
        nodes = summary.get("volume_nodes" if stage == "volume" else "section_nodes", "unknown")
        cells = sum(summary["cells"].values()) if stage == "volume" else summary.get("section_cells", "unknown")
        readme.extend([f"{mesh['file']} ({row['role']}, {dimension.upper()})",
            f"  Generated: {entry['generated_utc']}", f"  Nodes: {nodes}; cells: {cells}",
            f"  CGNS coordinate multiplier from geometry in mm: {entry['cgns_length_scale']:g}."])
        if stage == "volume" and assembly:
            placement = assembly["placement"]
            readme.append(
                f"  Machine placement applied: Z offset {placement['streamwise_offset_mm']:g} mm; "
                f"phase {placement['angular_phase_deg']:g} deg."
            )
        if stage == "volume":
            quality = summary.get("quality", {})
            if "min_sicn" in quality:
                readme.append(f"  Minimum SICN: {quality['min_sicn']:.6g}.")
        boundaries = row["mesh"]["cgns"]["boundary_types"]
        readme.append("  Boundaries: " + ", ".join(f"{name} ({kind})" for name, kind in boundaries.items()
            if stage == "volume" or name not in ("hub", "shroud")) + ".")
    report["interfaces"] = list(interfaces.values())
    if interfaces:
        readme.extend(["", "SHARED PHYSICAL INTERFACES"])
        for interface in interfaces.values():
            coordinate = "Z" if interface["geometry_type"] == "axial" else "radius"
            readme.append(f"{interface['id']}: {coordinate} = {interface['coordinate_mm']:g} mm; "
                          f"{interface['upstream_id']}/outlet to {interface['downstream_id']}/inlet.")
    readme.extend(["", "COORDINATES AND USE",
        "Geometry and dimensional mesh-report lengths use mm. CGNS unit metadata uses metres.",
        "The standard CGNS multiplier is 0.001 (mm to m); each mesh's configured multiplier is listed above.",
        "Each mesh is one periodic passage. Multirow 3D meshes use machine coordinates with the recorded placement applied; additional displayed blades are not exported.",
        "Single-row meshes and linear 2D meshes retain their original physical coordinates.",
        "Configure periodic_left/periodic_right pairing in the CFD solver (translational for linear, rotational for annular).",
        "Adjacent rows share a physical interface, not necessarily matching nodes or blade counts. The inlet/outlet boundary names are preserved; configure row-interface coupling in the CFD solver.",
        "2D CGNS files are included only for linear cascades; axial/radial exports contain physical 3D meshes.",
        "Only completed meshes matching the final settings and selected computation chain are included.",
        "The project YAML also retains any inactive rows; their meshes are not included."])
    if omitted:
        readme.extend(["", "UNAVAILABLE MESHES", *omitted])
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("project.yaml", yaml.safe_dump(project, sort_keys=False, allow_unicode=True))
        archive.writestr("README.txt", "\n".join(readme) + "\n")
        archive.writestr("mesh_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("activity.log", _logs(meshes))
        for mesh in meshes:
            archive.write(mesh["path"], mesh["file"])
    return stream.getvalue(), f"turbo_moc_project_{now:%Y-%m-%d_%H-%M-%S}.zip"
