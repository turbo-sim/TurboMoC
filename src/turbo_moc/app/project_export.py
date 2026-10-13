"""Package the current project's completed meshes and readable results."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from turbo_moc.configuration import project as projects
from turbo_moc.io.cgns_stage import combine_rows
from turbo_moc.app import workflow as flow


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


def stage_meshes(project, registry):
    """Require a complete stage; calculation scope never limits export scope."""
    meshes, omitted = collect_meshes(project, registry)
    active = {row["id"] for row in projects.active_rows(project)}
    complete = {stage: [mesh for mesh in meshes if mesh["stage"] == stage]
                for stage in ("volume", "reference")}
    complete = {stage: items for stage, items in complete.items()
                if {item["row"]["id"] for item in items} == active}
    if "volume" not in complete and (project["machine"]["geometry_type"] != "linear"
                                     or "reference" not in complete):
        raise ValueError(NO_MESH + " Every active row must be current. " + " ".join(omitted))
    for items in complete.values():
        if len({item["row"]["mesh"]["cgns"]["length_scale"] for item in items}) != 1:
            raise ValueError("All rows must use the same CGNS length_scale for a shared physical stage.")
    return meshes, omitted, complete


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


def _export_entries(project, registry, stage, folder_name, report_name, description):
    entries, readme = [], []
    for row in projects.active_rows(project):
        references = (registry or {}).get(row["id"], {})
        reference = references.get(stage)
        if not reference:
            continue
        try:
            config = flow.resolved_configuration(project, row["id"], registry)
            if not flow.current(reference, flow.keys(config, references.get("nozzle"))[stage]):
                readme.append(f"{row['id']}: {description} out of date; not included.")
                continue
            folder = flow.directory(reference)
            if flow.read_json(folder / "status.json").get("state") != "complete":
                continue
            report = flow.read_json(folder / report_name)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
        names = [*report["files"].values(), report_name, *sorted(item.name for item in folder.glob("*.png"))]
        entries += [(f"{folder_name}/{row['id']}/{name}", folder / name) for name in names if (folder / name).is_file()]
        readme.append(f"{folder_name}/{row['id']}/ ({report['units']}, {report['frame']} frame): "
                      + ", ".join(report["files"].values()) + f"; {report_name}.")
    return entries, readme


def cad_entries(project, registry):
    """Current, completed CAD and BladeGen exports as (archive name, file) pairs, plus README lines."""
    cad, cad_readme = _export_entries(project, registry, "cad", "cad", "cad_report.json", "CAD export")
    bladegen, bladegen_readme = _export_entries(project, registry, "bladegen", "bladegen",
                                                "bladegen_report.json", "BladeGen export")
    readme = cad_readme + [line.replace("bladegen_report.json.", "bladegen_report.json (BladeGen files for TurboGrid).")
                           for line in bladegen_readme]
    return cad + bladegen, readme


def build_archive(project, registry):
    """Return the complete active stage plus settings and per-row diagnostics."""
    meshes, omitted, stages = stage_meshes(project, registry)
    combined = {}
    for stage, items in stages.items():
        name = "stage_2d.cgns" if stage == "reference" and "volume" in stages else "stage.cgns"
        combined[name] = combine_rows([(item["row"]["id"], item["path"]) for item in items])
    now = datetime.now(timezone.utc)
    generated = now.isoformat(timespec="seconds")
    report = {"created_utc": generated, "machine": project["machine"], "rows": {}, "interfaces": [],
              "stage_files": {name: manifest for name, (_, manifest) in combined.items()}}
    interfaces = {}
    readme = ["TurboMoC project", f"Packaged: {generated}",
              f"Machine: {project['machine']['geometry_type']}; design mode: {project['machine']['design_mode']}", "",
              "CONTENTS", "project.yaml - final settings; load this file in TurboMoC to reproduce the design.",
              "mesh_report.json - consolidated mesh statistics and quality results, grouped by row and dimension.",
              "activity.log - computation logs for the included meshes and their preceding stages.",
              "stage.cgns - the complete active machine, with a separate fluid zone for each row.",
              "stage_2d.cgns - optional separate true 2D linear mesh when a 3D mesh is also available.",
              "meshes/ - unchanged individual row exports for diagnostics.", "", "MESHES"]
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
    for interface in report["interfaces"]:
        for side in interface["exported_boundaries"]:
            stage_row = combined["stage.cgns"][1]["rows"][side["row_id"]]
            side["stage_file"] = "stage.cgns"
            side["zone"] = stage_row["zone"]
            side["stage_boundary"] = stage_row["boundaries"][side["boundary"]]
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
        "Stage zones, fluid element sections and boundary names are prefixed with their stable row IDs. Periodic donor-zone references are updated; local node/element indices are unchanged.",
        "Fluent 24.2 qualifies imported names with the zone, e.g. stator_1_fluid.stator_1_outlet. It initially imports these boundary patches as walls; assign the required inlet/outlet, periodic and inter-row interface types in the solver.",
        "Adjacent rows share a physical interface, not necessarily matching nodes or blade counts. The two interface sides stay separate; configure row-interface coupling in the CFD solver. No flow interface model is assigned by this archive.",
        "2D CGNS files are included only for linear cascades; axial/radial exports contain physical 3D meshes.",
        "Only completed meshes matching the final settings and selected computation chain are included.",
        "The project YAML also retains any inactive rows; their meshes are not included."])
    if omitted:
        readme.extend(["", "UNAVAILABLE MESHES", *omitted])
    cad, cad_readme = cad_entries(project, registry)
    if cad_readme:
        readme.extend(["", "CAD AND BLADEGEN EXPORTS", *cad_readme,
                       "Rows of a stage are written in the shared machine frame, so they open in place together."])
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in cad:
            archive.write(source, name)
        archive.writestr("project.yaml", yaml.safe_dump(project, sort_keys=False, allow_unicode=True))
        archive.writestr("README.txt", "\n".join(readme) + "\n")
        archive.writestr("mesh_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("activity.log", _logs(meshes))
        for name, (data, _) in combined.items():
            archive.writestr(name, data)
        for mesh in meshes:
            archive.write(mesh["path"], mesh["file"])
    return stream.getvalue(), f"turbo_moc_project_{now:%Y-%m-%d_%H-%M-%S}.zip"


def _current_job(references, stage, key):
    reference = references.get(stage)
    if not flow.current(reference, key):
        raise ValueError("not generated" if not reference else "out of date; regenerate for the current settings")
    folder = flow.directory(reference)
    if flow.read_json(folder / "status.json").get("state") != "complete":
        raise ValueError("not completed successfully")
    return reference, folder


def _csv(header, columns):
    rows = [",".join(header)] + [",".join(f"{float(value):.10g}" for value in row) for row in zip(*columns)]
    return "\n".join(rows) + "\n"


def nozzle_ready(project, registry):
    row = next(row for row in projects.active_rows(project) if row["role"] == "stator")
    references = (registry or {}).get(row["id"], {})
    config = flow.resolved_configuration(project, row["id"], registry)
    keys = flow.keys(config, references.get("nozzle"))
    try:
        _current_job(references, "nozzle", keys["nozzle"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Compute the nozzle for the current settings before downloading ({exc}).") from exc
    return row, references, config, keys


def nozzle_archive(project, registry):
    from turbo_moc.geometry import nozzle as geometry
    row, references, config, keys = nozzle_ready(project, registry)
    nozzle_reference, nozzle_folder = _current_job(references, "nozzle", keys["nozzle"])
    data = flow.read_json(nozzle_folder / "moc_result.json")
    parameters, settings = config["method_of_characteristics"], config["mesh"]["nozzle"]
    shape = geometry.contour(data, parameters, settings)
    now = datetime.now(timezone.utc)
    generated = now.isoformat(timespec="seconds")
    kind = "axisymmetric" if shape["axisymmetric"] else "planar"
    files = {
        "project.yaml": yaml.safe_dump(projects.nozzle_case(project), sort_keys=False, allow_unicode=True),
        "nozzle/moc_result.json": json.dumps(data, indent=2) + "\n",
        "nozzle/wall_contour.csv": _csv(("x_m", "r_m" if shape["axisymmetric"] else "y_m"), (shape["x"], shape["y"])),
        "nozzle/wall_distribution.csv": _csv(("x_m", "y_m", "p_Pa", "M", "rho_kg_m3", "T_K"),
            [data["wall_final"][name] for name in ("x", "y", "p", "M", "rho", "T")]),
        "nozzle/axis_distribution.csv": _csv(("x_m", "p_Pa", "M", "rho_kg_m3", "T_K"),
            [data["axis"][name] for name in ("x", "p", "M", "rho", "T")]),
    }
    nozzle = {"type": kind, "solver": parameters["solver"], "converged": data.get("converged"),
              "exit_mach": data.get("exit_M"), "design_exit_mach": data.get("design_Noz_Mach"),
              "exit_pressure_Pa": data.get("p_back"), "throat_half_height_m": parameters["y_t"],
              "exit_half_height_m": float(shape["y"][-1]), "length_m": float(shape["x"][-1] - shape["x"][0])}
    report = {"created_utc": generated, "nozzle": nozzle, "meshes": {}}
    jobs, omitted = [nozzle_folder], []
    for stage, dimension, summary_name in (("reference", "2d", "nozzle_mesh_2d.json"),
                                           ("volume", "3d", "nozzle_mesh_3d.json")):
        try:
            _, folder = _current_job(references, stage, keys[stage])
            cgns = folder / f"nozzle_mesh_{dimension}.cgns"
            summary = flow.read_json(folder / summary_name)
            if stage == "reference":
                summary, quality = summary["summary"], summary["report"]
            else:
                quality = flow.read_json(folder / "mesh_report.json")
            files[f"meshes/{cgns.name}"] = cgns.read_bytes()
            report["meshes"][dimension] = {"file": f"meshes/{cgns.name}", "summary": summary, "quality": quality}
            jobs.append(folder)
        except (OSError, ValueError, KeyError) as exc:
            omitted.append(f"{dimension.upper()} mesh: {exc}.")
    logs = []
    for folder in jobs:
        handle = flow.read_json(folder / "request.json")["handle"]
        title = f"{handle['row_id']} | {handle['action']} | {_timestamp(handle['started'])}"
        text = (folder / "activity.log").read_text(encoding="utf-8", errors="replace").rstrip()
        logs.append(f"{'=' * 72}\n{title}\n{'=' * 72}\n{text}\n")
    files["mesh_report.json"] = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    files["activity.log"] = "\n".join(logs)
    lower = "axis" if shape["axisymmetric"] else "symmetry plane"
    solid = ("a full 360 degree revolution." if shape["axisymmetric"] else
             f"a full-height slab of depth {settings['planar_depth']:g} m centred on z = 0.")
    readme = ["TurboMoC nozzle", f"Packaged: {generated}",
              f"Nozzle: {kind}, {parameters['solver']} contour; fluid {project['machine']['fluid_name']}", "",
              "CONTENTS",
              "project.yaml - nozzle case; load this file in TurboMoC to reproduce the design.",
              "nozzle/moc_result.json - complete method-of-characteristics solution.",
              "nozzle/wall_contour.csv - wall coordinates in metres, convergent inlet included, from inlet to exit.",
              "nozzle/wall_distribution.csv - wall coordinates and flow properties of the divergent section.",
              "nozzle/axis_distribution.csv - flow properties along the " + lower + ".",
              "mesh_report.json - nozzle summary, mesh statistics and quality results.",
              "activity.log - computation logs of the included jobs.",
              "meshes/ - structured CGNS meshes, coordinates in metres.", "",
              "NOZZLE",
              f"Exit Mach {nozzle['exit_mach']:.4f} (design {nozzle['design_exit_mach']:.4f}); "
              f"exit pressure {nozzle['exit_pressure_Pa']:.6g} Pa.",
              f"Throat half-height {nozzle['throat_half_height_m']:.6g} m; exit half-height "
              f"{nozzle['exit_half_height_m']:.6g} m; length {nozzle['length_m']:.6g} m.", "", "MESHES"]
    for dimension, entry in report["meshes"].items():
        patches = entry["summary"]["export"]["boundaries"]
        readme.append(f"{entry['file']} ({dimension.upper()}): {entry['summary']['nodes']} nodes, "
                      f"{entry['summary']['cells']} cells.")
        readme.append("  Boundaries: " + ", ".join(f"{name} ({patch['bc_type']})" for name, patch in patches.items()) + ".")
    if omitted:
        readme.extend(["", "UNAVAILABLE MESHES", *omitted])
    cad, cad_readme = cad_entries(project, registry)
    for name, source in cad:
        files[name] = source.read_bytes()
    if cad_readme:
        readme.extend(["", "CAD AND BLADEGEN EXPORTS", *cad_readme])
    readme.extend(["", "COORDINATES",
                   f"x is the axial distance from the throat. The 2D mesh is a half-domain bounded by the {lower}; "
                   f"the 3D mesh is {solid}"])
    files["README.txt"] = "\n".join(readme) + "\n"
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue(), f"turbo_moc_nozzle_{now:%Y-%m-%d_%H-%M-%S}.zip"
