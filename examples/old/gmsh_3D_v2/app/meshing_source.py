"""Independent reference and sweep stages, reusing the audited Gmsh core."""
import json
from pathlib import Path
from time import perf_counter

import gmsh
import numpy as np

from examples.gmsh import meshing_source as planar
from examples.gmsh_3D import meshing_source as shared
from . import geometry_source as geo
from .quality_source import analyze_volume

BOUNDARIES = shared.BOUNDARIES


def emit(progress, label, fraction=None):
    if progress:
        progress(label, fraction)


def save_section(directory, section, design):
    arrays = {"xy": section["xy"], "slave": section["slave"], "master": section["master"]}
    arrays.update({f"cells_{k}": v for k, v in section["cells"].items()})
    arrays.update({f"boundary_{k}": v for k, v in section["boundaries"].items()})
    np.savez_compressed(directory / "reference_mesh.npz", **arrays)
    metadata = {"profile": section["profile"], "transition": section["transition"],
                "design": geo.design_data(design)}
    (directory / "reference_definition.json").write_text(json.dumps(metadata), encoding="utf-8")


def load_section(directory):
    directory = Path(directory)
    with np.load(directory / "reference_mesh.npz", allow_pickle=False) as arrays:
        section = {name: arrays[name] for name in ("xy", "slave", "master")}
        section["cells"] = {int(name[6:]): arrays[name] for name in arrays.files if name.startswith("cells_")}
        section["boundaries"] = {name[9:]: arrays[name] for name in arrays.files if name.startswith("boundary_")}
    metadata = json.loads((directory / "reference_definition.json").read_text(encoding="utf-8"))
    section.update(profile=metadata["profile"], transition=metadata["transition"])
    return section, geo.load_design(metadata["design"])


def reference_mesh(design, config, output_dir, *, progress=None):
    """Generate and publish a 2D mesh only; no span sweep or volume export."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numerics = config["mesh"]["numerics"]
    planar.validate_mesh_settings(config["mesh"] | {"cgns": {"enabled": False}, "report": {"enabled": False}})
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    start = perf_counter()
    gmsh.initialize(["moc_v2_reference", "-v", str(max(4, numerics["verbosity"]))], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(numerics["terminal_output"]))
        emit(progress, "Build reference CAD", 0.05)
        gmsh.model.add("conformal_reference_section")
        blade = design.blade
        blade_curves = planar.add_blade_boundary(blade, numerics["blade_size"])
        surface, boundaries = planar.add_periodic_passage(
            design.curves, blade_curves, blade["pitch"], numerics["domain_size"])
        profile = planar.calculate_inflation_profile(numerics["inflation_layers"])
        planar.set_mesh_sizes(blade_curves, numerics, profile, blade["pitch"])
        planar.add_inflation_layers(blade_curves, numerics)
        planar.configure_recombination(surface, numerics)
        emit(progress, "Generate 1D mesh and match periodic nodes")
        gmsh.model.mesh.generate(1)
        planar.check_periodic_conformity(boundaries)
        emit(progress, "Prepare inflation/core grading", 0.25)
        transition = planar.configure_core_transition(
            blade_curves, numerics, profile, blade["pitch"], use_mesh=True)
        emit(progress, "Generate 2D reference mesh")
        gmsh.model.mesh.generate(2)
        emit(progress, "Validate and extract reference topology", 0.85)
        planar.check_periodic_conformity(boundaries)
        summary = planar.mesh_statistics(surface, boundaries)
        xy, cells, edges, local = planar.collect_fluid_mesh(surface, boundaries)
        slave, master, _ = planar.collect_periodic_nodes(boundaries, local, xy)
        blocks = {kind: np.asarray([c for c in cells if len(c) == width], dtype=np.int32).reshape(-1, width) - 1
                  for width, kind in ((3, 2), (4, 3))}
        blocks = {k: v for k, v in blocks.items() if len(v)}
        if numerics["recombine"] and 3 not in blocks:
            raise ValueError("Requested recombination produced no quadrilaterals.")
        section = {"xy": xy, "cells": blocks,
                   "boundaries": {name: e[:, :2] - 1 for name, e in edges.items()},
                   "slave": slave - 1, "master": master - 1,
                   "profile": profile, "transition": transition}
        gmsh.model.mesh.removeSizeCallback()
        gmsh.write(str(directory / "reference_mesh.msh"))
        save_section(directory, section, design)
        summary.update(section_nodes=len(xy), section_cells=sum(len(v) for v in blocks.values()),
                       elapsed_s=perf_counter() - start, shift_mm=design.curves["shift_mm"],
                       reference_radius_mm=design.channel.reference_radius,
                       inflation_profile=profile, core_transition=transition)
        (directory / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit(progress, "Reference mesh ready", 1.0)
        return summary
    finally:
        gmsh.model.mesh.removeSizeCallback()
        gmsh.finalize()


def collect_volume_arrays():
    """Retain real cell IDs/connectivity for histograms, slicing and worst cells."""
    nodes, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(nodes)
    nodes, xyz = nodes[order], xyz.reshape(-1, 3)[order]
    kinds, tag_arrays, node_arrays = gmsh.model.mesh.getElements(3, 1)
    blocks, tags = {}, {}
    for kind, ids, values in sorted(zip(kinds, tag_arrays, node_arrays), key=lambda item: item[0]):
        kind = int(kind)
        blocks[kind] = np.searchsorted(nodes, values).reshape(-1, 8 if kind == 5 else 6)
        tags[kind] = ids
    _, slave, master, rotation = gmsh.model.mesh.getPeriodicNodes(2, shared.SURFACES["periodic_right"])
    _, _, right = gmsh.model.mesh.getElements(2, shared.SURFACES["periodic_right"])
    return xyz, blocks, tags, {
        "periodic_faces": {"right": np.searchsorted(nodes, np.concatenate(right)).reshape(-1, 4)},
        "slave": np.searchsorted(nodes, slave), "master": np.searchsorted(nodes, master),
        "rotation": np.asarray(rotation).reshape(4, 4),
    }


def volume_mesh(section, design, config, output_dir, *, progress=None):
    """Sweep the supplied reference artifact; never refit, recenter or remesh."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numerics = config["mesh"]["numerics"]
    eta = geo.span_stations(numerics)
    start = perf_counter()
    deformation = None
    mapped = []
    for j, value in enumerate(eta):
        emit(progress, f"Map span section {j + 1}/{len(eta)}", 0.05 + 0.45 * j / len(eta))
        scale = design.channel.section_scale(float(value))
        if np.isclose(scale, 1, rtol=1e-12, atol=1e-12):
            q = section["xy"]
        else:
            if deformation is None:
                emit(progress, "Factor the sparse core deformation matrix")
                deformation = shared.CoreDeformation(section, numerics)
            _, target = design.section(float(value))
            q = deformation.deform(scale, design.curves, target, design.channel.pitch)
        mapped.append(design.channel.map_points(q, float(value)))
    xyz_sections = np.asarray(mapped)
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active.")
    gmsh.initialize(["moc_v2_volume", "-v", "4"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(numerics["terminal_output"]))
        emit(progress, "Assemble volume topology")
        counts = shared.assemble_volume(section, xyz_sections, design.channel)
        emit(progress, "Check Jacobians, boundary patches and periodicity", 0.65)
        validation = shared.validate_volume(section, xyz_sections, design.channel)
        emit(progress, "Calculate volume skewness, orthogonality and edge aspect ratio")
        xyz, blocks, tags, periodic = collect_volume_arrays()
        report, metrics = analyze_volume(xyz, blocks, tags, **periodic)
        arrays = {"xyz": xyz, **{f"cells_{k}": v for k, v in blocks.items()},
                  **{f"tags_{k}": v for k, v in tags.items()}, **metrics}
        np.savez_compressed(directory / "volume_data.npz", **arrays)
        (directory / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        gmsh.write(str(directory / "stator_mesh_3d.msh"))
        emit(progress, "Extract browser boundary datasets", 0.9)
        # Extract before closing Gmsh: no reopening the volume solely to draw it.
        points, faces = shared.collect_surface_mesh()
        surfaces = {}
        for name, arrays_for_surface in faces.items():
            used = np.unique(np.concatenate([a.ravel() for a in arrays_for_surface]))
            polys = np.concatenate([np.column_stack((np.full(len(a), a.shape[1]),
                                  np.searchsorted(used, a))).ravel() for a in arrays_for_surface])
            surfaces[name] = {"points": points[used].ravel().tolist(), "polys": polys.tolist()}
        (directory / "browser_surfaces.json").write_text(json.dumps(surfaces), encoding="utf-8")
        result = {
            "geometry": design.channel.definition.to_dict(), "cells": counts,
            "section_nodes": len(section["xy"]), "volume_nodes": len(xyz),
            "span_stations": eta.tolist(), "quality": validation,
            "reference_radius_mm": design.channel.reference_radius,
            "shift_mm": design.curves["shift_mm"], "elapsed_s": perf_counter() - start,
            "inflation_profile": section["profile"], "core_transition": section["transition"],
            "periodic_rotation_left_to_right": design.channel.rotation().tolist(),
        }
        (directory / "mesh_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(progress, "Volume mesh ready", 1.0)
        return result
    finally:
        gmsh.finalize()


def export_mesh(mesh_dir, config, output_dir, *, progress=None):
    """Export existing connectivity; filename/BC/unit edits do not remesh."""
    settings = config["mesh"]["cgns"]
    filename = settings["filename"]
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".cgns":
        raise ValueError("Choose a local .cgns filename.")
    if not np.isfinite(settings["length_scale"]) or settings["length_scale"] <= 0:
        raise ValueError("CGNS length scale must be positive.")
    shared.boundary_types(settings["boundary_types"])
    summary = json.loads((Path(mesh_dir) / "mesh_summary.json").read_text(encoding="utf-8"))
    definition = summary["geometry"]
    from scipy.interpolate import BSpline
    walls = [BSpline(definition[n]["knots"], definition[n]["control_points_rz_mm"],
                     definition[n]["degree"]) for n in ("hub", "shroud")]
    channel = geo.Channel(geo.GeometryDefinition(*walls, definition["num_blades"]))
    gmsh.initialize(["moc_v2_export", "-v", "2"], readConfigFiles=False)
    try:
        gmsh.open(str(Path(mesh_dir) / "stator_mesh_3d.msh"))
        emit(progress, "Audit boundary faces and write CGNS/ADF")
        return shared.export_cgns(Path(output_dir) / filename, settings, channel)
    finally:
        gmsh.finalize()

