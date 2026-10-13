"""Validate independently usable 2D design/mesh cases."""
from pathlib import Path
import numpy as np
import yaml
from turbo_moc.configuration.flow_domain import to_passage

def read_configuration(config_file, row_id=None):
    """Read and validate geometry, mesh and workflow settings from the YAML."""
    from turbo_moc.geometry import passage_2d as geo
    from turbo_moc.meshing import mesh_2d as msh

    with Path(config_file).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if isinstance(config, dict) and "rows" in config:
        return from_project(config, row_id=row_id)
    if isinstance(config, dict) and "flow_domain" in config:
        if "passage" in config:
            raise ValueError("Use flow_domain or legacy passage, not both.")
        config["passage"] = to_passage(config.pop("flow_domain"))
    if not isinstance(config, dict) or set(config) != {
        "nozzle",
        "blade",
        "passage",
        "mesh",
        "workflow",
    }:
        raise ValueError(
            "Configuration must contain nozzle, blade, passage, mesh, and workflow mappings."
        )
    if not all(isinstance(value, dict) for value in config.values()):
        raise ValueError("Each configuration section must be a mapping.")
    validate_workflow(config["workflow"])
    passage = config["passage"]
    for key in ("inlet_chords", "outlet_chords"):
        if not np.isfinite(passage[key]) or passage[key] <= 0:
            raise ValueError(f"passage.{key} must be finite and positive.")
    geo.validate_domain_shift(
        passage.get("domain_shift_mode", "manual"), passage.get("domain_shift", 0.0)
    )
    geo.validate_flow_domain_settings(passage.get("smooth", {}))

    msh.validate_mesh_settings(config["mesh"])
    return config


def validate_workflow(workflow):
    """Reject invalid execution/figure controls before solving or opening viewers."""
    if set(workflow) != {"generate_mesh", "geometry_figure", "reporting"}:
        raise ValueError("workflow must define generate_mesh, geometry_figure and reporting.")
    if type(workflow["generate_mesh"]) is not bool:
        raise ValueError("workflow.generate_mesh must be true or false.")
    validate_geometry_figure(workflow["geometry_figure"])
    reporting = workflow["reporting"]
    if not isinstance(reporting, dict) or set(reporting) != {"level", "filename"}:
        raise ValueError("workflow.reporting must define level and filename.")
    if reporting["level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("workflow.reporting.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    filename = reporting["filename"]
    if (not isinstance(filename, str) or Path(filename).name != filename or
            "/" in filename or "\\" in filename or not filename.endswith(".log")):
        raise ValueError("workflow.reporting.filename must be a filename ending in .log.")


def validate_geometry_figure(figure):
    """Validate construction-figure controls shared by planar and Fluent workflows."""
    from matplotlib.backend_bases import FigureCanvasBase

    if not isinstance(figure, dict) or set(figure) != {"save", "show", "size", "dpi", "formats"}:
        raise ValueError("workflow.geometry_figure must define save, show, size, dpi and formats.")
    for name in ("save", "show"):
        if type(figure[name]) is not bool:
            raise ValueError(f"workflow.geometry_figure.{name} must be true or false.")
    size = figure["size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or
            any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
                for value in size)):
        raise ValueError("workflow.geometry_figure.size must contain two positive finite dimensions.")
    if type(figure["dpi"]) is not int or figure["dpi"] <= 0:
        raise ValueError("workflow.geometry_figure.dpi must be a positive integer.")
    formats = figure["formats"]
    supported = FigureCanvasBase.get_supported_filetypes()
    if (not isinstance(formats, list) or not formats or
            any(not isinstance(extension, str) or not extension.startswith(".") or
                extension[1:] not in supported for extension in formats) or
            len(set(formats)) != len(formats)):
        raise ValueError("workflow.geometry_figure.formats must list unique supported extensions, e.g. [.png, .svg].")


def from_project(project, row_id=None):
    """Adapt the app project to a 2D reference workflow without changing its inputs.

    Span controls remain stored but are unused in 2D; only the five section
    boundary types are passed to the planar CGNS writer.
    """
    from copy import deepcopy
    from turbo_moc.configuration.row import configuration_from_data, to_shared
    row = configuration_from_data(project, row_id=row_id)
    shared = to_shared(row)
    mesh = deepcopy(row["mesh"])
    mesh["cgns"]["boundary_types"] = {key: value for key, value in
        mesh["cgns"]["boundary_types"].items() if key not in ("hub", "shroud")}
    workflow = row["workflow"]
    mesh["show_gui"] = workflow["show_gmsh"]
    execution = {
        "generate_mesh": workflow.get("generate_mesh", True),
        "geometry_figure": workflow.get("geometry_figure", {
            "save": workflow["save_figures"], "show": False,
            "size": [14, 8], "dpi": 300, "formats": [".png", ".svg"]}),
        "reporting": workflow.get("reporting", {
            "level": workflow["log_level"],
            "filename": workflow.get("log_filename", "mesh_generation.log")}),
    }
    validate_workflow(execution)
    from turbo_moc.meshing.mesh_2d import validate_mesh_settings
    validate_mesh_settings(mesh)
    return {"nozzle": shared["nozzle"], "blade": shared["blade"],
            "passage": shared["passage"], "mesh": mesh, "workflow": execution,
            "row_configuration": row}


def build_geometry(nozzle, config):
    """Use the app geometry/scaling when a project was supplied."""
    from turbo_moc.geometry import passage_2d, blade_row
    if "row_configuration" in config:
        design = blade_row.build_design(nozzle, config["row_configuration"])
        suction = passage_2d.xy(design.blade["suction"])
        count = len(suction) - len(nozzle["wall_final"]["x"]) - 3
        curves = design.curves | {
            "suction": suction, "pressure": passage_2d.xy(design.blade["pressure"]),
            "construction_camberline": suction[:count],
        }
        return design.blade, curves
    return passage_2d.build_geometry(nozzle, config)
