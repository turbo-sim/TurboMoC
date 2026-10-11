"""Validate independently usable 2D design/mesh cases."""
from pathlib import Path
import numpy as np
import yaml

def read_configuration(config_file):
    """Read and validate geometry, mesh and workflow settings from the YAML."""
    from .geometry import geometry_2d as geo
    from .meshing import meshing_2d as msh

    with Path(config_file).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
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
