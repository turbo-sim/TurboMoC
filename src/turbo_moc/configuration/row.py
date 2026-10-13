"""TurboMoC configuration names and validation for the shared numerical APIs."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from turbo_moc.configuration.flow_domain import normalize as normalize_domain, to_passage


def validate_mesh_configuration(mesh):
    """Validate reusable section/span/export controls independently of a blade provider."""
    from turbo_moc.geometry import meridional_channel as geo
    from turbo_moc.meshing import mesh_2d as msh2, mesh_3d as msh

    msh.pyvista_options(mesh.get("pyvista", {}))
    geo.span_stations(mesh["numerics"])
    # The section numerical/plot validation applies, but swept rows have seven BCs.
    msh2.validate_mesh_settings(mesh | {"cgns": {"enabled": False}, "report": {"enabled": False}})
    cgns = mesh["cgns"]
    if cgns["enabled"]:
        name = cgns["filename"]
        if Path(name).name != name or Path(name).suffix.lower() != ".cgns":
            raise ValueError("mesh.cgns.filename must be a local .cgns filename.")
        if not np.isfinite(cgns["length_scale"]) or cgns["length_scale"] <= 0:
            raise ValueError("mesh.cgns.length_scale must be positive.")
        if set(cgns["boundary_types"]) != set(msh.BOUNDARIES):
            raise ValueError("CGNS boundary_types must name all seven 3D boundaries.")
        msh.boundary_types(cgns["boundary_types"])
    for name in VIEW_DEFAULTS:
        value = mesh["visualization"][name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"mesh.visualization.{name} must be positive.")
    return mesh


def _validate_shared(config):
    """Validate a YAML-shaped mapping, including browser-edited configurations."""
    from turbo_moc.geometry import meridional_channel as geo
    from turbo_moc.geometry import passage_2d as planar_geometry

    if not isinstance(config, dict) or set(config) != {
        "nozzle",
        "blade",
        "passage",
        "machine",
        "mesh",
        "workflow",
    }:
        raise ValueError(
            "YAML requires nozzle, blade, passage, machine, mesh, and workflow mappings."
        )
    if not all(isinstance(value, dict) for value in config.values()):
        raise ValueError("Each top-level section must be a mapping.")
    workflow = config["workflow"]
    for name in ("save_figures", "show_gmsh", "show_pyvista"):
        if type(workflow.get(name)) is not bool:
            raise ValueError(f"workflow.{name} must be true or false.")
    if workflow.get("log_level") not in (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ):
        raise ValueError(
            "workflow.log_level must be DEBUG, INFO, WARNING, ERROR or CRITICAL."
        )
    geo.validate_machine(config["machine"])
    planar_geometry.validate_flow_domain_settings(config["passage"].get("smooth", {}))
    planar_geometry.validate_domain_shift(
        config["passage"]["domain_shift_mode"], config["passage"]["domain_shift"]
    )
    for name in ("inlet_chords", "outlet_chords"):
        if not np.isfinite(config["passage"][name]) or config["passage"][name] <= 0:
            raise ValueError(f"passage.{name} must be positive.")
    validate_mesh_configuration(config["mesh"])
    from turbo_moc.geometry.stator_blade import STATOR_PARAMETRIZATIONS
    if config["blade"].get("parametrization", "full") not in STATOR_PARAMETRIZATIONS:
        raise ValueError(f"blade_geometry.parametrization must be one of {tuple(STATOR_PARAMETRIZATIONS)}.")
    if "nozzle" in config["mesh"]:
        from turbo_moc.geometry.nozzle import validate_settings
        validate_settings(config["mesh"]["nozzle"])
    from turbo_moc.configuration.fluid_inputs import validate_inlet_inputs, validate_outlet_inputs
    validate_inlet_inputs(config["nozzle"])
    validate_outlet_inputs(config["nozzle"])
    return config


SECTIONS = {
    "nozzle": "method_of_characteristics",
    "machine": "meridional_channel",
    "blade": "blade_geometry",
    "passage": "flow_domain",
}
SMOOTH_FIELDS = (
    "inlet_extension_type", "outlet_extension_type",
    "inlet_axial_tangent", "inlet_blade_tangent",
    "outlet_blade_tangent", "outlet_axial_tangent", "num_points",
)
VIEW_DEFAULTS = {"section_edge_width": 0.35, "volume_edge_width": 0.35}


def normalize(config):
    """Accept old case files; publish current row names and nested domain inputs."""
    result = deepcopy(config)
    for old, new in SECTIONS.items():
        if old in result:
            if new in result:
                raise ValueError(f"Use either {old} or {new}, not both.")
            result[new] = result.pop(old)
    result["flow_domain"] = normalize_domain(result["flow_domain"])
    if isinstance(result.get("blade_geometry"), dict):
        result["blade_geometry"] = {"parametrization": result["blade_geometry"].get("parametrization", "full"),
                                    **result["blade_geometry"]}
    if isinstance(result["mesh"].get("numerics"), dict):
        result["mesh"]["numerics"].setdefault("recombine_minimum_quality", 0.1)
    view = result["mesh"].setdefault("visualization", {})
    for name, value in VIEW_DEFAULTS.items():
        view.setdefault(name, value)
    return result


def to_shared(config):
    """Adapt to numerical APIs; design objects and jobs retain UI-facing names."""
    result = deepcopy(config)
    for old, new in SECTIONS.items():
        result[old] = result.pop(new)
    result["passage"] = to_passage(result["passage"])
    return result


def validate_configuration(config):
    _validate_shared(to_shared(config))
    return config


def read_configuration(path, row_id=None):
    """Read one numerical row, accepting both flat cases and machine projects.

    The app uses load_project to retain all rows. Existing scripts/figures keep
    receiving the same flat contract, with an optional explicit row selection.
    """
    with Path(path).open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    return configuration_from_data(content, row_id=row_id)


def configuration_from_data(content, row_id=None):
    """Validate an in-memory project/row and select an active stator."""
    if isinstance(content, dict) and "rows" in content:
        from turbo_moc.configuration.project import normalize_project, validate_project, active_rows, get_row, row_configuration, row_assembly
        project = validate_project(normalize_project(content))
        if row_id is None:
            supported = [row for row in active_rows(project) if row["role"] == "stator"]
            if not supported:
                raise ValueError("This MoC workflow requires an active stator; use the app for rotor design and meshing.")
            row_id = supported[0]["id"]
        elif row_id not in {row["id"] for row in active_rows(project)}:
            raise ValueError(f"Blade row {row_id!r} is not active in this machine project.")
        if get_row(project, row_id)["role"] != "stator":
            raise ValueError("This MoC workflow requires a stator; use the app for rotor design and meshing.")
        config = validate_configuration(row_configuration(project, row_id))
        assembly = row_assembly(project, row_id)
        if assembly is not None:
            config["assembly"] = assembly
        return config
    if row_id is not None and row_id != "stator_1":
        raise ValueError("Historical flat cases contain only the stator_1 row.")
    return validate_configuration(normalize(content))

