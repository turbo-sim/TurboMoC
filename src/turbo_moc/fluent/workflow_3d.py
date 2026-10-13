"""Prepare and solve one stationary 3D stator passage using the shared mesher.

Paths resolve relative to the supplied workflow YAML. Importing this module
does not load PyFluent/barotropy or launch an external process.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

from . import common
from .barotropic_model import generate_barotropic_model, positive
from .graphics_3d import prepare_contour_graphics, export_contours
from .import_checks import validate_import
from turbo_moc.configuration.row import read_configuration, to_shared, configuration_from_data
from turbo_moc.configuration.automation import split_workflow
from turbo_moc.configuration.planar import validate_geometry_figure
from turbo_moc.configuration.project import get_row, load_project, normalize_project, validate_project
from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.geometry import blade_row as geo
from turbo_moc.workflows.nozzle import solve_nozzle
from turbo_moc.meshing import mesh_3d as mesh3
from ..reporting import (
    get_logger,
    logging_context,
    heading,
    step,
    summary,
    display_path,
)

logger = get_logger("fluent.3d")
BOUNDARIES = (
    "inlet",
    "outlet",
    "blade",
    "periodic_left",
    "periodic_right",
    "hub",
    "shroud",
)
WALLS = ("blade", "hub", "shroud")
RESIDUALS = {"continuity", "x-velocity", "y-velocity", "z-velocity", "k", "omega"}


def read_workflow(config_file):
    """Read the workflow and adapt its selected stator to the meshing contract."""
    config_file = Path(config_file).resolve()
    with config_file.open(encoding="utf-8") as stream:
        return validate_workflow(yaml.safe_load(stream), config_file)


def validate_workflow(workflow, config_file):
    required = {
        "design_file",
        "row_id",
        "output_dir",
        "geometry_figure",
        "show_gmsh",
        "show_pyvista",
        "save_mesh_figure",
        "prepare_only",
        "run_case",
        "barotropic",
        "fluent",
        "reporting",
    }
    workflow, inline_project = split_workflow(workflow, required)
    common.validate_reporting(workflow["reporting"])
    validate_geometry_figure(workflow["geometry_figure"])
    for name in (
        "show_gmsh",
        "show_pyvista",
        "save_mesh_figure",
        "prepare_only",
        "run_case",
    ):
        if type(workflow[name]) is not bool:
            raise ValueError(f"{name} must be true or false.")
    if not isinstance(workflow["row_id"], str) or not workflow["row_id"]:
        raise ValueError("row_id must explicitly select a stationary stator row.")
    config_file = Path(config_file).resolve()
    workflow["output_dir"] = common.workflow_output(workflow, config_file)
    if inline_project is None:
        workflow["design_file"] = (config_file.parent / workflow["design_file"]).resolve()
        project = load_project(workflow["design_file"])
    else:
        workflow["design_file"] = config_file
        project = validate_project(normalize_project(inline_project))
    row = get_row(project, workflow["row_id"])
    if row["role"] != "stator":
        raise ValueError(
            "The Fluent 3D workflow supports stationary stator rows only; "
            f"selected row {workflow['row_id']!r} is a {row['role']}."
        )
    config = configuration_from_data(project, row_id=workflow["row_id"])
    nozzle = config["method_of_characteristics"]
    if nozzle.get("delta_flow", 0.0) != 0:
        raise ValueError(
            "The stator template requires planar MoC flow (delta_flow: 0)."
        )
    p0 = positive(solver_parameters(nozzle)["P0"], "method_of_characteristics.P0")
    if positive(nozzle.get("p_back", np.nan), "method_of_characteristics.p_back") >= p0:
        raise ValueError("The stator expansion requires p_back < P0.")
    cgns = config["mesh"]["cgns"]
    if not cgns["enabled"] or not np.isclose(
        cgns["length_scale"], 0.001, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Enable mesh.cgns with length_scale: 0.001 for metre coordinates."
        )
    for name in ("barotropic", "fluent"):
        if not isinstance(workflow[name], dict):
            raise ValueError(f"{name} must be a mapping.")
    model = workflow["barotropic"]
    if model["calculation_type"] != "equilibrium" or model["efficiency"] != 1.0:
        raise ValueError(
            "Use calculation_type: equilibrium and efficiency: 1.0 to match MoC."
        )
    if positive(model["pressure_ratio"], "barotropic.pressure_ratio") <= 1:
        raise ValueError("barotropic.pressure_ratio must exceed 1.")
    if type(model["polynomial_degree"]) is not int or model["polynomial_degree"] < 1:
        raise ValueError("polynomial_degree must be a positive integer.")
    positive(model["ODE_tolerance"], "ODE_tolerance")
    fluent = workflow["fluent"]
    common.validate_solver_options(fluent, boundary_roles=BOUNDARIES)
    if set(fluent["residual_equations"]) != RESIDUALS:
        raise ValueError(f"3D SST requires residual criteria for {sorted(RESIDUALS)}.")
    if type(fluent["wall_yplus"]) is not bool:
        raise ValueError("fluent.wall_yplus must be true or false.")
    if fluent["view"] not in (
        "isometric",
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ):
        raise ValueError("fluent.view must name a standard Fluent view.")
    return workflow, config


def geometry_context(design, config):
    """Describe periodicity and the physical midspan surface in SI units."""
    mapping = design.meridional_channel
    scale = float(config["mesh"]["cgns"]["length_scale"])
    transform = np.asarray(mapping.periodic_transform(), dtype=float).copy()
    transform[:3, 3] *= scale
    kind = config["meridional_channel"]["type"]
    dimensions = config["meridional_channel"]
    section = (
        {
            "type": "cylinder",
            "radius_m": (dimensions["hub_radius"] + dimensions["blade_height"] / 2)
            * scale,
        }
        if kind == "axial"
        else {
            "type": "plane",
            "point_m": [0.0, 0.0, dimensions["blade_height"] * scale / 2],
            "normal": [0.0, 0.0, 1.0],
        }
    )
    return {
        "kind": kind,
        "transform": transform.tolist(),
        "axis_origin": [0.0, 0.0, 0.0],
        "axis_direction": [0.0, 0.0, 1.0],
        "angle_deg": (
            0.0 if mapping.is_linear else float(-np.degrees(mapping.pitch_angle))
        ),
        "translation_m": transform[:3, 3].tolist(),
        "section": section,
    }


def validate_periodic_vertices(vertices, transform):
    """Require a finite bijection under the geometry's left-to-right transform."""
    points = [np.asarray(value, dtype=float) for value in vertices]
    if len(points) != 2 or any(p.ndim != 2 or p.shape[1] != 3 for p in points):
        raise RuntimeError(
            "Periodic vertices must contain two arrays of XYZ coordinates."
        )
    if any(not np.isfinite(p).all() for p in points):
        raise RuntimeError("Imported periodic vertices contain nonfinite coordinates.")
    left, right = (np.unique(p, axis=0) for p in points)
    if left.shape != right.shape or len(left) < 3:
        raise RuntimeError(
            "Imported periodic boundaries have inconsistent vertex counts."
        )
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("The periodic transform must be a finite 4x4 matrix.")
    mapped = left @ matrix[:3, :3].T + matrix[:3, 3]
    length = max(float(np.ptp(np.vstack((left, right)), axis=0).max()), 1e-6)
    tolerance = max(1e-9, length * 1e-6)
    distances, indices = cKDTree(right).query(mapped)
    if len(np.unique(indices)) != len(left) or float(distances.max()) > tolerance:
        raise RuntimeError(
            "Imported periodic vertices do not match the geometry transform."
        )
    return {
        "node_count": len(left),
        "max_mismatch_m": float(distances.max()),
        "tolerance_m": tolerance,
    }


def configure_periodicity(settings, expected, geometry, vertices):
    """Validate imported coordinates, then establish a conformal periodic pair."""
    bc = settings.setup.boundary_conditions
    types = common.boundary_types(bc)
    left = common.resolve_zone(expected["periodic_left"], types)
    right = common.resolve_zone(expected["periodic_right"], types)
    audit = validate_periodic_vertices(vertices, geometry["transform"])
    rotational = geometry["kind"] != "linear"
    names = bc.periodic.get_object_names()
    if names and set(names) != {left, right}:
        raise RuntimeError(f"Expected one periodic pair; found {names}.")
    # Fluent 24.2 keeps conformal-periodic axes on the adjacent cell zone;
    # the periodic boundary's own axis fields are inactive in this model.
    fluids = settings.setup.cell_zone_conditions.fluid
    for name in fluids.get_object_names():
        frame = fluids[name].reference_frame
        frame.frame_motion = False
        if rotational:
            for i, value in enumerate(geometry["axis_origin"]):
                frame.reference_frame_axis_origin[i].value = value
            for i, value in enumerate(geometry["axis_direction"]):
                frame.reference_frame_axis_direction[i].value = value
    if not names:
        arguments = dict(
            zone_name=left,
            shadow_zone_name=right,
            rotate_periodic=rotational,
            create=True,
            auto_translation=rotational,
        )
        if not rotational:
            arguments["direction"] = geometry["translation_m"]
        settings.mesh.modify_zones.make_periodic(**arguments)
    names = bc.periodic.get_object_names()
    if set(names) != {left, right}:
        raise RuntimeError(
            f"Fluent did not create the expected conformal periodic pair: {names}."
        )
    for name in names:
        periodic = bc.periodic[name].periodic
        periodic.rotationally_periodic = "Rotational" if rotational else "Translational"
        if periodic.p_jump.is_active():
            periodic.p_jump = 0.0
    settings.mesh.check()
    geometry["periodic_validation"] = audit
    logger.info(
        "  Periodic coordinate audit: %d nodes, maximum mismatch %.3e m.",
        audit["node_count"],
        audit["max_mismatch_m"],
    )
    return left


def configure_fluent(solver, mesh_file, state, expressions, geometry, options):
    """Import, audit and configure one stationary volume without initializing."""
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    settings = solver.settings
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    common.configure_material(settings, expressions, options, dimension=3)
    bc = settings.setup.boundary_conditions
    types = common.boundary_types(bc)
    zones = {
        role: common.resolve_zone(options["boundaries"][role], types)
        for role in BOUNDARIES
    }
    geometry["import_validation"] = validate_import(solver, zones, geometry.get("mesh"))
    for role, kind in (
        ("inlet", "pressure-inlet"),
        ("outlet", "pressure-outlet"),
        *((role, "wall") for role in WALLS),
    ):
        if types[zones[role]].replace("_", "-") != kind:
            settings.mesh.modify_zones.zone_type(
                zone_names=[zones[role]], new_type=kind
            )
    periodic_zones = [zones[role] for role in ("periodic_left", "periodic_right")]
    surfaces = solver.fields.field_data.get_field_data(
        SurfaceFieldDataRequest(
            surfaces=periodic_zones,
            data_types=[SurfaceDataType.Vertices],
        )
    )
    vertices = tuple(surfaces[name].vertices for name in periodic_zones)
    zones["periodic"] = configure_periodicity(
        settings, options["boundaries"], geometry, vertices
    )
    common.configure_flow_boundaries(settings, zones, state, options, wall_roles=WALLS)
    common.configure_numerics(settings, options)
    settings.solution.initialization.initialization_type = "hybrid"
    prepare_contour_graphics(solver, options, geometry)
    summary(
        logger,
        "Fluent 3D configuration",
        {
            "Solver": "Steady, pressure-based coupled; 3D double precision",
            "Turbulence": "SST k-omega",
            "Energy equation": "Disabled",
            "Geometry": geometry["kind"],
            "Periodic angle [deg]": geometry["angle_deg"],
            **{f"Boundary: {role}": zones[role] for role in BOUNDARIES},
        },
    )
    return zones


def solution_diagnostics(solver, options):
    """Preserve shared flow checks and report y+ on each of the three walls."""
    result = common.solution_diagnostics(solver, options)
    if options["wall_yplus"]:
        reports = solver.settings.solution.report_definitions
        walls = solver.settings.setup.boundary_conditions.wall.get_object_names()
        names = []
        for role in WALLS:
            zone = common.resolve_zone(options["boundaries"][role], walls)
            for kind, report_type in (
                ("mean", "surface-areaavg"),
                ("max", "surface-facetmax"),
            ):
                name = f"workflow-{kind}-yplus-{role}"
                reports.surface[name] = {
                    "report_type": report_type,
                    "field": "y-plus",
                    "surface_names": [zone],
                }
                names.append(name)
        for item in reports.compute(report_defs=names):
            for name, values in item.items():
                result[name.removeprefix("workflow-")] = float(values[0])
    return result


def run_stages(solver, options, output_dir, state):
    return common.run_stages(
        solver, options, output_dir, state, diagnostics=solution_diagnostics
    )


def save_fluent_case(pyfluent, workflow, mesh_file, state, expressions, geometry):
    return common.save_fluent_case(
        pyfluent,
        workflow,
        mesh_file,
        state,
        expressions,
        geometry,
        dimension=3,
        configure=configure_fluent,
        run=run_stages,
        export=export_contours,
        metadata={"dimension": 3, "row_id": workflow["row_id"], "geometry": geometry},
    )


def plot_reference_geometry(design, nozzle, *, figure_size):
    """Adapt the scaled 3D reference blade to the planar construction plot."""
    # The 3D passage stores boundary curves only. Construction surfaces belong
    # to the scaled blade; plotting the original MoC wall would mix coordinates.
    suction = geo.planar.xy(design.blade["suction"])
    n_convergent = len(suction) - len(nozzle["wall_final"]["x"]) - 3
    if n_convergent < 2:
        raise RuntimeError("Unexpected stator suction-surface layout.")
    curves = design.curves | {
        "suction": suction,
        "pressure": geo.planar.xy(design.blade["pressure"]),
        "construction_camberline": suction[:n_convergent],
    }
    figure, axes = geo.planar.plot_construction(
        design.blade, curves, figure_size=figure_size
    )
    for axis in axes:
        if design.meridional_channel.is_linear:
            axis.set_xlabel("Pitchwise reference coordinate [mm]")
            axis.set_ylabel("Negative streamwise reference coordinate [mm]")
        else:
            axis.set_xlabel("Conformal pitchwise reference coordinate [mm]")
            axis.set_ylabel("Conformal streamwise reference coordinate [mm]")
    return figure, axes


def prepare_mesh(workflow, config, output_dir):
    """Use the unchanged 3D mesher and retain its one fresh MoC solution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 5 if workflow["prepare_only"] else (8 if workflow["run_case"] else 6)
    step(logger, 2, total, "Calculate the MoC design")
    nozzle = solve_nozzle(config["method_of_characteristics"])
    step(logger, 3, total, "Build the 3D stator geometry")
    design = geo.build_design(nozzle, config)
    geometry = geometry_context(design, config)
    figures = workflow["geometry_figure"]
    if figures["save"] or figures["show"]:
        figure, _ = plot_reference_geometry(design, nozzle, figure_size=figures["size"])
        try:
            if figures["save"]:
                for extension in figures["formats"]:
                    figure.savefig(
                        output_dir / f"stator_reference_geometry{extension}",
                        dpi=figures["dpi"],
                        bbox_inches="tight",
                    )
            if figures["show"]:
                plt.show()
        finally:
            plt.close(figure)
    step(logger, 4, total, "Generate, validate and export the 3D mesh")

    def progress(label, fraction=None):
        logger.info("  %s", label)

    mesh3.reference_mesh(design, config, output_dir, progress=progress)
    section, saved_design = mesh3.load_section(output_dir)
    mesh3.volume_mesh(section, saved_design, config, output_dir, progress=progress)
    mesh3.export_mesh(output_dir, config, output_dir, progress=progress)
    geometry["mesh"] = mesh_expectations(output_dir)
    (output_dir / "fluent_geometry.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )
    mesh_file = output_dir / config["mesh"]["cgns"]["filename"]
    if workflow["show_gmsh"]:
        import gmsh

        gmsh.initialize(readConfigFiles=False)
        try:
            gmsh.open(str(output_dir / "stator_mesh_3d.msh"))
            gmsh.fltk.run()
        finally:
            gmsh.finalize()
    if workflow["show_pyvista"] or workflow["save_mesh_figure"]:
        mesh3.view_mesh_pyvista(
            output_dir,
            config["mesh"]["pyvista"],
            show=workflow["show_pyvista"],
            save_figure=workflow["save_mesh_figure"],
        )
    return nozzle, geometry, mesh_file


def mesh_expectations(output_dir):
    """Record exported cell count and SI bounds for the Fluent import audit."""
    directory = Path(output_dir)
    report = json.loads((directory / "mesh_summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "volume_data.npz") as data:
        xyz = data["xyz"] * 0.001
        bounds = [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
    return {"cell_count": sum(report["cells"].values()), "bounds_m": bounds}


def main(config_file):
    """Prepare mesh/material, optionally configure and solve, and report outputs."""
    config_file = Path(config_file).resolve()
    with config_file.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Workflow YAML must be a mapping.")
    common.validate_reporting(raw.get("reporting"))
    output_dir = common.workflow_output(raw, config_file)
    with logging_context(
        output_dir / raw["reporting"]["filename"], raw["reporting"]["level"]
    ):
        heading(logger, "Fluent 3D stator workflow")
        total = 5 if raw.get("prepare_only") else (8 if raw.get("run_case") else 6)
        step(logger, 1, total, "Validate configuration and load dependencies")
        workflow, config = validate_workflow(raw, config_file)
        barotropy, pyfluent = common.load_dependencies(workflow["prepare_only"])
        nozzle, geometry, mesh_file = prepare_mesh(
            workflow, config, output_dir / "gmsh"
        )
        step(logger, 5, total, "Generate and validate the barotropic model")
        expressions, state = generate_barotropic_model(
            barotropy,
            to_shared(config),
            nozzle,
            workflow["barotropic"],
            output_dir / "barotropy",
        )
        results = {
            "Row": workflow["row_id"],
            "Mesh": display_path(mesh_file),
            "Output directory": display_path(output_dir),
        }
        if not workflow["prepare_only"]:
            case = save_fluent_case(
                pyfluent, workflow, mesh_file, state, expressions, geometry
            )
            results["Case"] = display_path(case)
            if workflow["run_case"]:
                run = json.loads(
                    (case.parent / "run_summary.json").read_text(encoding="utf-8")
                )
                results.update(
                    {
                        "Converged": run["converged"],
                        "Data": run["data_file"],
                        "Relative mass imbalance": run["diagnostics"][
                            "relative_mass_imbalance"
                        ],
                    }
                )
        summary(logger, "Fluent 3D workflow completed", results)
