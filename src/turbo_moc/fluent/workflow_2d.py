"""Generate a 2D MoC mesh, barotropic model and staged Fluent solution.

Run ``python -m examples.fluent_2D.run_fluent_case``.
Settings live in the adjacent workflow YAML; the numerical callable accepts its path.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

from .barotropic_model import generate_barotropic_model, positive
from . import common
from .common import (
    MACH_EXPRESSION,
    SolverStage,
    boundary_types,
    configure_material,
    configure_flow_boundaries,
    configure_numerics,
    configure_stage,
    contour_definitions,
    initialize_solution,
    load_dependencies,
    record_stage_iterations,
    residual_history,
    resolve_zone,
    set_order,
    solution_diagnostics,
    solution_stages,
    validate_reporting,
    validate_solver_options,
    workflow_output,
)
from turbo_moc.geometry import passage_2d as geo
from turbo_moc.workflows.nozzle import solve_nozzle
from turbo_moc.meshing.mesh_2d import create_mesh
from turbo_moc.configuration.planar import read_configuration, validate_geometry_figure, from_project, build_geometry
from turbo_moc.configuration.automation import split_workflow
from turbo_moc.configuration.fluid_inputs import solver_parameters
from turbo_moc.reporting import (
    display_path,
    get_logger,
    heading,
    step,
    summary as log_summary,
    logging_context,
)

logger = get_logger("fluent.runner")


def read_workflow(config_file):
    """Load an embedded app project or a historical design_file workflow."""
    config_file = Path(config_file).resolve()
    with config_file.open(encoding="utf-8") as stream:
        workflow = yaml.safe_load(stream)
    return validate_workflow(workflow, config_file)


def validate_workflow(workflow, config_file):
    """Validate parsed inputs separately so configuration errors can be logged."""
    config_file = Path(config_file).resolve()
    required = {
        "design_file",
        "output_dir",
        "geometry_figure",
        "show_gmsh",
        "prepare_only",
        "run_case",
        "barotropic",
        "fluent",
        "reporting",
    }
    workflow, project = split_workflow(workflow, required, optional={"row_id"})
    validate_reporting(workflow["reporting"])
    validate_geometry_figure(workflow["geometry_figure"])
    for section in ("barotropic", "fluent"):
        if not isinstance(workflow[section], dict):
            raise ValueError(f"{section} must be a mapping.")
    for name in (
        "show_gmsh",
        "prepare_only",
        "run_case",
    ):
        if type(workflow[name]) is not bool:
            raise ValueError(f"{name} must be true or false.")
    workflow["output_dir"] = workflow_output(workflow, config_file)
    if project is None:
        workflow["design_file"] = (config_file.parent / workflow["design_file"]).resolve()
        config = read_configuration(workflow["design_file"], row_id=workflow.get("row_id"))
    else:
        workflow["design_file"] = config_file
        config = from_project(project, row_id=workflow.get("row_id"))
    if ("row_configuration" in config and
            config["row_configuration"]["meridional_channel"]["type"] != "linear"):
        raise ValueError("Fluent 2D requires a linear-cascade app project. "
                         "Use Fluent 3D for axial/radial passages; their 2D grids are conformal references.")
    nozzle = config["nozzle"]
    if nozzle.get("delta_flow", 0.0) != 0:
        raise ValueError(
            "The Fluent stator workflow requires planar MoC flow (delta_flow: 0)."
        )
    p0 = positive(solver_parameters(nozzle)["P0"], "nozzle.P0")
    from turbo_moc.configuration.fluid_inputs import exit_pressure
    p_back = positive(exit_pressure(nozzle) if "p_back" in nozzle else np.nan, "nozzle.p_back")
    if p_back >= p0:
        raise ValueError("The stator expansion requires nozzle.p_back < nozzle.P0.")
    cgns = config["mesh"].get("cgns", {})
    if not cgns.get("enabled", False) or not np.isclose(
        cgns["length_scale"], 0.001, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Enable mesh.cgns with length_scale: 0.001 to import metre coordinates in Fluent."
        )
    barotropic, fluent = workflow["barotropic"], workflow["fluent"]
    if (
        barotropic["calculation_type"] != "equilibrium"
        or barotropic["efficiency"] != 1.0
    ):
        raise ValueError(
            "Use calculation_type: equilibrium and efficiency: 1.0 to match the MoC isentrope."
        )
    if positive(barotropic["pressure_ratio"], "barotropic.pressure_ratio") <= 1:
        raise ValueError("barotropic.pressure_ratio must exceed 1.")
    if (
        type(barotropic["polynomial_degree"]) is not int
        or barotropic["polynomial_degree"] < 1
    ):
        raise ValueError("polynomial_degree must be a positive integer.")
    positive(barotropic["ODE_tolerance"], "ODE_tolerance")
    validate_solver_options(fluent)
    return workflow, config


def configure_periodicity(settings, expected, pitch_m, vertices):
    """Use imported conformity or explicitly pair the two pitchwise boundaries."""
    bc = settings.setup.boundary_conditions
    types = boundary_types(bc)
    left = resolve_zone(expected["periodic_left"], types)
    right = resolve_zone(expected["periodic_right"], types)
    left_points, right_points = (
        np.unique(np.asarray(points, dtype=float), axis=0) for points in vertices
    )
    if (
        left_points.shape != right_points.shape
        or len(left_points) < 2
        or not np.isfinite(left_points).all()
        or not np.isfinite(right_points).all()
    ):
        raise RuntimeError(
            "The imported periodic boundaries have inconsistent vertices."
        )
    translated = left_points.copy()
    translated[:, 0] += pitch_m
    distances, indices = cKDTree(right_points).query(translated)
    if len(np.unique(indices)) != len(left_points) or np.max(distances) > max(
        1e-9, pitch_m * 1e-6
    ):
        raise RuntimeError(
            f"Imported periodic translation does not match pitch {pitch_m} m."
        )
    periodic_names = bc.periodic.get_object_names()
    if periodic_names:
        # Fluent 2024 R2 includes both sides in the periodic collection.
        if set(periodic_names) != {left, right}:
            raise RuntimeError(f"Expected one periodic pair; found {periodic_names}.")
    else:
        settings.mesh.modify_zones.make_periodic(
            zone_name=left,
            shadow_zone_name=right,
            rotate_periodic=False,
            create=True,
            auto_translation=False,
            direction=[pitch_m, 0.0],
        )
        periodic_names = bc.periodic.get_object_names()
        if set(periodic_names) != {left, right}:
            raise RuntimeError("Fluent did not create the conformal periodic pair.")
    for name in periodic_names:
        periodic = bc.periodic[name].periodic
        periodic.rotationally_periodic = "Translational"
        # Pressure-jump controls belong to streamwise periodic flow. The stator
        # has inlet/outlet boundaries, so these controls are normally inactive.
        if periodic.p_jump.is_active():
            periodic.p_jump = 0.0
    return left


def configure_fluent(solver, mesh_file, state, expressions, pitch_m, options):
    """Configure a steady pressure-based coupled setup, without initialization."""
    settings = solver.settings
    # Import the metre-scaled CGNS with named boundaries and periodic metadata.
    logger.info(
        "  Import CGNS mesh and check its topology: %s", display_path(mesh_file)
    )
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    logger.info("  Select steady planar flow, SST k-omega and the barotropic material.")
    configure_material(settings, expressions, options, dimension=2)

    logger.info("  Assign inlet, outlet, blade and conformal periodic boundaries.")
    bc = settings.setup.boundary_conditions
    types = boundary_types(bc)
    zones = {
        key: resolve_zone(options["boundaries"][key], types)
        for key in ("inlet", "outlet", "blade")
    }
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    periodic_zones = [
        resolve_zone(options["boundaries"][key], types)
        for key in ("periodic_left", "periodic_right")
    ]
    surface_data = solver.fields.field_data.get_field_data(
        SurfaceFieldDataRequest(
            surfaces=periodic_zones,
            data_types=[SurfaceDataType.Vertices],
        )
    )
    zones["periodic"] = configure_periodicity(
        settings,
        options["boundaries"],
        pitch_m,
        tuple(surface_data[name].vertices for name in periodic_zones),
    )
    configure_flow_boundaries(settings, zones, state, options)

    logger.info(
        "  Configure coupled numerics, residual monitoring and contour definitions."
    )
    configure_numerics(settings, options)
    # Save the requested initialization method even in case-creation-only mode.
    settings.solution.initialization.initialization_type = "hybrid"
    # Store plot definitions and periodic display settings in every saved case.
    prepare_contour_graphics(solver, options, pitch_m)
    log_summary(
        logger,
        "Fluent configuration",
        {
            "Solver": "Steady, pressure-based coupled; 2D double precision",
            "Energy equation": "Disabled",
            "Turbulence model": "SST k-omega",
            "Initialization": "Hybrid (before the first solver stage)",
            "High-order relaxation": f"{options['high_order_relaxation_factor']:g} (all flow variables)",
            "Initial discretization": options["solution_strategy"][0][
                "order"
            ].capitalize()
            + " order",
            "Periodic pitch": f"{pitch_m:.8f} m",
            **{
                f"Boundary: {role}": zones[role]
                for role in ("inlet", "outlet", "blade")
            },
            "Periodic pair": " / ".join(periodic_zones),
            **{
                f"Residual: {name}": f"{tolerance:g}"
                for name, tolerance in options["residual_equations"].items()
            },
        },
    )
    return zones


def prepare_contour_graphics(solver, options, pitch_m):
    """Define whole-passage contours and native translational periodic instances.

    A cell zone needs an explicit zone surface to serve as a contour surface.
    Instancing is a display operation: the computation still has one passage.
    """
    if not any(settings["enabled"] for _, _, settings in contour_definitions(options)):
        return
    graphics = solver.settings.results.graphics
    surfaces = solver.settings.results.surfaces.zone_surface
    fluid_surfaces = []
    for index, zone in enumerate(
        solver.settings.setup.cell_zone_conditions.fluid.get_object_names()
    ):
        name = f"workflow-fluid-{index}"
        surfaces[name] = {"zone_name": zone}
        fluid_surfaces.append(name)

        if options["periodic_instancing"]["enabled"]:
            # Match the GUI's Periodic Instancing panel: copy in the negative
            # pitch direction, using metre coordinates imported from CGNS.
            graphics.periodic_instancing[zone] = {
                "periodic_type": "translational",
                "surfaces": [name],
                "translation": [-pitch_m, 0.0, 0.0],
                "repeats": options["periodic_instancing"]["repeats"],
            }
    for name, field, contour_options in contour_definitions(options):
        if contour_options["enabled"]:
            graphics.contour[name] = {
                "field": field,
                "filled": True,
                "node_values": True,
                "surfaces_list": fluid_surfaces,
            }


def export_contours(solver, options, output_dir):
    """Render static pressure and barotropic Mach number using Fluent graphics."""
    graphics = solver.settings.results.graphics
    picture = graphics.picture
    fluid_zones = solver.settings.setup.cell_zone_conditions.fluid.get_object_names()
    picture.driver_options.hardcopy_format = "png"
    picture.use_window_resolution = False
    filenames = []
    for name, _, contour_options in contour_definitions(options):
        if not contour_options["enabled"]:
            continue
        graphics.contour[name].display()
        if options["periodic_instancing"]["enabled"]:
            # Apply repeats to the active contour before fitting the camera.
            for zone in fluid_zones:
                graphics.periodic_instancing[zone].display()
        graphics.views.restore_view(view_name="front")
        graphics.views.auto_scale()
        picture.x_resolution = contour_options["width"]
        picture.y_resolution = contour_options["height"]
        filename = output_dir / contour_options["filename"]
        picture.save_picture(file_name=str(filename))
        if not filename.is_file() or filename.stat().st_size == 0:
            raise RuntimeError(f"Fluent did not produce the {name} contour image.")
        logger.info(
            "  %s contour saved: %s",
            name.replace("-", " ").capitalize(),
            display_path(filename),
        )
        filenames.append(filename)
    return filenames


def run_stages(solver, options, output_dir, state):
    """Run shared stages, retaining the 2D module's diagnostic extension points."""
    return common.run_stages(
        solver,
        options,
        output_dir,
        state,
        diagnostics=solution_diagnostics,
        residual_reader=residual_history,
    )


def save_fluent_case(pyfluent, workflow, mesh_file, state, expressions, pitch_m):
    """Create or solve a 2D case through the shared Fluent session lifecycle."""
    return common.save_fluent_case(
        pyfluent,
        workflow,
        mesh_file,
        state,
        expressions,
        pitch_m,
        dimension=2,
        configure=configure_fluent,
        run=run_stages,
        export=export_contours,
        metadata={"pitch_m": pitch_m},
    )


def main(config_file):
    """Run the workflow with numbered progress shared by the console and log."""
    config_file = Path(config_file).resolve()
    # Read just enough to locate the log. Full validation and dependency loading
    # happen inside the logging context, so their failures are recorded too.
    with config_file.open(encoding="utf-8") as stream:
        workflow = yaml.safe_load(stream)
    if not isinstance(workflow, dict):
        raise ValueError("Workflow YAML must be a mapping.")
    reporting = workflow.get("reporting")
    validate_reporting(reporting)
    output_dir = workflow_output(workflow, config_file)
    log_file = output_dir / reporting["filename"]
    total_steps = (
        5 if workflow.get("prepare_only") else (8 if workflow.get("run_case") else 6)
    )

    with logging_context(log_file, reporting["level"]):
        heading(logger, "Fluent stator workflow")
        step(
            logger,
            1,
            total_steps,
            "Validate configuration and load dependencies",
        )
        workflow, config = validate_workflow(workflow, config_file)
        barotropy, pyfluent = load_dependencies(workflow["prepare_only"])
        mode = (
            "Prepare mesh and model"
            if workflow["prepare_only"]
            else ("Create and solve" if workflow["run_case"] else "Create case only")
        )
        log_summary(
            logger,
            "Run settings",
            {
                "Mode": mode,
                "Workflow configuration": display_path(config_file),
                "Design configuration": display_path(workflow["design_file"]),
                "Output directory": display_path(output_dir),
                "Execution log": display_path(log_file),
            },
        )
        mesh_dir = output_dir / "gmsh"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        step(logger, 2, total_steps, "Calculate the MoC design")
        # Solve once for this invocation. Geometry and barotropy consume the
        # same result; an earlier run's nozzle file is never used as input.
        nozzle = solve_nozzle(config["nozzle"])
        step(logger, 3, total_steps, "Build blade geometry and periodic passage")
        blade, curves = build_geometry(nozzle, config)
        if not geo.assess_clearance(blade, curves)["contained"]:
            raise ValueError("Cannot mesh: a periodic boundary intersects the blade.")
        log_summary(
            logger,
            "Geometry",
            {
                "Stator pitch": f"{blade['pitch']:.4f} mm",
                "Domain shift method": curves["domain_shift_mode"],
                "Domain shift": f"{curves['shift_mm']:+.4f} mm ({curves['domain_shift']:+.5f} pitch)",
            },
        )
        figure_options = workflow["geometry_figure"]
        if figure_options["save"] or figure_options["show"]:
            figure, _ = geo.plot_construction(
                blade, curves, figure_size=figure_options["size"]
            )
            try:
                if figure_options["save"]:
                    for extension in figure_options["formats"]:
                        filename = mesh_dir / f"stator_geometry{extension}"
                        figure.savefig(
                            filename, dpi=figure_options["dpi"], bbox_inches="tight"
                        )
                        logger.info(
                            "  Geometry figure saved: %s", display_path(filename)
                        )
                if figure_options["show"]:
                    logger.info("  Opening the geometry viewer; close it to continue.")
                    plt.show()
            finally:
                plt.close(figure)

        step(logger, 4, total_steps, "Generate, validate and export the mesh")
        # This workflow owns its viewer switch independently of the mesh-only example.
        create_mesh(
            blade,
            curves,
            config["mesh"] | {"show_gui": workflow["show_gmsh"]},
            mesh_dir,
            show_gui=workflow["show_gmsh"],
        )
        step(logger, 5, total_steps, "Generate and validate the barotropic model")
        expressions, state = generate_barotropic_model(
            barotropy, config, nozzle, workflow["barotropic"], output_dir / "barotropy"
        )
        results = {"Mode": mode, "Status": "Completed"}
        if not workflow["prepare_only"]:
            case_file = save_fluent_case(
                pyfluent,
                workflow,
                mesh_dir / config["mesh"]["cgns"]["filename"],
                state,
                expressions,
                float(blade["pitch"]) * 0.001,
            )
            results["Final solution case" if workflow["run_case"] else "Setup case"] = (
                display_path(case_file)
            )
            results["Fluent transcript"] = display_path(
                case_file.with_name(case_file.name.removesuffix(".cas.h5") + ".trn")
            )
            if workflow["run_case"]:
                run_summary = json.loads(
                    (case_file.parent / "run_summary.json").read_text(encoding="utf-8")
                )
                diagnostics = run_summary["diagnostics"]
                results.update(
                    {
                        "Converged": "Yes" if run_summary["converged"] else "No",
                        "Total iterations": run_summary["last_iteration"],
                        "Relative mass imbalance": f"{diagnostics['relative_mass_imbalance']:.3e}",
                        "Final solution data": display_path(run_summary["data_file"]),
                        **{
                            f"Residual: {name}": f"{value:.3e}"
                            for name, value in run_summary["last_residuals"].items()
                        },
                    }
                )
                for name, _, contour in contour_definitions(workflow["fluent"]):
                    if contour["enabled"]:
                        results[name.replace("-", " ").capitalize() + " contour"] = (
                            display_path(case_file.parent / contour["filename"])
                        )
                if workflow["fluent"]["plot_residuals"]:
                    results["Residual plot"] = display_path(
                        case_file.parent / "residuals.png"
                    )
        results["Output directory"] = display_path(output_dir)
        results["Execution log"] = display_path(log_file)
        heading(logger, "Results")
        log_summary(logger, "Fluent workflow completed", results)
