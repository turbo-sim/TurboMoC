"""Generate a MoC mesh, barotropic model and staged Fluent solution.

Run ``poetry run python examples/fluent/run_fluent_case.py``.
All user settings live in CONFIG_FILE; no command-line arguments are needed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.fluent.barotropic_model import generate_barotropic_model, positive
from examples.gmsh.geometry_source import assess_clearance
from examples.gmsh.meshing_source import create_mesh
from examples.gmsh.run_mesh_generation import (
    build_geometry,
    load_or_solve_nozzle,
    plot_construction,
    read_configuration,
)
from examples.workflow_logging import (
    display_path,
    get_logger,
    heading,
    section,
    step,
    summary as log_summary,
    workflow_logging,
)

logger = get_logger("fluent.runner")

# User entry point: select the workflow YAML here; paths inside it are relative
# to that file. Solver stages, thermodynamics and image settings belong in YAML.
EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EXAMPLE_DIR / "stator_fluent.yaml"
MACH_EXPRESSION = "barotropic_mach_number"


@dataclass(frozen=True)
class SolverStage:
    """One solver block, including whether Fluent may stop it early."""

    name: str
    first_order: bool
    iterations: int
    time_scale_factor: float
    stop_on_convergence: bool


def solution_stages(options):
    """Read the YAML strategy; every block may stop once its residuals converge."""
    stages = []
    strategy = options["solution_strategy"]
    for index, stage in enumerate(strategy, start=1):
        stages.append(
            SolverStage(
                name=f"stage_{index}",
                first_order=stage["order"] == "first",
                iterations=stage["iterations"],
                time_scale_factor=stage["time_scale_factor"],
                stop_on_convergence=True,
            )
        )
    return stages


def read_workflow(config_file):
    """Load workflow settings and the existing, independently usable MoC YAML."""
    config_file = Path(config_file).resolve()
    with config_file.open(encoding="utf-8") as stream:
        workflow = yaml.safe_load(stream)
    return validate_workflow(workflow, config_file)


def validate_reporting(reporting):
    """Keep the execution log inside the output directory with a known level."""
    if not isinstance(reporting, dict) or set(reporting) != {"level", "filename"}:
        raise ValueError("reporting must define level and filename.")
    if reporting["level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            "reporting.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL."
        )
    filename = reporting["filename"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".log")
    ):
        raise ValueError("reporting.filename must be a filename ending in .log.")


def validate_workflow(workflow, config_file):
    """Validate parsed inputs separately so configuration errors can be logged."""
    config_file = Path(config_file).resolve()
    required = {
        "design_file",
        "output_dir",
        "moc_cache_file",
        "save_geometry_figure",
        "prepare_only",
        "run_case",
        "force_moc_solve",
        "barotropic",
        "fluent",
        "reporting",
    }
    if not isinstance(workflow, dict) or set(workflow) != required:
        raise ValueError(f"Workflow YAML must contain exactly {sorted(required)}.")
    validate_reporting(workflow["reporting"])
    for section in ("barotropic", "fluent"):
        if not isinstance(workflow[section], dict):
            raise ValueError(f"{section} must be a mapping.")
    for name in (
        "save_geometry_figure",
        "prepare_only",
        "run_case",
        "force_moc_solve",
    ):
        if type(workflow[name]) is not bool:
            raise ValueError(f"{name} must be true or false.")
    for name in ("design_file", "output_dir", "moc_cache_file"):
        workflow[name] = (config_file.parent / workflow[name]).resolve()
    config = read_configuration(workflow["design_file"])
    nozzle = config["nozzle"]
    if nozzle.get("delta_flow", 0.0) != 0:
        raise ValueError(
            "The Fluent stator workflow requires planar MoC flow (delta_flow: 0)."
        )
    p0 = positive(nozzle["P0"], "nozzle.P0")
    p_back = positive(nozzle.get("p_back", np.nan), "nozzle.p_back")
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


def validate_solver_options(fluent):
    """Check the solver schedule and export settings before launching Fluent."""
    if (
        not isinstance(fluent["solution_strategy"], list)
        or not fluent["solution_strategy"]
    ):
        raise ValueError("solution_strategy must contain at least one stage.")
    for stage in fluent["solution_strategy"]:
        if not isinstance(stage, dict) or set(stage) != {
            "iterations",
            "time_scale_factor",
            "order",
        }:
            raise ValueError(
                "Each solver stage must define iterations, time_scale_factor and order."
            )
        if type(stage["iterations"]) is not int or stage["iterations"] < 1:
            raise ValueError("Stage iterations must be a positive integer.")
        positive(stage["time_scale_factor"], "stage.time_scale_factor")
        if stage["order"] not in ("first", "second"):
            raise ValueError("Stage order must be first or second.")
    if (
        not isinstance(fluent["residual_equations"], dict)
        or not fluent["residual_equations"]
    ):
        raise ValueError("residual_equations must map equation names to tolerances.")
    for equation, tolerance in fluent["residual_equations"].items():
        if not isinstance(equation, str) or not equation:
            raise ValueError("Each residual equation must have a nonempty name.")
        positive(tolerance, f"residual_equations.{equation}")
    if type(fluent["plot_residuals"]) is not bool:
        raise ValueError("plot_residuals must be true or false.")
    if (
        not 0
        < positive(
            fluent["high_order_relaxation_factor"], "high_order_relaxation_factor"
        )
        <= 1
    ):
        raise ValueError("high_order_relaxation_factor must be in (0, 1].")
    for name in ("density_relaxation", "turbulence_relaxation"):
        if not 0 < positive(fluent[name], name) <= 1:
            raise ValueError(f"{name} must be in (0, 1].")
    if type(fluent["processor_count"]) is not int or fluent["processor_count"] < 1:
        raise ValueError("processor_count must be a positive integer.")
    for name in ("start_timeout", "turbulence_viscosity_ratio"):
        positive(fluent[name], f"fluent.{name}")
    if not 0 < positive(fluent["turbulence_intensity"], "turbulence_intensity") <= 1:
        raise ValueError("turbulence_intensity must be a fraction in (0, 1].")
    boundaries = fluent["boundaries"]
    # Roles identify how a boundary is used; actual mesh-zone names live in YAML.
    required_roles = {"inlet", "outlet", "blade", "periodic_left", "periodic_right"}
    if not isinstance(boundaries, dict) or set(boundaries) != required_roles:
        raise ValueError(
            "fluent.boundaries must define inlet, outlet, blade and both periodic sides."
        )
    if any(
        not isinstance(name, str) or not name for name in boundaries.values()
    ) or len(set(boundaries.values())) != len(boundaries):
        raise ValueError(
            "fluent.boundaries must specify distinct names for all five boundaries."
        )
    case_name = fluent["case_filename"]
    if Path(case_name).name != case_name or not case_name.endswith(".cas.h5"):
        raise ValueError("case_filename must be a filename ending in .cas.h5.")
    if type(fluent["print_transcript"]) is not bool:
        raise ValueError("print_transcript must be true or false.")
    for name in ("pressure_contour", "mach_contour"):
        contour = fluent[name]
        if type(contour["enabled"]) is not bool:
            raise ValueError(f"{name}.enabled must be true or false.")
        if (
            Path(contour["filename"]).name != contour["filename"]
            or Path(contour["filename"]).suffix.lower() != ".png"
        ):
            raise ValueError(f"{name}.filename must be a PNG filename.")
        for dimension in ("width", "height"):
            if type(contour[dimension]) is not int or contour[dimension] <= 0:
                raise ValueError(f"{name}.{dimension} must be a positive integer.")
    if fluent["pressure_contour"]["filename"] == fluent["mach_contour"]["filename"]:
        raise ValueError("Pressure and Mach contours must have different filenames.")
    instances = fluent["periodic_instancing"]
    if type(instances["enabled"]) is not bool:
        raise ValueError("periodic_instancing.enabled must be true or false.")
    if type(instances["repeats"]) is not int or instances["repeats"] < 1:
        raise ValueError("periodic_instancing.repeats must be a positive integer.")


def load_dependencies(prepare_only):
    """Keep the example importable without the optional Fluent installation."""
    try:
        import barotropy
    except ImportError as exc:
        raise RuntimeError(
            "Install the companion checkout: poetry run python -m pip install -e ../barotropy"
        ) from exc
    pyfluent = None
    if not prepare_only:
        try:
            import ansys.fluent.core as pyfluent
        except ImportError as exc:
            raise RuntimeError(
                "Install PyFluent with: poetry install -E fluent"
            ) from exc
    return barotropy, pyfluent


def resolve_zone(expected, names):
    """Find a named CGNS boundary, allowing a unique importer-added prefix."""
    if expected in names:
        return expected
    matches = [
        name
        for name in names
        if name.endswith((f"-{expected}", f"_{expected}", f":{expected}"))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify zone {expected!r}; available zones: {sorted(names)}"
        )
    return matches[0]


def boundary_types(boundary_conditions):
    """Return the imported face-zone names and their Fluent boundary types."""
    types = {}
    for kind, zones in boundary_conditions.get_state().items():
        if not isinstance(zones, dict):
            continue
        for name, data in zones.items():
            if isinstance(data, dict) and "name" in data:
                types[name] = kind
    return types


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
    general = settings.setup.general
    general.solver.type = "pressure-based"
    general.solver.time = "steady"
    general.solver.two_dim_space = "planar"
    general.operating_conditions.operating_pressure = 0.0
    general.operating_conditions.gravity.enable = False
    models = settings.setup.models
    models.energy.enabled = False
    models.viscous.model = "k-omega"
    models.viscous.k_omega_model = "sst"

    for name, definition in expressions.items():
        settings.setup.named_expressions[name] = {"definition": definition}
    # Use the barotropic acoustic-speed fit for the plotted Mach number.
    settings.setup.named_expressions[MACH_EXPRESSION] = {
        "definition": "VelocityMagnitude / barotropic_speed_of_sound",
    }
    material = settings.setup.materials.fluid.create(name=options["material_name"])
    material.density = {"option": "expression", "expression": "barotropic_density"}
    material.viscosity = {"option": "expression", "expression": "barotropic_viscosity"}
    # Fluent derives acoustic speed from d(rho)/dp for pressure-dependent density.
    # The exported speed_of_sound, void_fraction, and vapor_quality are diagnostics.
    fluids = settings.setup.cell_zone_conditions.fluid
    if len(fluids.get_object_names()) != 1:
        raise RuntimeError("Expected one fluid cell zone for the stator passage.")
    fluids[fluids.get_object_names()[0]].general.material = options["material_name"]

    logger.info("  Assign inlet, outlet, blade and conformal periodic boundaries.")
    bc = settings.setup.boundary_conditions
    types = boundary_types(bc)
    zones = {
        key: resolve_zone(options["boundaries"][key], types)
        for key in ("inlet", "outlet", "blade")
    }
    for key, kind in (
        ("inlet", "pressure-inlet"),
        ("outlet", "pressure-outlet"),
        ("blade", "wall"),
    ):
        if types[zones[key]].replace("_", "-") != kind:
            settings.mesh.modify_zones.zone_type(zone_names=[zones[key]], new_type=kind)
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
    inlet = bc.pressure_inlet[zones["inlet"]]
    inlet.momentum.gauge_total_pressure.value = state["P0"]
    inlet.momentum.supersonic_or_initial_gauge_pressure.value = state["P0"]
    # The smooth passage approaches axial flow at the inlet cap, toward -y.
    inlet.momentum.direction_specification_method = "Normal to Boundary"
    inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    inlet.turbulence.turbulent_intensity = options["turbulence_intensity"]
    inlet.turbulence.turbulent_viscosity_ratio = options["turbulence_viscosity_ratio"]
    outlet = bc.pressure_outlet[zones["outlet"]]
    outlet.momentum.gauge_pressure.value = state["p_back"]
    outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    outlet.turbulence.backflow_turbulent_intensity = options["turbulence_intensity"]
    outlet.turbulence.backflow_turbulent_viscosity_ratio = options[
        "turbulence_viscosity_ratio"
    ]
    wall = bc.wall[zones["blade"]]
    wall.momentum.wall_motion = "Stationary Wall"
    wall.momentum.shear_condition = "No Slip"

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


def configure_numerics(settings, options):
    """Set coupled pseudo time stepping and normalized residual monitoring."""
    methods = settings.solution.methods
    methods.p_v_coupling.flow_scheme = "Coupled"
    methods.gradient_scheme = "least-square-cell-based"
    set_order(settings, first_order=options["solution_strategy"][0]["order"] == "first")
    high_order = methods.high_order_term_relaxation
    high_order.enable = True
    # Fluent's "Flow Variables Only" group includes pressure, density, velocity
    # components and turbulence. Query its label for the installed Fluent version.
    flow_variables = next(
        value
        for value in high_order.options.select_variables.allowed_values()
        if "flow" in value.lower()
    )
    high_order.options.select_variables = flow_variables
    high_order.options.relaxation_factor = options["high_order_relaxation_factor"]
    methods.pseudo_time_method.formulation.coupled_solver = "global-time-step"
    time_step = settings.solution.run_calculation.pseudo_time_settings.time_step_method
    time_step.time_step_method = "automatic"
    time_step.length_scale_methods = "conservative"
    time_step.time_step_size_scale_factor = options["solution_strategy"][0][
        "time_scale_factor"
    ]
    relax = (
        settings.solution.controls.pseudo_time_explicit_relaxation_factor.global_dt_pseudo_relax
    )
    for equation, value in (
        ("density", options["density_relaxation"]),
        ("k", options["turbulence_relaxation"]),
        ("omega", options["turbulence_relaxation"]),
    ):
        relax[equation] = value
    monitor = settings.solution.monitor.residual
    monitor.options = {
        "normalize": True,
        "print": True,
        "plot": False,
        "criterion_type": "absolute",
    }
    residuals = monitor.equations
    available_equations = residuals.get_object_names()
    unknown = set(options["residual_equations"]) - set(available_equations)
    if unknown:
        raise ValueError(
            f"Residual equations are not active in this Fluent model: {sorted(unknown)}"
        )
    for equation in available_equations:
        residuals[equation].monitor = equation in options["residual_equations"]
        residuals[equation].check_convergence = False
    for equation, tolerance in options["residual_equations"].items():
        residuals[equation].monitor = True
        # Fluent hides the criterion when convergence checking is disabled.
        # Enable it before assigning the tolerance so saved cases work too.
        residuals[equation].check_convergence = True
        residuals[equation].absolute_criteria = tolerance
        # Every solution stage stops as soon as all configured criteria are met.
        # Keep these checks enabled in the original setup case as well.


def set_order(settings, first_order):
    """Switch all transported fields together to the requested spatial order."""
    scheme = "first-order-upwind" if first_order else "second-order-upwind"
    settings.solution.methods.discretization_scheme = {
        "pressure": "standard" if first_order else "second-order",
        **{name: scheme for name in ("density", "mom", "k", "omega")},
    }


def residual_history(transcript_file, output_dir, options, save_plot=False):
    """Read the overall transcript and optionally plot it with barotropy.

    Barotropy detects the residual-table header, so equation names and their
    column order come from Fluent rather than a fixed five-column parser.
    """
    import barotropy

    parsed = barotropy.read_residual_file(str(transcript_file))
    # The current barotropy reader returns an empty DataFrame (without its
    # status flag) before Fluent prints its first iteration table.
    history = parsed[0] if isinstance(parsed, tuple) else parsed
    if history.empty:
        return {}
    history = history.drop_duplicates("iter", keep="last").sort_values("iter")
    equations = list(options["residual_equations"])
    missing = set(equations) - set(history.columns)
    if missing:
        raise RuntimeError(
            f"Transcript does not contain configured residuals: {sorted(missing)}"
        )
    history[["iter", *equations]].rename(columns={"iter": "iteration"}).to_csv(
        output_dir / "residuals.csv", index=False
    )
    if save_plot and options["plot_residuals"]:
        figure, _ = barotropy.plot_residuals(str(transcript_file))
        try:
            # Save one PNG; barotropy's savefig=True also produces SVG and EPS.
            figure.savefig(output_dir / "residuals.png", dpi=160, bbox_inches="tight")
        finally:
            plt.close(figure)
        logger.info(
            "  Residual plot saved: %s", display_path(output_dir / "residuals.png")
        )
    last = history.iloc[-1]
    return {
        "last_iteration": int(last["iter"]),
        "last_residuals": {equation: float(last[equation]) for equation in equations},
    }


def solution_diagnostics(solver, options):
    """Record actual property extrema and boundary mass imbalance."""
    reports = solver.settings.solution.report_definitions
    cells = solver.settings.setup.cell_zone_conditions.fluid.get_object_names()
    names = []
    for label, field in (
        ("pressure", "pressure"),
        ("viscosity", "viscosity-lam"),
        ("density", "density"),
    ):
        for kind in ("min", "max"):
            name = f"workflow-{kind}-{label}"
            reports.volume[name] = {
                "report_type": f"volume-{kind}",
                "field": field,
                "cell_zones": cells,
            }
            names.append(name)
    for kind, collection in (
        ("inlet", solver.settings.setup.boundary_conditions.pressure_inlet),
        ("outlet", solver.settings.setup.boundary_conditions.pressure_outlet),
    ):
        name = f"workflow-mass-{kind}"
        reports.flux[name] = {
            "report_type": "flux-massflow",
            "boundaries": [
                resolve_zone(options["boundaries"][kind], collection.get_object_names())
            ],
        }
        names.append(name)
    result = {}
    for item in reports.compute(report_defs=names):
        for key, value in item.items():
            result[key.removeprefix("workflow-")] = float(value[0])
    result["relative_mass_imbalance"] = abs(
        result["mass-inlet"] + result["mass-outlet"]
    ) / max(abs(result["mass-inlet"]), abs(result["mass-outlet"]), 1e-12)
    return result


def initialize_solution(settings):
    """Hybrid-initialize once, before the first solver stage."""
    initialization = settings.solution.initialization
    initialization.initialization_type = "hybrid"
    # Dispatch through the selected method. Fluent 2024 R2 can mark the separate
    # hybrid_initialize command inactive after selecting hybrid initialization.
    initialization.initialize()


def record_stage_iterations(transcript, stage_end_offsets, stage_results):
    """Recover stage iteration counts after solving, without intermediate checks.

    During solving we only record the transcript's byte position. Afterwards,
    the iteration column at each position identifies how far that stage ran.
    Barotropy still reads and plots the actual residual history.
    """
    position = 0
    previous_iteration = 0
    last_iteration = 0
    row_length = None
    for result, end_position in zip(stage_results, stage_end_offsets):
        for line in transcript[position:end_position].splitlines():
            words = line.split()
            if not words:
                continue
            if words[0] == b"iter" and words[-1] == b"time/iter":
                row_length = len(words) + 1
            elif len(words) == row_length and words[0].isdigit():
                last_iteration = int(words[0])
        result["actual_iterations"] = last_iteration - previous_iteration
        previous_iteration = last_iteration
        position = end_position


def configure_stage(settings, stage, options):
    """Apply spatial order, pseudo timescale and convergence stopping together."""
    set_order(settings, first_order=stage.first_order)
    time_step = settings.solution.run_calculation.pseudo_time_settings.time_step_method
    time_step.time_step_size_scale_factor = stage.time_scale_factor
    for equation in options["residual_equations"]:
        settings.solution.monitor.residual.equations[equation].check_convergence = (
            stage.stop_on_convergence
        )


def run_stages(solver, options, output_dir, state):
    """Run the YAML strategy and save the final case and solution data together.

    Every block stops when Fluent meets all configured normalized residual
    criteria or exhausts its iteration budget, then advances to the next block.
    Field, mass-balance and transcript checks run once, after the final block.
    Exceptions from Fluent abort immediately without saving failed solution data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = solution_stages(options)
    output_stem = options["case_filename"].removesuffix(".cas.h5")
    case_file = output_dir / options["case_filename"]
    transcript_file = output_dir / f"{output_stem}.trn"
    data_file = output_dir / f"{output_stem}.dat.h5"
    summary_file = output_dir / "run_summary.json"
    stage_end_offsets = []
    residuals_saved = False
    summary = {
        "status": "initializing",
        "initialization": "hybrid",
        "completed_stages": [],
        "stage_results": [],
        "requested_iterations": {
            "first_order": sum(
                stage.iterations for stage in stages if stage.first_order
            ),
            "second_order": sum(
                stage.iterations for stage in stages if not stage.first_order
            ),
        },
    }
    try:
        section(logger, "Hybrid initialization")
        configure_stage(solver.settings, stages[0], options)
        initialize_solution(solver.settings)

        for index, stage in enumerate(stages, start=1):
            summary["status"] = stage.name
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            spatial_order = "first" if stage.first_order else "second"
            stopping = (
                "full block" if not stage.stop_on_convergence else "until convergence"
            )
            section(
                logger,
                f"Solver stage {index}/{len(stages)} | {spatial_order.capitalize()} order | "
                f"Timescale {stage.time_scale_factor:g} | Up to {stage.iterations} iterations ({stopping})",
            )
            configure_stage(solver.settings, stage, options)
            solver.settings.solution.run_calculation.iterate(
                iter_count=stage.iterations
            )

            # Bookkeeping only: go directly to the next stage without computing
            # reports or reading residuals. Analyse these positions at the end.
            stage_end_offsets.append(
                transcript_file.stat().st_size if transcript_file.exists() else 0
            )
            summary["completed_stages"].append(stage.name)
            summary["stage_results"].append(
                {
                    "name": stage.name,
                    "order": spatial_order,
                    "time_scale_factor": stage.time_scale_factor,
                    "requested_iterations": stage.iterations,
                    "stop_on_convergence": stage.stop_on_convergence,
                }
            )

        summary["status"] = "final_checks"
        section(logger, "Final solution checks")
        # Fluent sometimes returns normally after a solver error. Check its
        # transcript once, at the end, before saving the final solution data.
        transcript = transcript_file.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        if (
            "floating point exception" in transcript
            or "divergence detected" in transcript
        ):
            raise RuntimeError(f"Fluent diverged; see {transcript_file}")
        summary.update(
            residual_history(transcript_file, output_dir, options, save_plot=True)
        )
        residuals_saved = True
        diagnostics = solution_diagnostics(solver, options)
        summary["diagnostics"] = diagnostics
        if (
            not all(np.isfinite(value) for value in diagnostics.values())
            or diagnostics["min-density"] <= 0
            or diagnostics["min-viscosity"] <= 0
        ):
            raise RuntimeError("Invalid property fields in the final solution.")
        summary["pressure_outside_fit"] = (
            diagnostics["min-pressure"] < state["p_min"]
            or diagnostics["max-pressure"] > state["P0"]
        )
        last_residuals = summary.get("last_residuals", {})
        summary["converged"] = (
            bool(last_residuals)
            and all(
                last_residuals.get(equation, float("inf")) <= tolerance
                for equation, tolerance in options["residual_equations"].items()
            )
            and diagnostics["relative_mass_imbalance"] < 1e-3
        )
        # Replace the initial setup case with the final solver settings and
        # write its matching data in the same Fluent command. This runs once,
        # including when the last stage reaches convergence before its limit.
        summary["status"] = "saving_final_solution"
        logger.info("  Save the final case and data: %s", display_path(case_file))
        solver.settings.file.write_case_data(file_name=str(case_file))
        summary["case_file"] = str(case_file)
        summary["data_file"] = str(data_file)
        summary["status"] = "completed"
        if not summary["converged"]:
            logger.warning(
                "The solution strategy finished without meeting all convergence criteria."
            )
        if summary["pressure_outside_fit"]:
            logger.warning(
                "Local pressure range %.1f to %.1f Pa exceeds the barotropic fit range %.1f to %.1f Pa; "
                "properties use extrapolation or endpoint limits outside the fit.",
                diagnostics["min-pressure"],
                diagnostics["max-pressure"],
                state["p_min"],
                state["P0"],
            )
    except Exception as exc:
        summary["failed_stage"] = summary["status"]
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        if transcript_file.exists():
            record_stage_iterations(
                transcript_file.read_bytes(),
                stage_end_offsets,
                summary["stage_results"],
            )
            if not residuals_saved:
                summary.update(
                    residual_history(
                        transcript_file, output_dir, options, save_plot=True
                    )
                )
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            "  Residual CSV and run summary saved: %s", display_path(output_dir)
        )
    return output_dir


def contour_definitions(options):
    """Map YAML image settings to Fluent fields and readable plot names."""
    return (
        ("static-pressure", "pressure", options["pressure_contour"]),
        ("barotropic-mach-number", f"expr:{MACH_EXPRESSION}", options["mach_contour"]),
    )


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


def save_fluent_case(pyfluent, workflow, mesh_file, state, expressions, pitch_m):
    """Save the setup, optionally solve and save final case/data, then close."""
    options = workflow["fluent"]
    output_dir = workflow["output_dir"] / "fluent"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_file = output_dir / options["case_filename"]
    transcript_file = output_dir / f"{case_file.name.removesuffix('.cas.h5')}.trn"
    total_steps = 8 if workflow["run_case"] else 6
    step(
        logger,
        6,
        total_steps,
        "Launch Fluent, configure physics and save the setup case",
    )
    logger.info(
        "  Launch Fluent %s with %d processor(s).",
        options["product_version"] or "latest",
        options["processor_count"],
    )
    solver = pyfluent.launch_fluent(
        product_version=options["product_version"],
        mode="solver",
        dimension=2,
        precision="double",
        processor_count=options["processor_count"],
        ui_mode=options["ui_mode"],
        start_timeout=options["start_timeout"],
        cwd=str(output_dir),
        cleanup_on_exit=True,
        start_transcript=False,
    )
    transcript_started = False
    stream_started = False
    failed = False
    try:
        if options["print_transcript"]:
            solver.transcript.start(write_to_stdout=True)
            stream_started = True
        # Stable output filenames replace previous results without accumulating
        # timestamped cases or prompting during this unattended workflow.
        solver.settings.file.batch_options.confirm_overwrite = False
        # Fluent can retain old content in an existing transcript. Clear it so
        # residuals and actual stage iterations cannot include an earlier run.
        transcript_file.write_text("", encoding="utf-8")
        solver.settings.file.start_transcript(file_name=str(transcript_file))
        transcript_started = True
        logger.info(
            "  Record Fluent's native console output: %s", display_path(transcript_file)
        )
        zones = configure_fluent(
            solver, mesh_file, state, expressions, pitch_m, options
        )
        logger.info(
            "  Save the original setup case before initialization: %s",
            display_path(case_file),
        )
        solver.settings.file.write_case(file_name=str(case_file))
        (output_dir / "setup_summary.json").write_text(
            json.dumps(
                {
                    "fluent_version": str(solver.get_fluent_version()),
                    "case_file": str(case_file),
                    "zones": zones,
                    "inlet_state": state,
                    "pitch_m": pitch_m,
                    "initialized": False,
                    "solution_iterations": 0,
                    "run_case": workflow["run_case"],
                    "transcript_file": str(transcript_file),
                    "high_order_relaxation": solver.settings.solution.methods.high_order_term_relaxation.get_state(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if workflow["run_case"]:
            step(logger, 7, total_steps, "Initialize and solve using the YAML strategy")
            run_stages(solver, options, output_dir, state)
            step(logger, 8, total_steps, "Export final pressure and Mach contours")
            if any(
                settings["enabled"] for _, _, settings in contour_definitions(options)
            ):
                export_contours(solver, options, output_dir)
            else:
                logger.info("  Both contour exports are disabled in the YAML.")
    except Exception:
        failed = True
        raise
    finally:
        try:
            if transcript_started and not failed:
                solver.settings.file.stop_transcript()
            if stream_started and not failed:
                solver.transcript.stop()
        finally:
            logger.info("  Close the Fluent session.")
            if failed:
                solver.exit(timeout=5, timeout_force=True)
            else:
                solver.exit()
    logger.info(
        "  Fluent session closed; %s case available: %s",
        "final solution" if workflow["run_case"] else "setup",
        display_path(case_file),
    )
    return case_file


def main():
    """Run the workflow with numbered progress shared by the console and log."""
    config_file = Path(CONFIG_FILE).resolve()
    # Read just enough to locate the log. Full validation and dependency loading
    # happen inside the logging context, so their failures are recorded too.
    with config_file.open(encoding="utf-8") as stream:
        workflow = yaml.safe_load(stream)
    if not isinstance(workflow, dict):
        raise ValueError("Workflow YAML must be a mapping.")
    reporting = workflow.get("reporting")
    validate_reporting(reporting)
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    log_file = output_dir / reporting["filename"]
    total_steps = (
        5 if workflow.get("prepare_only") else (8 if workflow.get("run_case") else 6)
    )

    with workflow_logging(log_file, reporting["level"]):
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

        step(logger, 2, total_steps, "Load or calculate the MoC design")
        nozzle = load_or_solve_nozzle(
            config["nozzle"],
            workflow["moc_cache_file"],
            force_solve=workflow["force_moc_solve"],
        )
        step(logger, 3, total_steps, "Build blade geometry and periodic passage")
        blade, curves = build_geometry(nozzle, config)
        if not assess_clearance(blade, curves)["contained"]:
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
        if workflow["save_geometry_figure"]:
            logger.info("  Export the blade and passage construction figures.")
            figure, _ = plot_construction(blade, curves)
            try:
                for extension in ("png", "svg"):
                    filename = mesh_dir / f"stator_geometry.{extension}"
                    figure.savefig(filename, dpi=180, bbox_inches="tight")
                    logger.info("  Geometry figure saved: %s", display_path(filename))
            finally:
                plt.close(figure)

        step(logger, 4, total_steps, "Generate, validate and export the mesh")
        create_mesh(blade, curves, config["mesh"], mesh_dir, show_gui=False)
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


if __name__ == "__main__":
    main()
