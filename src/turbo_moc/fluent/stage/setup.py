"""Design the stage with the app pipeline and open a configured, unsolved 3D Fluent case."""

import json
from pathlib import Path

import yaml

from . import case, design

VELOCITIES = ("x-velocity", "y-velocity", "z-velocity")


def start_from_rest(settings, start, velocities=VELOCITIES):
    """Fluid at rest at the inlet total pressure, then the outlet pressure lowered step by step.

    Near the saturated liquid the density falls by orders of magnitude through
    the stage; a hybrid initialization then starts far from a consistent field
    and the inlet reverses or the solver diverges. Ramping the outlet establishes
    the flow subsonically first.
    """
    from ..common import set_order
    inlet = settings.setup.boundary_conditions.pressure_inlet
    outlet = settings.setup.boundary_conditions.pressure_outlet
    total = inlet[inlet.get_object_names()[0]].momentum.gauge_total_pressure.value()
    back = outlet[outlet.get_object_names()[0]].momentum.gauge_pressure
    target = back.value()
    initialization = settings.solution.initialization
    initialization.initialization_type = "standard"
    initialization.reference_frame = "absolute"
    initialization.defaults["pressure"] = total
    for component in velocities:
        initialization.defaults[component] = 0.0
    initialization.standard_initialize()
    set_order(settings, first_order=True)
    settings.solution.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor = (
        start["time_scale_factor"])
    for pressure in start["outlet_pressures"]:
        back.value = pressure
        print(f"    start-up: outlet {pressure / 1e3:.0f} kPa, {start['iterations']} iterations")
        settings.solution.run_calculation.iterate(iter_count=start["iterations"])
    back.value = target


def solve(solver, options, velocities=VELOCITIES):
    """Initialize and run the staged solution strategy, each stage with its own order and time scale."""
    from ..common import set_order
    settings = solver.settings
    if "start" in options:
        start_from_rest(settings, options["start"], velocities)
    else:
        settings.solution.initialization.initialize()
    for stage in options["solution_strategy"]:
        set_order(settings, first_order=stage["order"] == "first")
        settings.solution.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor = (
            stage["time_scale_factor"])
        print(f"    {stage['iterations']} iterations, {stage['order']} order")
        settings.solution.run_calculation.iterate(iter_count=stage["iterations"])


def main(config_file):
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if "project_file" in workflow:
        workflow["project_file"] = str((config_file.parent / workflow["project_file"]).resolve())
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/5 Blade designs and 3D meshes (app jobs)")
    project, registry = design.prepare(workflow, output_dir)
    point, summary = design.write_operating_point(output_dir, project, registry)
    print("    operating point:", json.dumps({key: value for key, value in summary.items() if key != "design"},
                                              default=float)[:400])

    from ..barotropic_model import generate_barotropic_model
    from ..common import load_dependencies
    barotropy, pyfluent = load_dependencies(False)
    print("2/5 Barotropic model along the stator inlet isentrope")
    parameters = dict(point["stator_config"]["method_of_characteristics"])
    if "backend" in workflow["barotropic"]:
        parameters["backend"] = workflow["barotropic"]["backend"]
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, point["nozzle"],
                                                   workflow["barotropic"], output_dir / "barotropy")

    options = workflow["fluent"]
    print(f"3/5 Launch Fluent ({options['ui_mode']}) and import the stage mesh")
    fluent_dir = output_dir / "fluent"
    fluent_dir.mkdir(exist_ok=True)
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=3,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode=options["ui_mode"], start_timeout=options["start_timeout"],
                                    cwd=str(fluent_dir), cleanup_on_exit=False)
    print("4/5 Configure materials, rotating rotor zone, mixing plane, periodicity and boundaries")
    configured = case.configure_stage(solver, output_dir / "mesh" / "stage.cgns", state, expressions, point, options)
    case_file = fluent_dir / options["case_filename"]
    solver.settings.file.batch_options.confirm_overwrite = False
    solver.settings.file.write_case(file_name=str(case_file))
    (fluent_dir / "setup_summary.json").write_text(json.dumps(
        {**configured, "inlet_state": state, **summary, "case_file": str(case_file)}, indent=2, default=float),
        encoding="utf-8")
    print(f"5/5 Case saved: {case_file}")
    if options.get("solve", False):
        print("    Solve (hybrid initialization, staged strategy) and save case and data")
        solve(solver, options)
        solver.settings.file.write_case_data(file_name=str(case_file))
        print(f"    Solved case and data saved: {case_file}")
        if options["ui_mode"] == "no_gui":
            solver.exit()
    else:
        print("    Fluent is open: check the setup, initialize (hybrid) and run the calculation.")
    return case_file
