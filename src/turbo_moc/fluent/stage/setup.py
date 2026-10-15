"""Design the stage with the app pipeline and open a configured, unsolved 3D Fluent case."""

import json
from pathlib import Path

import yaml

from . import case, design


def main(config_file):
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
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
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, point["nozzle"],
                                                   workflow["barotropic"], output_dir / "barotropy")

    print("3/5 Launch Fluent (GUI) and import the stage mesh")
    options = workflow["fluent"]
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
    print("    Fluent is open: check the setup, initialize (hybrid) and run the calculation.")
    return case_file
