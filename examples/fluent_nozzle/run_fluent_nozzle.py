"""Design a nozzle with the MoC, mesh it, build the barotropic model and open a configured Fluent case.

The case is configured and saved but not solved: Fluent stays open with its GUI
so the setup can be checked and the calculation started by hand. MoC wall and
axis distributions are written next to the case for comparison with the CFD.

Run from the repository root:
    python examples/fluent_nozzle/run_fluent_nozzle.py
"""
import json
from pathlib import Path

import numpy as np
import yaml

from turbo_moc.configuration.fluid_inputs import validate_outlet_inputs
from turbo_moc.fluent.barotropic_model import generate_barotropic_model
from turbo_moc.fluent.common import (boundary_types, configure_flow_boundaries, configure_material,
                                     configure_numerics, load_dependencies, resolve_zone, set_order)
from turbo_moc.geometry import nozzle as nozzle_geometry
from turbo_moc.meshing import nozzle_mesh
from turbo_moc.workflows.nozzle import solve_nozzle

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_nozzle.yaml"


def write_moc_distributions(result, directory):
    for name in ("axis", "wall_final"):
        data = result[name]
        columns = [key for key in ("x", "y", "p", "M", "rho", "T") if key in data]
        table = np.column_stack([np.asarray(data[key], dtype=float) for key in columns])
        np.savetxt(directory / f"moc_{name.replace('_final', '')}.csv", table, delimiter=",",
                   header=",".join(columns), comments="")


def configure_case(solver, mesh_file, state, expressions, options):
    settings = solver.settings
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    configure_material(settings, expressions, options, dimension=2)
    types = boundary_types(settings.setup.boundary_conditions)
    zones = {role: resolve_zone(role, types) for role in ("inlet", "outlet", "wall", "axis")}
    if types[zones["axis"]] != "symmetry":
        settings.mesh.modify_zones.zone_type(zone_names=[zones["axis"]], new_type="symmetry")
    configure_flow_boundaries(settings, zones, state, options, wall_roles=("wall",))
    configure_numerics(settings, options)
    if options.get("viscous_model", "sst") == "inviscid":
        settings.setup.models.viscous.model = "inviscid"
    settings.solution.initialization.initialization_type = "hybrid"
    settings.solution.run_calculation.iter_count = options["solution_strategy"][0]["iterations"]
    return zones


def solve(solver, options):
    settings = solver.settings
    settings.solution.initialization.initialize()
    for stage in options["solution_strategy"]:
        set_order(settings, first_order=stage["order"] == "first")
        settings.solution.run_calculation.pseudo_time_settings.time_step_method.time_step_size_scale_factor = (
            stage["time_scale_factor"])
        settings.solution.run_calculation.iterate(iter_count=stage["iterations"])


def extract_surfaces(solver, directory):
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest
    field_data = solver.fields.field_data
    for surface in ("wall", "axis"):
        xyz = np.asarray(field_data.get_field_data(SurfaceFieldDataRequest(
            surfaces=[surface], data_types=[SurfaceDataType.Vertices]))[surface].vertices).reshape(-1, 3)
        columns = {"x": xyz[:, 0], "y": xyz[:, 1]}
        for name in ("absolute-pressure", "expr:barotropic_mach_number", "velocity-magnitude",
                     "expr:barotropic_density"):
            columns[name] = np.asarray(field_data.get_field_data(ScalarFieldDataRequest(
                field_name=name, surfaces=[surface], node_value=True))[surface])
        order = np.argsort(columns["x"])
        np.savetxt(directory / f"cfd_{surface}.csv", np.column_stack([v[order] for v in columns.values()]),
                   delimiter=",", header=",".join(columns), comments="")


def main(config_file=CONFIG_FILE):
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    for name in ("moc", "mesh", "barotropy", "fluent"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    barotropy, pyfluent = load_dependencies(False)

    parameters = dict(workflow["method_of_characteristics"], fluid_name=workflow["fluid_name"])
    validate_outlet_inputs(parameters)
    print("1/4 MoC design")
    result = solve_nozzle(parameters)
    (output_dir / "moc" / "moc_result.json").write_text(json.dumps(result, default=float), encoding="utf-8")
    write_moc_distributions(result, output_dir / "moc")
    print(f"    exit Mach {result['exit_M']:.4f} (design {result['design_Noz_Mach']:.4f}), "
          f"exit pressure {result['p_back'] / 1e5:.2f} bar")

    print("2/4 Structured 2D mesh")
    settings = dict(workflow["mesh"])
    summary = nozzle_mesh.mesh_2d(result, parameters, settings, output_dir / "mesh")
    print(f"    {summary['cells']:,} cells, first wall cell at the throat {summary['first_wall_height'] * 1e6:.2f} um")

    print("3/4 Barotropic model")
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, result,
                                                   workflow["barotropic"], output_dir / "barotropy")

    print("4/4 Fluent setup (left open, not solved)")
    options = workflow["fluent"]
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=2,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode=options["ui_mode"], start_timeout=options["start_timeout"],
                                    cwd=str(output_dir / "fluent"), cleanup_on_exit=False)
    zones = configure_case(solver, output_dir / "mesh" / "nozzle_mesh_2d.cgns", state, expressions, options)
    case_file = output_dir / "fluent" / options["case_filename"]
    solver.settings.file.batch_options.confirm_overwrite = False
    solver.settings.file.write_case(file_name=str(case_file))
    (output_dir / "fluent" / "setup_summary.json").write_text(json.dumps(
        {"case_file": str(case_file), "zones": zones, "inlet_state": state,
         "moc_exit_mach": result["exit_M"], "mesh_cells": summary["cells"]}, indent=2, default=float),
        encoding="utf-8")
    print(f"    case saved: {case_file}")
    if options.get("solve", False):
        print("5/5 Solve, save case and data, extract wall and axis distributions")
        solve(solver, options)
        solver.settings.file.write_case_data(file_name=str(case_file))
        extract_surfaces(solver, output_dir / "fluent")
        print(f"    solved case and data saved: {case_file}")
    print("    Fluent is open for inspection.")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
