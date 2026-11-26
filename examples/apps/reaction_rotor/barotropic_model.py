"""Barotropic HEM model (Fluent expressions) for the stator + reaction-rotor stage of build_stage_mesh.py.

    python examples/apps/reaction_rotor/barotropic_model.py

Same step as the Fluent stage pipeline (turbo_moc.fluent.stage.setup, step 2/5): the stator inlet state of the
computed MoC design is expanded isentropically (equilibrium two-phase) and rho(p), a(p), ... are fitted with
piecewise polynomials. The fit runs from the inlet pressure down to p_in / PRESSURE_RATIO, which must lie below
the lowest outlet pressure you will impose on the rotor. Output: output/stage/barotropy/ (Fluent expression
files, fit report and figures), plus inlet_state.json with the values for the inlet boundary.
"""
import json
from pathlib import Path

import yaml

from turbo_moc.app import workflow as flow
from turbo_moc.fluent.barotropic_model import generate_barotropic_model
from turbo_moc.fluent.common import load_dependencies
from turbo_moc.fluent.stage import design

HERE = Path(__file__).resolve().parent
STAGE = HERE / "output" / "stage"
SETTINGS = {
    "calculation_type": "equilibrium",  # HEM, as in the TurboExpo stage cases
    "efficiency": 1.0,                  # Isentropic
    "pressure_ratio": 10.0,             # 251.3 kPa / 10 = 25 kPa: covers rotor outlet pressures down to ~30 kPa
    "polynomial_degree": 8,
    "segment_pressure_ratio": 1.3,
    "ODE_solver": "LSODA",
    "ODE_tolerance": 1.0e-9,
    "save_figures": True,
}

if __name__ == "__main__":
    barotropy, _ = load_dependencies(True)
    flow.OUTPUT = STAGE / "jobs"
    project = yaml.safe_load((STAGE / "project.yaml").read_text(encoding="utf-8"))
    registry = design.registry_from_jobs(STAGE / "jobs")  # Latest session: the jobs behind mesh/stage.cgns
    config = flow.resolved_configuration(project, "stator_1", registry)
    nozzle = flow.load_nozzle(registry["stator_1"]["nozzle"], config)
    parameters = dict(config["method_of_characteristics"])
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, nozzle, SETTINGS,
                                                   STAGE / "barotropy")
    (STAGE / "barotropy" / "inlet_state.json").write_text(json.dumps(state, indent=2, default=float),
                                                          encoding="utf-8")
    print("Barotropic model written to", STAGE / "barotropy")
    print("Inlet state:", json.dumps(state, default=float))
