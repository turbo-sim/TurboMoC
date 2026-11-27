"""From the stage mesh to a ready-to-run Fluent case (the automatic version of fluent_setup.md):

    python examples/apps/reaction_rotor/setup_fluent_case.py --rpm 5000 --p-out 95000 [--gui]

Inputs (made by build_stage_mesh.py and barotropic_model.py; other paths with --mesh / --barotropy):
  output/stage/mesh/stage.cgns                    both rows, zones <row>_fluid.<row>_<patch>
  output/stage/barotropy/fluent_expressions_safe.json   barotropic rho(p), a(p), mu(p) as Fluent expressions
  output/stage/barotropy/inlet_state.json         inlet total pressure P0
Output: output/stage/fluent/stage_fluent.cas.h5 (not initialized, not solved). With --gui Fluent stays open, so you
can check every step of fluent_setup.md, initialize (hybrid) and run.

It calls the same function as the TurboExpo pipeline (turbo_moc.fluent.stage.case.configure_stage): SST,
energy off, barotropic material, rotating rotor zone (MRF), mixing plane, rotational periodicity, total-pressure
inlet (P0, normal to the boundary) and static-pressure outlet. Try the manual setup first (fluent_setup.md).
"""
import argparse
import json
from pathlib import Path

from turbo_moc.fluent.common import load_dependencies
from turbo_moc.fluent.stage import case

HERE = Path(__file__).resolve().parent
STAGE = HERE / "output" / "stage"
OPTIONS = {
    "material_name": "barotropic_fluid",
    "interface": "mixing_plane",
    "turbulence_intensity": 0.05,
    "turbulence_viscosity_ratio": 10.0,
    "high_order_relaxation_factor": 0.15,
    "density_relaxation": 0.5,
    "turbulence_relaxation": 0.3,
    "residual_equations": {name: 1.0e-4 for name in ("continuity", "x-velocity", "y-velocity", "z-velocity", "k", "omega")},
    "slip_endwalls": False,  # Viscous hub and shroud
    "solution_strategy": [{"iterations": 300, "time_scale_factor": 0.5, "order": "first"},
                          {"iterations": 1000, "time_scale_factor": 1.0, "order": "second"}],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpm", type=float, required=True, help="rotor speed [rpm]")
    parser.add_argument("--p-out", type=float, required=True, help="rotor outlet static pressure [Pa]")
    parser.add_argument("--swirl-sign", type=int, default=-1, choices=(-1, 1),
                        help="rotation sign about +z (the TurboMoC stages use -1)")
    parser.add_argument("--mesh", default=str(STAGE / "mesh" / "stage.cgns"))
    parser.add_argument("--barotropy", default=str(STAGE / "barotropy"))
    parser.add_argument("--out", default=str(STAGE / "fluent"))
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--gui", action="store_true", help="keep Fluent open with its GUI")
    args = parser.parse_args()

    baro = Path(args.barotropy)
    expressions = json.loads((baro / "fluent_expressions_safe.json").read_text(encoding="utf-8"))
    state = json.loads((baro / "inlet_state.json").read_text(encoding="utf-8"))
    if args.p_out < state["p_min"]:
        raise SystemExit(f"Outlet pressure {args.p_out:.0f} Pa is below the barotropic fit ({state['p_min']:.0f} Pa): "
                         "regenerate the model with a larger pressure_ratio.")
    point = {"rpm": args.rpm, "swirl_sign": args.swirl_sign, "rotor_exit_pressure": args.p_out}

    _, pyfluent = load_dependencies(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    solver = pyfluent.launch_fluent(mode="solver", dimension=3, precision="double", processor_count=args.processes,
                                    ui_mode="gui" if args.gui else "no_gui", cwd=str(out), cleanup_on_exit=not args.gui)
    configured = case.configure_stage(solver, Path(args.mesh), state, expressions, point, OPTIONS)
    case_file = out / "stage_fluent.cas.h5"
    solver.settings.file.batch_options.confirm_overwrite = False
    solver.settings.file.write_case(file_name=str(case_file))
    (out / "setup_summary.json").write_text(json.dumps({**configured, "inlet_state": state, **point}, indent=2,
                                                       default=float), encoding="utf-8")
    print("Case saved:", case_file)
    if args.gui:
        print("Fluent is open: check the setup, initialize (hybrid) and run.")
    else:
        solver.exit()


if __name__ == "__main__":
    main()
