"""
moc.api -- computational API layer.

Mirrors the pattern used by turbodash's core.py: one plain function that
takes primitive arguments (no MOCSolver/FluidManager objects, no UI
concerns) and returns a single flat, JSON-safe dict. A Dash (or any other)
UI layer can sit on top of this exactly like turbodash's app.py sits on top
of core.py's compute_stage_meanline() -- call design_nozzle() inside a
callback, fan its returned dict out into plot/table builders.

This module owns no state itself: FluidManager caching (moc.core.classes.
get_fluid_manager) already lives one level down and is reused as-is, so
repeated calls at the same operating point (the common case while a UI
sweeps geometry) stay cheap.
"""

import numpy as np
import jaxprop as jxp

from moc.core.classes import get_fluid_manager
from moc.solvers.conventional import MOCSolver as ConventionalSolver
from moc.solvers.mln import MOCSolverMLN

__all__ = ["design_nozzle", "list_supported_fluids", "NozzleDesignError"]


class NozzleDesignError(Exception):
    """Raised when a nozzle design request is invalid or the solve fails
    before reaching a usable result (as opposed to the solver's own
    graceful early-stop, which still returns a partial-but-usable design --
    see the 'converged' flag in design_nozzle's returned dict)."""


def list_supported_fluids() -> list:
    """CoolProp's fluid list, for populating a fluid-selection UI control."""
    import CoolProp.CoolProp as CP
    return sorted(CP.FluidsList())


def _inlet_state(fluid, P0, T0, Q0):
    if Q0 is None or (isinstance(Q0, float) and np.isnan(Q0)):
        if T0 is None:
            raise NozzleDesignError("T0 is required when Q0 is not given (subcooled-liquid or single-phase inlet).")
        return fluid.get_state(jxp.PT_INPUTS, P0, T0)
    if not (0.0 <= Q0 <= 1.0):
        raise NozzleDesignError(f"Q0={Q0} out of range -- must be in [0, 1] for a saturated two-phase inlet.")
    return fluid.get_state(jxp.PQ_INPUTS, P0, Q0)


def design_nozzle(
    fluid_name: str,
    P0: float,
    T0: float = None,
    Q0: float = float("nan"),
    p_back: float = None,
    Noz_Mach: float = None,
    solver: str = "conventional",
    y_t: float = 0.01,
    n: int = 15,
    rho_t: float = 0.2,
    rho_d: float = 0.1,
    delta_flow: float = 0.0,
    tau_max: float = 90.0,
    tau_step: float = 1.0,
    n_tau_mln: int = 60,
    use_true_sauer_line: bool = False,
    backend: str = "HEOS",
    show_plot: bool = False,
    progress_callback=None,
) -> dict:
    """
    Runs a full MOC nozzle design (all solver stages) and returns one flat,
    JSON-safe result dict -- no MOCSolver/FluidManager objects, so this is
    safe to call directly from a UI callback or an HTTP handler.

    progress_callback: optional callable(fraction: float, stage_label: str),
    threaded through every solver stage. Pass moc.progress.
    make_progress_reporter(set_progress)'s return value here to drive a
    Dash background-callback progress bar across the whole solve.

    Inlet condition (same convention throughout moc):
      - Q0 = nan (default) + T0: subcooled-liquid (flashing) or single-phase
        superheated-vapor inlet, set via (P0, T0).
      - Q0 in [0, 1]: saturated two-phase inlet, set via (P0, Q0); T0 is
        derived automatically and does not need to be supplied.

    Design target: give p_back (recommended -- pressure-based termination
    is unambiguous even where two-phase HEM sound speed makes Mach
    non-monotonic along the expansion, see MOCSolver.run_stage_3_kernel's
    docstring) and/or Noz_Mach directly. At least one is required. p_back
    is required for solver="mln" (used to compute the corner fan's target
    turning angle theta*).

    solver: "conventional" (rounded-arc kernel) or "mln" (minimum-length,
    sharp-corner kernel) -- both share the same FluidManager cache key, so
    calling both for the same (fluid_name, P0, T0, Q0) only builds the
    fluid table once.

    Returns
    -------
    dict with keys:
      wall_final, div_wall, kernel_wall, sauer, IVP, mesh_data, axis
        -- same schema as MOCSolver.build_result_data / export_full_result_data
      fluid_name, solver, design_Noz_Mach, p_back, converged, runtime_s
    """
    import time

    if solver not in ("conventional", "mln"):
        raise NozzleDesignError(f"solver must be 'conventional' or 'mln', got {solver!r}")
    if p_back is None and Noz_Mach is None:
        raise NozzleDesignError("At least one of p_back or Noz_Mach is required.")
    if solver == "mln" and p_back is None:
        raise NozzleDesignError("solver='mln' requires p_back (used to compute the corner fan's target turning angle).")

    t0 = time.time()

    fluid = jxp.Fluid(fluid_name)
    state_0 = _inlet_state(fluid, P0, T0, Q0)
    T0_resolved = float(state_0.T)

    design_Noz_Mach = Noz_Mach
    if p_back is not None:
        state_out = fluid.get_state(jxp.PSmass_INPUTS, p_back, state_0.s)
        v_out = float(np.sqrt(abs(2 * (state_0.h - state_out.h))))
        design_Noz_Mach = float(v_out / state_out.a)

    fm = get_fluid_manager(fluid_name, backend, P0=P0, T0=T0_resolved, Q0=Q0)

    tau = np.linspace(0, tau_max, int(tau_max / tau_step) + 1)
    base_config = dict(
        y_t=y_t, n=n, rho_t=rho_t, P0=float(P0), Q0=float(Q0), T0=T0_resolved,
        tau=tau, rho_d=rho_d, Noz_Mach=design_Noz_Mach, P_back=p_back,
        fluid_name=fluid_name, delta_flow=delta_flow, n_reflex=float("nan"),
    )

    try:
        if solver == "conventional":
            s = ConventionalSolver(base_config, show_plot=show_plot, fluid_manager=fm)
            s.run_stage_1_sauer(use_true_sauer_line=use_true_sauer_line, progress_callback=progress_callback)
            s.run_stage_2_ivp(progress_callback=progress_callback)
            s.run_stage_3_kernel(progress_callback=progress_callback)
            s.run_stage_4_reflex(progress_callback=progress_callback)
        else:
            state_out = fluid.get_state(jxp.PSmass_INPUTS, p_back, state_0.s)
            v_out = float(np.sqrt(abs(2 * (state_0.h - state_out.h))))
            nu_f = fm.nu_of_V(v_out)
            theta_star_deg = float(np.degrees(0.5 * nu_f))

            mln_config = dict(base_config)
            mln_config["tau_mln"] = np.linspace(0, theta_star_deg, n_tau_mln)
            mln_config["flashing"] = not use_true_sauer_line
            s = MOCSolverMLN(mln_config, show_plot=show_plot, fluid_manager=fm)
            s.run_stage_1_sauer(progress_callback=progress_callback)
            # sauer_direct, not ivp_march, for BOTH branches: for flashing,
            # it collapses the flat front to just the corner point, matching
            # the reference algorithm exactly (Zebbiche & Youbi 2007 Fig.
            # 3a: ray 1 is corner -> axis directly, zero interior points --
            # see MOCSolverMLN.run_stage_2_sauer_direct's docstring). Marching
            # first would feed Stage 3 a multi-point front instead, breaking
            # that property without changing the converged exit condition --
            # this is what method_of_characteristics/moc_min_length's own
            # validated generate_laes_cases.py does.
            s.run_stage_2_sauer_direct(progress_callback=progress_callback)
            s.run_stage_3_kernel_mln(progress_callback=progress_callback)
            s.run_stage_4_reflex(progress_callback=progress_callback)
    except Exception as e:
        raise NozzleDesignError(f"MOC solve failed for solver={solver!r}: {e}") from e

    exit_M = float(s.axis_points_vec[-1].M) if s.axis_points_vec else float("nan")
    converged = bool(np.isfinite(exit_M) and abs(exit_M - design_Noz_Mach) / max(design_Noz_Mach, 1e-9) < 0.05)

    data = s.build_result_data(extra_meta=dict(
        solver=solver, P0=float(P0), T0=T0_resolved, Q0=float(Q0),
        p_back=p_back, design_Noz_Mach=design_Noz_Mach,
        y_t=y_t, n=n, rho_t=rho_t, rho_d=rho_d, delta_flow=delta_flow,
    ))
    data["solver"] = solver
    data["design_Noz_Mach"] = design_Noz_Mach
    data["p_back"] = p_back
    data["converged"] = converged
    data["exit_M"] = exit_M
    data["runtime_s"] = time.time() - t0
    # Stagnation/total state (s0, T0) -- the isentrope's actual starting
    # point, needed for a T-s diagram alongside the axis-array's own first/
    # last points (~sonic/MoC-start, exit). Not derivable from `axis` alone:
    # the isentrope's total point sits upstream of where Stage 1 seeds the
    # solve, at essentially zero velocity, so it isn't one of the solved
    # points at all.
    data["s0"] = float(state_0.s)
    data["T0_stagnation"] = float(state_0.T)
    return data
