"""
turbo_moc.api -- computational API layer.

Mirrors the pattern used by turbodash's core.py: one plain function that
takes primitive arguments (no MOCSolver/FluidManager objects, no UI
concerns) and returns a single flat, JSON-safe dict. A Dash (or any other)
UI layer can sit on top of this exactly like turbodash's app.py sits on top
of core.py's compute_stage_meanline() -- call design_nozzle() inside a
callback, fan its returned dict out into plot/table builders.

This module owns no state itself: FluidManager caching (turbo_moc.core.classes.
get_fluid_manager) already lives one level down and is reused as-is, so
repeated calls at the same operating point (the common case while a UI
sweeps geometry) stay cheap.
"""

import numpy as np
import jaxprop as jxp

from turbo_moc.core.classes import get_fluid_manager
from turbo_moc.solvers.conventional import MOCSolver as ConventionalSolver
from turbo_moc.solvers.mln import MOCSolverMLN

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
    auto_nudge_q0: bool = True,
) -> dict:
    """Run every stage of a method-of-characteristics nozzle design.

    The function accepts ordinary values and returns no solver or fluid-manager
    objects, so its result can be passed directly to a UI callback or HTTP
    handler.

    Parameters
    ----------
    fluid_name : str
        CoolProp fluid name.
    P0 : float
        Stagnation pressure in pascals.
    T0 : float, optional
        Stagnation temperature in kelvin for a single-phase or flashing inlet.
    Q0 : float, optional
        Inlet quality for a saturated two-phase state. Leave as NaN when using
        ``T0``.
    p_back : float, optional
        Back pressure in pascals. This is the recommended design target and is
        required by the minimum-length solver.
    Noz_Mach : float, optional
        Direct nozzle Mach-number target. At least one of ``p_back`` and
        ``Noz_Mach`` is required.
    solver : {"conventional", "mln"}
        Rounded-arc conventional solver or sharp-corner minimum-length solver.
    y_t : float
        Throat half-height.
    n : int
        Number of points used to discretize the initial characteristic line.
    rho_t, rho_d : float
        Geometric radii used by the nozzle construction.
    delta_flow : float
        Axisymmetric-flow switch or coefficient passed to the solver.
    tau_max, tau_step : float
        Angular extent and increment for the conventional solver.
    n_tau_mln : int
        Number of angular samples for the minimum-length expansion fan.
    use_true_sauer_line : bool
        Use the classical parabolic Sauer sonic line instead of the default
        flat post-jump throat front.
    backend : str
        Thermodynamic backend passed to the fluid manager.
    show_plot : bool
        Enable the solver's diagnostic plotting.
    progress_callback : callable, optional
        Called as ``progress_callback(fraction, stage_label)`` throughout
        the solver stages.
    auto_nudge_q0 : bool
        If a saturated-liquid classical Sauer solve fails, retry with small
        positive inlet qualities and retain the first converged result.

    Returns
    -------
    dict
        JSON-safe solver data. It includes wall, kernel, Sauer-line, mesh, and
        axis data together with the resolved inputs, convergence state, and
        runtime.
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

    q0_is_zero = Q0 is not None and not (isinstance(Q0, float) and np.isnan(Q0)) and float(Q0) == 0.0
    q0_candidates = [Q0]
    if use_true_sauer_line and auto_nudge_q0 and q0_is_zero:
        q0_candidates += [0.005, 0.01, 0.02, 0.05, 0.1]

    last_exc = None
    for attempt_i, q0_try in enumerate(q0_candidates):
        state_0 = _inlet_state(fluid, P0, T0, q0_try)
        T0_resolved = float(state_0.T)

        design_Noz_Mach = Noz_Mach
        if p_back is not None:
            state_out = fluid.get_state(jxp.PSmass_INPUTS, p_back, state_0.s)
            v_out = float(np.sqrt(abs(2 * (state_0.h - state_out.h))))
            design_Noz_Mach = float(v_out / state_out.a)

        fm = get_fluid_manager(fluid_name, backend, P0=P0, T0=T0_resolved, Q0=q0_try)

        tau = np.linspace(0, tau_max, int(tau_max / tau_step) + 1)
        base_config = dict(
            y_t=y_t, n=n, rho_t=rho_t, P0=float(P0), Q0=float(q0_try), T0=T0_resolved,
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
                # this is what turbo_moc/moc_min_length's own
                # validated generate_laes_cases.py does.
                s.run_stage_2_sauer_direct(progress_callback=progress_callback)
                s.run_stage_3_kernel_mln(progress_callback=progress_callback)
                s.run_stage_4_reflex(progress_callback=progress_callback)
        except Exception as e:
            last_exc = e
            if attempt_i == len(q0_candidates) - 1:
                raise NozzleDesignError(f"MOC solve failed for solver={solver!r}: {e}") from e
            continue

        exit_M = float(s.axis_points_vec[-1].M) if s.axis_points_vec else float("nan")
        converged = bool(np.isfinite(exit_M) and abs(exit_M - design_Noz_Mach) / max(design_Noz_Mach, 1e-9) < 0.05)
        if converged or attempt_i == len(q0_candidates) - 1:
            break

    data = s.build_result_data(extra_meta=dict(
        solver=solver, P0=float(P0), T0=T0_resolved, Q0=float(q0_try),
        p_back=p_back, design_Noz_Mach=design_Noz_Mach,
        y_t=y_t, n=n, rho_t=rho_t, rho_d=rho_d, delta_flow=delta_flow,
    ))
    data["solver"] = solver
    data["design_Noz_Mach"] = design_Noz_Mach
    data["p_back"] = p_back
    data["converged"] = converged
    data["exit_M"] = exit_M
    data["runtime_s"] = time.time() - t0
    if q0_try != Q0:
        data["q0_nudged_from"] = float(Q0)
    # Stagnation/total state (s0, T0) -- the isentrope's actual starting
    # point, needed for a T-s diagram alongside the axis-array's own first/
    # last points (~sonic/MoC-start, exit). Not derivable from `axis` alone:
    # the isentrope's total point sits upstream of where Stage 1 seeds the
    # solve, at essentially zero velocity, so it isn't one of the solved
    # points at all.
    data["s0"] = float(state_0.s)
    data["T0_stagnation"] = float(state_0.T)
    return data
