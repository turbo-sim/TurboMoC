"""
turbo_moc.api -- computational API layer.

Mirrors the pattern used by turbodash's core.py: one plain function that
takes primitive arguments (no MOCSolver/FluidManager objects, no UI
concerns) and returns a single flat, JSON-safe dict. A Dash (or any other)
UI layer can sit on top of this exactly like turbodash's app.py sits on top
of core.py's compute_stage_meanline() -- call design_nozzle() inside a
callback, fan its returned dict out into plot/table builders.

This module owns no state itself: FluidManager caching (turbo_moc.core.thermo_characteristics.
get_fluid_manager) already lives one level down and is reused as-is, so
repeated calls at the same operating point (the common case while a UI
sweeps geometry) stay cheap.
"""

import logging
import time

import numpy as np
from scipy.optimize import brentq
import jaxprop as jxp

from turbo_moc.core import thermo_characteristics as _classes
from turbo_moc.core.thermo_characteristics import get_fluid_manager
from turbo_moc.reporting import format_pressure, summary
from turbo_moc.core.moc_conventional import MOCSolver as ConventionalSolver
from turbo_moc.core.moc_minimum_length import MOCSolverMLN

__all__ = ["design_nozzle", "list_supported_fluids", "NozzleDesignError"]

logger = logging.getLogger(__name__)


class NozzleDesignError(Exception):
    """Raised when a nozzle design request is invalid or the solve fails
    before reaching a usable result (as opposed to the solver's own
    graceful early-stop, which still returns a partial-but-usable design --
    see the 'converged' flag in design_nozzle's returned dict)."""


def list_supported_fluids() -> list:
    """CoolProp's fluid list, for populating a fluid-selection UI control."""
    import CoolProp.CoolProp as CP
    return sorted(CP.FluidsList())


def _is_set(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _attempt_callback(progress_callback, attempt, attempts):
    """The caller's callback, its stage labels tagged with the retry number
    ("1 of 4 (retry 2 of 6)") -- the leading stage digit is unchanged, so
    turbo_moc.progress.make_progress_reporter routes it as before."""
    if progress_callback is None or attempt == 0:
        return progress_callback

    def tagged(fraction, stage_label):
        retry = f"retry {attempt + 1} of {attempts}"
        if stage_label.endswith(")"):  # merge into an existing qualifier: "(finished; retry 2 of 6)"
            progress_callback(fraction, f"{stage_label[:-1]}; {retry})")
        else:
            progress_callback(fraction, f"{stage_label} ({retry})")
    return tagged


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
        Back pressure in pascals. This is the recommended design target and
        takes precedence over ``Noz_Mach``.
    Noz_Mach : float, optional
        Exit Mach-number target, used when ``p_back`` is not given: the back
        pressure is the first isentrope pressure below the throat with this
        Mach number. At least one of ``p_back`` and ``Noz_Mach`` is required.
    solver : {"conventional", "mln"}
        Rounded-arc conventional solver or sharp-corner minimum-length solver.
    y_t : float
        Throat half-height.
    n : int
        Number of points used to discretize the initial characteristic line.
    rho_t, rho_d : float
        Geometric radii used by the nozzle construction.
    delta_flow : float
        0 for a planar nozzle, 1 for an axisymmetric nozzle (y is then the
        radius and the compatibility relations include the sin(theta)/y source
        term).
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
    if solver not in ("conventional", "mln"):
        raise NozzleDesignError(f"solver must be 'conventional' or 'mln', got {solver!r}")
    if p_back is None and Noz_Mach is None:
        raise NozzleDesignError("At least one of p_back or Noz_Mach is required.")

    t0 = time.perf_counter()

    fluid = jxp.Fluid(fluid_name)
    inlet = f"Q0 {float(Q0):.4g}" if _is_set(Q0) else (f"T0 {T0:.2f} K" if T0 is not None else "T0 not set")
    target = f"target pressure {format_pressure(p_back)}" if p_back is not None else f"target Mach {Noz_Mach:.4f}"
    logger.info("  MoC: %s; %s; P0 %s, %s; %d seed points; %s", solver, fluid_name, format_pressure(P0),
                inlet, n, target)

    q0_is_zero = Q0 is not None and not (isinstance(Q0, float) and np.isnan(Q0)) and float(Q0) == 0.0
    q0_candidates = [Q0]
    if use_true_sauer_line and auto_nudge_q0 and q0_is_zero:
        q0_candidates += [0.005, 0.01, 0.02, 0.05, 0.1]

    last_exc = None
    p_back_input = p_back
    for attempt_i, q0_try in enumerate(q0_candidates):
        callback = _attempt_callback(progress_callback, attempt_i, len(q0_candidates))
        state_0 = _inlet_state(fluid, P0, T0, q0_try)
        T0_resolved = float(state_0.T)

        design_Noz_Mach = Noz_Mach
        p_back = p_back_input
        if p_back is None:
            p_back = pressure_for_mach(fluid, state_0, Noz_Mach)
            logger.info("  MoC: exit Mach %.4f corresponds to %s", Noz_Mach, format_pressure(p_back))
        if p_back is not None:
            state_out = fluid.get_state(jxp.PSmass_INPUTS, p_back, state_0.s)
            v_out = float(np.sqrt(abs(2 * (state_0.h - state_out.h))))
            design_Noz_Mach = float(v_out / state_out.a)

        cached = len(_classes._fluid_manager_cache)
        start = time.perf_counter()
        fm = get_fluid_manager(fluid_name, backend, P0=P0, T0=T0_resolved, Q0=q0_try)
        if len(_classes._fluid_manager_cache) > cached:
            logger.info("  MoC: fluid properties initialized in %.2f s (first-call setup and JIT "
                        "compilation included)", time.perf_counter() - start)
        else:
            logger.info("  MoC: reusing cached fluid properties")

        tau = np.linspace(0, tau_max, int(tau_max / tau_step) + 1)
        base_config = dict(
            y_t=y_t, n=n, rho_t=rho_t, P0=float(P0), Q0=float(q0_try), T0=T0_resolved,
            tau=tau, rho_d=rho_d, Noz_Mach=design_Noz_Mach, P_back=p_back,
            fluid_name=fluid_name, delta_flow=delta_flow, n_reflex=float("nan"),
        )

        try:
            if solver == "conventional":
                s = ConventionalSolver(base_config, show_plot=show_plot, fluid_manager=fm)
                s.run_stage_1_sauer(use_true_sauer_line=use_true_sauer_line, progress_callback=callback)
                s.run_stage_2_ivp(progress_callback=callback)
                s.run_stage_3_kernel(progress_callback=callback)
                s.run_stage_4_reflex(progress_callback=callback)
            else:
                state_out = fluid.get_state(jxp.PSmass_INPUTS, p_back, state_0.s)
                v_out = float(np.sqrt(abs(2 * (state_0.h - state_out.h))))
                nu_f = fm.nu_of_V(v_out)
                theta_star_deg = float(np.degrees(0.5 * nu_f))

                mln_config = dict(base_config)
                if delta_flow:
                    mln_config["tau_mln"] = np.linspace(0, 2 * theta_star_deg, 2 * n_tau_mln - 1)
                else:
                    mln_config["tau_mln"] = np.linspace(0, theta_star_deg, n_tau_mln)
                mln_config["flashing"] = not use_true_sauer_line
                s = MOCSolverMLN(mln_config, show_plot=show_plot, fluid_manager=fm)
                s.run_stage_1_sauer(progress_callback=callback)
                # sauer_direct, not ivp_march, for BOTH branches: for flashing,
                # it collapses the flat front to just the corner point, matching
                # the reference algorithm exactly (Zebbiche & Youbi 2007 Fig.
                # 3a: ray 1 is corner -> axis directly, zero interior points --
                # see MOCSolverMLN.run_stage_2_sauer_direct's docstring). Marching
                # first would feed Stage 3 a multi-point front instead, breaking
                # that property without changing the converged exit condition --
                # this is what turbo_moc/moc_min_length's own
                # validated generate_laes_cases.py does.
                s.run_stage_2_sauer_direct(progress_callback=callback)
                s.run_stage_3_kernel_mln(progress_callback=callback)
                s.run_stage_4_reflex(progress_callback=callback)
        except Exception as e:
            last_exc = e
            if attempt_i == len(q0_candidates) - 1:
                # Propagated, not logged here: the caller's boundary reports it once.
                raise NozzleDesignError(f"MOC solve failed for solver={solver!r}: {e}") from e
            logger.warning("  MoC attempt %d of %d failed with Q0 = %g (%s). Retrying with Q0 = %g.",
                           attempt_i + 1, len(q0_candidates), float(q0_try), e, q0_candidates[attempt_i + 1])
            continue

        exit_M = float(s.axis_points_vec[-1].M) if s.axis_points_vec else float("nan")
        converged = bool(np.isfinite(exit_M) and abs(exit_M - design_Noz_Mach) / max(design_Noz_Mach, 1e-9) < 0.05)
        if converged or attempt_i == len(q0_candidates) - 1:
            break
        logger.warning("  MoC attempt %d of %d with Q0 = %g did not converge (exit Mach %.4f, design %.4f). "
                       "Retrying with Q0 = %g.", attempt_i + 1, len(q0_candidates), float(q0_try), exit_M,
                       design_Noz_Mach, q0_candidates[attempt_i + 1])

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
    data["runtime_s"] = time.perf_counter() - t0
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
    _report_result(data, s, converged, exit_M, design_Noz_Mach, p_back, callback)
    return data


def pressure_for_mach(fluid, state_0, mach, n_scan=400, p_min_ratio=1e-4):
    if mach is None or not np.isfinite(mach) or mach <= 1:
        raise NozzleDesignError(f"The exit Mach number must be larger than 1, got {mach!r}.")
    P0 = float(state_0.p)

    def mach_at(p):
        state = fluid.get_state(jxp.PSmass_INPUTS, float(p), state_0.s)
        return float(np.sqrt(abs(2 * (state_0.h - state.h))) / state.a)

    pressures = np.geomspace(0.999 * P0, p_min_ratio * P0, n_scan)
    previous = None
    for p in pressures:
        value = mach_at(p) - mach
        if np.isfinite(value) and value >= 0:
            if previous is None:
                return float(p)
            return float(brentq(lambda q: mach_at(q) - mach, p, previous[0], xtol=1e-9 * P0))
        if np.isfinite(value):
            previous = (p, value)
    raise NozzleDesignError(f"Exit Mach {mach:.4f} is not reached on the isentrope above "
                            f"{format_pressure(p_min_ratio * P0)}.")


def _report_result(data, solver, converged, exit_M, design_Noz_Mach, p_back, callback):
    """Final summary, a warning for a non-converged result, and the closing
    progress update -- which names the outcome rather than implying
    convergence: all stages ran, the design target may still be missed."""
    exit_p = float(solver.axis_points_vec[-1].p) if solver.axis_points_vec else float("nan")
    deviation = abs(exit_M - design_Noz_Mach) / max(design_Noz_Mach, 1e-9) if np.isfinite(exit_M) else float("nan")
    rows = {"Status": "Converged" if converged else f"Not converged (exit Mach {100 * deviation:.1f}% off design)"}
    rows["Exit pressure"] = format_pressure(exit_p)
    if p_back is not None:
        rows["Target pressure"] = format_pressure(p_back)
    rows["Exit Mach"] = f"{exit_M:.4f} (design {design_Noz_Mach:.4f})"
    if _is_set(data.get("q0_nudged_from")):  # NaN when Q0 was NaN (not a retry)
        rows["Inlet quality"] = f"{data['meta']['Q0']:g} (retried from {data['q0_nudged_from']:g})"
    rows["Total runtime"] = f"{data['runtime_s']:.2f} s"
    summary(logger, "MoC results", rows, indent="  ")
    if not converged:
        logger.warning("  MoC result not converged: exit Mach %.4f vs design %.4f (tolerance 5%%). "
                       "The geometry is returned as computed.", exit_M, design_Noz_Mach)
    if callback is not None:
        callback(1.0, "4 of 4 (finished)" if converged else "4 of 4 (finished, not converged)")
