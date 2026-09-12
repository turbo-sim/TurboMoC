import time
import numpy as np
from jax import numpy as jnp
from moc.core.classes import *
from moc.io.post_process import printProgress
import matplotlib.pyplot as plt
import pandas as pd
import json

jxp.set_plot_options()
colors = jxp.COLORS_PYTHON

class MOCSolver:
    """
    Encapsulates the configuration, state, and sequential logic for a 
    Method of Characteristics (MOC) simulation.
    """
    def __init__(self, config: dict, show_plot=False, fluid_manager: 'FluidManager' = None):
        # --- Configuration & State ---
        self.y_t = config['y_t']
        self.n = config['n']
        self.rho_t = config['rho_t']
        self.P0 = config['P0']
        self.Q0 = config['Q0']
        self.T0 = config['T0']
        self.tau = config['tau']
        self.rho_d = config['rho_d']
        self.Noz_Mach = config['Noz_Mach']
        self.fluid_name = config['fluid_name']
        self.delta_flow = config['delta_flow']
        self.n_reflex = config['n_reflex']
        self.P_back = config.get('P_back', None)  # target exit static pressure
        
        self.IN = {'NozzleType': 'AXI', 'n_ref': self.n_reflex} # Configuration dict for MassFlowCalc
       
        self.mesh_data = {
            "stage2_c_plus": [],
            "stage2_c_minus": [],
            "stage2_fronts": [], # NEW: To capture the vertical fronts in Stage 2
            "stage3_mesh": [],
            "stage3_fronts": []  # NEW: To capture the fronts in Stage 3
        }

        # --- Graphics ---
        self.show_plot = show_plot
        if self.show_plot is True:
            self.fig, self.ax1 = plt.subplots(figsize=(6, 5))
        
        # --- MOC Internal State (Lists/Tuples) ---
        self.Sauer = []
        self.sauer_main = []
        self.IVP = []
        self.nozzle_wall = []
        self.nozzle_wall_divergent = []
        self.nozzle_wall_kernel = []

        self.IVP_MAIN = []
        self.axis_points_vec = [] 
        self.length = []
        
        # --- Initialization ---
        # A pre-built FluidManager can be passed in and reused across many
        # geometry evaluations at the same (fluid, P0, T0/Q0) -- it depends
        # only on those, not on nozzle geometry, and building it (notably
        # its Prandtl-Meyer table) is the dominant one-time cost per case.
        # Rebuilding it on every solver instantiation is wasted work in a
        # design/optimization loop that holds the operating point fixed.
        if fluid_manager is not None:
            print('Reusing provided Fluid Manager...')
            self.fluid_manager = fluid_manager
        else:
            print('Initializing Fluid Manager...')
            self.fluid_manager = FluidManager(
                fluid_name=self.fluid_name, backend="HEOS", P0=self.P0, T0=self.T0, Q0=self.Q0
            )
        self.t0 = time.time()
        
        # 1b. Initialize Global Reusable Calculation Objects (State is copied to these)
        self.p_calc = self._init_point_calc()
        self.point_calc = self._init_point_calc()
        self.wall_calc = self._init_point_calc() 
        self.pt3_calc = self._init_point_calc()
        self.pt1_calc = self._init_point_calc()
        self.pt2_calc = self._init_point_calc()
        self.pt_result = self._init_point_calc()

    def _init_point_calc(self):
        # Helper to initialize reusable objects with stagnation properties
        p = InternalPoint(self.fluid_manager)
        p.P0, p.T0, p.Q0 = self.P0, self.T0, self.Q0
        return p
        
    # --------------------------------------------------------------------------
    # STAGE 1: SAUER ANALYSIS
    # --------------------------------------------------------------------------
    def run_stage_1_sauer(self, use_true_sauer_line=False, progress_callback=None):
        """
        use_true_sauer_line=False (default): flat throat front (x=0 for every
        y), uniform post-jump velocity -- see the long comment below for why
        this exists (avoids the Q=u^2-a^2=0 stall at the M~1 singularity).

        use_true_sauer_line=True: the actual classical Sauer (1947) transonic
        small-perturbation solution (InternalPoint.SauerMain -- a parabolic
        sonic line x(y), built from an expansion around a*), only physically
        applicable when the isentrope crosses M=1 smoothly
        (``jax_solve_critical_state``'s ``SMOOTH_SONIC`` mode) -- i.e.
        no discontinuous flashing
        jump to violate the small-perturbation assumption. Kept as an
        explicit opt-in alongside the flat-front default (not a replacement)
        so both can be generated and compared for the same case.

        progress_callback: optional callable(fraction: float, stage_label: str)
        -- see printProgress's on_progress and moc.progress. Threaded through
        every run_stage_* method the same way, so a UI can drive one overall
        progress bar across the whole solve.
        """
        if use_true_sauer_line:
            self._run_stage_1_sauer_true_line(progress_callback=progress_callback)
            return

        print('\n\n\t========= Flashing Throat Initialization =========\n')

        # 1. Pull the REAL jumped velocity from the critical state
        # We want the V that corresponds to Mach 13, not Mach 1.
        # jax_solve_critical_state can return EITHER a genuine flashing jump
        # (comfortably above M=1) OR, when subcooling is small enough that
        # the isentrope never actually crosses the saturation dome before
        # reaching sonic conditions, a smooth M=1.0000 root (SMOOTH_SONIC
        # mode -- see that function). The mode isn't threaded through to
        # here, so this can't tell which case it got; but treating the
        # exact-M=1 case identically to a real jump starts the whole throat
        # front at Q=u^2-a^2=0 EXACTLY, which silently stalls Stage 2/3's
        # linearised compatibility solve (division by zero) instead of
        # crashing -- the solver just returns the unchanged initial state
        # (verified: dT=2.5K's "converged" result was M=1.0000, x=0, i.e.
        # nothing was computed at all). A small multiplicative margin fixes
        # this: negligible for a real jump (already well above M=1), but
        # nudges an exact-M=1 start just barely supersonic, matching the
        # same defensive pattern SauerMain() already uses elsewhere
        # (a_star = self.a*(1+1e-3)) for the identical class of problem.
        # A flat 1e-3 margin was tried first and was not enough -- dT=2.5K
        # only inched to M=1.0005 before stalling again almost immediately,
        # since Q=u^2-a^2 stays small (not just exactly zero) for a good
        # stretch after the start, which the linearised compatibility solve
        # is still sensitive to. A flat 5% margin cleared that stretch
        # (dT=2.5K then reached M=1.99 vs design 1.95), but it also
        # distorted the other cases, which don't need it at all: their
        # v_jump is already a REAL, physically correct flashing jump
        # (M_jump = 1.4-2.4), so multiplying an already-correct velocity by
        # 1.05 just injects a needless ~1-1.5% error (verified: their exit
        # Mach match went from ~0.3% off to ~1-1.5% off). So the margin is
        # scaled adaptively instead: large only right at M_jump~1 (where
        # the actual singularity lives), decaying to ~0 within a few
        # percent of M=1 -- keeps the fix local to the case that actually
        # needs it.
        h0 = self.fluid_manager.h0
        h_crit = self.fluid_manager.critical_state["h"]
        a_crit = self.fluid_manager.critical_state["a"]
        v_jump_raw = jnp.sqrt(2 * (h0 - h_crit))
        M_jump_raw = v_jump_raw / a_crit
        margin = 1e-3 + 0.05 * jnp.exp(-(M_jump_raw - 1.0) / 0.02)
        v_jump = v_jump_raw * (1 + margin)
        
        y_coords = np.linspace(0, self.y_t, self.n)
        for i, y_val in enumerate(y_coords):
            p = InternalPoint(self.fluid_manager) 
            
            # --- GEOMETRY ---
            p.x = 0.0 # Straight line at throat
            p.y = y_val
            
            # --- PHYSICS ---
            # We manually set the state to the 'Jumped' condition 
            # instead of letting SauerMain() calculate it.
            p.u = v_jump
            p.v = 0.0
            
            # Sync all other properties (P, rho, a, M, theta, alpha, slopes)
            # Using the actual high velocity
            p.ThermoFlowProp() 
            
            self.Sauer.append(p)
            self.sauer_main.append(p)
            printProgress(i+1, self.n, '1 of 4', on_progress=progress_callback)

        self.axis_points_vec.append(self.Sauer[0])

    def _run_stage_1_sauer_true_line(self, progress_callback=None):
        """
        Genuine Sauer (1947) transonic small-perturbation sonic line:
        InternalPoint.SauerMain() builds a parabolic x(y) from an expansion
        around the sonic speed a*, using the throat radius of curvature
        rho_t -- the classical construction, as opposed to run_stage_1_sauer's
        flat/uniform-jump-velocity front. Only meaningful when the isentrope
        actually passes through M=1 smoothly (no flashing jump to break the
        small-perturbation assumption around M=1); the caller is responsible
        for only invoking this for SMOOTH_SONIC cases.

        gamma at the sonic state has to be set on each point BEFORE calling
        SauerMain() (its epsilon/alpha algebra uses self.gamma before the
        internal compute_sonic_state() call would otherwise populate it) --
        precomputed once here from the manager's own critical_state.
        """
        print('\n\n\t========= Flashing Throat Initialization (true Sauer parabolic sonic line) =========\n')

        gamma_star = float(self.fluid_manager.critical_state["cp"] / self.fluid_manager.critical_state["cv"])

        y_coords = np.linspace(0, self.y_t, self.n)
        for i, y_val in enumerate(y_coords):
            p = InternalPoint(self.fluid_manager)
            p.y = y_val
            p.y_t = self.y_t
            p.rho_t = self.rho_t
            p.gamma = gamma_star
            p.a = self.fluid_manager.a_star
            p.delta = 0.0
            p.SauerMain()

            self.Sauer.append(p)
            self.sauer_main.append(p)
            printProgress(i + 1, self.n, '1 of 4', on_progress=progress_callback)

        self.axis_points_vec.append(self.Sauer[0])

    # --------------------------------------------------------------------------
    # STAGE 2: SAUER TO IVP (IVP Marching)
    # --------------------------------------------------------------------------
    def run_stage_2_ivp(self, progress_callback=None):
        self.IVP.append(self.Sauer[-1])
        print('\n\n========= Sauer to IVP (Stage 2)=========\n')
        num_iterations = self.n*2 - 2

        for j in range(0, num_iterations):
            printProgress(j, num_iterations - 1, '2 of 4', on_progress=progress_callback)
            k = len(self.Sauer)
            Sauer_old = self.Sauer 
            Sauer_new = []
            
            for i in range(k-1): 
                self.p_calc.compute_from_characteristics(Sauer_old[i], Sauer_old[i+1])
                p = InternalPoint(self.fluid_manager) 
                p.__dict__.update(self.p_calc.__dict__) 

                # --- CAPTURE HERE (Inside i loop) ---
                self.mesh_data["stage2_c_plus"].append(([float(Sauer_old[i].x), float(p.x)], [float(Sauer_old[i].y), float(p.y)]))
                self.mesh_data["stage2_c_minus"].append(([float(Sauer_old[i+1].x), float(p.x)], [float(Sauer_old[i+1].y), float(p.y)]))
                                
                if self.show_plot is True:
                    self.ax1.plot(p.x, p.y, "o", color="black", markersize=0.5)
                    self.ax1.plot((Sauer_old[i].x, p.x),(Sauer_old[i].y, p.y), "-", color=colors[3], linewidth=1.5) 
                    self.ax1.plot((Sauer_old[i+1].x, p.x),(Sauer_old[i+1].y, p.y), "-", color=colors[2], linewidth=1.5) 

                Sauer_new.append(p)
            
            # This is necessary to have the IVP only on one half
            if j == 0 or j % 2 == 0:
                mirror_point = InternalPoint(self.fluid_manager)
                mirror_point.__dict__.update(Sauer_new[0].__dict__)
                mirror_point.y = -mirror_point.y
                mirror_point.v = -mirror_point.v
                mirror_point.ThermoFlowProp()

                axis_point = InternalPoint(self.fluid_manager)
                axis_point.solve_centerline_point(mirror_point, Sauer_new[0])
                Sauer_new.insert(0, axis_point)
                self.axis_points_vec.append(axis_point)
            
            self.Sauer = Sauer_new
            self.IVP.append(self.Sauer[-1])

            self.IVP_MAIN = self.IVP

            self.mesh_data["stage2_fronts"].append(([float(p.x) for p in Sauer_new], [float(p.y) for p in Sauer_new]))

        if self.show_plot is True:
            self.ax1.plot(p.x, p.y, "o", color="black", markersize=0.5)

    # --------------------------------------------------------------------------
    # STAGE 3: IVP 2 KERNEL (Kernel Marching)
    # --------------------------------------------------------------------------
    def _kernel_step_core(self, front, ang, collect_segments=False):
        """
        Advance `front` (a plain list of InternalPoint states, NOT
        necessarily self.IVP) by one MOC unit-process step to angle `ang`,
        returning the new front (or (new front, segments) if
        collect_segments, where segments is the list of (pt2->new_point)
        line pairs used for the stage3_mesh visualization -- matching the
        per-J capture the original inline loop used to do). Pure
        extraction of steps 1-4 of run_stage_3_kernel's main loop body,
        reused for every step (coarse and fine/adaptive alike -- see
        run_stage_3_kernel) so there is a single source of truth for the
        unit-process marching logic.
        """
        IVP_New = []
        segments = [] if collect_segments else None
        self.pt3_calc.__dict__.update(front[0].__dict__)
        self.pt1_calc.__dict__.update(front[1].__dict__)
        self.wall_calc.x, self.wall_calc.y = rotation_around_center(self.y_t, ang, self.rho_d)

        flag = 0; k = 0
        while flag != 1:
            try:
                intermediate_pt_state = self.wall_calc.Inv_Wall_pnt(self.wall_calc, self.pt3_calc, self.pt1_calc, ang)
                self.pt_result.__dict__.update(intermediate_pt_state.__dict__)
                flag = 1
            except NameError:
                k += 1
                self.pt3_calc.__dict__.update(front[0 + k].__dict__)
                self.pt1_calc.__dict__.update(front[1 + k].__dict__)

        p_wall_store = InternalPoint(self.fluid_manager)
        p_wall_store.__dict__.update(self.wall_calc.__dict__)
        IVP_New.append(p_wall_store)

        self.pt3_calc.__dict__.update(self.pt_result.__dict__)

        for j in range(0, int(len(front) - 1) - k):
            if j == 0: self.pt3_calc.__dict__.update(p_wall_store.__dict__)
            self.pt2_calc.__dict__.update(front[j + 1 + k].__dict__)
            self.point_calc.compute_from_characteristics(self.pt2_calc, self.pt3_calc)
            p_result_store = InternalPoint(self.fluid_manager)
            p_result_store.__dict__.update(self.point_calc.__dict__)
            if collect_segments:
                segments.append(([float(self.pt2_calc.x), float(self.point_calc.x)], [float(self.pt2_calc.y), float(self.point_calc.y)]))
            IVP_New.append(p_result_store)
            self.pt3_calc.__dict__.update(p_result_store.__dict__)

        p_result_store_last = p_result_store
        self.pt2_calc.__dict__.update(p_result_store_last.__dict__)
        self.pt3_calc.__dict__.update(p_result_store_last.__dict__)
        self.pt3_calc.v = -self.pt3_calc.v
        self.pt3_calc.y = -self.pt3_calc.y
        self.point_calc.compute_from_characteristics(self.pt3_calc, self.pt2_calc)

        p_result_final = InternalPoint(self.fluid_manager)
        p_result_final.__dict__.update(self.point_calc.__dict__)
        IVP_New.append(p_result_final)
        return (IVP_New, segments) if collect_segments else IVP_New

    @staticmethod
    def _interp_front(front_old, front_new, frac, fluid_manager):
        interpolated = []
        for p_old, p_new in zip(front_old, front_new):
            p = InternalPoint(fluid_manager)
            p.x = float(p_old.x) + frac * (float(p_new.x) - float(p_old.x))
            p.y = float(p_old.y) + frac * (float(p_new.y) - float(p_old.y))
            p.u = float(p_old.u) + frac * (float(p_new.u) - float(p_old.u))
            p.v = float(p_old.v) + frac * (float(p_new.v) - float(p_old.v))
            p.ThermoFlowProp()
            interpolated.append(p)
        return interpolated

    @staticmethod
    def _adaptive_step(diff_prev, rate_estimate, coarse_step, min_step=0.005, fraction=0.25):
        """
        Marching step size (deg), shrinking as the sweep approaches the
        target -- refines the KERNEL MARCHING itself over the final
        approach (accumulated truncation error shrinks along with the
        step), not just a one-off interpolation at the last coarse step
        (a single fine bracket at the very end can't correct for error
        already baked into the front by ~25-33 deg of coarse marching
        beforehand).

        A static relative-distance threshold ("shrink once within X% of
        target") was tried first and failed: verified directly (dT=5K)
        that the metric can still be ~11-12% from target at one step
        before crossing, comfortably above a 10% "start shrinking"
        threshold -- so the single next coarse step jumped straight past
        the crossing without ever engaging the fine steps, since the
        decision was made looking at the CURRENT distance, not whether
        the upcoming step is large enough to overshoot it. Sized instead
        from the estimated LOCAL rate of change (from the previous
        step's actual result): predicted_remaining_ang = the angle
        distance still needed to reach the target at the current rate,
        and the next step is capped to a fraction of that -- so it
        naturally shrinks several steps before the crossing, however
        steep the local metric-vs-angle slope happens to be for a given
        case, rather than relying on a fixed threshold tuned to one case.
        """
        if rate_estimate is None or rate_estimate == 0:
            return coarse_step
        predicted_remaining_ang = abs(diff_prev / rate_estimate)
        return min(coarse_step, max(min_step, fraction * predicted_remaining_ang))

    def run_stage_3_kernel(self, progress_callback=None):
        tau_max = float(self.tau[-1])
        coarse_step = float(self.tau[1] - self.tau[0]) if len(self.tau) > 1 else 1.0
        print('\n ========= Developing Kernal Region (Stage 3) ========= \n')

        # Termination metric: stop on STATIC PRESSURE reaching the target back
        # pressure (self.P_back) when provided, not on Mach. Pressure decreases
        # monotonically along any expansion, but the two-phase HEM sound speed
        # is not monotonic in quality, so Mach vs pressure can be non-monotonic
        # (see NICFD_2026/.../02_flashing_preliminary/complete_flashing.pdf,
        # panel c: Mach dips after the post-jump peak and can flatten out
        # before reaching the design value, or approach it from either side --
        # "stop when M reaches Noz_Mach" is ambiguous there). Pressure has no
        # such ambiguity. Falls back to Mach-based stopping if P_back isn't set.
        use_pressure = self.P_back is not None
        if use_pressure:
            metric = lambda pt: float(pt.p)
            target = self.P_back
        else:
            metric = lambda pt: float(pt.M)
            target = self.Noz_Mach
        diff_prev = metric(self.IVP[0]) - target
        diff_initial = diff_prev if diff_prev != 0 else 1.0
        ang = 0.0
        rate_estimate = None  # d(diff)/d(angle), estimated from the previous step

        c = 0
        while True:
            c += 1
            step = self._adaptive_step(diff_prev, rate_estimate, coarse_step)
            ang_next = ang + step
            if ang_next > tau_max:
                metric_name = 'p_target' if use_pressure else 'Noz_Mach'
                print(f'\n\n\t***** WARNING: reached the end of tau '
                      f'({tau_max:.2f} deg) without crossing {metric_name} — '
                      f'result may not have reached the design exit condition *****\n\n')
                break

            ivp_before_step = self.IVP

            try:
                IVP_New, segments = self._kernel_step_core(self.IVP, ang_next, collect_segments=True)
                self.mesh_data["stage3_mesh"].extend(segments)
                if self.show_plot is True:
                    for (sx, sy) in segments:
                        self.ax1.plot(sx, sy, "-", color=colors[3], linewidth=1.5)
                self.nozzle_wall.append(self.IVP[0])
                self.nozzle_wall_kernel.append(self.IVP[0])
            except Exception as e:
                # Near-sonic starts (post-jump Mach only slightly above 1)
                # make the axis-point compatibility relation (Q = u^2-a^2 -> 0)
                # ill-conditioned; large accumulated turning can eventually
                # drive a point outside the fluid table's valid range. Stop
                # gracefully at the last converged front instead of crashing
                # the whole batch -- self.IVP still holds that last-good state.
                print(f'\n\n\t***** WARNING: Stage 3 stopped at angle {ang_next:.2f} deg '
                      f'due to a property-solve failure ({e!s:.120}) — '
                      f'design Mach not reached, using last valid state *****\n\n')
                break

            diff_curr = metric(IVP_New[-1]) - target

            if diff_prev * diff_curr <= 0:
                # Even with the adaptive shrink, the crossing lands inside
                # (last, ang_next] rather than exactly at ang_next -- do one
                # final linear interpolation within this now-small step
                # (<=0.01 deg once rel_dist is small) for the exact target.
                frac = diff_prev / (diff_prev - diff_curr)
                interpolated = self._interp_front(ivp_before_step, IVP_New, frac, self.fluid_manager)
                self.IVP = interpolated
                self.axis_points_vec.append(interpolated[-1])
                self.nozzle_wall.pop(); self.nozzle_wall_kernel.pop()
                self.nozzle_wall.append(interpolated[0])
                self.nozzle_wall_kernel.append(interpolated[0])
                self.mesh_data["stage3_fronts"].append(([float(p.x) for p in interpolated], [float(p.y) for p in interpolated]))
                print(f'\n\n\t***** Kernel Region Done (adaptive step reached {step:.4f} deg, '
                      f'exact target within ({ang:.3f}, {ang_next:.3f}] deg, '
                      f'p={interpolated[-1].p:.1f} M={interpolated[-1].M:.4f}) *****\n\n')
                if progress_callback is not None:
                    progress_callback(1.0, '3 of 4')
                break

            self.IVP = IVP_New
            self.axis_points_vec.append(IVP_New[-1])

            if self.show_plot is True:
                self.ax1.plot([element.x for element in IVP_New], [element.y for element in IVP_New], "-", color = colors[2], linewidth=1.5)

            self.mesh_data["stage3_fronts"].append(([float(p.x) for p in IVP_New], [float(p.y) for p in IVP_New]))

            t2 = time.time()
            if c % 10 == 0: print('\n\tProgress\t\tAngle\t\tMa_C\t\tp_axis[Pa]\tp_target[Pa]\tx\t\tMass\t\tClockTime')
            printProgress(IVP_New[-1].M - 1, self.Noz_Mach - 1, '3 of 4', '', 2, 15)
            print("\t\t\t\t%2.3f\t\t%1.2f\t\t%.1f\t\t%s\t\t%1.4f\t\t%.4f\t\t%.2f\t\t" % (
                ang_next, IVP_New[-1].M, IVP_New[-1].p,
                f"{target:.1f}" if use_pressure else "n/a",
                IVP_New[-1].x, MassFlowCalc(IVP_New) / 2, t2 - self.t0,))
            if progress_callback is not None:
                # Fraction of the initial (metric - target) gap closed so far
                # -- uses whichever metric (pressure or Mach) Stage 3 is
                # actually terminating on, unlike the printProgress call
                # above (always Mach-based, even under pressure-based
                # termination -- fine as a rough terminal indicator, not
                # accurate enough to drive a real UI progress bar).
                frac = 1.0 - min(1.0, max(0.0, abs(diff_curr) / abs(diff_initial)))
                progress_callback(frac, '3 of 4')

            rate_estimate = (diff_curr - diff_prev) / step
            diff_prev = diff_curr
            ang = ang_next


    # --------------------------------------------------------------------------
    # STAGE 4: REFLEX ZONE
    # --------------------------------------------------------------------------
    def run_stage_4_reflex(self, progress_callback=None):
            print('\nStart: Reflex Zone (Stage 4)\n')
            
            print("Optimizing: Pre-calculating Fluid Properties for Stage 4...")
            for p in self.IVP:
                p.ThermoFlowProp() 
                
            final_IVP_line = self.IVP

            # A: Finding Boundary (Wall Contour)
            # MassIn is the cumulative mass-flow integral over final_IVP_line
            # [0:i+1]. The original code recomputed this via MassFlowCalc()
            # over the whole (growing) prefix from scratch every iteration --
            # O(i) work per step, O(n^2) total over the sweep. Each step only
            # adds ONE new segment (i-1, i) to the integral, so accumulate it
            # incrementally instead (O(1) per step, O(n) total), reusing
            # MassFlowCalc's own per-segment formula (delta=0, matching its
            # hardcoded planar assumption) rather than re-deriving it.
            mass_cumulative = 0.0
            for i in range(1, len(final_IVP_line)):
                p_prev, p_cur = final_IVP_line[i - 1], final_IVP_line[i]
                seg_len = DistCalc(p_prev.x, p_prev.y, p_cur.x, p_cur.y)
                rot_vec = RotVec(MakeVec(p_prev.x, p_prev.y, p_cur.x, p_cur.y), -90)
                rho_avg = (p_prev.rho + p_cur.rho) / 2
                u_avg = (p_prev.u + p_cur.u) / 2
                v_avg = (p_prev.v + p_cur.v) / 2
                V_avg = vecDotVec(u_avg, v_avg, rot_vec[0], rot_vec[1])
                mass_cumulative = mass_cumulative + (rho_avg * seg_len * V_avg)

                Ref_P = (final_IVP_line[i],)
                MassIn = mass_cumulative
                MassOut = MassFlowCalc(Ref_P)

                l = MassIn / MassOut
                self.length.append(l)

                p_reflex_input = InternalPoint(self.fluid_manager)
                p_reflex_input.__dict__.update(Ref_P[0].__dict__)
                LocWall = p_reflex_input.ReflexLine(1, l)

                self.nozzle_wall.append(LocWall)
                self.nozzle_wall_divergent.append(LocWall)
                printProgress(i, len(self.IVP)-1, '4A of 4', on_progress=progress_callback)


            # Ensure the nozzle wall starts at the throat (y_t) and ends at the axis (x_end, 0)
            # self.nozzle_wall_final = [InternalPoint(self.fluid_manager, x=0.0, y=0.0)] + \
            #                     self.nozzle_wall + \
            #                     [InternalPoint(self.fluid_manager, x=self.nozzle_wall[-1].x, y=0.0)]

            self.nozzle_wall_final = self.nozzle_wall


    # --------------------------------------------------------------------------
    # PLOTTING AND REPORTING
    # --------------------------------------------------------------------------
    def plot_results(self):
        
        if self.show_plot is True:
            # Nozzle plot
            wall_x_coords = [p.x for p in self.nozzle_wall_final]
            wall_y_coords = [p.y for p in self.nozzle_wall_final]
            div_wall_x_coords = [p.x for p in self.nozzle_wall_divergent]
            div_wall_y_coords = [p.y for p in self.nozzle_wall_divergent]
            kernel_wall_x_coords = [p.x for p in self.nozzle_wall_kernel]
            kernel_wall_y_coords = [p.y for p in self.nozzle_wall_kernel]
            sauer_x_coords = [p.x for p in self.sauer_main]
            sauer_y_coords = [p.y for p in self.sauer_main]
            IVP_x = [p.x for p in self.IVP[1:]]
            IVP_y = [p.y for p in self.IVP[1:]]

            self.ax1.plot(wall_x_coords, wall_y_coords, "-", color="black", linewidth=1.5, label="Nozzle Wall Contour")
            self.ax1.plot((IVP_x, div_wall_x_coords), (IVP_y, div_wall_y_coords), "-", color = colors[3], linewidth=1.5)
            self.ax1.plot(div_wall_x_coords, div_wall_y_coords, "o", color="black")
            self.ax1.plot((wall_x_coords[0], wall_x_coords[-1]), (0.0, 0.0), "--", color="black", linewidth=1.5)
            self.ax1.plot(kernel_wall_x_coords, kernel_wall_y_coords, "o", color = "black")
            self.ax1.plot(sauer_x_coords, sauer_y_coords, "-o", color = colors[0], linewidth=1.5)
            self.ax1.set_aspect('equal')

            # Axis propertis plot
            # --- Prepare data once ---
            p_axis_data = [p.p for p in self.axis_points_vec]
            Ma_axis_data = [p.M for p in self.axis_points_vec]
            rho_axis_data = [p.rho for p in self.axis_points_vec]
            s_axis_data = [p.s for p in self.axis_points_vec]
            T_axis_data = [p.T for p in self.axis_points_vec]
            coord_axis_data = [p.x for p in self.axis_points_vec]

            # --- Create a single figure with 3 rows and 1 column of subplots ---
            fig_properties, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True)
            plt.subplots_adjust(hspace=0.4) # Add space between plots

            # --- Plot 1: Pressure ---
            ax2.plot(coord_axis_data, p_axis_data, color=colors[2], linewidth=2)
            ax2.plot((coord_axis_data[-1], wall_x_coords[-1]), (p_axis_data[-1], p_axis_data[-1]), color=colors[2], linewidth=2)
            ax2.set_ylabel("Pressure (Pa)")

            # --- Plot 2: Mach Number ---
            ax3.plot(coord_axis_data, Ma_axis_data, color=colors[4], linewidth=2)
            ax3.plot((coord_axis_data[-1], wall_x_coords[-1]), (Ma_axis_data[-1], Ma_axis_data[-1]), color=colors[4], linewidth=2)
            ax3.set_ylabel("Mach Number (-)")

            # --- Plot 3: Density ---
            ax4.plot(coord_axis_data, rho_axis_data, color=colors[6], linewidth=2)
            ax4.plot((coord_axis_data[-1], wall_x_coords[-1]), (rho_axis_data[-1], rho_axis_data[-1]), color=colors[6], linewidth=2)
            ax4.set_xlabel("Axial Distance (m)")
            ax4.set_ylabel(r"Density ($\mathrm{kg/m}^3$)")

            # T-s digram plot
            fluid = jxp.Fluid(self.fluid_name)
            fig, ax = fluid.plot_phase_diagram(x_prop="s", y_prop="T", plot_quality_isolines=False)
            ax.plot(s_axis_data, T_axis_data, "-", color="black")


        # Final formatting
        total_runtime = time.time() - self.t0
        print(f"\nCompleted MOC Simulation in {total_runtime:.4f} seconds")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Method of Characteristics Nozzle Design")
        # plt.legend(loc="best")
        # plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def build_result_data(self, extra_meta=None):
            """
            Returns the full result dict (wall/axis/mesh data, all JSON-safe
            plain floats/lists) without touching the filesystem -- the part
            of export_full_result_data that's actually reusable in-memory
            (e.g. by moc.api), split out so that function can just json.dump
            this instead of duplicating the dict-building logic.
            """
            def to_l(pts, attr): return [float(getattr(p, attr)) for p in pts]

            # Wall points carry u,v (set by ReflexLine) but ThermoFlowProp()
            # was never called on them, so p/M/T/rho/s were never populated
            # -- do that now so wall property distributions can be exported.
            for p in self.nozzle_wall_final:
                p.ThermoFlowProp()

            data = {
                "wall_final": {
                    "x": to_l(self.nozzle_wall_final, 'x'), "y": to_l(self.nozzle_wall_final, 'y'),
                    "p": to_l(self.nozzle_wall_final, 'p'), "M": to_l(self.nozzle_wall_final, 'M'),
                    "rho": to_l(self.nozzle_wall_final, 'rho'), "T": to_l(self.nozzle_wall_final, 'T'),
                },
                "div_wall": {"x": to_l(self.nozzle_wall_divergent, 'x'), "y": to_l(self.nozzle_wall_divergent, 'y')},
                "kernel_wall": {"x": to_l(self.nozzle_wall_kernel, 'x'), "y": to_l(self.nozzle_wall_kernel, 'y')},
                "sauer": {"x": to_l(self.sauer_main, 'x'), "y": to_l(self.sauer_main, 'y')},
                "IVP": {"x": to_l(self.IVP[1:], 'x'), "y": to_l(self.IVP[1:], 'y')},
                "mesh_data": self.mesh_data,
                "axis": {
                    "x": to_l(self.axis_points_vec, 'x'),
                    "p": to_l(self.axis_points_vec, 'p'),
                    "M": to_l(self.axis_points_vec, 'M'),
                    "rho": to_l(self.axis_points_vec, 'rho'),
                    "s": to_l(self.axis_points_vec, 's'),
                    "T": to_l(self.axis_points_vec, 'T')
                },
                "fluid_name": self.fluid_name
            }
            if extra_meta:
                data["meta"] = extra_meta
            return data

    def export_full_result_data(self, filename="moc_result_data.json", extra_meta=None):
            data = self.build_result_data(extra_meta=extra_meta)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
