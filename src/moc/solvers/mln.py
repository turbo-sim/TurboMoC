"""
Minimum-Length Nozzle (MLN) solver.

Inherits Stages 1, 2, and 4 from MOCSolver in moc/srcMOC/moc_solver.py.
Replaces Stage 3 (kernel) with a centred-expansion-fan construction:
  - No circular arc wall; the throat corner is a sharp point at (0, y_t).
  - All right-running (C+) characteristics originate from the corner.
  - Interior points and axis points are computed exactly as before.
  - Termination: axis Mach number >= Noz_Mach.
Stage 4 (reflex / turning section) is inherited unchanged.
"""

import time
import json
import numpy as np
from jax import numpy as jnp
from moc.core.classes import (
    InternalPoint, FluidManager, MassFlowCalc,
    rotation_around_center, jitted_calculate_static_properties,
    DistCalc, RotVec, MakeVec, vecDotVec,
)
from moc.io.post_process import printProgress
import matplotlib.pyplot as plt
import jaxprop as jxp
import jax

jxp.set_plot_options()
colors = jxp.COLORS_PYTHON


# ---------------------------------------------------------------------------
# Re-use the JAX vectorised stage-2 step from the parent solver
# ---------------------------------------------------------------------------
@jax.jit
def march_line_jax_step(carry, _):
    """Vectorised MoC step with predictor-corrector (identical to moc/)."""
    ld, j, delta, s0, h0, fluid = carry

    xP, yP, uP, vP, aP, thP, alP = [ld[k][:-1] for k in ['x','y','u','v','a','theta','alpha']]
    xM, yM, uM, vM, aM, thM, alM = [ld[k][1:]  for k in ['x','y','u','v','a','theta','alpha']]

    # Predictor slopes
    lamP_pre = jnp.tan(thP + alP)
    lamM_pre = jnp.tan(thM - alM)

    denom_geom = lamM_pre - lamP_pre
    safe_denom = jnp.where(jnp.abs(denom_geom) < 1e-12, 1.0, denom_geom)
    x_pre = (lamM_pre * xM - lamP_pre * xP + yP - yM) / safe_denom
    y_pre = yP + lamP_pre * (x_pre - xP)

    QP, QM   = uP**2 - aP**2, uM**2 - aM**2
    R1P, R1M = 2*uP*vP - QP*lamP_pre, 2*uM*vM - QM*lamM_pre
    SP = jnp.where(jnp.abs(yP) <= 1e-9, 0.0, carry[2] * (aP**2 * vP) / yP)
    SM = jnp.where(jnp.abs(yM) <= 1e-9, 0.0, carry[2] * (aM**2 * vM) / yM)

    TP = SP*(x_pre - xP) + QP*uP + R1P*vP
    TM = SM*(x_pre - xM) + QM*uM + R1M*vM

    det_pre  = QP*R1M - QM*R1P
    safe_det = jnp.where(jnp.abs(det_pre) < 1e-12, 1.0, det_pre)
    u_pre = (TP*R1M - TM*R1P) / safe_det
    v_pre = (QP*TM  - QM*TP)  / safe_det

    # Intermediate thermo
    V_pre  = jnp.sqrt(u_pre**2 + v_pre**2)
    _, _, _, _, _, a_int, _ = jitted_calculate_static_properties(fluid, V_pre, s0, h0)
    th_int = jnp.arctan2(v_pre, u_pre)
    al_int = jnp.arcsin(jnp.clip(a_int / jnp.where(V_pre == 0, 1.0, V_pre), 0, 1))

    # Corrector slopes (averaged)
    lamP_cor = 0.5*(lamP_pre + jnp.tan(th_int + al_int))
    lamM_cor = 0.5*(lamM_pre + jnp.tan(th_int - al_int))
    safe_cor = jnp.where(jnp.abs(lamM_cor - lamP_cor) < 1e-12, 1.0, lamM_cor - lamP_cor)
    new_x = (lamM_cor*xM - lamP_cor*xP + yP - yM) / safe_cor
    new_y = yP + lamP_cor*(new_x - xP)

    valid_mask = jnp.abs(new_x) > 1e-9
    def mask(arr):   return jnp.where(valid_mask, arr, 0.0)
    def pad(arr):    return jnp.pad(arr, (0, 1))

    marching_ld = {
        'x': pad(mask(new_x)), 'y': pad(mask(new_y)),
        'u': pad(mask(u_pre)), 'v': pad(mask(v_pre)),
        'a': pad(mask(a_int)), 'theta': pad(mask(th_int)),
        'alpha': pad(mask(al_int))
    }

    # Axis reflection every 2nd step
    def do_reflect(ld_in):
        lam_r  = jnp.tan(ld_in['theta'][0] - ld_in['alpha'][0])
        lam_r  = jnp.where(jnp.abs(lam_r) < 1e-12, 1.0, lam_r)
        x_ax   = ld_in['x'][0] - ld_in['y'][0] / lam_r
        u_ax   = jnp.sqrt(ld_in['u'][0]**2 + ld_in['v'][0]**2)
        _, _, _, _, _, a_ax, _ = jitted_calculate_static_properties(fluid, u_ax, s0, h0)
        def shift(arr, val): return jnp.concatenate([jnp.array([val]), arr[:-1]])
        return {
            'x': shift(ld_in['x'], x_ax), 'y': shift(ld_in['y'], 0.0),
            'u': shift(ld_in['u'], u_ax), 'v': shift(ld_in['v'], 0.0),
            'a': shift(ld_in['a'], a_ax), 'theta': shift(ld_in['theta'], 0.0),
            'alpha': shift(ld_in['alpha'], jnp.arcsin(jnp.clip(a_ax/u_ax, 0, 1)))
        }

    final_ld = jax.lax.cond(j % 2 == 0, do_reflect, lambda x: x, marching_ld)
    return (final_ld, j+1, carry[2], s0, h0, fluid), final_ld


class MOCSolverMLN:
    """
    Minimum-Length Nozzle solver.
    Stages 1, 2, 4 are identical to MOCSolver in moc/.
    Stage 3 uses a centred expansion fan from the throat corner.
    """

    def __init__(self, config: dict, show_plot=False, fluid_manager: 'FluidManager' = None):
        self.y_t        = config['y_t']
        self.n          = config['n']
        self.rho_t      = config['rho_t']
        self.P0         = config['P0']
        self.Q0         = config['Q0']
        self.T0         = config['T0']
        self.tau        = config['tau']          # standard arc kernel (kept for reference)
        self.tau_mln    = config['tau_mln']      # fan angles for MLN — can be coarser
        self.Noz_Mach   = config['Noz_Mach']
        self.fluid_name = config['fluid_name']
        self.delta_flow = config['delta_flow']
        self.flashing   = config.get('flashing', False)  # subcooled-liquid inlet w/ discontinuous a
        self.P_back     = config.get('P_back', None)     # target exit static pressure (flashing)

        self.show_plot = show_plot
        if show_plot:
            self.fig, self.ax1 = plt.subplots(figsize=(10, 6))

        self.Sauer               = []
        self.sauer_main          = []
        self.IVP                 = []
        self.nozzle_wall         = []
        self.nozzle_wall_divergent = []
        self.nozzle_wall_kernel  = []
        self.axis_points_vec     = []
        self.length              = []

        # Full mesh capture, so any figure can be reproduced from the saved
        # JSON without re-running the solve (see export_full_result_data).
        self.mesh_data = {
            "stage2_c_plus": [],
            "stage2_c_minus": [],
            "stage2_fronts": [],
            "stage3_mesh": [],
            "stage3_fronts": [],
        }

        # See moc.solvers.conventional.MOCSolver's __init__ for why a
        # pre-built FluidManager can be reused across a geometry sweep.
        if fluid_manager is not None:
            print('Reusing provided Fluid Manager...')
            self.fluid_manager = fluid_manager
        else:
            print('Initializing Fluid Manager...')
            self.fluid_manager = FluidManager(
                fluid_name=self.fluid_name, backend="HEOS",
                P0=self.P0, T0=self.T0, Q0=self.Q0
            )
        self.t0 = time.time()

        self.p_calc     = self._init_pt()
        self.point_calc = self._init_pt()
        self.wall_calc  = self._init_pt()
        self.pt1_calc   = self._init_pt()
        self.pt2_calc   = self._init_pt()
        self.pt3_calc   = self._init_pt()
        self.pt_result  = self._init_pt()

    def _init_pt(self):
        p = InternalPoint(self.fluid_manager)
        p.P0, p.T0, p.Q0 = self.P0, self.T0, self.Q0
        return p

    # ------------------------------------------------------------------
    # STAGE 1 — Throat line: smooth Sauer curve, or flashing flat front
    # ------------------------------------------------------------------
    def run_stage_1_sauer(self, progress_callback=None):
        if self.flashing:
            self._stage1_flashing_front(progress_callback=progress_callback)
        else:
            self._stage1_sauer_smooth(progress_callback=progress_callback)

        # Stage 2 (marching variant) reassigns self.Sauer, so the true corner
        # position must be captured here while self.Sauer is still pristine.
        self.throat_corner_x = float(self.Sauer[-1].x)
        self.axis_points_vec.append(self.Sauer[0])

    def _stage1_sauer_smooth(self, progress_callback=None):
        """Real Sauer transonic sonic line (identical to moc/)."""
        print('\n\t=== Stage 1: Sauer sonic line ===\n')
        y_vals = np.linspace(0, self.y_t, self.n)
        for i, y_val in enumerate(y_vals):
            p = InternalPoint(self.fluid_manager)
            p.y_t = self.y_t; p.rho_t = self.rho_t; p.delta = self.delta_flow
            p.compute_sonic_state()
            p.y = y_val
            p.SauerMain()
            self.Sauer.append(p)
            self.sauer_main.append(p)
            printProgress(i+1, self.n, '1 of 4', on_progress=progress_callback)

    def _stage1_flashing_front(self, progress_callback=None):
        """
        Flashing throat initialization (ported from moc_tool_flashing).

        The Sauer curvature formula assumes a continuous sound-speed field near
        the sonic point, which does not hold across the discontinuous jump in a
        when a subcooled liquid inlet crosses the saturation line. Instead, the
        whole throat plane (x=0) is set to the isentropic 'jumped' velocity
        corresponding to the flashing critical state found by
        jax_solve_critical_state's FLASHING_JUMP fallback.
        """
        print('\n\t=== Stage 1: Flashing throat front ===\n')
        v_jump = float(jnp.sqrt(2 * (self.fluid_manager.h0 - self.fluid_manager.critical_state["h"])))

        y_vals = np.linspace(0, self.y_t, self.n)
        for i, y_val in enumerate(y_vals):
            p = InternalPoint(self.fluid_manager)
            p.x = 0.0
            p.y = y_val
            p.u = v_jump
            p.v = 0.0
            p.ThermoFlowProp()
            self.Sauer.append(p)
            self.sauer_main.append(p)
            printProgress(i+1, self.n, '1 of 4', on_progress=progress_callback)

    # ------------------------------------------------------------------
    # STAGE 2 — MLN: use Sauer line directly as IVP (no marching needed)
    # ------------------------------------------------------------------
    def run_stage_2_sauer_direct(self, progress_callback=None):
        """
        For the MLN, the expansion fan originates entirely from the sharp
        throat corner: within a centred simple wave, the corner's own
        (theta, nu) at any fan angle follows directly from that angle alone
        (see InternalPoint.corner_point_prop), with no dependence on the
        rest of the throat line -- PROVIDED the corner's own reference state
        has theta=0 exactly (the nu(V) table's zero point). This holds
        exactly for the flashing front (_stage1_flashing_front sets v=0,
        i.e. theta=0, at every point, including the corner), but NOT for
        the smooth Sauer sonic line (_stage1_sauer_smooth): the transonic
        curvature formula (SauerMain) gives a small but genuinely nonzero
        flow angle even at the wall point, so collapsing to a single point
        would silently drop that curvature. Only the flashing case is
        therefore simplified to a single-point IVP; the smooth case keeps
        the full multi-point front as before.

        For the flashing case, dropping the other (redundant -- all
        identical state, just spread over y) front points also makes the
        first fan angle collapse exactly to the reference algorithm's ray 1
        (corner -> axis directly, zero interior points, see Zebbiche &
        Youbi 2007 Fig. 3a) -- previously ray 1 was seeded from the full
        multi-point front, adding "interior" points that carried no
        independent new information.
        """
        if self.flashing:
            print('\n=== Stage 2 MLN (flashing): corner point only (no marching) ===\n')
            self.IVP = [self.Sauer[-1]]   # single corner point (y = y_t)
            print('  IVP line set to the throat corner point only.')
        else:
            print('\n=== Stage 2 MLN: using Sauer line as IVP (no marching) ===\n')
            self.IVP = list(reversed(self.Sauer))   # IVP[0] = Sauer wall pt (y = y_t)
            print(f'  IVP line set to Sauer line: {len(self.IVP)} points.')
        if progress_callback is not None:
            progress_callback(1.0, '2 of 4')  # instantaneous, no marching loop to report mid-progress on

    # ------------------------------------------------------------------
    # STAGE 2 (flashing variant) — IVP marching, ported from moc_tool_flashing
    # ------------------------------------------------------------------
    def run_stage_2_ivp_march(self, progress_callback=None):
        """
        The flashing throat front is flat (x=0 for every y), so it carries no
        geometric spread in x. Stage 3's Inv_Wall_pnt needs two IVP points with
        different x to define the line 3-1 slope — feeding the flat front to it
        directly divides by zero. Marching the front forward via interior-point
        compatibility (mixing C+/C- characteristics, always at different slopes
        since alpha != 0) builds a proper x-spread characteristics mesh before
        Stage 3 runs, exactly as moc_tool_flashing does for the arc-wall kernel.

        NOTE: mutates self.Sauer in place; run_stage_1_sauer already cached the
        true throat-corner x in self.throat_corner_x before this runs.
        """
        print('\n=== Stage 2 MLN (flashing): IVP marching ===\n')
        self.IVP.append(self.Sauer[-1])
        num_iterations = self.n * 2 - 2

        for j in range(num_iterations):
            printProgress(j, num_iterations - 1, '2 of 4', on_progress=progress_callback)
            k = len(self.Sauer)
            Sauer_old = self.Sauer
            Sauer_new = []

            for i in range(k - 1):
                self.p_calc.compute_from_characteristics(Sauer_old[i], Sauer_old[i + 1])
                p = InternalPoint(self.fluid_manager)
                p.__dict__.update(self.p_calc.__dict__)

                self.mesh_data["stage2_c_plus"].append(([float(Sauer_old[i].x), float(p.x)], [float(Sauer_old[i].y), float(p.y)]))
                self.mesh_data["stage2_c_minus"].append(([float(Sauer_old[i + 1].x), float(p.x)], [float(Sauer_old[i + 1].y), float(p.y)]))

                if self.show_plot:
                    self.ax1.plot(p.x, p.y, "o", color="black", markersize=0.5)
                    self.ax1.plot((Sauer_old[i].x, p.x), (Sauer_old[i].y, p.y), "-", color=colors[3], linewidth=1.0)
                    self.ax1.plot((Sauer_old[i + 1].x, p.x), (Sauer_old[i + 1].y, p.y), "-", color=colors[2], linewidth=1.0)

                Sauer_new.append(p)

            # Keep the IVP on one half via periodic axis reflection
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

            self.mesh_data["stage2_fronts"].append(([float(p.x) for p in Sauer_new], [float(p.y) for p in Sauer_new]))

        print(f'  IVP line built via marching: {len(self.IVP)} points.')

    # ------------------------------------------------------------------
    # STAGE 3 — MLN kernel: centred expansion fan from throat corner
    # ------------------------------------------------------------------
    def run_stage_3_kernel_mln(self, progress_callback=None):
        """
        Minimum-length nozzle kernel.

        Identical logic to the standard Stage 3 kernel EXCEPT that the
        wall position is fixed at the throat corner (x_corner, y_t) for
        every fan angle, instead of moving along a circular arc.

        For flashing cases, the corner point at each fan angle is computed
        DIRECTLY via InternalPoint.corner_point_prop (no iterative wall
        search), and self.IVP is seeded (in run_stage_2_sauer_direct) with
        just the single corner point rather than the full flat front --
        together these make ray 1 collapse exactly to corner -> axis
        directly (zero interior points), matching Zebbiche & Youbi (2007)
        Fig. 3a's reference algorithm.

        The corner position is taken from IVP[0] after Stage 2 (the
        wall-side point on the IVP line, which is already supersonic —
        avoiding the M=1 singularity that caused NaN in the previous
        implementation).

        All C+ characteristics thus originate from this single corner
        point. The flow angle at the corner grows step-by-step through
        the wall compatibility equation (Inv_Wall_pnt), which is the
        standard procedure for any wall boundary.

        FIXED TARGET, NOT A SEARCH. self.tau_mln is expected to already span
        [0, theta*] where theta* = 0.5*nu(M_f) is the classical SSL-MLN
        corner turn angle (Argrow & Emanuel 1991, eq. for the straight
        sonic line MLN), computed from the same real-gas Prandtl-Meyer
        integral used in NICFD_2026/python_codes/04_moc_flashing.py, referenced
        from the post-jump state (nu=0 there) rather than the classical M=1
        point -- exactly the origin of this corner's fan. Sweeping a KNOWN,
        bounded angle range (rather than an open-ended Mach/pressure-crossing
        search) avoids accumulating turning far past where the design
        actually ends, which is what previously drove the axis-point solve
        into the near-M=1 singularity and the fluid table's valid range.
        """
        tau = self.tau_mln[1:]
        print('\n=== Stage 3 MLN: centred expansion fan (fixed sweep to theta*) ===\n')

        # Fixed corner: the original Stage-1 wall point, cached in
        # run_stage_1_sauer as self.throat_corner_x (self.Sauer is mutated by
        # the flashing marching variant of Stage 2, so it can't be read here).
        x_corner = self.throat_corner_x
        y_corner = float(self.y_t)

        # Store the corner once as the start of the kernel wall contour
        self.nozzle_wall.append(self.IVP[0])
        self.nozzle_wall_kernel.append(self.IVP[0])

        # The flat flashing front (x=0 for every y) has no x-spread, so the
        # standard Inv_Wall_pnt (which interpolates along the "line 3-1" by
        # x) is degenerate there. For flashing, the corner point is instead
        # computed DIRECTLY (InternalPoint.corner_point_prop): within a
        # centred simple wave with theta=0 at the throat/jump reference
        # (exactly where flashing's flat front sits), theta=nu(V) holds at
        # the corner for any prescribed fan angle, with no dependence on
        # neighbouring front data -- so no shooting search is needed at all.
        # Non-flashing (curved Sauer) fronts keep the original iterative
        # wall unit process, since the real Sauer curvature means the
        # corner's own flow angle is not exactly zero there.

        c = 0
        for ang in tau:
            c += 1
            ang_rad = float(np.radians(ang))
            IVP_New = []

            try:
                if self.flashing:
                    # 1. Corner point: direct, no search (see docstring above).
                    p_wall_store = InternalPoint(self.fluid_manager)
                    p_wall_store.corner_point_prop(x_corner, y_corner, ang_rad)
                    IVP_New.append(p_wall_store)

                    self.pt3_calc.__dict__.update(p_wall_store.__dict__)

                    # 2. Interior marching against the rest of the current front.
                    p_result_store = p_wall_store  # default when there are no interior points (ray 1)
                    for j in range(0, len(self.IVP) - 1):
                        if j == 0:
                            self.pt3_calc.__dict__.update(p_wall_store.__dict__)

                        self.pt2_calc.__dict__.update(self.IVP[j + 1].__dict__)
                        self.point_calc.compute_from_characteristics(self.pt2_calc, self.pt3_calc)

                        p_result_store = InternalPoint(self.fluid_manager)
                        p_result_store.__dict__.update(self.point_calc.__dict__)
                        IVP_New.append(p_result_store)
                        self.pt3_calc.__dict__.update(p_result_store.__dict__)

                        # Both characteristic feet contributed to this interior point
                        # (paramP=pt2_calc, the C+ foot; paramM=pt3_calc, the C- foot --
                        # for j=0 this is the corner itself). Only the pt2_calc segment
                        # used to be recorded, which left the corner (and every other
                        # C- connection) visually disconnected from the mesh, even
                        # though it was correctly used in the actual compatibility
                        # solve -- drawing both makes the true two-family mesh visible.
                        self.mesh_data["stage3_mesh"].append(([float(self.pt2_calc.x), float(self.point_calc.x)], [float(self.pt2_calc.y), float(self.point_calc.y)]))
                        self.mesh_data["stage3_mesh"].append(([float(self.pt3_calc.x), float(self.point_calc.x)], [float(self.pt3_calc.y), float(self.point_calc.y)]))

                        if self.show_plot:
                            self.ax1.plot(
                                [self.pt2_calc.x, self.point_calc.x],
                                [self.pt2_calc.y, self.point_calc.y],
                                '-', color='#1f77b4', linewidth=0.6, zorder=2
                            )
                            self.ax1.plot(
                                [self.pt3_calc.x, self.point_calc.x],
                                [self.pt3_calc.y, self.point_calc.y],
                                '-', color='#1f77b4', linewidth=0.6, zorder=2
                            )
                else:
                    # 1. Setup foot points from current IVP line
                    self.pt3_calc.__dict__.update(self.IVP[0].__dict__)
                    self.pt1_calc.__dict__.update(self.IVP[1].__dict__)

                    # MLN: wall is FIXED at corner (not moving along an arc)
                    self.wall_calc.x = x_corner
                    self.wall_calc.y = y_corner

                    # 2. Wall-point intersection loop (same as standard kernel)
                    flag = 0; k = 0
                    while flag != 1:
                        try:
                            intermediate_pt_state = self.wall_calc.Inv_Wall_pnt(
                                self.wall_calc, self.pt3_calc, self.pt1_calc, ang)
                            self.pt_result.__dict__.update(intermediate_pt_state.__dict__)
                            flag = 1
                        except NameError:
                            k += 1
                            self.pt3_calc.__dict__.update(self.IVP[0 + k].__dict__)
                            self.pt1_calc.__dict__.update(self.IVP[1 + k].__dict__)

                    p_wall_store = InternalPoint(self.fluid_manager)
                    p_wall_store.__dict__.update(self.wall_calc.__dict__)
                    IVP_New.append(p_wall_store)

                    self.pt3_calc.__dict__.update(self.pt_result.__dict__)

                    # 3. Interior marching (identical to standard kernel)
                    p_result_store = p_wall_store  # default when there are no interior points
                    for j in range(0, int(len(self.IVP) - 1) - k):
                        if j == 0:
                            self.pt3_calc.__dict__.update(p_wall_store.__dict__)

                        self.pt2_calc.__dict__.update(self.IVP[j + 1 + k].__dict__)
                        self.point_calc.compute_from_characteristics(self.pt2_calc, self.pt3_calc)

                        p_result_store = InternalPoint(self.fluid_manager)
                        p_result_store.__dict__.update(self.point_calc.__dict__)
                        IVP_New.append(p_result_store)
                        self.pt3_calc.__dict__.update(p_result_store.__dict__)

                        # Both characteristic feet contributed to this interior point
                        # (paramP=pt2_calc, the C+ foot; paramM=pt3_calc, the C- foot --
                        # for j=0 this is the corner itself). Only the pt2_calc segment
                        # used to be recorded, which left the corner (and every other
                        # C- connection) visually disconnected from the mesh, even
                        # though it was correctly used in the actual compatibility
                        # solve -- drawing both makes the true two-family mesh visible.
                        self.mesh_data["stage3_mesh"].append(([float(self.pt2_calc.x), float(self.point_calc.x)], [float(self.pt2_calc.y), float(self.point_calc.y)]))
                        self.mesh_data["stage3_mesh"].append(([float(self.pt3_calc.x), float(self.point_calc.x)], [float(self.pt3_calc.y), float(self.point_calc.y)]))

                        if self.show_plot:
                            self.ax1.plot(
                                [self.pt2_calc.x, self.point_calc.x],
                                [self.pt2_calc.y, self.point_calc.y],
                                '-', color='#1f77b4', linewidth=0.6, zorder=2
                            )
                            self.ax1.plot(
                                [self.pt3_calc.x, self.point_calc.x],
                                [self.pt3_calc.y, self.point_calc.y],
                                '-', color='#1f77b4', linewidth=0.6, zorder=2
                            )

                # 4. Axis point via image-reflection (identical to standard kernel)
                p_result_store_last = p_result_store
                self.pt2_calc.__dict__.update(p_result_store_last.__dict__)
                self.pt3_calc.__dict__.update(p_result_store_last.__dict__)
                self.pt3_calc.v = -self.pt3_calc.v
                self.pt3_calc.y = -self.pt3_calc.y

                self.point_calc.compute_from_characteristics(self.pt3_calc, self.pt2_calc)

                p_result_final = InternalPoint(self.fluid_manager)
                p_result_final.__dict__.update(self.point_calc.__dict__)
                IVP_New.append(p_result_final)
            except Exception as e:
                # Near-sonic starts (post-jump Mach only slightly above 1)
                # make the axis-point compatibility relation (Q = u^2-a^2 -> 0)
                # ill-conditioned; large accumulated turning can eventually
                # drive a point outside the fluid table's valid range. Stop
                # gracefully at the last converged front instead of crashing
                # the whole batch -- self.IVP still holds that last-good state.
                print(f'\n\t***** WARNING: Stage 3 stopped at angle {ang:.2f} deg '
                      f'due to a property-solve failure ({e!s:.120}) — '
                      f'design Mach not reached, using last valid state *****\n')
                break

            self.IVP = IVP_New
            self.axis_points_vec.append(p_result_final)

            self.mesh_data["stage3_fronts"].append(([float(p.x) for p in IVP_New], [float(p.y) for p in IVP_New]))

            if self.show_plot:
                self.ax1.plot(
                    [p.x for p in IVP_New], [p.y for p in IVP_New],
                    '-', color='#1f77b4', linewidth=0.6, zorder=2
                )

            # Progress (fixed sweep -- no early termination search)
            t2 = time.time()
            if c % 10 == 0:
                print('\n\tAngle\t\tMa_axis\t\tp_axis[Pa]\tx\t\tTime')
            print(f'\t{ang:.2f}\t\t{IVP_New[-1].M:.4f}\t\t{IVP_New[-1].p:.1f}'
                  f'\t\t{IVP_New[-1].x:.6f}\t\t{t2-self.t0:.1f}s')
            if progress_callback is not None:
                # Fixed sweep, no early-termination search -- angle fraction
                # swept is already an exact progress fraction, unlike the
                # metric-gap-closed proxy the conventional solver's adaptive
                # Stage 3 needs.
                progress_callback(c / len(tau), '3 of 4')

        else:
            # Loop completed without a property-solve failure: reached theta*
            print(f'\n\t***** MLN Kernel Done: swept full range to theta* = '
                  f'{tau[-1]:.2f} deg *****\n')

        final = self.axis_points_vec[-1] if self.axis_points_vec else self.IVP[-1]
        print(f'\tFinal state: M={final.M:.4f} (design M_f={self.Noz_Mach:.4f}), '
              f'p={final.p:.1f} Pa'
              + (f' (target p_out={self.P_back:.1f} Pa)' if self.P_back is not None else ''))

    # ------------------------------------------------------------------
    # STAGE 4 — Reflex zone (identical to moc/)
    # ------------------------------------------------------------------
    def _refine_ivp_near_corner(self, n_sub=8, n_segments=3):
        """
        Insert n_sub interpolated points into each of the first n_segments
        segments of self.IVP (the final kernel front), for Stage 4's wall
        construction specifically.

        This is the missing piece kernel-angle grid compression does NOT
        provide: run_stage_4_reflex produces exactly one wall point per
        segment of self.IVP via a single mass-flow integral, regardless of
        how many kernel rays contributed to that segment -- so the first
        wall point (corner -> IVP[1]) is always one big mass-flow "jump",
        no matter how finely the kernel angles are resolved (verified:
        Nj=5 compression only moved that first point by ~2%). This mirrors
        Argrow & Emanuel (1988)'s separate transition-region aspect-ratio
        control (AR = S+/Save, Fig. 4-5), which subdivides the wall-marching
        steps in the transition region independently of the kernel grid.

        Interpolation is linear in (x, y, u, v) between the two endpoints,
        consistent with the paper's own transition-region formulation
        (eqs. 3a-3d explicitly treat the connecting characteristic and wall
        segments as straight lines between consecutive points).
        """
        refined = [self.IVP[0]]
        for seg_idx in range(len(self.IVP) - 1):
            p0, p1 = self.IVP[seg_idx], self.IVP[seg_idx + 1]
            n = n_sub if seg_idx < n_segments else 1
            for k in range(1, n + 1):
                frac = k / n
                p = InternalPoint(self.fluid_manager)
                p.x = float(p0.x) + frac * (float(p1.x) - float(p0.x))
                p.y = float(p0.y) + frac * (float(p1.y) - float(p0.y))
                p.u = float(p0.u) + frac * (float(p1.u) - float(p0.u))
                p.v = float(p0.v) + frac * (float(p1.v) - float(p0.v))
                p.ThermoFlowProp()
                refined.append(p)
        self.IVP = refined

    def run_stage_4_reflex(self, refine_near_corner=False, n_sub=8, n_segments=3, progress_callback=None):
        print('\n=== Stage 4: Reflex zone ===\n')
        # refine_near_corner defaults to False: measured to leave the wall
        # SHAPE unchanged in that region (segment angles drift <1 deg across
        # the whole refined stretch -- it's still effectively the same
        # straight line at theta*, since the sub-points are linearly
        # interpolated between the corner and the first real kernel-front
        # point). Its only effect was a marginal accuracy improvement to
        # the mass-flow integral for points further downstream, not
        # visible in the wall contour -- not worth the added complexity.
        if self.flashing and refine_near_corner:
            self._refine_ivp_near_corner(n_sub=n_sub, n_segments=n_segments)
        for p in self.IVP:
            p.ThermoFlowProp()

        # MassIn is the cumulative mass-flow integral over self.IVP[0:i+1].
        # Accumulated incrementally (O(1)/step, O(n) total) instead of
        # recomputing MassFlowCalc over the whole growing prefix from
        # scratch every step (O(n^2) total) -- same fix as
        # moc.solvers.conventional.MOCSolver.run_stage_4_reflex, ported here
        # since MLN's kernel front can carry more points (finer angle
        # sweeps) than the conventional solver's, making the O(n^2) pattern
        # more costly here, not less.
        mass_cumulative = 0.0
        for i in range(1, len(self.IVP)):
            p_prev, p_cur = self.IVP[i - 1], self.IVP[i]
            seg_len = DistCalc(p_prev.x, p_prev.y, p_cur.x, p_cur.y)
            rot_vec = RotVec(MakeVec(p_prev.x, p_prev.y, p_cur.x, p_cur.y), -90)
            rho_avg = (p_prev.rho + p_cur.rho) / 2
            u_avg = (p_prev.u + p_cur.u) / 2
            v_avg = (p_prev.v + p_cur.v) / 2
            V_avg = vecDotVec(u_avg, v_avg, rot_vec[0], rot_vec[1])
            mass_cumulative = mass_cumulative + (rho_avg * seg_len * V_avg)

            Ref_P = (self.IVP[i],)
            MassIn = mass_cumulative
            MassOut = MassFlowCalc(Ref_P)
            l = MassIn / MassOut
            self.length.append(l)

            p_in = InternalPoint(self.fluid_manager)
            p_in.__dict__.update(Ref_P[0].__dict__)
            LocWall = p_in.ReflexLine(1, l)

            self.nozzle_wall.append(LocWall)
            self.nozzle_wall_divergent.append(LocWall)

            if self.show_plot:
                self.ax1.plot((Ref_P[0].x, LocWall.x), (Ref_P[0].y, LocWall.y),
                              '-', color='lightgray', linewidth=0.6, zorder=1)

            printProgress(i, len(self.IVP)-1, '4 of 4', on_progress=progress_callback)

        self.nozzle_wall_final = self.nozzle_wall

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_results(self):
        if not self.show_plot:
            return

        wx = [p.x for p in self.nozzle_wall_final]
        wy = [p.y for p in self.nozzle_wall_final]

        self.ax1.plot(wx, wy, '-', color='black', linewidth=2, label='MLN wall')
        self.ax1.plot([wx[0], wx[-1]], [0, 0], '--', color='black', linewidth=1)
        if not self.flashing:
            # For flashing, the fan now departs directly from the corner
            # (see run_stage_2_sauer_direct / run_stage_3_kernel_mln), so
            # the flat throat front carries no independent information and
            # isn't drawn. The smooth (non-flashing) Sauer line remains a
            # real physical curve used by Stage 3, so it's still shown.
            self.ax1.plot([p.x for p in self.sauer_main],
                          [p.y for p in self.sauer_main],
                          '-o', color=colors[0], linewidth=1.5, label='Sonic line')
        self.ax1.set_xlabel('x (m)')
        self.ax1.set_ylabel('y (m)')
        self.ax1.set_title('Minimum-Length Nozzle — MoC')
        self.ax1.legend()
        self.ax1.set_aspect('equal')

        # Axis properties
        if self.axis_points_vec:
            fig2, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
            xax = [p.x for p in self.axis_points_vec]
            axes[0].plot(xax, [p.p   for p in self.axis_points_vec], color=colors[2])
            axes[0].set_ylabel('Pressure (Pa)')
            axes[1].plot(xax, [p.M   for p in self.axis_points_vec], color=colors[4])
            axes[1].set_ylabel('Mach number')
            axes[2].plot(xax, [p.rho for p in self.axis_points_vec], color=colors[6])
            axes[2].set_ylabel('Density (kg/m³)')
            axes[2].set_xlabel('x (m)')
            plt.tight_layout()

        total = time.time() - self.t0
        print(f'\nMLN simulation completed in {total:.2f} s')
        plt.show()

    # ------------------------------------------------------------------
    # DATA EXPORT — full mesh + wall + axis data, so any figure can be
    # reproduced later without re-running the solve. Same schema as
    # ConventionalSolver's export_full_result_data, for interchangeability.
    # ------------------------------------------------------------------
    def build_result_data(self, extra_meta=None):
        """Returns the full result dict without touching the filesystem --
        see moc.solvers.conventional.MOCSolver.build_result_data (same
        schema, reused as-is by moc.api)."""
        def to_l(pts, attr): return [float(getattr(p, attr)) for p in pts]

        # Wall points carry u,v (set by ReflexLine) but ThermoFlowProp() was
        # never called on them, so p/M/T/rho/s were never populated -- do
        # that now so wall property distributions can be exported too.
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
