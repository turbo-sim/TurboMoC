import jaxprop as jxp
from jax import jit
from jax import numpy as jnp
import pdb
import numpy as np
import jax
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


# =============================================================================
# 1. JAX-COMPATIBLE HELPER FUNCTIONS (Outside Classes)
# ============================================================================= 

@jax.jit
def jitted_calculate_static_properties(fluid, V: jnp.ndarray, s0: float, h0: float):
    # Calculate Raw Enthalpy
    h_static_raw = h0 - jnp.square(V) / 2.0

    # SAFETY MASK: only guard against actual NaN (padding, etc). Do NOT treat
    # h_static_raw <= 0 as invalid -- CoolProp's reference state makes h
    # negative for many fluids (e.g. nitrogen: h0 ~ -71000 J/kg here), so that
    # check used to fire on every real point and force a=0 everywhere.
    is_invalid = jnp.isnan(h_static_raw)
    h_safe = jnp.where(is_invalid, h0, h_static_raw)
    
    # Get state from JAXProp
    state = fluid.get_state(jxp.HmassSmass_INPUTS, h_safe, s0)
    
    # Calculate Gamma
    gamma = state["cp"] / state["cv"]
    
    # MASK THE OUTPUT: If it was an invalid point, return 0.0 for speed of sound, etc.
    a_out = jnp.where(is_invalid, 0.0, state["a"])
    
    return (h_safe, state["T"], state["p"], state["rho"], gamma, a_out, s0)

def jax_solve_critical_state(fluid, h0, s0, mach_tol=0.05):
    """
    Critical-state solver with a flashing fallback.

    1. Tries to find the smooth sonic point (Ma=1) via bracketed root-finding.
    2. If no smooth root exists — or the "root" brentq returns actually sits on
       a discontinuous jump in sound speed rather than a true zero-crossing,
       which happens when a subcooled-liquid isentrope crosses the saturation
       line: a(P) jumps from the high single-phase liquid branch to the low
       collapsed two-phase branch over an arbitrarily small pressure step, so
       bisection can satisfy its P-tolerance while landing on either side of
       the jump with a residual nowhere near zero — falls back to locating the
       saturation pressure and nudging slightly into the two-phase dome to
       pick up the collapsed (HEM) speed of sound. Mirrors the approach in
       moc_tool_flashing/srcMOC/moc_classes.py, with an added post-hoc
       validation of the smooth root (that file trusts brentq unconditionally,
       which silently returns a bogus state when the isentrope crosses a jump
       like this one).
    """

    def residual_sonic_point_p(P):
        st = fluid.get_state(jxp.PSmass_INPUTS, float(P), float(s0))
        return h0 - st["h"] - 0.5 * (st["a"] ** 2)

    stagnation_state = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s0)
    P0_guess = stagnation_state["p"]

    lower = 0.05 * P0_guess
    upper = 0.999 * P0_guess

    def flashing_fallback():
        """
        Locate the saturation crossing by grid-scanning P downward from
        stagnation. A direct root-find on Q itself (as in the original
        moc_tool_flashing implementation) is unreliable on backends that
        report Q=0.0 (not NaN) across the *entire* single-phase subcooled
        liquid branch, not only at the saturation boundary -- there is then
        no clean sign change to bracket over the full [lower, upper] range,
        so brentq can converge anywhere on the flat Q=0 plateau.
        """
        def quality(P):
            q = float(fluid.get_state(jxp.PSmass_INPUTS, float(P), float(s0))["Q"])
            return 0.0 if np.isnan(q) else q

        P_grid = np.linspace(upper, lower, 400)
        P_liquid_side, P_two_phase_side = None, None
        for P in P_grid:
            if quality(P) > 1e-4:
                P_two_phase_side = P
                break
            P_liquid_side = P

        if P_two_phase_side is None:
            state = fluid.get_state(jxp.PSmass_INPUTS, lower, s0)
            return state, "LIMIT_FALLBACK"

        if P_liquid_side is None:
            P_sat = P_two_phase_side
        else:
            try:
                P_sat = brentq(lambda P: quality(P) - 1e-4,
                                P_two_phase_side, P_liquid_side, xtol=1e-3)
            except ValueError:
                P_sat = P_two_phase_side

        P_jump = P_sat * 0.995
        state = fluid.get_state(jxp.PSmass_INPUTS, P_jump, s0)
        return state, "FLASHING_JUMP"

    try:
        # --- PATH 1: SMOOTH ROOT ---
        P_critical = brentq(residual_sonic_point_p, lower, upper, xtol=1e-6)
        state = fluid.get_state(jxp.PSmass_INPUTS, P_critical, s0)

        # Validate: a true sonic point has V == a (Ma == 1) at convergence.
        # brentq only guarantees P-tolerance, so a bracket straddling a
        # discontinuity (rather than a smooth zero-crossing) can "converge"
        # while the residual there is still far from zero.
        v_check = np.sqrt(2 * max(0.0, h0 - state["h"]))
        ma_check = v_check / state["a"]
        if abs(ma_check - 1.0) <= mach_tol:
            mode = "SMOOTH_SONIC"
        else:
            state, mode = flashing_fallback()

    except ValueError:
        # --- PATH 2: FLASHING FALLBACK (no sign change at all) ---
        state, mode = flashing_fallback()

    v_final = np.sqrt(2 * max(0.0, h0 - state["h"]))
    ma_final = v_final / state["a"]
    print(f"CRIT SOLVE | Mode: {mode} | P: {state['p']:.1f} | Mach: {ma_final:.4f} | a: {state['a']:.2f}")

    return state


# @jit
def jitted_corrector_algebra(
    # ... (Implementation remains the same) ...
    xM, yM, uM, vM, aM, thetaM, 
    xP, yP, uP, vP, aP, thetaP,
    u_P_ref, v_P_ref, u_M_ref, v_M_ref,
    delta
):
    # --- (Function implementation remains exactly as provided) ---
    VM = jnp.sqrt(jnp.square(uM) + jnp.square(vM)); VP = jnp.sqrt(jnp.square(uP) + jnp.square(vP))
    alphaM = jnp.arcsin(aM / VM); lamM = jnp.tan(thetaM - alphaM); QM = jnp.square(uM) - jnp.square(aM)
    R1M = 2.0 * uM * vM - QM * lamM; SM = jnp.where(jnp.abs(yM) <= 1e-9, 0.0, delta * (jnp.square(aM) * vM) / yM)
    alphaP = jnp.arcsin(aP / VP); lamP = jnp.tan(thetaP + alphaP); QP = jnp.square(uP) - jnp.square(aP)
    R1P = 2.0 * uP * vP - QP * lamP; SP = jnp.where(jnp.abs(yP) <= 1e-9, 0.0, delta * (jnp.square(aP) * vP) / yP)

    denom_geom = lamM - lamP 
    x_int = (lamM * xM - lamP * xP + yP - yM) / denom_geom; y_int = yM + lamM * (x_int - xM)

    TP = SP * (x_int - xP) + (QP * u_P_ref) + (R1P * v_P_ref)
    TM = SM * (x_int - xM) + (QM * u_M_ref) + (R1M * v_M_ref)

    num_v = (TP * QM) - (TM * QP); denom_v = (QM * R1P) - (QP * R1M)
    
    is_denom_sing = jnp.abs(denom_v) <= 1e-12
    v_full = jnp.where(is_denom_sing, 0.0, num_v / denom_v)
    is_M_sing = jnp.abs(QM) <= 1e-9
    u_full = jnp.where(is_M_sing, u_P_ref, (TM - (R1M * v_full)) / QM)
    
    v_final = jnp.where(is_denom_sing | is_M_sing, 0.0, v_full)
    u_final = jnp.where(is_M_sing, u_P_ref, u_full)
    
    return x_int, y_int, u_final, v_final

# @jit
def jitted_moc_predictor_algebra(
    xM, yM, lamM, SM, QM, R1M, uM, vM,
    xP, yP, lamP, SP, QP, R1P, uP, vP
):
    """
    JIT-compiled function to solve MOC geometry and linear compatibility system 
    (Predictor Step).
    
    Parameters are properties of the C- foot point (M) and the C+ foot point (P).
    """
    
    # --- A. Geometry (Intersection) ---
    denom_geom = lamM - lamP
    
    # Calculate intersection point (x, y)
    # y - yP = lamP * (x - xP)  (C+ line)
    # y - yM = lamM * (x - xM)  (C- line)
    
    x = (lamM * xM - lamP * xP + yP - yM) / denom_geom
    y = yM + lamM * (x - xM)

    # --- B. Compatibility Terms (T) ---
    # T = S*(x_new - x_foot) + Q*u_foot + R1*v_foot (Linear approximation)
    TP = SP * (x - xP) + (QP * uP) + (R1P * vP)
    TM = SM * (x - xM) + (QM * uM) + (R1M * vM)

    # --- C. Solve Linear System for u, v (Cramer's Rule) ---
    # The system is:
    # 1) QM * u + R1M * v = TM (C- line)
    # 2) QP * u + R1P * v = TP (C+ line)
    
    num_v = (TP * QM) - (TM * QP)
    denom_v = (QM * R1P) - (QP * R1M)

    # 1. Solve for v and u normally (allowing NaN if denom is zero)
    v_full = num_v / denom_v
    u_full = (TM - (R1M * v_full)) / QM
    
    # 2. Apply robust singularity checks using jnp.where
    
    # Check for near-zero denominator in velocity solution
    is_denom_v_sing = jnp.abs(denom_v) <= 1e-9
    
    # Check for near-zero denominator in u solution (occurs when u^2 = a^2, Q=0)
    is_QM_sing = jnp.abs(QM) <= 1e-9
    is_QP_sing = jnp.abs(QP) <= 1e-9

    # Final v: If compatibility matrix is singular, or QM is singular, set v=0 
    # (a simplification often used for stability in MOC singularities)
    v_final = jnp.where(is_denom_v_sing | is_QM_sing | is_QP_sing, 0.0, v_full)

    # Final u: If QM is singular, use the P-point u as a placeholder, otherwise use u_full
    u_final = jnp.where(is_QM_sing, uP, u_full)

    # If the compatibility matrix was singular, reset u to the average (another stability measure)
    u_final = jnp.where(is_denom_v_sing, (uP + uM) / 2.0, u_final)

    return x, y, u_final, v_final

# =============================================================================
# 4. FLUID MANAGER CLASS (Optimized for Speed)
# ============================================================================= 

class FluidManager:
    """
    Manages the single instance of the jaxprop fluid handler and 
    stores the persistent stagnation properties (s0, h0) and critical state.
    """
    def __init__(self, fluid_name: str, backend: str, P0, T0, Q0):
        # 1. Initialize fluid ONCE (Slow step done only at start)
        # self.method = "pTm_equilibrium"
        self.method = "pT_equilibrium"
        self.P0 = P0; self.T0 = T0; self.Q0 = Q0
        self.fluid = jxp.FluidJAX(fluid_name, backend)
        # Raw (non-JAX) Fluid object, cached once: calculate_static_properties
        # and _build_nu_table both call this directly (bypassing FluidJAX's
        # JIT/pure_callback layer, see their docstrings) on every single
        # point during marching -- re-walking self.fluid.fluid each time
        # pays equinox's Module __getattribute__ overhead thousands of times
        # for no reason, since it never changes after construction.
        self.raw_fluid = self.fluid.fluid
        if np.isnan(self.Q0):
            # Subcooled liquid (flashing) inlet: quality is undefined, use (P0, T0).
            self.state = self.fluid.get_state(jxp.PT_INPUTS, self.P0, self.T0)
        else:
            # Saturated two-phase inlet: use (P0, Q0), matching existing configs.
            self.state = self.fluid.get_state(jxp.PQ_INPUTS, self.P0, self.Q0)


        # self.fluid = jxp.FluidBicubic(fluid_name=fluid_name,
        #                          backend="HEOS",
        #                          h_max=self.state.h*1.2,
        #                          h_min=self.state.h*0.2,
        #                          p_min=0.5e5,
        #                          p_max=3.0e5,
        #                          N_h=50,
        #                          N_p=50
        #                          )
        
        self.s0 = self.state["s"]
        self.h0 = self.state["h"]
        
        self.calculate_stagnation_state()
        
        # 2. Calculate critical state once (smooth sonic root, or flashing fallback)
        self.critical_state = jax_solve_critical_state(self.fluid, self.h0, self.s0)
        self.P_star = self.critical_state["p"]
        self.a_star = self.critical_state["a"]

        # 3. Build a real-gas Prandtl-Meyer table nu(V) once, referenced to the
        # critical/jump state (nu=0 there). This is the EXACT invariant used
        # by the planar (delta=0) MOC compatibility relations
        # theta+nu=const on C-, theta-nu=const on C+ (Zebbiche & Youbi 2007,
        # eqs. 1-2 -- the same real-gas integral already used for theta* in
        # generate_laes_cases.py, tabulated here so every unit process can
        # evaluate/invert it without re-integrating from scratch each time).
        self._build_nu_table()

    def _build_nu_table(self, n_points=10000, p_floor_ratio=1e-5):
        """
        Tabulates nu(V) along the s=s0 isentrope, starting at the critical/
        jump state (V*, nu=0) and descending in pressure toward vacuum
        (increasing V, increasing M). Stops early (truncates the table) at
        the first pressure step the fluid backend cannot resolve, which is
        always well beyond any physical state the solver should encounter.

        n_points=20000 (up from 4000) and cubic-spline interpolation (below,
        replacing np.interp/linear) both reduce the table's own resolution
        error -- the residual imprecision left over once nu_of_V/V_of_nu are
        used self-consistently everywhere (see generate_laes_cases.py's
        theta_star_ssl), since a piecewise-linear table has O(dV^2) error at
        each node regardless of how densely it's sampled at the ends.
        """
        V_star = float(np.sqrt(2 * max(0.0, self.h0 - self.critical_state["h"])))
        p_vec = np.geomspace(self.P_star, self.P_star * p_floor_ratio, n_points)

        # Per-point loop, same structure/semantics as before (natural
        # try/except truncation at the first point the backend can't
        # resolve, typically deep in the near-vacuum tail this table
        # deliberately geomspaces toward) -- but calling the RAW CoolProp
        # wrapper (self.fluid.fluid, jaxprop's non-JAX Fluid class) instead
        # of self.fluid (jxp.FluidJAX), which routes every single-point call
        # through jax.pure_callback + jit. Profiling showed that JIT layer
        # was recompiling on nearly every call (~4200 recompiles for ~10000
        # points, ~43s of pure compilation with zero cache reuse) on top of
        # the real per-point CoolProp cost (~8-10ms/point, not negligible
        # near the two-phase region, confirmed separately) -- batching the
        # whole/partial range in one JAX call was tried and rejected: any
        # bisection/exponential search over batched JAX calls re-evaluates
        # each probed prefix from scratch (no incremental caching underneath
        # pure_callback), so total point-evaluations balloon past what the
        # simple per-point loop ever needed. Going around the JIT layer
        # entirely removes the recompile tax while keeping cost strictly
        # O(valid points), matching the original loop's cost profile minus
        # the JIT overhead that was never buying anything here anyway.
        raw_fluid = self.raw_fluid
        V_list = [V_star]
        a_list = [self.a_star]
        rho_list = [self.critical_state["rho"]]
        for p in p_vec[1:]:
            try:
                st = raw_fluid.get_state(jxp.PSmass_INPUTS, float(p), float(self.s0))
                h, a, rho = float(st["h"]), float(st["a"]), float(st["rho"])
                if not np.isfinite(h) or not np.isfinite(a) or a <= 0:
                    break
                V = float(np.sqrt(2 * max(0.0, self.h0 - h)))
                if V <= V_list[-1]:
                    break
                V_list.append(V)
                a_list.append(a)
                rho_list.append(rho)
            except Exception:
                break

        V_arr = np.array(V_list)
        a_arr = np.array(a_list)
        rho_arr = np.array(rho_list)
        integrand = np.sqrt(np.clip(1.0 / a_arr**2 - 1.0 / V_arr**2, 0.0, None))
        nu_arr = np.zeros_like(V_arr)
        for i in range(1, len(V_arr)):
            nu_arr[i] = nu_arr[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * (V_arr[i] - V_arr[i - 1])

        self._nu_V_table = V_arr
        self._nu_nu_table = nu_arr
        self._nu_a_table = a_arr
        self._nu_rho_table = rho_arr

        # Cubic splines replace linear np.interp for all three lookups, so
        # values *between* table nodes are no longer piecewise-linear
        # approximations. V_arr and nu_arr are both strictly increasing by
        # construction (V_list only appends V > V_list[-1]; integrand >= 0
        # everywhere so nu_arr is non-decreasing) except for possible flat
        # nu segments right at V*, where the integrand can be ~0 -- dedupe
        # those before building the nu -> V inverse spline.
        self._spline_nu_of_V = CubicSpline(V_arr, nu_arr)
        self._spline_a_of_V = CubicSpline(V_arr, a_arr)
        self._spline_rho_of_V = CubicSpline(V_arr, rho_arr)
        nu_unique, idx_unique = np.unique(nu_arr, return_index=True)
        self._spline_V_of_nu = CubicSpline(nu_unique, V_arr[idx_unique])
        self._nu_max = float(nu_arr[-1])

    def nu_of_V(self, V):
        """Real-gas Prandtl-Meyer angle (radians) at velocity V, referenced to the critical/jump state."""
        V = float(np.clip(float(V), self._nu_V_table[0], self._nu_V_table[-1]))
        return float(self._spline_nu_of_V(V))

    def a_of_V(self, V):
        """Speed of sound at velocity V along the same s=s0 isentrope used for nu_of_V (cheap table lookup, no fluid call)."""
        V = float(np.clip(float(V), self._nu_V_table[0], self._nu_V_table[-1]))
        return float(self._spline_a_of_V(V))

    def rho_of_V(self, V):
        """Density at velocity V along the same s=s0 isentrope used for nu_of_V (cheap table lookup, no fluid call)."""
        V = float(np.clip(float(V), self._nu_V_table[0], self._nu_V_table[-1]))
        return float(self._spline_rho_of_V(V))

    def V_of_nu(self, nu_val):
        """Inverse of nu_of_V: velocity at a given real-gas Prandtl-Meyer angle (radians)."""
        nu_val = float(nu_val)
        if nu_val <= 0.0:
            return float(self._nu_V_table[0])
        nu_val = min(nu_val, self._nu_max)
        return float(self._spline_V_of_nu(nu_val))


    def calculate_stagnation_state(self):

        total_state = self.fluid.get_state(jxp.PSmass_INPUTS, self.P0, self.s0)

        self.h0 = total_state["h"]

    # def calculate_static_properties(self, V: float, s0: float, h0: float):
    #     h, T, p, rho, gamma, a, s = jitted_calculate_static_properties(self.fluid, V, s0, h0)
    #     # a = self.compute_mixture_sound_speed(p)
        
    #     return {"h": h, "T": T, "p": p, "rho": rho, "gamma": gamma, "a": a, "s": s}

    def calculate_static_properties(self, V, s0, h0):
        """Calculates static properties for a single scalar velocity V.

        Calls the raw (non-JAX) Fluid object directly instead of routing
        through jitted_calculate_static_properties + FluidJAX.get_state's
        nested JIT/pure_callback layers. This is the sole caller of that
        path and V is always a single scalar here (ThermoFlowProp, called
        once per point throughout Stage 2/3 marching -- never batched), so
        the JIT layer was paying a full XLA trace+compile tax on nearly
        every call for zero reuse benefit (same diagnosis and fix as
        FluidManager._build_nu_table's per-point loop).
        """
        V_f = float(V)
        h0_f, s0_f = float(h0), float(s0)
        h_static_raw = h0_f - V_f**2 / 2.0

        # Guard against actual NaN (e.g. padding) only -- do NOT treat
        # h_static_raw <= 0 as invalid, CoolProp's reference state makes h
        # negative for many fluids (e.g. nitrogen: h0 ~ -71000 J/kg here).
        is_invalid = not np.isfinite(h_static_raw)
        h_safe = h0_f if is_invalid else h_static_raw

        state = self.raw_fluid.get_state(jxp.HmassSmass_INPUTS, h_safe, s0_f)
        gamma = float(state["cp"]) / float(state["cv"])
        a = 0.0 if is_invalid else float(state["a"])

        return {"h": h_safe, "T": float(state["T"]), "p": float(state["p"]),
                "rho": float(state["rho"]), "gamma": gamma, "a": a, "s": s0_f}


_fluid_manager_cache = {}


def get_fluid_manager(fluid_name: str, backend: str, P0, T0, Q0) -> "FluidManager":
    """
    Returns a cached FluidManager for (fluid_name, backend, P0, T0, Q0),
    building it only on first use for that combination.

    FluidManager depends only on the fluid and inlet condition, not on
    nozzle geometry -- building it (notably its Prandtl-Meyer table, see
    _build_nu_table) is the dominant one-time cost per solver instantiation.
    A design/optimization loop that sweeps geometry at a FIXED operating
    point should call this once and pass the result to MOCSolver/
    MOCSolverMLN's fluid_manager= argument, instead of letting each one
    rebuild it from scratch. Changing the fluid or the operating point is a
    different cache key -- it does NOT reuse a stale manager.

    Q0 = float('nan') marks a subcooled-liquid/single-phase inlet (use P0,T0
    instead) -- this is the common case here, so NaN needs explicit
    handling: two separately-constructed float('nan') objects are neither
    `is` nor `==` to each other (NaN != NaN by IEEE 754), so using Q0
    directly as a dict-key component would silently never hit the cache for
    this case (every call would look like a distinct, never-before-seen
    key). Collapsed to a fixed sentinel string instead so all NaN-Q0 calls
    at the same (fluid, P0, T0) correctly share one cached manager.
    """
    q0_f = float(Q0)
    q0_key = "nan" if q0_f != q0_f else q0_f
    key = (fluid_name, backend, float(P0), float(T0), q0_key)
    if key not in _fluid_manager_cache:
        _fluid_manager_cache[key] = FluidManager(fluid_name, backend, P0, T0, Q0)
    return _fluid_manager_cache[key]


class InternalPoint:
    """
    Class representing a single point in a Method of Characteristics (MoC) grid.
    Stores geometry and flow properties.
    Includes a method to compute the sonic condition (Sauer real-gas approach)
    using a 1D root finder and the main Sauer function for the sonic line.
    """

    # -----------------------------
    # Constructor
    # -----------------------------
    def __init__(self, fluid_manager: 'FluidManager', x=None, y=None):        # Fluid setup
        self.manager = fluid_manager

        # Geometry
        self.x = x
        self.y = y

        # Flow properties
        self.M = None
        self.theta = None
        self.nu = None
        self.lamP = None
        self.lamM = None

        self.u = None
        self.v = None
        self.V = None
        self.p = None
        self.T = None
        self.rho = None
        self.a = None
        self.gamma = None
        self.h = None
        self.s = None

        # Stagnation properties (set externally)
        self.P0 = self.manager.P0
        self.T0 = self.manager.T0
        self.Q0 = self.manager.Q0
        self.h0 = self.manager.h0
        self.s0 = self.manager.Q0

        # Additional MoC parameters
        self.y_t = None
        self.rho_t = None
        self.delta = 0.0
        self.GasEqu = "CoolProp"

        # Label for identification
        self.label = "Sauer"

        # Tracks whether h/T/p/rho/a/gamma/s/M/theta/alpha/lamP/lamM are
        # consistent with the CURRENT u,v (set True at the end of
        # ThermoFlowProp, invalidated automatically by __setattr__ below
        # whenever u or v is reassigned). Lets interior_point_prop/
        # solve_centerline_point skip re-deriving already-fresh foot points
        # (see ensure_fresh) instead of unconditionally recomputing them --
        # foot points are frequently reused unchanged from the previous
        # marching step, so this was pure duplicate CoolProp work. Automatic
        # invalidation on any u/v write (rather than trusting call sites to
        # flag staleness themselves) is what makes this safe: e.g.
        # _kernel_step_core's axis-mirror step negates pt3_calc.v in place
        # right before reuse, which correctly re-dirties it and forces a
        # real recompute (theta/lamM depend on v's sign; h/T/p/rho/a/gamma
        # don't, but the point must not be treated as fresh regardless).
        self._thermo_fresh = False

    def __setattr__(self, name, value):
        if name in ("u", "v") and self.__dict__.get("_thermo_fresh", False):
            self.__dict__["_thermo_fresh"] = False
        self.__dict__[name] = value

    def ensure_fresh(self):
        """Recompute thermo/flow properties only if u or v changed since the
        last ThermoFlowProp call -- see _thermo_fresh above."""
        if not self.__dict__.get("_thermo_fresh", False):
            self.ThermoFlowProp()

    # -----------------------------
    # Pretty-print
    # -----------------------------
    def __str__(self):
        return (
            f"Point(x={self.x:.2f}, y={self.y:.2f}, label='{getattr(self, 'label', '')}',\n"
            f"  M={self.M:.2f}, theta={self.theta:.2f}, nu={self.nu}, lamdbaP={self.lamP}, lamdbaM={self.lamM},\n"
            f"  u={self.u:.2f}, v={self.v:.2f}, V={self.V:.2f},\n"
            f"  p={self.p:.2f}, T={self.T:.2f}, rho={self.rho:.2f}, a={self.a:.2f}, gamma={self.gamma:.2f}, h={self.h:.2f}, s={self.s:.2f})"
        )
    
    # -----------------------------
    # Convert to dictionary
    # -----------------------------
    def to_dict(self):
        return self.__dict__.copy()
    
    def copy_from(self, other):
        self.x, self.y = other.x, other.y
        self.u, self.v = other.u, other.v
        self.a, self.theta, self.alpha = other.a, other.theta, other.alpha
        self.M, self.p, self.rho = other.M, other.p, other.rho

    # -----------------------------
    # Thermodynamic properties helper
    # -----------------------------
    def ThermoFlowProp(self):

        self.V = jnp.sqrt(self.u**2 + self.v**2)

        results = self.manager.calculate_static_properties(
            V=self.V, 
            s0=self.manager.s0, 
            h0=self.manager.h0
        )
        
        self.h = results["h"]
        self.T = results["T"]
        self.p = results["p"]
        self.rho = results["rho"]
        self.a = results["a"]
        self.gamma = results["gamma"]
        self.s = results["s"]

        self.M = self.V / self.a
        if hasattr(self.u, "__len__") or self.u != 0:
            self.theta = jnp.atan(self.v / self.u)
        else:
            self.theta = 0.0
        self.alpha = jnp.asin(self.a/self.V)
        self.lamP = jnp.tan(self.theta+self.alpha)
        self.lamM = jnp.tan(self.theta-self.alpha)

        self.__dict__["_thermo_fresh"] = True

    # -----------------------------
    # Compute sonic (critical) state
    # -----------------------------
    def compute_sonic_state(self):
            """
            Calculates the critical (sonic, M=1) state.
            FIX: Reads the pre-calculated critical state properties from the Manager.
            """
            # Read pre-calculated critical values (faster and avoids redundant root_scalar)
            # print(self.h0, self.manager.a_star)
            self.h = self.manager.h0 - 0.5 * self.manager.a_star**2 # h* = h0 - 0.5 * a*^2
            self.a = self.manager.a_star # a*
            self.p = self.manager.P_star # P*
            
            # Re-fetch the full state for other properties (T, rho, gamma) using h* and s0
            state = self.manager.fluid.get_state(jxp.HmassSmass_INPUTS, self.h, self.manager.s0)
            
            self.T = state["T"]
            self.rho = state["rho"]
            self.gamma = state["cp"] / state["cv"]
            self.s = self.manager.s0
            
            # Note: M is undefined (1.0) and V is V* (a*) at this point
            
            return self

    # -----------------------------
    # Sauer main method
    # -----------------------------
    def SauerMain(self):
        
        a_star = self.a * (1 + 1e-3)
        epsilon = - (self.y_t / (2 * (3 + self.delta))) * (
            ((self.gamma + 1) * (1 + self.delta) / (self.rho_t / self.y_t)) ** 0.5
        )
        alpha = ((1 + self.delta) / ((self.gamma + 1) * self.rho_t * self.y_t)) ** 0.5

        # Equation for the sonic line
        self.x = - (self.gamma + 1) * alpha * self.y ** 2 / (2 * (3 + self.delta))
        u_dash = alpha * self.x + (((self.gamma + 1) * alpha ** 2 * self.y ** 2) / (2 * (1 + self.delta)))
        v_dash = (((self.gamma + 1) * alpha ** 2 * self.y * self.x) / (1 + self.delta)) + (
            ((self.gamma + 1) ** 2 * alpha ** 3 * self.y ** 3) / (2 * (1 + self.delta) * (3 + self.delta))
        )
        self.u = a_star * (1 + u_dash)
        self.v = a_star * v_dash

        self.compute_sonic_state()
        self.x = self.x - epsilon
        self.V = jnp.sqrt(self.u ** 2 + self.v ** 2)
        self.M = self.V / self.a

        self.theta = jnp.atan(self.v / self.u)
        self.alpha = jnp.asin(self.a/self.V)
        self.lamP = jnp.tan(self.theta+self.alpha)
        self.lamM = jnp.tan(self.theta-self.alpha)


    # -----------------------------
    # Compute interior point
    # -----------------------------
    def compute_from_characteristics(self, paramP, paramM, EPC=0):
        """
        Computes a new MOC point.

        EPC defaults to 0 (no corrector): interior_point_prop /
        solve_centerline_point now solve the EXACT planar (theta, nu)
        compatibility relations directly (see their docstrings), so there
        is no thermodynamic linearisation error left for a corrector step
        to reduce. euler_predictor_corrector still uses the old linearised
        Q/R1/S algebra and would only reintroduce that error if applied
        here, so it is skipped unless explicitly requested.
        """

        # Check if point is on the AXIS. A genuine axis-reflection pair is
        # constructed by mirroring one foot point to negative y (see the
        # Stage-3 axis call in moc_solver_mln.py: paramM.y = -paramM.y, and
        # Stage-2's periodic wraparound pairs), so it always has OPPOSITE
        # signs with matching magnitude. Checking |y| closeness alone is not
        # enough: two genuinely distinct INTERIOR points (same sign) can
        # coincidentally land within 1e-5 of each other in y once the fan
        # sweep is fine enough, which was observed to falsely trigger the
        # axis solve mid-flow (forcing y=0, theta=0 on an ordinary interior
        # point) once the compatibility solve became accurate enough to
        # produce such close y matches.
        same_side = (float(paramP.y) * float(paramM.y)) > 0
        is_axis_point = (not same_side) and (abs(abs(float(paramP.y)) - abs(float(paramM.y))) < 1e-5)

        # --- 1. PREDICTOR STEP (exact for planar theta/nu compatibility) ---
        if is_axis_point:
            # Calls the dedicated axis solution (v=0 boundary)
            self.solve_centerline_point(paramP, paramM)
        else:
            # Calls the interior 2x2 algebraic solution (JIT optimized predictor)
            self.interior_point_prop(paramP, paramM)

        # --- 2. CORRECTOR STEP (opt-in only, see docstring) ---
        if EPC == 1:
            # Calls the optimized corrector loop (JIT optimized algebra)
            self.euler_predictor_corrector(paramP, paramM)

        return self

    def solve_centerline_point(self, paramP, paramM):
        """
        Calculates the properties of a new point on the Axis of Symmetry (y=0).

        Exact planar (delta=0) compatibility: theta + nu(V) = const along the
        C- characteristic through paramM (Zebbiche & Youbi 2007, eqs. 1, 9,
        27-28). With the axis reflection boundary condition theta=0, this
        gives nu(V_axis) = paramM.theta + nu(paramM.V) directly -- no
        linearised Q/R1/S algebra needed. Geometry (intersection) is
        unchanged: paramP supplies the mirror-image C+ line used to locate
        the axis point.
        """
        paramP.ensure_fresh()
        paramM.ensure_fresh()

        nu_M = self.manager.nu_of_V(paramM.V)
        nu_axis = paramM.theta + nu_M
        V_axis = self.manager.V_of_nu(nu_axis)
        a_axis = self.manager.a_of_V(V_axis)

        # Geometry: slope evaluated at the AVERAGE of paramM's and the new
        # axis point's (theta, mu) -- mirrors interior_point_prop's averaged-
        # slope corrector. Both end states are already known exactly here
        # (theta_axis=0 by the reflection BC, mu_axis from V_axis/a_axis via
        # the exact thermo solve above), so this is a single evaluation, not
        # an iteration. Using only paramM's own (un-averaged) slope was
        # found to introduce small position errors that compound step-over-
        # step across a long fan sweep (diagnosed on the LAES MLN cases:
        # isolated wall/mesh points drifting further off the smooth trend
        # with each subsequent angle step).
        theta_M = float(paramM.theta)
        mu_M = float(jnp.arcsin(jnp.clip(paramM.a / paramM.V, -1.0, 1.0)))
        mu_axis = float(np.arcsin(np.clip(a_axis / V_axis, -1.0, 1.0)))
        theta_avg = 0.5 * theta_M
        mu_avg = 0.5 * (mu_M + mu_axis)
        lamM_avg = np.tan(theta_avg - mu_avg)
        x_axis = float(paramM.x) - float(paramM.y) / lamM_avg

        self.x = x_axis
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.u = V_axis

        self.ThermoFlowProp()

        return self
    
    def interior_point_prop(self, paramP, paramM):
        """
        Exact planar (delta=0) interior-point unit process.

        Zebbiche & Youbi (2007), eqs. 1-2, 9-23: for planar flow the
        characteristic compatibility relations are the EXACT invariants
            theta + nu(V) = const   along C- (paramM's characteristic)
            theta - nu(V) = const   along C+ (paramP's characteristic)
        so theta3 and nu3 (hence V3) at the new point follow directly from
        the two foot points' theta and nu, with no linearisation at all in
        the thermodynamic part. This replaces the general (axisymmetric-
        capable) linearised Q/R1/S solve, which is only approximate for
        this delta=0 problem and was the source of accumulating error in
        earlier attempts. Geometry (the (x, y) intersection) still needs
        the standard linearised treatment, unchanged from before.
        """
        paramP.ensure_fresh()
        paramM.ensure_fresh()

        nu_P = self.manager.nu_of_V(paramP.V)
        nu_M = self.manager.nu_of_V(paramM.V)

        I_minus = paramM.theta + nu_M   # invariant along C- through paramM
        I_plus = paramP.theta - nu_P    # invariant along C+ through paramP

        theta3 = 0.5 * (I_minus + I_plus)
        nu3 = 0.5 * (I_minus - I_plus)
        V3 = self.manager.V_of_nu(nu3)
        a3 = self.manager.a_of_V(V3)

        # Geometry: averaged-slope corrector. A single-pass intersection
        # using only the foot points' own slopes is first-order accurate in
        # the turning angle; over a long fan sweep (dozens of degrees in
        # small steps) this was found to introduce small position errors
        # near locally shallow characteristic angles that then compound
        # step-over-step, since each new front point's position feeds the
        # next as an anchor (diagnosed on the LAES MLN cases: isolated
        # points visibly drifting off the smooth wall/mesh trend, growing
        # worse over dozens of subsequent angle steps). Iterating the slope
        # at the average of each foot point's state and the new point's
        # (now-known-exactly) state removes that drift. Only mu depends on
        # the (fixed) new-point V3/a3; theta3 is already exact, so this
        # converges in 1-2 passes.
        theta_M, mu_M = float(paramM.theta), float(jnp.arcsin(jnp.clip(paramM.a / paramM.V, -1.0, 1.0)))
        theta_P, mu_P = float(paramP.theta), float(jnp.arcsin(jnp.clip(paramP.a / paramP.V, -1.0, 1.0)))
        mu_3 = float(np.arcsin(np.clip(a3 / V3, -1.0, 1.0)))
        theta3_f = float(theta3)

        lamM = float(paramM.lamM)
        lamP = float(paramP.lamP)
        x3 = (lamM * float(paramM.x) - lamP * float(paramP.x) + float(paramP.y) - float(paramM.y)) / (lamM - lamP)
        y3 = float(paramM.y) + lamM * (x3 - float(paramM.x))

        for _ in range(6):
            lamM_avg = np.tan(0.5 * (theta_M + theta3_f) - 0.5 * (mu_M + mu_3))
            lamP_avg = np.tan(0.5 * (theta_P + theta3_f) + 0.5 * (mu_P + mu_3))
            denom = lamM_avg - lamP_avg
            x_new = (lamM_avg * float(paramM.x) - lamP_avg * float(paramP.x) + float(paramP.y) - float(paramM.y)) / denom
            y_new = float(paramM.y) + lamM_avg * (x_new - float(paramM.x))
            if abs(x_new - x3) < 1e-10 and abs(y_new - y3) < 1e-10:
                x3, y3 = x_new, y_new
                break
            x3, y3 = x_new, y_new

        self.x = x3
        self.y = y3

        self.u = V3 * jnp.cos(theta3)
        self.v = V3 * jnp.sin(theta3)

        self.ThermoFlowProp()

        return self

    def corner_point_prop(self, x_corner, y_corner, theta_rad):
        """
        Direct (search-free) sharp-corner unit process for a centred
        Prandtl-Meyer expansion fan.

        Within a centred simple wave emanating from a sharp corner, theta
        and nu are related EXACTLY by theta - theta_ref = nu(V) - nu_ref
        (Zebbiche & Youbi 2007, eqs. 1-2 restricted to a simple wave). Since
        the corner's reference state (the throat/post-jump flow, before any
        turning) has theta_ref=0 and is exactly where this class's nu(V)
        table is zeroed (see FluidManager._build_nu_table), this collapses
        to theta = nu(V) at the corner itself for ANY prescribed turning
        angle theta -- no dependence on neighbouring front data at all.

        This replaces Inv_Wall_pnt / Inv_Wall_pnt_vertical's iterative
        shooting search (which interpolates against bracketing front
        points to locate a wall intersection) for the MLN sharp-corner
        case, where the wall position is fixed and the flow angle at each
        fan angle is prescribed by construction -- there is nothing to
        search for. Used by moc_solver_mln.py's run_stage_3_kernel_mln.
        """
        V = self.manager.V_of_nu(theta_rad)
        self.x = x_corner
        self.y = y_corner
        self.u = V * jnp.cos(theta_rad)
        self.v = V * jnp.sin(theta_rad)
        self.ThermoFlowProp()
        return self

    def euler_predictor_corrector(self, paramP, paramM, max_iter=1000, tol_x=1e-6, tol_v=1e-6):
        """
        Euler Predictor-Corrector optimized with Object Reuse and Thermophysical Properties 
        (a, theta) calculated ONLY ONCE before the iteration loop.
        """

        # --- Initialization (Object Reuse setup remains) ---
        param_old = InternalPoint(self.manager)
        param_old.x, param_old.y, param_old.u, param_old.v = self.x, self.y, self.u, self.v
        param_new = InternalPoint(self.manager)
        param_new.x, param_new.y, param_new.u, param_new.v = self.x, self.y, self.u, self.v
        
        P_avg = InternalPoint(self.manager)
        M_avg = InternalPoint(self.manager)
        delta = self.delta
        
        # --- NEW Step A: Calculate Fixed Averaged Properties (ONCE) ---
        
        # 1. Calculate initial kinematic averages
        P_avg.x = paramP.x; P_avg.y = 0.5 * (paramP.y + param_old.y)
        P_avg.u = 0.5 * (paramP.u + param_old.u); P_avg.v = 0.5 * (paramP.v + param_old.v)

        M_avg.x = paramM.x; M_avg.y = 0.5 * (paramM.y + param_old.y)
        M_avg.u = 0.5 * (paramM.u + param_old.u); M_avg.v = 0.5 * (paramM.v + param_old.v)
        
        # 2. CRITICAL: Calculate ALL Thermodynamic properties (a, theta) ONCE.
        # These properties are now FIXED throughout the iteration.
        P_avg.ThermoFlowProp() 
        M_avg.ThermoFlowProp()

        # ---------------------------------------------------------------------
        # --- Local Helper Functions (Modified - ThermoFlowProp REMOVED) ---
        # ---------------------------------------------------------------------
        
        def _interior_prop(target_point, Pp, Mm, P_ref, M_ref):
            # Pp and Mm are now FIXED and contain 'a' and 'theta'.
            # OLD SLOW CALLS REMOVED: Pp.ThermoFlowProp(); Mm.ThermoFlowProp()
            
            # JIT function call relies on the fixed Pp.a, Pp.theta, etc.
            x_new, y_new, u_new, v_new = jitted_corrector_algebra(
                # ... (JIT call parameters remain the same) ...
                xM=Mm.x, yM=Mm.y, uM=Mm.u, vM=Mm.v, aM=Mm.a, thetaM=Mm.theta, 
                xP=Pp.x, yP=Pp.y, uP=Pp.u, vP=Pp.v, aP=Pp.a, thetaP=Pp.theta,
                u_P_ref=P_ref.u, v_P_ref=P_ref.v, u_M_ref=M_ref.u, v_M_ref=M_ref.v,
                delta=target_point.delta
            )

            target_point.x = x_new.item(); target_point.y = y_new.item()
            target_point.u = u_new.item(); target_point.v = v_new.item()
            return target_point


        def _axis_prop(target_point, Pp, Mm, P_ref, M_ref):
            # OLD SLOW CALLS REMOVED: Pp.ThermoFlowProp(); Mm.ThermoFlowProp()
            
            # 1. Characteristic terms calculation (Use FIXED properties)
            Pp.alpha = jnp.asin(Pp.a / Pp.V); Pp.lamP = jnp.tan(Pp.theta + Pp.alpha); Pp.Q = Pp.u**2 - Pp.a**2; Pp.R1 = 2.0 * Pp.u * Pp.v - Pp.Q * Pp.lamP; Pp.S = 0.0
            Mm.alpha = jnp.asin(Mm.a / Mm.V); Mm.lamM = jnp.tan(Mm.theta - Mm.alpha); Mm.Q = Mm.u**2 - Mm.a**2; Mm.R1 = 2.0 * Mm.u * Mm.v - Mm.Q * Mm.lamM; Mm.S = 0.0

            # 2. Intersection and Solve (Standard Python/Intersection helper)
            target_point.x, target_point.y = Intersection([Mm.x, Mm.y], [Pp.x, Pp.y], [Mm.lamM, Pp.lamP])
            target_point.y = 0.0
            Mm.T = Mm.Q * M_ref.u + Mm.R1 * M_ref.v
            target_point.v = 0.0
            target_point.u = Mm.T / Mm.Q
            return target_point

        # ---------------------------------------------------------------------
        # --- Main corrector loop (Kinematic Update Only) ---
        # ---------------------------------------------------------------------
        
        for _ in range(max_iter):
            # Convergence check remains the same
            if (jnp.abs(param_old.y - param_new.y) < tol_x) and \
            (jnp.abs(param_old.u - param_new.u) < tol_v) and \
            (jnp.abs(param_old.v - param_new.v) < tol_v):
                break

            # 1. Update Averaged Kinematic State (u, v, y) ONLY. a, theta remain FIXED in P_avg, M_avg.
            # This kinematic averaging is required for convergence of the spatial update.
            P_avg.y = 0.5 * (paramP.y + param_old.y); P_avg.u = 0.5 * (paramP.u + param_old.u); P_avg.v = 0.5 * (paramP.v + param_old.v)
            M_avg.y = 0.5 * (paramM.y + param_old.y); M_avg.u = 0.5 * (paramM.u + param_old.u); M_avg.v = 0.5 * (paramM.v + param_old.v)
            
            # 2. Calculate new point (param_new is updated in place)
            if jnp.abs(jnp.abs(paramP.y) - jnp.abs(paramM.y)) < 1e-5:
                param_new = _axis_prop(param_new, P_avg, M_avg, paramP, paramM)
            else:
                param_new = _interior_prop(param_new, P_avg, M_avg, paramP, paramM)

            # 3. Update old for next iteration (Fast attribute swap)
            param_old.x, param_old.y, param_old.u, param_old.v = param_new.x, param_new.y, param_new.u, param_new.v

        # Final thermodynamic update (REQUIRED ONCE)
        param_new.ThermoFlowProp()

        # Update the 'self' object
        self.x, self.y, self.u, self.v = param_new.x, param_new.y, param_new.u, param_new.v
        self.ThermoFlowProp()
        
        return self
    
    def Inv_Wall_pnt(self, pwall, p33, p11, wall_ang):
            Corr_v = 1e-6
            max_iter = 50  # Prevent infinite hangs
            iters = 0
            
            # Initialize p2 as the foot point on the line 3-1
            p2 = InternalPoint(self.manager)
            p2.P0, p2.T0, p2.Q0 = p33.P0, p33.T0, p33.Q0
            
            # Set initial guess for p2 properties
            p2.u, p2.v = p33.u, p33.v
            
            # Flags to check for convergence (initialized to force first loop)
            flagv = p2.v + 10.0
            flagu = p2.u + 10.0

            # --- Iterative loop to find intersection point p2 ---
            # Condition: Keep running while the change is GREATER than the tolerance
            while (abs(p2.v - flagv) > Corr_v or abs(p2.u - flagu) > Corr_v) and iters < max_iter:
                iters += 1
                
                # Store current values to calculate the delta at end of loop
                flagv = float(p2.v)
                flagu = float(p2.u)

                # Slope of the line between points 3 and 1
                denom_31 = p33.x - p11.x
                lam31 = (p33.y - p11.y) / (denom_31 if abs(denom_31) > 1e-12 else 1e-12)
                
                # Update local properties to get the Mach line slope (lamP)
                p2.ThermoFlowProp()
                lam42 = p2.lamP

                # Find intersection of line 3-1 and Mach line from wall
                [p2.x, p2.y] = Intersection([p33.x, p33.y], [pwall.x, pwall.y], [lam31, lam42]) 

                # Safety check for geometric validity
                if p2.x > p11.x:
                    # If this happens, we usually need to pick the next segment in MAIN
                    raise NameError('BeyondPointp33xIWP')
                
                # Interpolate velocity at the new intersection x-coordinate
                p2.u = lin_interp(p33.x, p11.x, p33.u, p11.u, p2.x)
                p2.v = lin_interp(p33.x, p11.x, p33.v, p11.v, p2.x)

            # --- Final Property Calculation at the Wall Point (self) ---
            self.x = pwall.x
            self.y = pwall.y
            
            # Refresh p2 one last time for compatibility coefficients
            p2.ThermoFlowProp()

            # Exact planar (delta=0) wall compatibility: theta - nu(V) = const
            # along the C+ characteristic from p2 to the wall point (Zebbiche
            # & Youbi 2007, eqs. 1-2). wall_ang is the prescribed wall flow
            # angle (exact geometric BC), so nu_wall follows directly -- no
            # linearised Q/R1/S/T algebra needed.
            rad_wall = jnp.radians(wall_ang)
            nu_p2 = self.manager.nu_of_V(p2.V)
            nu_wall = rad_wall - p2.theta + nu_p2
            V_wall = self.manager.V_of_nu(nu_wall)

            self.u = V_wall * jnp.cos(rad_wall)
            self.v = V_wall * jnp.sin(rad_wall)

            # Update all remaining thermo properties (M, a, p, rho, etc.)
            self.ThermoFlowProp()

            # Return the interpolated foot point p2 for potential use
            return p2

    def Inv_Wall_pnt_vertical(self, pwall, p33, p11, wall_ang):
            """
            Wall-point unit process for an initial data line that is VERTICAL
            (constant x -- e.g. the flat flashing throat front, x=0 for every
            y). Inv_Wall_pnt interpolates position along the "line 3-1" by x,
            which is degenerate when x33 == x11; here the same shooting
            method is redone parametrizing by y instead (x is fixed along the
            line, so any y between p33.y and p11.y is a valid position).
            Otherwise identical to Inv_Wall_pnt.
            """
            Corr_v = 1e-6
            max_iter = 50
            iters = 0

            p2 = InternalPoint(self.manager)
            p2.P0, p2.T0, p2.Q0 = p33.P0, p33.T0, p33.Q0
            p2.u, p2.v = p33.u, p33.v

            flagv = p2.v + 10.0
            flagu = p2.u + 10.0

            x_line = p33.x  # constant along the vertical front
            y_lo, y_hi = (p11.y, p33.y) if p11.y < p33.y else (p33.y, p11.y)

            while (abs(p2.v - flagv) > Corr_v or abs(p2.u - flagu) > Corr_v) and iters < max_iter:
                iters += 1
                flagv = float(p2.v)
                flagu = float(p2.u)

                # Update local properties to get the Mach line slope (lamP)
                p2.ThermoFlowProp()
                lam42 = p2.lamP

                # Intersection of the vertical line x = x_line with the C+
                # line through pwall of slope lam42:
                #   y - pwall.y = lam42 * (x - pwall.x)  ->  x = x_line
                p2.x = x_line
                p2.y = pwall.y - lam42 * (pwall.x - x_line)

                # Safety check: p2 must lie within the known segment 3-1
                if not (y_lo - 1e-9 <= p2.y <= y_hi + 1e-9):
                    raise NameError('BeyondPointp33yIWP')

                p2.u = lin_interp(p33.y, p11.y, p33.u, p11.u, p2.y)
                p2.v = lin_interp(p33.y, p11.y, p33.v, p11.v, p2.y)

            # --- Final Property Calculation at the Wall Point (self) ---
            self.x = pwall.x
            self.y = pwall.y

            p2.ThermoFlowProp()

            # Exact planar (delta=0) wall compatibility, see Inv_Wall_pnt.
            rad_wall = jnp.radians(wall_ang)
            nu_p2 = self.manager.nu_of_V(p2.V)
            nu_wall = rad_wall - p2.theta + nu_p2
            V_wall = self.manager.V_of_nu(nu_wall)

            self.u = V_wall * jnp.cos(rad_wall)
            self.v = V_wall * jnp.sin(rad_wall)

            self.ThermoFlowProp()

            return p2

    def ReflexLine(self, n, l):
        """
        Reflex Line position calculator (ReflexZone).
        Calculates the position of the new wall point (or reflection point) 
        a distance 'l' away, projected along the Mach angle plus flow angle.
        
        Parameters:
            n (float): Step multiplier (1 in the case of wall projection, n_step in others).
            l (float): Calculated distance based on mass flow ratio.
        """
        self.ThermoFlowProp()
        angle_rad = (jnp.arcsin(1 / self.M) + self.theta) 
        
        RetObj = InternalPoint(self.manager) 
        RetObj.x = self.x + n * l * jnp.cos(angle_rad)
        RetObj.y = self.y + n * l * jnp.sin(angle_rad)
        RetObj.u = self.u
        RetObj.v = self.v

        return RetObj
            

# -----------------------------
# Helper functions
# ----------------------------- 
def Intersection(p1_coords, p2_coords, slopes):
        x1, y1 = p1_coords
        x2, y2 = p2_coords
        lam1, lam2 = slopes        
        denom = lam1 - lam2

        if abs(denom) < 1e-12: 
            raise ValueError("MOC Error: Characteristic lines are near-parallel. Cannot find intersection.")
            
        x = (lam1 * x1 - lam2 * x2 + y2 - y1) / denom
        y = y1 + lam1 * (x - x1)
        
        return x, y
    
def lin_interp(x1,x2,y1,y2,x):
    return y1 + (x - x1) * ((y2 - y1) / (x2 - x1))

def rotation_around_center(y_t, ang, rho_d):
    """
    Rotate the point (0, y_t) about the center located at (0, rho_d + y_t)
    by an angle `ang` (in degrees), counter-clockwise.

    This is used in the kernel-region construction of the 
    Method of Characteristics to generate rotated characteristic 
    boundaries starting from the throat radius.

    Parameters
    ----------
    y_t : float
        The initial y-coordinate of the point to be rotated (the x-coordinate is fixed to 0).
    ang : float
        Rotation angle in degrees. Positive = counter-clockwise rotation.
    rho_d : float
        Kernels offset distance; the rotation center is at (0, rho_d + y_t).

    Returns
    -------
    x_rot : float
        Rotated x-coordinate of the point.
    y_rot : float
        Rotated y-coordinate of the point.

    Notes
    -----
    The transformation performed is:
        1. Translate the point so the center becomes the origin
        2. Apply rotation matrix R(ang)
        3. Translate back to the original center
    """
    # Original point
    x = jnp.array([0.0, y_t])

    # Rotation center
    center = jnp.array([0.0, rho_d + y_t])

    # Shift to center
    xt = x - center

    # Rotation matrix using standard NumPy arrays
    ang_rad = jnp.radians(ang)
    M = jnp.array([
        [jnp.cos(ang_rad), -jnp.sin(ang_rad)],
        [jnp.sin(ang_rad),  jnp.cos(ang_rad)]
    ])

    # Rotate
    xr = M @ xt

    # Shift back
    x_rot = xr + center

    return x_rot[0], x_rot[1]


def build_convergent_inlet(y_t, rho_d, L_conv, n_points=30):
    """
    Purely geometric (no MOC/flow physics) upstream extension of the wall,
    from the inlet plane (x=-L_conv) back to the throat (x=0, y=y_t) --
    an "S-curve" convergent contour made of two tangent circular arcs of
    EQUAL radius rho_d, meeting at a single inflection point where
    concavity flips (a reversed-curvature/ogive convergent inlet, the
    same construction long-radius flow nozzles use).

    Arc 1 (throat -> inflection) is literally the SAME circle as the
    kernel region's own throat rounding (moc.core.classes.
    rotation_around_center, center (0, rho_d+y_t), same radius rho_d) --
    continued backward (negative angle) instead of forward, so the wall
    is curvature-continuous across the throat, not just position/slope-
    continuous.

    Arc 2 (inflection -> inlet) is arc 1's own point reflection through
    the inflection point P1 (Q -> 2*P1 - Q for every point Q on arc 1).
    A point reflection through a point ON a curve preserves the tangent
    LINE there while flipping curvature sign -- exactly "changes
    concavity" -- and, since the throat point itself (ang=0) has zero
    slope by the throat's own left-right symmetry, its reflection (the
    inlet-plane end point) also has EXACTLY zero slope: the wall meets
    the inlet duct already parallel to the axis, not just "almost"
    parallel, with no separate solve needed for where that happens.

    Splitting L_conv exactly in half between the two arcs is what makes
    arc 2's radius come out equal to arc1's (rho_d) automatically --
    algebraically, R2 = L2/sin(phi1) = L2/(L1/rho_d) = rho_d when
    L1 == L2 -- rather than requiring a free second radius.

    Parameters
    ----------
    y_t : float
        Throat half-height (m).
    rho_d : float
        Throat/kernel radius of curvature (m) -- same value already used
        for the divergent side's own throat rounding (rotation_around_
        center's rho_d).
    L_conv : float
        Total axial length of the convergent inlet section (m), split
        evenly between the two arcs (L1 = L2 = L_conv/2). Must satisfy
        L_conv/2 < rho_d (arc 1 cannot reach further upstream than its
        own quarter-circle) -- raises ValueError otherwise.
    n_points : int
        Number of points per arc (2*n_points total, excluding the
        duplicated throat point).

    Returns
    -------
    dict
        {"x", "y"}, ordered from the inlet plane (x=-L_conv) to just
        before the throat (x=0 itself is NOT included, so this can be
        directly prepended to a wall array that already starts there).
    """
    L1 = 0.5 * float(L_conv)
    if not (0.0 < L1 < rho_d):
        raise ValueError(
            f"L_conv/2={L1:.6g} must be strictly between 0 and rho_d={rho_d:.6g} "
            "-- the same-radius throat arc cannot reach further upstream than a "
            "quarter circle; reduce L_conv or increase rho_d."
        )
    phi1 = float(np.arcsin(L1 / rho_d))

    ang = np.linspace(-phi1, 0.0, n_points + 1)
    x1c = rho_d * np.sin(ang)
    y1c = y_t + rho_d * (1.0 - np.cos(ang))

    # x1c/y1c are ordered junction (ang=-phi1) -> throat (ang=0).
    x1_junction, y1_junction = x1c[0], y1c[0]
    x2c = 2.0 * x1_junction - x1c
    y2c = 2.0 * y1_junction - y1c
    # x2c/y2c (arc 1's point reflection through the junction) are then
    # ordered junction -> inlet end (x2c[0]==junction, x2c[-1]==-L_conv).

    # Inlet -> throat order: arc 2 reversed (inlet end -> junction), then
    # arc 1's interior points (junction excluded, already the last point
    # of the arc-2 half; throat excluded, the caller's own wall array
    # already starts there).
    x = np.concatenate([x2c[::-1], x1c[1:-1]])
    y = np.concatenate([y2c[::-1], y1c[1:-1]])

    return {"x": x.tolist(), "y": y.tolist()}


# -----------------------------------------------------------------------------#
# --- JAX-COMPATIBLE SUPPORTING FUNCTIONS (Must be defined here or imported) ---
# -----------------------------------------------------------------------------#



# These operate on plain Python-float scalars from InternalPoint attributes,
# called one segment at a time inside Python loops (MassFlowCalc, the Stage-4
# incremental mass sum) -- never batched. @jit here bought nothing (no
# fusion/vectorization across a Python for-loop) and only added a JAX
# dispatch/compile round trip per call, so these are plain numpy now.
def DistCalc(x1, y1, x2, y2):
    """Calculates distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def vecDotVec(u0, u1, x0, x1):
    """Calculates the dot product of two 2D vectors [u0, u1] and [x0, x1]."""
    return u0 * x0 + u1 * x1

def MakeVec(x1, y1, x2, y2):
    """
    Converts coordinates into a normalized unit vector along the segment P2 -> P1.
    """
    a_x = x1 - x2
    a_y = y1 - y2
    mag = np.sqrt(a_x**2 + a_y**2)

    if mag > 1e-12:
        return [a_x / mag, a_y / mag]
    return [np.nan, np.nan]

def RotVec(vec, angle_deg):
    """Rotates a vector (u, v) by an angle (degrees). Returns [u_rot, v_rot]."""
    angle_rad = np.radians(angle_deg)

    u_rot = vec[0] * np.cos(angle_rad) - vec[1] * np.sin(angle_rad)
    v_rot = vec[0] * np.sin(angle_rad) + vec[1] * np.cos(angle_rad)

    return [u_rot, v_rot]

# -----------------------------------------------------------------------------#
#
# Mass Flow Calculator (MAIN FUNCTION)
#
# -----------------------------------------------------------------------------#

def MassFlowCalc(IO):
    """
    Calculates the mass flow rate based on the original formula,
    FIXED to use ThermoFlowProp() and JAX math functions (jnp).
    """
    flag = 1
    m = len(IO)
    mass = 0.0
    
    # Determine Axisymmetry (delta=1 for AXI, delta=0 for PLANAR)
    delta = 0.0
    
    # --- Mode 1: Single Point Calculation (m=1) ---
    if m == 1:
        # FIX 1: Use the correct method name
        M = IO[0].M
        ang = (jnp.arcsin(1/M) + jnp.arctan(IO[0].v / IO[0].u) - (jnp.pi / 2))
        V = vecDotVec(IO[0].u, IO[0].v, jnp.cos(ang), jnp.sin(ang))
        
        mass = mass + IO[0].rho * V 
    
    else:
        for i in range(1,m):
            l = DistCalc(IO[i-flag].x,IO[i-flag].y,IO[i].x,IO[i].y)

            try:
                vec = MakeVec(IO[i-flag].x,IO[i-flag].y,IO[i].x,IO[i].y)
            except: 
                pdb.set_trace()

            rotVec =  RotVec(vec, -90)
            rho = (IO[i-flag].rho + IO[i].rho)/2
            u = (IO[i-flag].u + IO[i].u)/2
            v = (IO[i-flag].v + IO[i].v)/2
            x = (IO[i-flag].x + IO[i].x)/2
            y = (IO[i-flag].y + IO[i].y)/2
            V =  vecDotVec(u,v,rotVec[0],rotVec[1])

            if delta:
                R = jnp.sqrt(x**2 + y**2)
                A = jnp.pi*2*R
            else:
                A = 1

            mass = mass + (rho * l * V * A)

    return mass

