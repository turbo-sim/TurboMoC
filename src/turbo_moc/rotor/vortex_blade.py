"""
turbo_moc.rotor.vortex_blade -- supersonic impulse/reaction rotor blade design by
the vortex-flow method of Goldman & Scullin (1968), "Analytical Investigation
of Supersonic Turbomachinery Blading, I - Computer Program for Blading
Design", NASA TN D-4421 -- the original, fully-worked, FORTRAN-validated
source both Boxer et al. (1952, NACA RM L52B06) and Paniagua et al. (2014)
build on. Generalized to real gases the same way Bufi & Cinnella (2018,
DGMOC/RODEC) generalized it for dense gases.

This module is a direct, line-for-line port of NASA TN D-4421's FORTRAN
listing (its Main Program's transition-arc marching loops, Eqs. 6/7/9/10/13,
and the GSTAR/CSTAR/SIGMA geometric-parameter calculations), NOT a
reimplementation from the paper's prose or from Paniagua (2014)'s shorter,
closed-form-looking Eqs. 11-19 -- those turned out to describe something
different from what they appear to (see "History" below). Validated to 4
decimal places against the report's own worked example (Table II: BETAN=65,
VIN=39, VOUT=35, VLOW=18, VUP=59, GAM=1.4 -> BETAT=-68.7395 deg,
GSTAR=0.5544, CSTAR=1.6281, individual transition-arc coordinates matching
point-for-point) before any real-gas generalization was attempted.

Physical picture (TN D-4421 Fig. 1): a uniform inlet flow (far upstream,
flow angle beta_inlet) is turned into a supersonic free-vortex flow
(M*R*=const) by an INLET TRANSITION ARC, held at constant Mach number
through a CIRCULAR ARC concentric with the vortex center, then turned back
into uniform flow (beta_outlet) by an OUTLET TRANSITION ARC. The pressure
(lower/concave) surface and suction (upper/convex) surface are built
independently this way, each with its own constant surface Mach number
(M_lower, M_upper).

IMPORTANT -- what `pressure` and `suction` actually bound: they are NOT the
two sides of one solid blade closing into a simple polygon. Past their far
corners the flow is uniform, so each surface asymptotes to a straight line
at beta_inlet/beta_outlet; extending `suction`'s corners that way lands
exactly `pitch` short of `pressure`'s corners (by construction -- see the
GSTAR discussion below) and the two extensions are parallel, so they never
meet. This is the ordinary cascade far-field picture: THIS blade's suction
surface runs parallel, upstream and downstream, to the NEXT (pitch-shifted)
blade's pressure surface -- together they bound one flow passage. The solid
single-blade cross-section (`blade` in the returned dict) closes pressure
to suction+pitch with a rounded leading/trailing edge: the upper wall is
shifted a bit further than +pitch (by `translate`, solved so the
trailing-edge fillet's radius hits a target pitch/le_te_ratio), then each
end is closed by a straight tangent segment plus a circular arc (built as
a semicircle on that segment's own diameter -- see `_close_blade_le_te`).
TN D-4421 itself only mentions closing segments (BE, GJ in its Fig. 1) and
rounded LE/TE without giving a formula for their size; this module's
choice (pitch/le_te_ratio, default 25) is this project's own, not TN
D-4421's.

History (why this replaced several earlier, unsuccessful attempts in this
module): Paniagua (2014)'s own Eq. 17 (x*=-R*sin(phi), y*=R*cos(phi)) looks
like a closed-form wall-position formula, and was tried literally first --
it produced large, unphysical kinks at the transition-arc/circular-arc
junction. TN D-4421's FORTRAN reveals why: that formula gives the position
along a REFERENCE characteristic curve (the "major vortex-expansion/
compression characteristic"), used only to construct MACH LINES -- the
actual wall position is found by intersecting each such Mach line with the
PREVIOUS wall segment (a genuine two-line intersection, marching step by
step), i.e. real characteristics marching, not a smooth closed-form curve.
Implementing that literal marching (this module) fixes the kink entirely.
Likewise, the earlier "each surface has no natural relationship to the
other, pitch must be guessed" problem is resolved by TN D-4421's own GSTAR
calculation (Eq.-adjacent, "CALCULATE G* -- THE DIMENSIONLESS BLADE
SPACING"): extend the upper surface's own inlet-far point along
beta_inlet until it reaches the same x-coordinate as the lower surface's
own inlet-far point; the resulting y-gap IS the blade pitch -- not a free
input to guess, a derived output of the geometry.

Real-gas generalization: only the thermodynamic relations are replaced --
R*(nu) = a_star/V(nu) (already shown numerically equivalent to inverting
this report's own perfect-gas f(R*) relation, Eq. 10b, for the perfect-gas
case) and local Mach number/Mach angle via this package's own
V_of_nu/a_of_V real-gas Prandtl-Meyer table. The marching recursion itself
(position via Mach-line/wall-segment intersection), the rotation formulas
(Eqs. 6/7), and the GSTAR/CSTAR/SIGMA geometry are pure kinematics/geometry,
unchanged by the thermodynamic model.

beta_outlet: ALWAYS derived from beta_inlet and M_inlet/M_outlet via
mass-flux continuity across the transition, rho*V*cos(beta) = const --
see _solve_beta_outlet's docstring. This is the real-gas generalization
of TN D-4421's own Eq. 9 ("BETAT"), verified to reproduce the report's
own Table II value (-68.7395 deg) to 10+ significant figures when
evaluated with perfect-gas relations, so it is the same physics as
Eq. 9, not a different formula standing in for it. It is deliberately
NOT an overridable free input (unlike Bufi & Cinnella (2018)'s
convention, which this module followed in earlier versions): an
inconsistent beta_outlet doesn't just distort the geometry, it produces
a self-intersecting wall with no warning that anything is wrong --
e.g. M_inlet=M_outlet with beta_inlet=0 has NO consistent solution
except beta_outlet=0 itself, and there is no way to tell from the
output alone that a self-intersecting result traces back to this rather
than a bug. If a specific exit flow angle is the actual design target,
choose M_inlet/M_outlet to hit it via continuity instead.

Units: this module works entirely in the SAME units as its FluidManager
(SI: Pa, K, m/s) for thermodynamic quantities, and returns GEOMETRY
nondimensionalized by r* (the physical radius of the sonic-velocity
streamline for whichever surface's vortex flow, in meters) unless an
explicit `r_star` length scale is supplied to rescale to physical units --
see `design_rotor_vortex_blade`'s docstring.
"""

import numpy as np

from turbo_moc.core.classes import get_fluid_manager

__all__ = ["design_rotor_vortex_blade"]


def _solve_beta_outlet(manager, V_inlet, V_outlet, beta_inlet_rad):
    """
    Real-gas generalization of TN D-4421's Eq. 9 ("BETAT"): beta_outlet
    derived from beta_inlet and the inlet/outlet Mach numbers via mass-flux
    continuity across the transition, rho*V*cos(beta) = const, rather than
    taken as an independent free input. Eq. 9 itself is exactly this
    relation specialized to a perfect gas (rho/rho0, V/a0 as closed-form
    functions of M) -- verified numerically here: reproduces TN D-4421's
    own Table II value (BETAT=-68.7395 deg) to 10+ significant figures
    when evaluated with the perfect-gas relations, confirming the general
    (real-gas) form below is the same physics, not a different formula.

    Without this constraint, beta_outlet is a free parameter that can be
    picked inconsistently with beta_inlet/M_inlet/M_outlet for a given
    fluid -- e.g. M_inlet=M_outlet with beta_inlet=0 and a nonzero
    beta_outlet has no valid solution (continuity forces beta_outlet=0
    too) -- and the resulting transition-arc rotations produce a self-
    intersecting wall. Solving for beta_outlet from continuity instead
    guarantees a geometrically valid (though not necessarily still-sane
    for extreme inputs) result whenever one exists.

    Returns beta_outlet in radians, with the same sign convention as the
    rest of this module (negative for the "ordinary" turning direction
    matching TN D-4421's own worked example).
    """
    lhs = manager.rho_of_V(V_inlet) * V_inlet * np.cos(beta_inlet_rad)
    rhs_rate = manager.rho_of_V(V_outlet) * V_outlet
    cos_beta_outlet = lhs / rhs_rate
    if abs(cos_beta_outlet) > 1.0:
        raise ValueError(
            f"No real beta_outlet satisfies mass-flux continuity for these inputs "
            f"(cos(beta_outlet)={cos_beta_outlet:.4f} outside [-1,1]) -- "
            f"M_inlet/M_outlet/beta_inlet are too inconsistent with each other."
        )
    return -np.arccos(np.clip(cos_beta_outlet, -1.0, 1.0))


def _line_intersect(P1, d1, P2, d2):
    """Intersection of line P1+t*d1 with line P2+s*d2."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([P2[0] - P1[0], P2[1] - P1[1]])
    t, _s = np.linalg.solve(A, b)
    return np.array(P1) + t * np.array(d1)


def _arc_from_diameter(L, T, interior_ref, num_points=40):
    """
    LE/TE closure arc: a semicircle with L and T as the two ends of a
    DIAMETER (center = midpoint(L,T), radius = |L-T|/2). Tangent to the
    lower wall's own tangent line at L and to the upper wall's tangent
    line at T automatically, with no separate slope/projection step --
    provided T itself was already built as the intersection of the upper
    wall's own tangent line with the normal to the lower wall through L
    (see the LE/TE construction in design_rotor_vortex_blade). This is
    the exactly-determined case of "circle tangent to one point, tangent
    to a second line": 3 constraints on a circle's 3 DOF (center x,y,
    radius), unlike "circle through two points AND tangent at both"
    which is 4 constraints on 3 DOF and has no solution in general.

    Sweep direction is chosen via the sign of a cross product against
    interior_ref (a point inside the passage) so the arc bulges away from
    the passage -- this is what makes the LE and TE arcs come out with
    opposite (CW/CCW) handedness automatically. A raw distance comparison
    between the two candidate centers is NOT used here: since the two
    candidate centers are only 2R apart, their distances to a far
    interior_ref are nearly equal and the comparison is numerically
    unreliable.
    """
    L = np.asarray(L, dtype=float)
    T = np.asarray(T, dtype=float)
    center = 0.5 * (L + T)
    R = np.linalg.norm(T - L) / 2.0
    ang1 = np.arctan2(L[1] - center[1], L[0] - center[0])
    to_center = center - np.asarray(interior_ref, dtype=float)
    radial = L - center
    cross_z = to_center[0] * radial[1] - to_center[1] * radial[0]
    sign = -1.0 if cross_z >= 0 else 1.0
    angles = ang1 + sign * np.linspace(0, np.pi, num_points)
    return center[0] + R * np.cos(angles), center[1] + R * np.sin(angles)


def _close_blade_le_te(lower, upper, pitch, beta_inlet_rad, beta_outlet_rad,
                        le_te_ratio, num_points):
    """
    Close the pressure (`lower`) and suction+pitch (`upper`) open walls
    into a finite solid blade with rounded leading/trailing edges, using
    the validated line-first-then-arc construction (see _arc_from_diameter).

    The upper wall is shifted a bit further than the bare `+pitch` used to
    build `upper` itself -- by `translate`, beyond pitch -- so a circular
    arc can be tangent to both walls' own tangent lines at the corners.
    `translate` is not a free guess: it is the unique value that makes the
    trailing-edge arc's radius equal a target r_TE = pitch/le_te_ratio
    (a ratio, so this works whether or not the geometry has been
    dimensionalized yet). The leading-edge radius then falls out as
    whatever that SAME translate produces at the other end -- it is a
    consequence, not an independently chosen target, and will differ from
    r_TE whenever beta_inlet != beta_outlet (a reacting blade).

    Returns (closed_x, closed_y, diagnostics) where diagnostics has
    translate/le_radius/te_radius.
    """
    lower_x = np.asarray(lower["x"], dtype=float)
    lower_y = np.asarray(lower["y"], dtype=float)
    upper_x = np.asarray(upper["x"], dtype=float)
    upper_y = np.asarray(upper["y"], dtype=float)

    tan_bi = np.tan(beta_inlet_rad)
    tan_bo = np.tan(beta_outlet_rad)

    r_TE_target = pitch / le_te_ratio

    u_TE_unit = np.array([1.0, tan_bo])
    u_TE_unit /= np.linalg.norm(u_TE_unit)
    n_TE_unit = np.array([-u_TE_unit[1], u_TE_unit[0]])
    L1_check = np.array([lower_x[-1], lower_y[-1]])
    U1_notranslate = np.array([upper_x[-1], upper_y[-1]])
    D_TE_base = np.dot(U1_notranslate - L1_check, n_TE_unit)
    translate = (2.0 * r_TE_target - D_TE_base) / n_TE_unit[1]

    L0 = np.array([lower_x[0], lower_y[0]])
    L1 = np.array([lower_x[-1], lower_y[-1]])
    U0 = np.array([upper_x[0], upper_y[0] + translate])
    U1 = np.array([upper_x[-1], upper_y[-1] + translate])

    u_LE, n_LE = np.array([1.0, tan_bi]), np.array([-tan_bi, 1.0])
    u_TE, n_TE = np.array([1.0, tan_bo]), np.array([-tan_bo, 1.0])

    T_LE = _line_intersect(U0, u_LE, L0, n_LE)
    T_TE = _line_intersect(U1, u_TE, L1, n_TE)

    interior_ref = np.array([
        0.0,
        0.5 * (lower_y[len(lower_y) // 2] + upper_y[len(upper_y) // 2] + translate),
    ])
    le_x, le_y = _arc_from_diameter(L0, T_LE, interior_ref, num_points=num_points)
    te_x, te_y = _arc_from_diameter(L1, T_TE, interior_ref, num_points=num_points)
    le_radius = float(0.5 * np.linalg.norm(T_LE - L0))
    te_radius = float(0.5 * np.linalg.norm(T_TE - L1))

    upper_rev_x = upper_x[::-1]
    upper_rev_y = (upper_y + translate)[::-1]
    le_arc_rev_x = np.asarray(le_x)[::-1]
    le_arc_rev_y = np.asarray(le_y)[::-1]

    closed_x = np.concatenate([
        lower_x, np.asarray(te_x)[1:], [U1[0]], upper_rev_x[1:], [T_LE[0]], le_arc_rev_x[1:],
    ])
    closed_y = np.concatenate([
        lower_y, np.asarray(te_y)[1:], [U1[1]], upper_rev_y[1:], [T_LE[1]], le_arc_rev_y[1:],
    ])

    diagnostics = {"translate": float(translate), "le_radius": le_radius, "te_radius": te_radius}
    return closed_x, closed_y, diagnostics


def _V_of_M(manager, M_target):
    """
    Real-gas velocity at a prescribed Mach number, along the FluidManager's
    own isentrope (self.s0). Inverts M(V) = V / a_of_V(V) by bisection --
    a_of_V is a cheap spline lookup (see FluidManager._build_nu_table), so
    this is not a real thermodynamic call, just root-finding on the table.
    """
    V_lo = manager._nu_V_table[0]
    V_hi = manager._nu_V_table[-1]

    def resid(V):
        return V / manager.a_of_V(V) - M_target

    r_lo, r_hi = resid(V_lo), resid(V_hi)
    if r_lo > 0:
        raise ValueError(f"M_target={M_target} is below the table's minimum resolvable Mach number")
    if r_hi < 0:
        raise ValueError(f"M_target={M_target} exceeds the table's range -- rebuild with a wider table")

    lo, hi = V_lo, V_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _R_star(manager, nu):
    """R*(nu) = a_star/V(nu) -- real-gas generalization of TN D-4421 Eq. 10b/13,
    verified numerically equivalent to inverting that perfect-gas closed form."""
    V = manager.V_of_nu(float(nu))
    return manager.a_star / V


def _mach_angle(manager, R_star):
    """Mach angle mu = arcsin(1/M) at the state whose vortex-flow radius is R_star."""
    V = manager.a_star / R_star
    M = V / manager.a_of_V(V)
    return np.arcsin(np.clip(1.0 / M, -1.0, 1.0))


def _lower_arc_march(manager, nu_far, nu_wall, delnu, num_points):
    """
    Port of TN D-4421's 'CALCULATE COORDINATES FOR LOWER TRANSITION ARC'
    loop: marches from the wall (nu=nu_wall, local (x,y)=(0,R_wall)) toward
    nu_far, finding each wall point as the intersection of the local Mach
    line (through a reference point on the major vortex characteristic)
    with the previous wall segment. Real-gas generalized via _R_star/
    _mach_angle; the marching recursion itself is unchanged from the
    original FORTRAN. Returns (x, y) arrays, local unrotated frame, index
    0 = wall, index -1 = far/uniform-flow end.
    """
    KMN = num_points - 1
    delv = (nu_far - nu_wall) / KMN if KMN > 0 else 0.0
    R_wall = _R_star(manager, nu_wall)
    V = nu_far

    PHIKP1 = -(V - nu_wall) + KMN * delv
    UMKP1 = _mach_angle(manager, R_wall)
    TXLO, TYLO = 0.0, R_wall
    xs, ys = [TXLO], [TYLO]
    ref_xs, ref_ys = [], []

    for KK in range(1, KMN + 1):
        K = (KMN + 1) - KK
        PHIK = PHIKP1 - delv
        nu_k = V - (K - 1) * delv  # matches F(V,FN=K)=2V-...-2(K-1)delv <-> nu_k = V-(K-1)*delv
        TR = _R_star(manager, nu_k)
        TX, TY = TR * np.sin(PHIK), TR * np.cos(PHIK)
        EMWK = np.tan(-PHIKP1)
        UMK = _mach_angle(manager, TR)
        EMK = -np.tan((PHIK + UMK + PHIKP1 + UMKP1) / 2.0)
        TEMP = TYLO - EMWK * TXLO
        TEMPP = TY - EMK * TX
        TEMPPP = EMK - EMWK
        TXLO = (TEMP - TEMPP) / TEMPPP
        TYLO = (EMK * TEMP - EMWK * TEMPP) / TEMPPP
        PHIKP1, UMKP1 = PHIK, UMK
        xs.append(TXLO)
        ys.append(TYLO)
        ref_xs.append(TX)
        ref_ys.append(TY)

    return np.array(xs), np.array(ys), np.array(ref_xs), np.array(ref_ys)


def _upper_arc_march(manager, nu_far, nu_wall, delnu, num_points):
    """
    Port of TN D-4421's 'CALCULATE COORDINATES FOR UPPER TRANSITION ARC'
    loop -- genuinely different sign conventions from the lower-arc block
    in the original FORTRAN (not a simple mirror image), reproduced
    faithfully here. nu_wall > nu_far (the upper/suction surface increases
    the Mach number relative to the far/uniform-flow value).
    """
    JMN = num_points - 1
    delv = (nu_wall - nu_far) / JMN if JMN > 0 else 0.0
    R_wall = _R_star(manager, nu_wall)
    V = nu_far

    PHIJP1 = -(nu_wall - V) + JMN * delv
    UMJP1 = _mach_angle(manager, R_wall)
    TXUP, TYUP = 0.0, R_wall
    xs, ys = [TXUP], [TYUP]
    ref_xs, ref_ys = [], []

    for JJ in range(1, JMN + 1):
        J = (JMN + 1) - JJ
        PHIJ = PHIJP1 - delv
        FN = -J + 2
        nu_j = V - (FN - 1) * delv
        TR = _R_star(manager, nu_j)
        TX, TY = TR * np.sin(PHIJ), TR * np.cos(PHIJ)
        EMWJ = np.tan(-PHIJP1)
        UMJ = _mach_angle(manager, TR)
        EMJ = np.tan((-PHIJ + UMJ - PHIJP1 + UMJP1) / 2.0)
        TEMP = TYUP - EMWJ * TXUP
        TEMPP = TY - EMJ * TX
        TEMPPP = EMJ - EMWJ
        TXUP = (TEMP - TEMPP) / TEMPPP
        TYUP = (EMJ * TEMP - EMWJ * TEMPP) / TEMPPP
        PHIJP1, UMJP1 = PHIJ, UMJ
        xs.append(TXUP)
        ys.append(TYUP)
        ref_xs.append(TX)
        ref_ys.append(TY)

    return np.array(xs), np.array(ys), np.array(ref_xs), np.array(ref_ys)


def _rotate_inlet(x, y, alpha):
    """TN D-4421: XLOWN=Y*sin(a)+X*cos(a), YLOWN=Y*cos(a)-X*sin(a)."""
    s, c = np.sin(alpha), np.cos(alpha)
    return y * s + x * c, y * c - x * s


def _rotate_outlet(x, y, alpha):
    """TN D-4421: XLOWO=Y*sin(a)-X*cos(a), YLOWO=Y*cos(a)+X*sin(a) (note:
    genuinely different sign pattern from _rotate_inlet, not a typo)."""
    s, c = np.sin(alpha), np.cos(alpha)
    return y * s - x * c, y * c + x * s


def _surface(manager, nu_inlet, nu_outlet, nu_wall, beta_inlet_rad, beta_outlet_rad,
             is_upper, num_points):
    """
    One full surface (pressure or suction): inlet transition arc + circular
    arc + outlet transition arc, in a single local frame (own vortex center
    at the origin), per TN D-4421 Eqs. 6/7 (rotation angles) + the marching
    functions above (position).
    """
    if is_upper:
        alpha_in = (nu_wall - nu_inlet) - beta_inlet_rad
        alpha_out = -((nu_wall - nu_outlet) + beta_outlet_rad)
        x_in_l, y_in_l, refx_in_l, refy_in_l = _upper_arc_march(manager, nu_inlet, nu_wall, None, num_points)
        x_out_l, y_out_l, refx_out_l, refy_out_l = _upper_arc_march(manager, nu_outlet, nu_wall, None, num_points)
    else:
        alpha_in = (nu_inlet - nu_wall) - beta_inlet_rad
        alpha_out = -((nu_outlet - nu_wall) + beta_outlet_rad)
        x_in_l, y_in_l, refx_in_l, refy_in_l = _lower_arc_march(manager, nu_inlet, nu_wall, None, num_points)
        x_out_l, y_out_l, refx_out_l, refy_out_l = _lower_arc_march(manager, nu_outlet, nu_wall, None, num_points)

    x_in, y_in = _rotate_inlet(x_in_l, y_in_l, alpha_in)
    x_out, y_out = _rotate_outlet(x_out_l, y_out_l, alpha_out)
    refx_in, refy_in = _rotate_inlet(refx_in_l, refy_in_l, alpha_in)
    refx_out, refy_out = _rotate_outlet(refx_out_l, refy_out_l, alpha_out)

    # Mach-line (characteristic) segments: reference point on the major
    # vortex characteristic -> the wall point it locates via intersection
    # with the previous wall segment. Wall points on the inlet arc are
    # xs[1:] in the ORIGINAL (unreversed) marching order, i.e. x_in[1:]
    # before the [::-1] flip used for the final wall polyline.
    mach_in = [
        (float(refx_in[k]), float(refy_in[k]), float(x_in[k + 1]), float(y_in[k + 1]))
        for k in range(len(refx_in))
    ]
    mach_out = [
        (float(refx_out[k]), float(refy_out[k]), float(x_out[k + 1]), float(y_out[k + 1]))
        for k in range(len(refx_out))
    ]

    R_wall_star = _R_star(manager, nu_wall)
    n_circ = max(num_points, int(abs(np.degrees(alpha_out - alpha_in))) + 2)
    angle_circ = np.linspace(alpha_in, alpha_out, n_circ)
    x_circ = R_wall_star * np.sin(angle_circ)
    y_circ = R_wall_star * np.cos(angle_circ)

    x = np.concatenate([x_in[::-1], x_circ[1:-1], x_out])
    y = np.concatenate([y_in[::-1], y_circ[1:-1], y_out])

    return {
        "x": x, "y": y,
        "x_inlet_far": float(x_in[-1]), "y_inlet_far": float(y_in[-1]),
        "x_outlet_far": float(x_out[-1]), "y_outlet_far": float(y_out[-1]),
        "R_wall_star": float(R_wall_star),
        "alpha_inlet": float(alpha_in), "alpha_outlet": float(alpha_out),
        "mach_lines_inlet": mach_in, "mach_lines_outlet": mach_out,
    }


def design_rotor_vortex_blade(
    fluid_name,
    P0_rel, T0_rel,
    M_inlet, M_outlet,
    M_lower, M_upper,
    beta_inlet,
    backend="HEOS",
    Q0_rel=float("nan"),
    r_star=None,
    num_points=60,
    le_te_ratio=25.0,
):
    """
    Design a supersonic rotor blade section by the vortex-flow method,
    following NASA TN D-4421 (Goldman & Scullin, 1968) -- see this module's
    docstring for exactly how, why it replaced earlier (Paniagua Eq. 17
    based) attempts in this module.

    Parameters
    ----------
    fluid_name : str
        CoolProp fluid name.
    P0_rel, T0_rel : float
        Relative-frame stagnation pressure [Pa] and temperature [K] at the
        rotor inlet (subcooled-liquid/single-phase convention -- pass
        Q0_rel in [0,1] instead for a saturated two-phase inlet).
    M_inlet, M_outlet : float
        Relative Mach number far upstream/downstream of the rotor (in the
        uniform-flow regions, before/after the transition arcs). Together
        with beta_inlet these set the degree of reaction: M_outlet=M_inlet
        is the pure-impulse case (no static enthalpy drop across the
        rotor); M_outlet != M_inlet gives a reacting blade.
    M_lower, M_upper : float
        Constant Mach number held on the pressure (concave/lower) and
        suction (convex/upper) circular arcs.
    beta_inlet : float
        Relative flow angle [deg] from the axial direction at inlet --
        the only free flow-angle input. beta_outlet is ALWAYS derived
        from beta_inlet/M_inlet/M_outlet via mass-flux continuity
        (rho*V*cos(beta) = const across the transition) -- the real-gas
        generalization of TN D-4421's own Eq. 9, see `_solve_beta_outlet`'s
        docstring -- not a free input, by design: an inconsistent
        beta_outlet doesn't just distort the geometry, it produces a
        self-intersecting wall (mass isn't actually conserved), with no
        warning otherwise. If you want a specific exit flow angle, choose
        M_inlet/M_outlet to hit it via continuity rather than overriding
        beta_outlet directly (not currently supported -- would mean
        solving continuity for M_outlet given both betas instead).
    r_star : float or None
        Physical radius [m] of the sonic-velocity streamline to rescale
        the (nondimensional) geometry by. None returns x*, y* (units of
        r*, i.e. dimensionless).
    num_points : int
        Points per transition arc (endpoints included) -- a fine-grained
        stand-in for TN D-4421's explicit Delta-nu marching.
    le_te_ratio : float
        Target pitch/r_TE ratio used to size the rounded leading/trailing
        edges that close the solid blade (default 25). The upper wall's
        extra shift beyond `pitch` is solved so the trailing-edge arc's
        radius hits pitch/le_te_ratio exactly; the leading-edge radius is
        then whatever that same shift produces at the other end (equal to
        the TE radius only for the pure-impulse, beta_inlet==beta_outlet
        case) -- see `_close_blade_le_te`'s docstring.

    Returns
    -------
    dict
        Flat, JSON-safe dict: ``pressure``/``suction`` (each {"x","y"},
        nondimensional by r* unless r_star given, in each surface's OWN
        local frame -- own vortex center at the origin -- plus
        ``mach_lines_inlet``/``mach_lines_outlet``, lists of
        (x0,y0,x1,y1) characteristic segments from the marching recursion).
        ``suction`` additionally carries ``extension_inlet``/
        ``extension_outlet`` (x0,y0,x1,y1): straight stubs continuing the
        suction surface at beta_inlet/beta_outlet out to pressure's far-
        corner x -- see the module docstring's "IMPORTANT" note for why
        these do NOT meet pressure (they bound the passage to the NEXT
        blade, offset by `pitch`, not this blade's own pressure side).
        Also: ``pitch`` (TN D-4421's GSTAR: the natural blade-to-blade
        spacing derived from the geometry, not a free input), ``chord``
        (CSTAR, the lower surface's own inlet-far-to-outlet-far distance),
        ``solidity`` (chord/pitch), and the design parameters echoed back.
    """
    manager = get_fluid_manager(fluid_name, backend, P0_rel, T0_rel, Q0_rel)

    V_inlet = _V_of_M(manager, M_inlet)
    V_outlet = _V_of_M(manager, M_outlet)
    V_lower = _V_of_M(manager, M_lower)
    V_upper = _V_of_M(manager, M_upper)

    nu_inlet = manager.nu_of_V(V_inlet)
    nu_outlet = manager.nu_of_V(V_outlet)
    nu_lower = manager.nu_of_V(V_lower)
    nu_upper = manager.nu_of_V(V_upper)

    beta_inlet_rad = np.radians(beta_inlet)
    beta_outlet_rad = _solve_beta_outlet(manager, V_inlet, V_outlet, beta_inlet_rad)
    beta_outlet = float(np.degrees(beta_outlet_rad))

    pressure = _surface(manager, nu_inlet, nu_outlet, nu_lower,
                         beta_inlet_rad, beta_outlet_rad, is_upper=False, num_points=num_points)
    suction = _surface(manager, nu_inlet, nu_outlet, nu_upper,
                        beta_inlet_rad, beta_outlet_rad, is_upper=True, num_points=num_points)

    # TN D-4421 "CALCULATE G*": extend the upper surface's own inlet-far
    # point along beta_inlet until it reaches the lower surface's own
    # inlet-far x-coordinate; the y-gap is the natural blade pitch.
    tan_bi = np.tan(beta_inlet_rad)
    y_last_i = suction["y_inlet_far"] + tan_bi * (pressure["x_inlet_far"] - suction["x_inlet_far"])
    pitch = pressure["y_inlet_far"] - y_last_i

    chord = float(np.hypot(pressure["x_outlet_far"] - pressure["x_inlet_far"],
                            pressure["y_outlet_far"] - pressure["y_inlet_far"]))
    solidity = chord / pitch if pitch != 0 else float("inf")

    scale = 1.0 if r_star is None else float(r_star)

    def _pack(surf):
        def _scale_lines(lines):
            return [(x0 * scale, y0 * scale, x1 * scale, y1 * scale) for x0, y0, x1, y1 in lines]
        return {
            "x": (surf["x"] * scale).tolist(),
            "y": (surf["y"] * scale).tolist(),
            "R_wall_star": surf["R_wall_star"] * scale,
            "mach_lines_inlet": _scale_lines(surf["mach_lines_inlet"]),
            "mach_lines_outlet": _scale_lines(surf["mach_lines_outlet"]),
        }

    pressure_packed = _pack(pressure)
    suction_packed = _pack(suction)

    # Beyond each transition arc's far corner, the flow is uniform (straight,
    # at beta_inlet/beta_outlet), so the surface itself asymptotes to a
    # straight line at that angle -- NOT a diagonal jump to the other
    # surface's corner (that earlier "sharp LE/TE" closure was wrong: pressure
    # and suction of the SAME blade extended this way are PARALLEL lines,
    # offset by exactly `pitch`, so they never meet -- this is the ordinary
    # cascade far-field picture, pressure of one blade running parallel to
    # suction of the next). Extend the suction surface's own corners by a
    # straight stub in the local flow direction, out to the pressure
    # surface's far-corner x, so the two curves can be compared directly;
    # the vertical gap between the stub's end and pressure's actual corner
    # is exactly `pitch` by construction.
    tan_bo = np.tan(beta_outlet_rad)
    x0i, y0i = suction["x_inlet_far"], suction["y_inlet_far"]
    x1i = pressure["x_inlet_far"]
    y1i = y0i + tan_bi * (x1i - x0i)
    x0o, y0o = suction["x_outlet_far"], suction["y_outlet_far"]
    x1o = pressure["x_outlet_far"]
    y1o = y0o + tan_bo * (x1o - x0o)
    suction_packed["extension_inlet"] = (x0i * scale, y0i * scale, x1i * scale, y1i * scale)
    suction_packed["extension_outlet"] = (x0o * scale, y0o * scale, x1o * scale, y1o * scale)

    # Lower/upper walls of the blade as separate OPEN curves (not merged
    # into the closed "blade" polygon above) -- lower is just `pressure`
    # again; upper is `suction` shifted by +pitch (the curve the blade
    # closure actually uses), which isn't otherwise exposed on its own
    # (the plain "suction" key is the UNshifted surface). Handy for
    # scripts that want to translate/compare the two walls independently
    # rather than only the pre-closed polygon.
    blade_lower = {"x": pressure_packed["x"], "y": pressure_packed["y"]}
    blade_upper = {
        "x": suction_packed["x"],
        "y": [y + pitch * scale for y in suction_packed["y"]],
    }

    # Solid single-blade cross-section, LE/TE closed by circular-arc
    # fillets: lower wall (pressure) -> TE arc -> TE straight stub ->
    # upper wall (suction+pitch, reversed) -> LE straight stub -> LE arc
    # (reversed) -> back to the lower wall's own start. See
    # `_close_blade_le_te`/`_arc_from_diameter` for the construction
    # (line built first via plain 2-line intersection, then the arc is a
    # semicircle on that line as its diameter -- exactly-determined, no
    # circle-through-two-tangent-points overconstraint).
    blade_x, blade_y, le_te_diag = _close_blade_le_te(
        blade_lower, blade_upper, pitch * scale, beta_inlet_rad, beta_outlet_rad,
        le_te_ratio=le_te_ratio, num_points=num_points,
    )

    return {
        "pressure": pressure_packed,
        "suction": suction_packed,
        "blade_lower": blade_lower,
        "blade_upper": blade_upper,
        "blade": {
            "x": blade_x.tolist(), "y": blade_y.tolist(),
            "closure": "circular_arc_le_te_fillet",
            "le_radius": le_te_diag["le_radius"],
            "te_radius": le_te_diag["te_radius"],
            "translate": le_te_diag["translate"],
        },
        "fluid_name": fluid_name,
        "M_inlet": float(M_inlet), "M_outlet": float(M_outlet),
        "M_lower": float(M_lower), "M_upper": float(M_upper),
        "beta_inlet": float(beta_inlet), "beta_outlet": float(beta_outlet),
        "nu_inlet_deg": float(np.degrees(nu_inlet)),
        "nu_outlet_deg": float(np.degrees(nu_outlet)),
        "nu_lower_deg": float(np.degrees(nu_lower)),
        "nu_upper_deg": float(np.degrees(nu_upper)),
        "pitch": pitch * scale,
        "chord": chord * scale,
        "solidity": solidity,
        "units": "m" if r_star is not None else "r*",
        "method": "goldman_1968_tn_d4421",
        "num_points": int(num_points),
    }
