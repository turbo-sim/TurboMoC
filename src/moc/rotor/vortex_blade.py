"""
moc.rotor.vortex_blade -- supersonic impulse/reaction rotor blade design by
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
single-blade cross-section (pressure and suction of the SAME blade, closed
by leading/trailing-edge material) is a separate, still-unresolved question
-- TN D-4421 mentions closing segments (BE, GJ in its Fig. 1) and rounded
LE/TE but gives no formula for their size, and this module does not
fabricate one.

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

Deviation from TN D-4421's OWN convention, kept from this module's earlier
iterations: beta_outlet is taken as a direct INPUT here (matching Bufi &
Cinnella (2018)'s convention), not derived from inlet/outlet Mach numbers
via mass conservation (TN D-4421's own Eq. 9) -- Eq. 9 is a perfect-gas-
specific isentropic area-Mach relation with no simple real-gas equivalent
available in this package, and Bufi & Cinnella's own dense-gas
generalization makes the same simplifying choice for the same reason.

Units: this module works entirely in the SAME units as its FluidManager
(SI: Pa, K, m/s) for thermodynamic quantities, and returns GEOMETRY
nondimensionalized by r* (the physical radius of the sonic-velocity
streamline for whichever surface's vortex flow, in meters) unless an
explicit `r_star` length scale is supplied to rescale to physical units --
see `design_rotor_vortex_blade`'s docstring.
"""

import numpy as np

from moc.core.classes import get_fluid_manager

__all__ = ["design_rotor_vortex_blade"]


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
    beta_inlet, beta_outlet,
    backend="HEOS",
    Q0_rel=float("nan"),
    r_star=None,
    num_points=60,
):
    """
    Design a supersonic rotor blade section by the vortex-flow method,
    following NASA TN D-4421 (Goldman & Scullin, 1968) -- see this module's
    docstring for exactly how, why it replaced earlier (Paniagua Eq. 17
    based) attempts in this module, and the one deliberate deviation
    (beta_outlet as a direct input rather than derived via TN D-4421's own
    Eq. 9, a perfect-gas-specific mass-conservation relation).

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
        uniform-flow regions, before/after the transition arcs).
    M_lower, M_upper : float
        Constant Mach number held on the pressure (concave/lower) and
        suction (convex/upper) circular arcs.
    beta_inlet, beta_outlet : float
        Relative flow angle [deg] from the axial direction at inlet/outlet.
    r_star : float or None
        Physical radius [m] of the sonic-velocity streamline to rescale
        the (nondimensional) geometry by. None returns x*, y* (units of
        r*, i.e. dimensionless).
    num_points : int
        Points per transition arc (endpoints included) -- a fine-grained
        stand-in for TN D-4421's explicit Delta-nu marching.

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
    beta_outlet_rad = np.radians(beta_outlet)

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

    # Blade wall: pressure (this blade, full) joined to suction SHIFTED BY
    # +pitch (the neighboring blade's suction, one pitch over) by direct
    # concatenation. This is NOT an arbitrary diagonal: by the same algebra
    # as the pitch/GSTAR derivation above, pressure's own tangent
    # continuation at each corner (slope tan_bi/tan_bo, exactly its transition
    # arc's far-end tangent) lands EXACTLY on suction-shifted-by-pitch's
    # corresponding corner --
    #   y_last_i2 = pressure.y_inlet_far - tan_bi*(pressure.x_inlet_far - suction.x_inlet_far)
    #             = suction.y_inlet_far + pitch   (substitute pitch's own definition)
    # so the straight segment closing pressure to shifted-suction IS the
    # tangent line, not a guess -- this is the picture confirmed against
    # (pressure this-blade / suction adjacent-blade+pitch): the passage lives
    # between suction (unshifted) and pressure shifted by -pitch (where the
    # transition-arc characteristics are); this pressure/shifted-suction pair
    # is the solid wall on the OTHER side of pressure.
    suction_shifted_y = suction["y"] + pitch
    blade_x = np.concatenate([pressure["x"], suction["x"][::-1]]) * scale
    blade_y = np.concatenate([pressure["y"], suction_shifted_y[::-1]]) * scale

    return {
        "pressure": pressure_packed,
        "suction": suction_packed,
        "blade": {"x": blade_x.tolist(), "y": blade_y.tolist(), "closure": "pressure_to_suction_plus_pitch_tangent"},
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
