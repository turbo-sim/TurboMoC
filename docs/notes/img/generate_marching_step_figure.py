import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jaxprop as jxp

jxp.set_plot_options()

# Perfect-gas formulas, same convention as the validated NASA TN D-4421
# Table II case (gamma=1.4). Standalone re-implementation (not importing
# the module's private functions) purely to expose per-step intermediate
# quantities for this figure.
gamma = 1.4
gamm1 = (gamma - 1.0) / 2.0
gamp1 = (gamma + 1.0) / 2.0
perm = np.sqrt(gamp1 / gamm1)


def fofrs(X):
    arg1 = np.clip(2 * gamm1 / (X * X) - gamma, -1, 1)
    arg2 = np.clip(2 * gamp1 * X * X - gamma, -1, 1)
    return perm * np.arcsin(arg1) + np.arcsin(arg2)


def F_of_V_FN(V, FN, DELV):
    return 2.0 * V - (np.pi / 2.0) * (perm - 1.0) - 2.0 * (FN - 1.0) * DELV


def root_find(fofx):
    lo, hi = 1.0 / perm, 0.999999999
    flo = fofrs(lo) - fofx
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = fofrs(mid) - fofx
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def R_of_nu(nu):
    if nu == 0.0:
        return 1.0
    return root_find(F_of_V_FN(nu, 1.0, 0.0))


def mach_angle(R):
    return np.arcsin(np.sqrt(gamp1 * R * R - gamm1))


# --- Lower (pressure) transition arc, coarse resolution, keeping every
#     intermediate quantity for one highlighted step.
nu_far = np.radians(39.0)   # VIN
nu_wall = np.radians(18.0)  # VLOW
KMN = 5
delv = (nu_far - nu_wall) / KMN
R_wall = R_of_nu(nu_wall)
V = nu_far

PHIKP1 = -(V - nu_wall) + KMN * delv
UMKP1 = mach_angle(R_wall)
TXLO, TYLO = 0.0, R_wall
wall_x, wall_y = [TXLO], [TYLO]
steps = []

for KK in range(1, KMN + 1):
    K = (KMN + 1) - KK
    PHIK = PHIKP1 - delv
    nu_k = V - (K - 1) * delv
    TR = R_of_nu(nu_k)
    TX, TY = TR * np.sin(PHIK), TR * np.cos(PHIK)
    EMWK = np.tan(-PHIKP1)
    UMK = mach_angle(TR)
    EMK = -np.tan((PHIK + UMK + PHIKP1 + UMKP1) / 2.0)
    TEMP = TYLO - EMWK * TXLO
    TEMPP = TY - EMK * TX
    TEMPPP = EMK - EMWK
    TXLO_new = (TEMP - TEMPP) / TEMPPP
    TYLO_new = (EMK * TEMP - EMWK * TEMPP) / TEMPPP
    steps.append(dict(k=KK, P_prev=(TXLO, TYLO), Q=(TX, TY), P_new=(TXLO_new, TYLO_new),
                       EMWK=EMWK, EMK=EMK, R=TR, nu=nu_k))
    TXLO, TYLO = TXLO_new, TYLO_new
    PHIKP1, UMKP1 = PHIK, UMK
    wall_x.append(TXLO)
    wall_y.append(TYLO)


def march_fine(nu_far, nu_wall, n):
    KMN = n - 1
    delv = (nu_far - nu_wall) / KMN
    R_wall = R_of_nu(nu_wall)
    V = nu_far
    PHIKP1 = -(V - nu_wall) + KMN * delv
    UMKP1 = mach_angle(R_wall)
    TXLO, TYLO = 0.0, R_wall
    xs, ys = [TXLO], [TYLO]
    for KK in range(1, KMN + 1):
        K = (KMN + 1) - KK
        PHIK = PHIKP1 - delv
        nu_k = V - (K - 1) * delv
        TR = R_of_nu(nu_k)
        TX, TY = TR * np.sin(PHIK), TR * np.cos(PHIK)
        EMWK = np.tan(-PHIKP1)
        UMK = mach_angle(TR)
        EMK = -np.tan((PHIK + UMK + PHIKP1 + UMKP1) / 2.0)
        TEMP = TYLO - EMWK * TXLO
        TEMPP = TY - EMK * TX
        TEMPPP = EMK - EMWK
        TXLO = (TEMP - TEMPP) / TEMPPP
        TYLO = (EMK * TEMP - EMWK * TEMPP) / TEMPPP
        PHIKP1, UMKP1 = PHIK, UMK
        xs.append(TXLO)
        ys.append(TYLO)
    return np.array(xs), np.array(ys)


fine_x, fine_y = march_fine(nu_far, nu_wall, 200)


def reference_locus(nu_far, nu_wall, n):
    KMN = n - 1
    delv = (nu_far - nu_wall) / KMN
    V = nu_far
    PHI_start = -(V - nu_wall) + KMN * delv
    xs, ys = [], []
    for idx in range(KMN + 1):
        K = KMN + 1 - idx
        PHI = PHI_start - idx * delv
        nu_k = V - (K - 1) * delv
        TR = R_of_nu(nu_k)
        xs.append(TR * np.sin(PHI))
        ys.append(TR * np.cos(PHI))
    return np.array(xs), np.array(ys)


ref_x, ref_y = reference_locus(nu_far, nu_wall, 200)

step = steps[2]  # highlight step k=3 of 5
Pp = np.array(step['P_prev'])
Q = np.array(step['Q'])
Pn = np.array(step['P_new'])


def line_through(P, slope, half_len=0.25):
    d = np.array([1.0, slope])
    d = d / np.linalg.norm(d)
    return np.array([P - half_len * d, P + half_len * d])


mach_line = line_through(Q, step['EMK'], half_len=0.22)
wall_tangent = line_through(Pp, step['EMWK'], half_len=0.22)

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(fine_x, fine_y, color='0.75', lw=1.5, zorder=1, label='transition arc (fine, for context)')
ax.plot(ref_x, ref_y, color='#3b6fb6', lw=1.0, ls=(0, (4, 2)), zorder=1,
        label='major characteristic locus (step 3 formula, swept continuously)')
ax.plot(wall_x, wall_y, 'o', color='black', ms=5, zorder=4, mfc='white', mew=1.3)
ax.plot(0, 0, '+', color='black', ms=10, mew=1.5, zorder=5)
ax.annotate('vortex center O', (0, 0), textcoords='offset points', xytext=(8, -14), fontsize=10)

ax.plot([0, Q[0]], [0, Q[1]], color='#3b6fb6', lw=1.0, ls=':', zorder=2)

ax.plot(*mach_line.T, color='#d9822b', lw=2.2, zorder=3)
ax.plot(*wall_tangent.T, color='#3ea55e', lw=2.2, zorder=3)

ax.plot(*Pp, 'o', color='black', ms=9, zorder=6)
ax.plot(*Q, 'o', color='#3b6fb6', ms=9, zorder=6)
ax.plot(*Pn, '*', color='#c0392b', ms=18, zorder=7)

# Short point labels only -- the full step-by-step description lives in
# the numbered legend box below, keyed by these same colors/symbols.
ax.annotate(r'$P_{i-1}$', Pp, textcoords='offset points', xytext=(-10, 10),
            fontsize=12, ha='right', fontweight='bold')
ax.annotate(r'$Q_i$', Q, textcoords='offset points', xytext=(10, -14),
            fontsize=12, ha='left', color='#3b6fb6', fontweight='bold')
ax.annotate(r'$P_i$', Pn, textcoords='offset points', xytext=(-14, 6),
            fontsize=12, ha='right', color='#c0392b', fontweight='bold')

step_text = (
    r"$\bf{(1)}$ interpolate $\nu_i$ between $\nu_{wall}$ and $\nu_{far}$" "\n"
    r"$\bf{(2)}$ $R^*(\nu_i)=a^*/V(\nu_i)$,  $\mu_i=\arcsin(1/M_i)$    " "(dotted: O" r"$\to Q_i$" ")\n"
    r"$\bf{(3)}$ $Q_i=(R^*\sin\phi_i,\ R^*\cos\phi_i)$    (blue)" "\n"
    r"$\bf{(4)}$ Mach line through $Q_i$, slope from $\mu_i$    (orange)" "\n"
    r"$\bf{(5)}$ $P_i$ = Mach line $\cap$ tangent at $P_{i-1}$    (red star)"
)
ax.text(0.02, 0.02, step_text, transform=ax.transAxes, fontsize=10,
        va='bottom', ha='left', linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='0.5', lw=0.8))

ax.set_xlabel('x  (r*)')
ax.set_ylabel('y  (r*)')
ax.set_aspect('equal')
ax.set_title('One marching step of a transition arc (NASA TN D-4421)', fontsize=13)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
fig.canvas.draw()
ax.relim()
ax.autoscale_view()
x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()
x_pad = 0.08 * (x_hi - x_lo)
ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
ax.set_ylim(y_lo - 0.85 * (y_hi - y_lo), y_hi + 0.55 * (y_hi - y_lo))

fig.tight_layout()
out = r'c:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\python_scripts\method_of_characteristics\moc\notes\img\marching_step.png'
fig.savefig(out, dpi=160)
print('saved', out)
