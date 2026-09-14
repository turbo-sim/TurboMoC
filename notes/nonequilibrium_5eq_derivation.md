# Steady 2D (x,y) MOC for the single-velocity 5-equation nonequilibrium model

This is the derivation backing `moc/src/moc/core/nonequilibrium_5eq_fast.py`,
written down properly instead of existing only in chat history. Everything
here should be checked against the actual code, not assumed from this
document -- if they disagree, one of them is wrong and needs fixing.

Status: Sections 1-4 (governing equations through the interior-point
compatibility relation) are re-derived from first principles and cross-
checked against `moc.core.classes.jitted_moc_predictor_algebra` (exact
match). Section 5 (pathline/source treatment) matches the implementation.
**Section 6 (open failure mode) is NOT resolved** -- documented here so the
next debugging pass has a written target instead of blind trial-and-error.

---

## 1. Governing equations

Steady, 2D, planar (delta=0) mixture mass + momentum (identical in form to
single-phase Euler -- interphase mass/heat transfer moves mass *between*
phases, not out of the mixture, and there is no external force):

```
d(rho u)/dx + d(rho v)/dy = 0
rho (u du/dx + v du/dy) = -dp/dx
rho (u dv/dx + v dv/dy) = -dp/dy
```

closed by the **frozen** acoustic relation `dp = a_mix^2 drho` (composition
alpha_g, h_g, h_l held fixed -- this is exactly the Wallis mixture sound
speed already used everywhere in the codebase,
`moc_unsteady.nonequilibrium.wallis_sound_speed`).

Mixture stagnation enthalpy is conserved along streamlines (adiabatic,
frictionless, interphase exchange is internal to the mixture):

```
h0 = h_mix(p, alpha_g, h_g, h_l) + (u^2+v^2)/2 = const along a streamline
```

Phase transport (steady form of Staedtke's mass/heat exchange, along the
streamline, d/ds = (1/V) D/Dt):

```
V d(alpha_g)/ds ~= sigma_Mg / rho_g            (leading order)
V d(h_g)/ds = [sigma_Mg*(h_ex-h_g) + Q_g] / (alpha_g rho_g)
V d(h_l)/ds = [-sigma_Mg*(h_ex-h_l) + Q_l] / (alpha_l rho_l)
```

with `sigma_Mg, Q_g, Q_l` from `moc_unsteady.nonequilibrium.
five_eq_source_rates` (TA-BD interfacial area + Ranz-Marshall/energy-jump
closure), unmodified.

## 2. Characteristic directions

Write the state as `(rho, u, v)` and use `dp = a^2 drho` (a = a_mix) to
eliminate p from the momentum equations:

```
u rho_x + v rho_y + rho u_x + rho v_y = 0
u u_x + v u_y + (a^2/rho) rho_x = 0
u v_x + v v_y + (a^2/rho) rho_y = 0
```

In matrix form `A q_x + B q_y = 0`, `q = (rho, u, v)^T`:

```
A = [[u, rho, 0], [a^2/rho, u, 0], [0, 0, u]]
B = [[v, 0, rho], [0, v, 0], [a^2/rho, 0, v]]
```

Characteristics are `lambda = dy/dx` satisfying `det(B - lambda A) = 0`.
Let `w = v - lambda u`:

```
det(B - lambda A) = w [w^2 - a^2(1+lambda^2)]
```

**First factor**, `w = 0`:

```
lambda = v/u = tan(theta)          -- the STREAMLINE, carries (alpha_g, h_g, h_l)
```

**Second factor**, `w^2 = a^2(1+lambda^2)`, expands to a quadratic in
lambda:

```
lambda^2 (u^2-a^2) - 2 lambda uv + (v^2-a^2) = 0
lambda = [uv +- a^2 sqrt(M^2-1)] / (u^2-a^2)
```

which is algebraically the same statement as the textbook result
`lambda = tan(theta +- mu)`, `mu = asin(a/V)` -- the two **Mach lines**
(C+, C-). No assumption of irrotationality or isentropy was used anywhere
in this section; it holds for any local `a`, including the two-phase
Wallis sound speed.

**Code check**: `_theta_mu` and the `lamP = tan(theta+mu)`, `lamM =
tan(theta-mu)` lines in `nonequilibrium_5eq_fast.py` match this exactly.

## 3. Compatibility relation along the Mach lines

The left-null-vector of `(B - lambda A)` at each Mach-line root gives the
compatibility relation. Skipping the by-hand null-vector algebra (routine
but long), the standard result (matches Zucrow & Hoffman Vol. 2, the
general/non-isentropic MOC unit process) is:

```
(u^2-a^2) du + (2uv - (u^2-a^2) lambda) dv = S dx      along dy/dx = lambda
```

with `S = 0` for planar flow with no external force (our case -- the
two-phase source terms act on composition/enthalpy via the pathline
equation, Section 5, NOT as a direct momentum source, so they do not
enter S).

**This is exactly** `Q, R1` in `moc.core.classes.jitted_moc_predictor_algebra`
(`Q = u^2-a^2`, `R1 = 2uv - Q*lambda`), confirming that reusing that
function (feeding it `a = a_mix` instead of a single-phase `a(rho,s)`) is
mathematically justified, not a shortcut. **This was checked once, in
chat, and is now checked here in writing** -- re-verify by hand against
this document if the interior-point code ever changes.

Discretized (finite-difference) form, foot point at `(x_foot, u_foot,
v_foot)`:

```
Q u_new + R1 v_new = S*(x_new - x_foot) + Q*u_foot + R1*v_foot =: T
```

Two such equations (C+ foot P, C- foot M) solve for `(u_new, v_new)` at
an interior point; `jitted_moc_predictor_algebra` does exactly this via
Cramer's rule.

**Convention** (confirmed by debugging, see git history / chat): C+ (the
`+mu` family) must be the foot with the LOWER y; C- (`-mu`) the foot with
the HIGHER y, so the two lines intersect DOWNSTREAM of both feet. Getting
this backwards for the wall unit process was a real, confirmed bug (fixed
by comparing against `classes.py`'s `Inv_Wall_pnt`, which uses `lamP` for
exactly this construction).

## 4. Wall point (one Mach line + prescribed boundary condition)

At the wall, `theta` is prescribed (design choice: the local wall tangent
angle), not solved for. Only ONE Mach line is usable (C+, from the
interior neighbor below): with `u = V cos(theta)`, `v = V sin(theta)`,
theta fixed, the single compatibility equation becomes:

```
Q u + R1 v = T   =>   V (Q cos(theta) + R1 sin(theta)) = T   =>   V = T / (Q cos(theta) + R1 sin(theta))
```

**Sanity check (Prandtl-Meyer limit)**: along C+, the exact single-phase
invariant is `theta - nu(V) = const`, so an expanding wall (theta
increasing) must give nu increasing, i.e. V increasing (flow accelerates
turning outward) -- confirmed directly against the code's output (V and M
increase, p decreases, matching Prandtl-Meyer expansion physics). This
part of the wall construction is verified correct.

## 5. Pathline (alpha_g, h_g, h_l) update

Implemented as `_pathline_step` (mass-conservative mixing, NOT naive
Euler -- see that function's docstring for why the naive version blows up
as either phase's mass -> 0). Key point verified this session: `alpha_g`
is a **volume** fraction (`alpha_g = mass_g/rho_g`), and since vapor
density is often ~100x smaller than liquid density, `alpha_g -> 1` does
**not** mean the liquid is gone by mass -- checked directly, `alpha_g =
0.999` corresponded to only ~88% mass quality in the cyclopentane case.
The correct "fully evaporated" gate is on **quality** (mass fraction),
not `alpha_g`.

## 6. OPEN, UNRESOLVED: geometry blowup when the wall angle decreases

**Symptom**: in a kernel (theta increasing) + turning-back (theta
decreasing to 0, standard nozzle "straightening" region) design, the
kernel segment is stable and validated (M, p, alpha_g all smooth and
physical over 100+ steps). The turning-back segment reliably blows up:
interior ray x,y positions diverge to O(1e6) mm within ~50-100 steps of
theta starting to decrease.

**Ruled out** (tested directly, not assumed):
- Two different wall-shape formulas (rough tangent integration, and an
  exact closed-form second circular arc) give near-identical blowups ->
  not a wall-geometry-formula bug.
- Disabling the two-phase source terms entirely (`mass_transfer=False`,
  frozen composition) does not prevent the blowup -> not a source-term/
  closure bug.

**Confirmed by direct instrumentation** (kernel THETA_MAX=6deg, N_KERNEL=60,
turning-back R_ARC2=3*R_ARC, N_TURN=100, cyclopentane case): `min|lamM-lamP|`
across the ray pairs is healthy (~1.6-1.8) through step 74, then decreases
smoothly and monotonically once the turning-back segment is well underway:

```
step107: min|lamM-lamP| = 1.271
step120: min|lamM-lamP| = 0.980
step130: min|lamM-lamP| = 0.711
step140: min|lamM-lamP| = 0.339
step144: min|lamM-lamP| = 0.161   (still falling, run stopped here)
```

heading to zero -- confirms the near-parallel-characteristics diagnosis.
Also notable: wall pressure is ALREADY unphysically escalating (132 -> 1629
kPa over the same steps) well before the denominator actually reaches
zero -- the 2x2 solve's conditioning degrades smoothly as the determinant
shrinks, so the corruption is gradual, not a sudden cliff at the exact
singular point.

**Still not done**: WHY the denominator trends to zero specifically once
theta starts decreasing has not been explained, only confirmed to happen.
Candidate mechanism (not yet checked): the outermost ray pair's own
(theta, mu) may not track the prescribed wall theta schedule closely
enough, letting their C+/C- slopes drift toward each other over many
steps -- structurally the same class of issue as the earlier (separately
diagnosed, never fully resolved) narrow-fan-collapse problem from earlier
in the session, but triggered by the turning-back schedule specifically
rather than initial fan narrowness. Next actual step: print each ray's
own (theta, mu, lamP, lamM) individually (not just the pairwise minimum)
through this range to see whether it's the wall-adjacent pair specifically
that's converging, or something happening further into the interior net.
