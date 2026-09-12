# Nozzle design

The nozzle solver constructs the divergent section from an initial supersonic
line. An expansion kernel first establishes the required downstream state. A
turning section then returns the flow toward the nozzle axis and completes the
wall. The conventional and minimum-length solvers share the thermodynamic
closure and characteristic point calculations, but use different expansion
geometries.

## Inputs and geometric scale

Stagnation pressure and either temperature or quality define $h_0,s_0$. The exit
pressure defines the target state through Eq. {eq}`eq-outlet-state`. The throat
half-height $y_t$ sets a length scale; the full throat width is $2y_t$. At fixed
thermodynamic conditions and dimensionless geometric ratios, uniformly scaling
all lengths scales the inviscid contour without changing its dimensionless
flow field.

The conventional method also uses a throat radius $R_t$ in its curved initial
line and a downstream arc radius $R_d$ in the prescribed expansion wall. These
are the API arguments `rho_t` and `rho_d`. They are distinct construction
parameters and should not be interchanged merely because both have units of
length.

## Initial line for a smooth sonic transition

The Sauer construction approximates transonic flow near a rounded throat by a
small-perturbation expansion {cite:p}`sauerNozzles1947`. The implementation's
planar working relations can be written using

```{math}
:label: eq-sauer-coefficients
k_s=\frac{1}{\sqrt{(\gamma_t+1)R_t y_t}},\qquad
\epsilon=-\frac{y_t}{6}\sqrt{\frac{(\gamma_t+1)y_t}{R_t}},\qquad
\widehat x=-\frac{(\gamma_t+1)k_s y^2}{6}.
```

The initial-line coordinates are $x=\widehat x-\epsilon$. Before this axial
shift, the velocity perturbations are evaluated as

```{math}
:label: eq-sauer-velocity
\frac{u}{a_t^\dagger}=1+k_s\widehat x
+\frac{(\gamma_t+1)k_s^2y^2}{2},\qquad
\frac{v}{a_t^\dagger}=(\gamma_t+1)k_s^2y\widehat x
+\frac{(\gamma_t+1)^2k_s^3y^3}{6}.
```

Here $\gamma_t$ is the local $c_p/c_v$ used by the code and
$a_t^\dagger=1.001a_t$ is its numerical offset from the sonic reference speed.
This offset starts the march just into the supersonic region. The thermodynamic
properties are then obtained from the real-fluid closure.

```{admonition} Review note — real-fluid Sauer coefficients
The equations above describe `InternalPoint.SauerMain`. The use of local
$c_p/c_v$ in coefficients inherited from a perfect-gas perturbation analysis
needs to be checked against the intended real-fluid derivation. In general,
isentropic sound-speed derivatives are not determined by $c_p/c_v$ alone.
The initialization should therefore not be described as an exact real-fluid
transonic solution. Check consistency of the two curvature parameters when
matching the initial line to the downstream wall. The Sauer report is from
1947; the stator manuscript's reference year of 1992 should be corrected.
```

## Flat initial front

When choking occurs through a discontinuous flashing transition, an expansion
about $M=1$ cannot represent the initial state. The alternative is a straight
line at $x=0$ with uniform downstream velocity and $v=0$. The
[saturation-crossing chapter](saturation_crossings.md) explains its physical
basis.

The public API selects this flat front by default; `use_true_sauer_line=True`
selects the curved construction. The flat-front option can also initialize a
smooth-sonic case. In the conventional solver its velocity offset is larger
when the reference Mach number is very close to unity and rapidly decreases
away from that singular region. This is numerical regularization, not an
additional physical phase-change model. A physically discontinuous flashing
state should not be initialized using a sonic perturbation merely by changing
the option.

## Conventional rounded-arc nozzle

The initial-value region is built by intersecting characteristics from adjacent
initial-line points and imposing symmetry at the axis. Repeating this process
produces the front that seeds the expansion kernel.

The upper kernel wall follows a circular arc. With its tangent horizontal at
the throat, its geometry has the form

```{math}
:label: eq-kernel-arc
x_w=x_t+R_d\sin\tau,\qquad
y_w=y_t+R_d(1-\cos\tau),\qquad \theta_w=\tau.
```

Increasing $\tau$ turns the flow away from the axis. At each wall point, the
wall compatibility condition determines the state; interior and axis points
complete the new characteristic front. The conventional solver adjusts its
angular increment as the selected exit target is approached and interpolates
the final front when a step passes the target. With a pressure target, the
stopping criterion uses pressure, avoiding ambiguity in a non-monotonic
Mach-number history.

The retained terminal construction line carries the information needed to
recover the turning wall. Up to this point the kernel wall has been prescribed;
the downstream wall is determined by the desired aligned flow.

## Wall recovery by mass conservation

For a curve $\mathcal C$ traversed from the axis toward the upper wall, the
flow crossing it per unit span is

```{math}
:label: eq-curve-mass-flow
\dot m'(\mathcal C)=\int_\mathcal C\rho(u\,\mathrm dy-v\,\mathrm dx).
```

The code integrates this flux cumulatively along the retained construction
line using segment-averaged density and velocity. From each point $P$, it
projects a straight characteristic segment toward a wall point $W$:

```{math}
:label: eq-turning-wall
\boldsymbol x_W=\boldsymbol x_P
+\ell_P\begin{pmatrix}\cos(\theta_P+\mu_P)\\\sin(\theta_P+\mu_P)\end{pmatrix},
\qquad
\ell_P=\frac{\dot m'_P}{\rho_P V_P\sin\mu_P}.
```

The denominator is the mass flux normal to a segment inclined at the Mach
angle to the velocity. Equating the flux through this segment to the cumulative
incoming flux determines its length. Repeating the operation recovers a wall
that encloses the corresponding streamtube. The implementation prepares the
terminal states for aligned outflow and copies the projected state to each wall
point. These discrete operations approximate the continuous turning region.

At the last point, the recovered wall encloses the full half-channel mass flow.
The exit height, exit direction, and conservation of mass provide complementary
checks on the constructed nozzle. A small target-Mach residual alone is not a
complete check of the wall geometry.

## Minimum-length nozzle

The minimum-length construction replaces the finite expansion arc with a sharp
corner. It concentrates the initial turning into a centered Prandtl–Meyer fan
and then supplies the wall needed to cancel the reflected waves. Generalized
Prandtl–Meyer functions have also been used for minimum-length nozzles with
temperature-dependent gas properties {cite:p}`zebbicheMinimumLength2007`.
The equations below explain the planar construction used by this package.

For a uniform, axis-aligned reference state with $\nu_\mathrm{ref}=0$, the
simple-wave relation at the upper corner is $\theta=\nu$. A characteristic
entering the domain from a fan state carries
$I_-=\theta+\nu=2\theta$. At its intersection with the symmetry axis,
$\theta=0$, so the corresponding axis value is $\nu_\mathrm{axis}=2\theta$.
Requiring the last fan ray to establish the desired exit state gives

```{math}
:label: eq-mln-half-turn
\theta_\mathrm{max}=\frac{\nu(V_\mathrm{out})}{2}.
```

More generally the numerator is the Prandtl–Meyer increment from the uniform
reference state. It is not necessary for that reference state to be sonic,
provided the entire construction remains on an admissible supersonic branch.

The API computes this angle from the outlet pressure and samples it with
`n_tau_mln` fan states. At each sample, the corner position remains fixed and
the state is found from $V=V(\theta)$. For the flat-front variant, the first ray
runs directly from the corner to the axis. Subsequent rays intersect the
previous front and its reflected characteristics, using the unit processes in
[the MoC chapter](method_of_characteristics.md). The turning contour is then
recovered by the same mass-conservation construction used for the conventional
nozzle.

```{admonition} Review note — curved-front minimum-length option
Equation {eq}`eq-mln-half-turn` assumes an initially uniform, aligned flow.
The API also uses the half-turning target with the curved Sauer option, where
the initial direction varies across the front and the wall process is
iterative. Check the required correction for that option before claiming that
it produces a strict minimum-length contour. The 2007 reference's metadata
and abstract were checked; its complete derivation was not accessible during
this draft. The detailed formulas here follow the current code and the planar
compatibility equations.
```

```{admonition} Figure placeholder — compare the two nozzle constructions
Place the rounded-arc and sharp-corner constructions side by side at the same
throat height and exit state. Label the initial line, kernel, terminal
characteristic, turning wall, symmetry axis, and uniform exit region. Show
$\tau$ distributed along the conventional arc and
$0\leq\theta\leq\theta_\mathrm{max}$ concentrated at the minimum-length corner.
```

## Resolution and resulting geometry

Initial-line resolution, angular sampling, and thermodynamic-table resolution
affect different parts of the calculation. Refining one cannot compensate for
an unsuitable initial state or a characteristic branch that becomes subsonic.
A useful refinement comparison examines outlet pressure, direction, mass flow,
and the complete wall contour at fixed physical inputs.

The returned `wall_final` is the divergent contour. Convergent inlet geometry
and stator closure are additional constructions. See the
[nozzle example](../examples/nozzle_design.md) for coordinate exports and
[stator design](stator_design.md) for the conversion into a blade passage.
