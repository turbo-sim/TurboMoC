# Rotor design by the vortex-flow method

The rotor method turns a uniform supersonic inlet flow into a vortex field,
guides that field between circular surfaces, and returns it to uniform flow
at the outlet. Each surface contains an inlet transition arc, a circular arc,
and an outlet transition arc. The package implements the construction of
Goldman and Scullin, with real-fluid properties replacing the perfect-gas
relations in the original report {cite:p}`goldmanBlading1968`.

Unlike the nozzle construction, the rotor method starts from a supersonic inlet
and prescribes Mach numbers on both circular surfaces. Its objective is to
produce the required turning and passage shape, rather than to accelerate a
subsonic inlet through a nozzle throat.

## Relative frame and thermodynamic states

The inputs `P0_rel`, `T0_rel`, and `Q0_rel` specify relative stagnation
conditions. The thermodynamic path used in the implementation is

```{math}
:label: eq-rotor-relative-energy
h+\frac{W^2}{2}=h_{0,\mathrm{rel}},\qquad s=s_{0,\mathrm{rel}}.
```

This is a local blade-section approximation. In a rotating row, the conserved
rothalpy is $h+W^2/2-U^2/2$ under the usual adiabatic inviscid assumptions.
At constant machine radius, the blade speed $U$ is constant and this reduces
to Eq. {eq}`eq-rotor-relative-energy`. The package does not solve the general
rotating, three-dimensional flow field or convert a complete stage's absolute
inlet conditions into these relative inputs.

Four Mach numbers specify the inlet, outlet, lower circular surface, and upper
circular surface. The implementation finds each corresponding $W$ on the
relative isentrope and evaluates the generalized Prandtl–Meyer angle
$\nu(W)$ using Eq. {eq}`eq-generalized-pm`. Typically, the lower surface
decelerates the inlet flow and the upper surface accelerates it, while both
remain supersonic.

```{admonition} Review note — inversion of rotor Mach targets
The rotor helper `_V_of_M` uses a bisection update that assumes Mach number
increases with velocity over its search interval. The
[isentropic-expansion chapter](isentropic_expansions.md) shows why this need
not hold for non-ideal or saturation-crossing paths. Check branch selection
before applying the rotor method to those cases. Sharing the nozzle property
backend does not establish that every flashing or wet-to-dry rotor input is
supported by this inversion.
```

## Vortex field and circular surfaces

An irrotational circular flow satisfies

```{math}
:label: eq-vortex-law
WR=K.
```

The code uses $K=a_\mathrm{ref}R_\mathrm{ref}$, so its dimensionless vortex
radius is

```{math}
:label: eq-vortex-radius
\widehat R(\nu)=\frac{R}{R_\mathrm{ref}}
=\frac{a_\mathrm{ref}}{W(\nu)}.
```

For a smooth sonic reference, $a_\mathrm{ref}=W_\mathrm{ref}$ and
$R_\mathrm{ref}$ is the radius of the sonic-velocity streamline in the
underlying vortex field. Along either circular wall, $R$ is constant, so
$W$, $a$, and $M$ are constant. Across the passage, these quantities vary
with radius.

The free-vortex law constrains velocity times radius, not local Mach number
times radius. The report's relation $M^*R^*=1$ uses
$M^*=W/a_\mathrm{ref}$; it must not be read as $(W/a)R^*=1$.

Each circular surface is represented in normalized coordinates by
$x=\widehat R_w\sin\zeta$, $y=\widehat R_w\cos\zeta$. Its angular extent
is obtained after subtracting the turning supplied by the transition arcs
from the required total flow turning.

## Outlet angle from continuity

For equal blade spacing and span at inlet and outlet, conservation of axial
mass flux gives

```{math}
:label: eq-rotor-continuity
\rho_\mathrm{in}W_\mathrm{in}\cos\beta_\mathrm{in}
=\rho_\mathrm{out}W_\mathrm{out}\cos\beta_\mathrm{out}.
```

The implemented turning branch is

```{math}
:label: eq-rotor-outlet-angle
\beta_\mathrm{out}=-\cos^{-1}\!\left(
\frac{\rho_\mathrm{in}W_\mathrm{in}}
{\rho_\mathrm{out}W_\mathrm{out}}\cos\beta_\mathrm{in}\right).
```

The argument must lie in $[-1,1]$. A value outside that interval means that the
specified states and inlet angle cannot satisfy the assumed passage
continuity. Thus `beta_outlet` is derived rather than accepted as an
independent target. Continuity is a necessary condition; it does not by itself
guarantee a nonintersecting blade for arbitrary surface Mach numbers.

If inlet and outlet correspond to the same state on the selected branch, their
relative speeds are equal and the rotor has no static enthalpy drop in this
model, as in an impulse section. For unequal speeds,
$h_\mathrm{in}-h_\mathrm{out}=(W_\mathrm{out}^2-W_\mathrm{in}^2)/2$.
A stage degree of reaction additionally requires the stage enthalpy drop,
which this isolated section calculation does not supply.

## Marching a transition arc

The following equations describe the current numerical construction in its
local normalized frame. They make explicit the distinction between a point on
a reference characteristic and a point on the blade wall.

For each inlet or outlet transition, sample $\nu_i$ uniformly from the circular
wall value $\nu_w$ to the corresponding far-field value $\nu_f$. Define
$\sigma=-1$ for the lower surface and $\sigma=+1$ for the upper surface. The
reference angle and point are

```{math}
:label: eq-rotor-reference-point
\phi_i=\sigma(\nu_i-\nu_w),\qquad
\boldsymbol Q_i=\widehat R(\nu_i)
\begin{pmatrix}\sin\phi_i\\\cos\phi_i\end{pmatrix}.
```

The march begins at the wall point
$\boldsymbol P_0=(0,\widehat R_w)$, where $\phi_0=0$. The local flow
direction is $-\phi$. The previous wall segment and the current Mach segment
therefore have slopes

```{math}
:label: eq-rotor-marching-slopes
m_{w,i}=\tan(-\phi_{i-1}),\qquad
m_{c,i}=\tan\!\left[-\frac{\phi_i+\phi_{i-1}}{2}
+\sigma\frac{\mu_i+\mu_{i-1}}{2}\right].
```

The new wall point is the intersection of the line through
$\boldsymbol P_{i-1}$ with slope $m_{w,i}$ and the line through
$\boldsymbol Q_i$ with slope $m_{c,i}$:

```{math}
:label: eq-rotor-wall-march
x_i=\frac{y_{i-1}-m_{w,i}x_{i-1}-Q_{y,i}+m_{c,i}Q_{x,i}}
{m_{c,i}-m_{w,i}},\qquad
y_i=y_{i-1}+m_{w,i}(x_i-x_{i-1}).
```

The dependence on $\boldsymbol P_{i-1}$ makes this a genuine sequential
construction. Plotting only $\boldsymbol Q_i$ would plot the reference
characteristic, not the transition wall. Increasing `num_points` reduces the
turning represented by each straight segment, but extreme input combinations
can still require a geometry check.

```{admonition} Figure placeholder — rotor transition construction
Show a circular wall joined to an inlet transition arc. Mark the previous
wall point $P_{i-1}$, the reference point $Q_i$, the local Mach segment, and
their intersection $P_i$. A second panel should show the complete lower and
upper surfaces with inlet transitions, circular arcs, and outlet transitions.
Use separate line styles for the reference characteristic and the blade wall.
```

## Orienting the surfaces

The code uses the following circular-arc parameters, expressed in the guide's
notation:

| Surface | $\zeta_\mathrm{in}$ | $\zeta_\mathrm{out}$ |
| --- | --- | --- |
| Lower | $(\nu_\mathrm{in}-\nu_\mathrm{lower})-\beta_\mathrm{in}$ | $-(\nu_\mathrm{out}-\nu_\mathrm{lower}+\beta_\mathrm{out})$ |
| Upper | $(\nu_\mathrm{upper}-\nu_\mathrm{in})-\beta_\mathrm{in}$ | $-(\nu_\mathrm{upper}-\nu_\mathrm{out}+\beta_\mathrm{out})$ |

For a local transition point $(x,y)$, the inlet transformation is

```{math}
:label: eq-rotor-inlet-transform
X=x\cos\zeta+y\sin\zeta,\qquad
Y=-x\sin\zeta+y\cos\zeta.
```

The outlet transformation includes reflection:

```{math}
:label: eq-rotor-outlet-transform
X=-x\cos\zeta+y\sin\zeta,\qquad
Y=x\sin\zeta+y\cos\zeta.
```

These signs match the implementation's local frames. The assembled surface
traverses the inlet transition from the far field to the circular arc and then
the outlet transition back to uniform flow.

## Pitch and finite blade closure

Let $L_\mathrm{in}$ and $U_\mathrm{in}$ be the inlet endpoints of the lower
and upper passage surfaces. Extend the upper endpoint in the inlet flow
direction to the lower endpoint's axial station. The vertical gap is the pitch:

```{math}
:label: eq-rotor-pitch
t_b=y_{L,\mathrm{in}}-y_{U,\mathrm{in}}
-\tan\beta_\mathrm{in}(x_{L,\mathrm{in}}-x_{U,\mathrm{in}}).
```

The reported chord is the distance between the lower surface's inlet and
outlet endpoints, and solidity is $c/t_b$. The returned `pressure` and
`suction` curves describe the open passage construction. To form a solid blade,
the upper surface is translated by one pitch and then by a further offset
needed for the finite edge thickness.

The target trailing-edge radius is $r_\mathrm{TE}=t_b/\texttt{le\_te\_ratio}$.
At the trailing edge, a common tangent direction defines the unit normal
$\boldsymbol n=(-\sin\beta_\mathrm{out},\cos\beta_\mathrm{out})$.
If $D_0$ is the normal separation before the extra pitchwise translation,
the required translation is $(2r_\mathrm{TE}-D_0)/n_y$. The code then
intersects one surface tangent with the normal through the other endpoint and
uses that normal segment as the diameter of a semicircular closure. The same
translation determines the leading-edge radius; it is not specified
independently.

```{admonition} Review note — finite-thickness geometry
The edge-radius rule is a package construction, not a result prescribed by
the original report. `blade_upper` contains the pitch translation, whereas
the final `blade` also includes the additional thickness translation. Check
the neighboring passage opening after this adjustment before treating the
finite blade as an unchanged realization of the ideal vortex flow.
```

All local lengths are multiplied by `r_star` when that dimensional scale is
provided. The [rotor example](../examples/rotor_design.md) exports both the
assembled blade and the separate passage surfaces. Radial placement is a
subsequent [geometric mapping](radial_mapping.md).
