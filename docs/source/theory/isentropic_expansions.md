# Isentropic expansions and choking

An isentrope connects the specified stagnation state to the states available
during an ideal expansion. Evaluating this path before constructing the nozzle
establishes the critical state, the outlet velocity, and whether the flow crosses
a saturation boundary.

## From pressure to velocity and area

For an outlet pressure $p_\mathrm{out}$, the equation of state and conservation
of total enthalpy give

```{math}
:label: eq-outlet-state
h_\mathrm{out}=h(p_\mathrm{out},s_0),\qquad
V_\mathrm{out}=\sqrt{2(h_0-h_\mathrm{out})},\qquad
M_\mathrm{out}=\frac{V_\mathrm{out}}{a(p_\mathrm{out},s_0)}.
```

The selected pressure must represent an expansion with an attainable
supersonic outlet state. In a uniform cross-section, continuity gives
$\dot m=\rho VA$. Consequently, the one-dimensional area ratio is

```{math}
:label: eq-area-ratio
\frac{A_\mathrm{out}}{A_t}
=\frac{\rho_t V_t}{\rho_\mathrm{out}V_\mathrm{out}}
=\frac{G_t}{G_\mathrm{out}}.
```

For a planar channel of fixed span, the area ratio equals the width ratio. The
relation provides a useful integral interpretation of the design. The
characteristic method additionally resolves the nonuniform cross-passage field
between the throat and the uniform outlet.

## Why a smooth maximum mass flux is sonic

Along a differentiable isentrope, the Gibbs relation gives $\mathrm dh=\mathrm
dp/\rho$, while the energy equation gives $V\,\mathrm dV=-\mathrm dp/\rho$.
Combining these with $\mathrm d\rho/\mathrm dp=1/a^2$ yields

```{math}
:label: eq-mass-flux-derivative
\frac{\mathrm dG}{\mathrm dp}
=V\frac{\mathrm d\rho}{\mathrm dp}+\rho\frac{\mathrm dV}{\mathrm dp}
=\frac{V}{a^2}-\frac{1}{V}
=\frac{M^2-1}{V}.
```

A smooth stationary point of $G(p)$ therefore occurs at $M=1$. In the usual
nozzle expansion it is the maximum mass flux and defines choking. This
derivation requires a well-defined derivative at the critical point.

At a saturation boundary, $G$ can remain continuous but have a corner because
its derivative changes discontinuously. If its maximum occurs at this corner,
the maximum-flow condition does not require either limiting Mach number to be
exactly one. The nitrogen flashing cases in {cite:t}`cioffiNonideal2026`
illustrate this distinction. The [initial-line construction](nozzle_design.md)
must account for the state reached downstream of that maximum.

The current critical-state routine first searches for a smooth sonic root and
checks its residual. If the apparent root is a jump, it locates the
saturation crossing and uses a nearby mixture state. This is a computational
procedure for the supported expansion regimes, rather than an exhaustive search
for every possible maximum of an arbitrary non-ideal isentrope.

## Mach number need not increase monotonically

Speed increases as static enthalpy decreases, but Mach number also depends on
the sound speed. Its differential is

```{math}
:label: eq-mach-change
\frac{\mathrm dM}{M}=\frac{\mathrm dV}{V}-\frac{\mathrm da}{a}.
```

An increase in sound speed can therefore reduce Mach number even while the flow
accelerates. Near-critical single-phase expansions can show smooth extrema in
$M(p)$. Wet-to-dry expansions can instead show a discontinuous decrease in Mach
number when the sound speed increases at the vapor saturation boundary.

For comparison, a calorically perfect gas with constant $\gamma$ obeys

```{math}
:label: eq-perfect-gas-pressure
\frac{p}{p_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)}.
```

Its monotonic pressure–Mach relation does not capture the extrema or jumps just
described. Using a locally evaluated $c_p/c_v$ in this perfect-gas formula does
not reproduce a general real-fluid isentrope.

```{figure} assets/mm_expansions.png
:name: fig-mm-expansions
:alt: Temperature–entropy paths and pressure–Mach curves for single-phase and wet-to-dry MM expansions.

Single-phase and wet-to-dry expansions of MM, extracted from Fig. 1 of
{cite:t}`cioffiNonideal2026`. The single-phase paths begin at twice the critical
pressure. The wet-to-dry family begins at saturated-liquid states. Read the
pressure plots from right to left as the expansion proceeds.
```

## Two-phase expansion paths

An expansion starting within the saturation dome can remain two-phase or reach
superheated vapor, depending on the fluid and inlet entropy. A subcooled-liquid
inlet first follows a single-phase path before entering the dome. The pressure,
velocity, and quality histories of these paths determine different nozzle
geometries even when the exit Mach number is the same.

For cyclopentane at approximately 2.51 bar stagnation pressure, the stator paper
finds that lower inlet quality requires a larger exit-to-throat area ratio and
a longer normalized nozzle for the same target Mach number. Changes in quality
have a greater effect than the temperature variations examined in that study
{cite:p}`cioffiStators2026`. These are case-specific trends associated with the
mixture density change, rather than a universal ordering for all fluids.

The guide uses cyclopentane to illustrate expansion within the dome, MM for
wet-to-dry behavior, and nitrogen for flashing. The latter two mechanisms are
developed in [saturation crossings](saturation_crossings.md).

## Selecting the design target

If $M(p)$ is non-monotonic or discontinuous, a requested Mach number can identify
multiple pressures on the same isentrope. Specifying pressure selects a state
more directly. For this reason `p_back` is the preferred input for
`design_nozzle`, and the minimum-length solver requires it to determine the
expansion-fan turning angle. When both pressure and Mach targets are supplied,
the API computes the design Mach number from the pressure target.

Here `p_back` defines the target for a newly generated contour. It does not ask
the package to solve the flow in a fixed nozzle at a changed operating pressure.
