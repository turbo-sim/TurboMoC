# Expansions across saturation boundaries

Crossing a saturation boundary changes the acoustic response of an equilibrium
fluid. The resulting change in Mach angle affects the characteristic net, and
can also change how the nozzle calculation is initialized. The relevant
distinction is whether the sonic transition is smooth and whether the flow
remains supersonic on both sides of the phase boundary
{cite:p}`cioffiNonideal2026`.

## Wet-to-dry expansion

A wet-to-dry expansion begins inside the saturation dome and reaches
superheated vapor. In the MM examples considered in the conference paper, the
sonic transition occurs within the mixture before the vapor saturation line is
reached. The flow remains supersonic across that boundary, so characteristic
marching remains applicable on both sides.

At the boundary, pressure and velocity can remain continuous while the sound
speed increases discontinuously. Since $M=V/a$, the Mach number decreases
abruptly and the Mach angle increases. This changes the directions of
characteristics that intersect the boundary and can cause neighboring
characteristics to cross.

The paper allows this crossover during the characteristic construction and
describes removing the resulting fold when processing the wall contour. The
construction resembles techniques used to handle weak shocks in characteristic
methods, but the equilibrium saturation crossing is not itself evidence of a
shock. A physical shock requires the appropriate conservation jump conditions
and entropy increase; a jump in $M$ caused solely by sound speed does not imply
either a pressure jump or entropy generation.

```{figure} assets/crossing_characteristics.png
:name: fig-crossing-characteristics
:alt: Wet-to-dry characteristic crossings and comparison of straight and Sauer initial lines for flashing flow.

Characteristic grids extracted from Fig. 3 of {cite:t}`cioffiNonideal2026`.
The left panels show local crossings near the wet-to-dry boundary. The right
panels compare nozzle contours initialized with a straight line and a Sauer
line for the low-subcooling nitrogen case.
```

```{admonition} Review note — folded wall and property interpolation
The conference manuscript does not specify a complete fold-removal algorithm.
The current conventional solver projects its retained characteristic line to
form `wall_final`; a dedicated implementation of the paper's fold correction
has not been identified. Locate the intended procedure and check its effect
on the exported contour. Also check interpolation around the phase boundary:
the present property tables use cubic splines, which do not represent an exact
jump unless the two sides are treated separately. A visually smooth contour
alone does not resolve these questions.
```

## Flashing with a smooth sonic transition

A flashing expansion starts from subcooled liquid and enters the two-phase
region at the liquid saturation boundary. The inlet degree of subcooling is

```{math}
:label: eq-subcooling
\Delta T_\mathrm{sub}=T_\mathrm{sat}(p_0)-T_0.
```

For nitrogen at $p_0=20$ bar and $\Delta T_\mathrm{sub}=2.5$ K, the conference
paper finds that the flow is still subsonic immediately after entering the
dome. The mass flux continues to increase until a smooth sonic state is
reached within the mixture. The saturation crossing therefore precedes the
MoC starting region, and a smooth-sonic initialization can be used.

The paper compares straight and Sauer initial lines for this particular case
and finds very similar nozzle contours. This comparison illustrates the small
influence of the initial-line choice in that case; it does not establish their
equivalence for all throat curvatures or inlet conditions.

## Flashing through a discontinuous critical state

At greater subcooling in the same nitrogen study, the maximum mass flux occurs
at the liquid saturation boundary. The sound speed falls enough that the
limiting Mach number changes from below unity to above unity without passing
smoothly through $M=1$. As shown by Eq. {eq}`eq-mass-flux-derivative`, a maximum
at a corner of the mass-flux curve is compatible with this behavior.

```{figure} assets/nitrogen_flashing.png
:name: fig-nitrogen-flashing
:alt: Nitrogen isentropes, mass-flux maxima, and Mach-number jumps for four degrees of inlet subcooling.

Nitrogen flashing expansions from 20 bar, extracted from Fig. 2 of
{cite:t}`cioffiNonideal2026`. The stars identify critical states. In the middle
panel the original notation $\rho v$ denotes the mass flux $G=\rho V$ used
here. The 2.5 K case reaches a smooth sonic state after the saturation crossing;
the other illustrated cases choke at the crossing.
```

The Sauer perturbation assumes a neighborhood of sonic flow, so it cannot be
continued through a finite subsonic-to-supersonic jump. The adopted
approximation is a flat initial front with the downstream mixture state. The
initial velocity is parallel to the nozzle axis, and the characteristic field
is built entirely on the supersonic side.

The straight front introduces a short region near the axis in which the flow
has not yet responded to the wall expansion. This accounts for the nearly
constant initial portion of the axial solution described in the conference
paper. Its extent and sensitivity to initialization should be assessed when
using a contour generated from this approximation.

## Applying these distinctions

| Expansion path | Critical transition | Initial construction |
| --- | --- | --- |
| Single-phase or within-dome expansion with a smooth sonic state | Continuous passage through $M=1$ | Smooth-sonic initialization; a flat approximation is also available |
| Wet-to-dry example with a supersonic vapor-boundary crossing | Smooth sonic state upstream of the crossing | Initialize upstream, then account for characteristic crossover |
| Low-subcooling flashing example | Entry into the mixture remains subsonic; choking follows | Initialize at the subsequent smooth sonic state |
| Flashing with choking at the liquid saturation boundary | Finite jump from $M<1$ to $M>1$ | Flat downstream front |

These categories describe the paths examined in the papers. If a candidate
path becomes subsonic within the region to be marched, the supersonic MoC
construction no longer applies there. Likewise, equilibrium geometry does not
resolve delayed flashing or phase slip. Such effects belong to the subsequent
flow analysis.

The [nozzle example](../examples/nozzle_design.md) uses the nitrogen flashing
conditions, while the [stator example](../examples/stator_design.md) uses
cyclopentane with a two-phase inlet.
