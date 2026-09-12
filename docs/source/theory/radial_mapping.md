# Radial mapping of blade sections

The radial utilities wrap a planar blade onto an annulus. They act on the
coordinates of an existing stator or rotor section and then repeat the mapped
blade at equally spaced azimuths. They do not recompute the characteristic
field after the geometry changes.

## Exponential map

Let $\xi$ be the chordwise coordinate and $\eta$ the pitchwise coordinate of
the planar section. Define the mapping coefficient

```{math}
:label: eq-radial-map-coefficient
k_r=\frac{\ln(r_2/r_1)}{c_a}.
```

The logarithmic mapping used by `apply_conformal_mapping` is

```{math}
:label: eq-radial-map
r=r_1\exp[k_r(\xi-\xi_1)],\qquad
\varphi=\varphi_0+k_r(\eta-\eta_1),\qquad
X=r\cos\varphi,\qquad Y=r\sin\varphi.
```

Thus the station $\xi=\xi_1$ maps to $r_1$ and
$\xi=\xi_1+c_a$ maps to $r_2$. A line with constant $\eta$ maps to a radial
line, while a line with constant $\xi$ maps to a circle. General oblique
straight lines map to logarithmic spirals.

The coefficient has units of inverse length, so every exponential and angle
argument is dimensionless. The radial inputs must be positive, the chordwise
span must be nonzero, and $r_1\ne r_2$ is needed for a nondegenerate mapping.

## Reference point used by the wrappers

The convenience wrappers derive the reference from the supplied point cloud:

```{math}
:label: eq-radial-reference
\xi_1=\max\xi,\qquad
c_a=\min\xi-\max\xi,
```

with $\eta_1$ taken from the point having the largest $\xi$. In this
implementation $c_a$ is therefore a **signed, negative span**, rather than a
positive chord length. The maximum chordwise coordinate maps to $r_1$ and the
minimum to $r_2$. This is a geometric convention; the code does not detect
which extreme is the physical inlet.

For stator output, the wrappers use $\xi=y$ and $\eta=x$. For rotor output,
they use $\xi=x$ and $\eta=y$. A separate trailing-edge curve must use the
same reference as its main blade contour; deriving another reference from
the edge alone would place it inconsistently. `wrap_blade_radial` explicitly
shares the reference for this reason.

## Local angle preservation

Writing the map in complex form,
$X+\mathrm iY=r_1\exp\{k_r[(\xi-\xi_1)+\mathrm i(\eta-\eta_1)]
+\mathrm i\varphi_0\}$, gives a nonzero complex derivative wherever the map
is nondegenerate. More directly, its line element satisfies

```{math}
:label: eq-radial-metric
\mathrm dX^2+\mathrm dY^2
=(k_r r)^2(\mathrm d\xi^2+\mathrm d\eta^2).
```

Both planar coordinate directions have the same local scale factor $|k_r|r$,
so angles are preserved locally in the selected mapping frame. Lengths and
areas are not preserved: the scale changes with radius. Swapping the stator's
input axes changes orientation, but retains local angle magnitudes.

## Blade count and pitch

A planar pitch translation $\eta\mapsto\eta+t_b$ maps to an angular
translation

```{math}
:label: eq-radial-pitch
\Delta\varphi_\mathrm{mapped}=k_r t_b.
```

The wrappers instead take an independently specified blade count $N_b$ and
repeat the blade at increments $2\pi/N_b$. To reproduce the original planar
periodicity exactly, the magnitudes must satisfy

```{math}
:label: eq-radial-periodicity
|k_r|t_b=\frac{2\pi}{N_b}.
```

The sign of $k_r t_b$ determines the direction in which adjacent planar copies
are ordered around the annulus. With arbitrary radii and blade count, the
returned array remains a geometric placement, but its spacing need not match
the passage for which the planar section was designed.

```{admonition} Review note — pitch consistency
The wrappers do not enforce Eq. {eq}`eq-radial-periodicity`. Check the selected
radii, signed chordwise span, and blade count together, and inspect overlap
and minimum passage opening. An optional future diagnostic could report the
difference between the mapped pitch and the imposed angular spacing.
```

## Use with the flow model

Conformal mapping preserves geometric angles locally; it does not preserve
the compressible-flow solution, density field, or mass-flow distribution.
For a rotor, changing machine radius also changes blade speed and the relation
between relative total enthalpy and rothalpy. The mapped section should
therefore be interpreted as a candidate geometry for subsequent flow analysis.

Stator coordinates and mapping radii normally use millimeters. Rotor data must
first be dimensionalized using `r_star` or a consistent `scale` in the wrapper.
Choosing $r_1$ and $r_2$ alone does not resolve an unspecified length scale in
the input profile.

```{admonition} Figure placeholder — planar to radial geometry
Show a rectangular coordinate grid beside its annular image, marking the
two chordwise extremes, $r_1$, $r_2$, and $\varphi_0$. Include two adjacent
planar blade copies separated by $t_b$ and compare their mapped angular
spacing with $2\pi/N_b$.
```
