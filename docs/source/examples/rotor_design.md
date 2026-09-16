# Design a two-phase vortex-flow rotor

This example constructs a supersonic rotor blade using the vortex-flow
design function. You will specify a relative inlet state and Mach numbers,
inspect the passage surfaces and assembled blade, and export dimensional
coordinates.

The rotor is an independent design example. It does not read the stator
output or match a complete turbine stage.

## Configure the relative flow

The inlet is a saturated nitrogen mixture at a relative stagnation pressure
of 20 bar and vapor mass fraction of 0.5. The script obtains its saturation
temperature through `jaxprop`, because the rotor API also requires a
temperature argument.

| Input | Value | Meaning |
| --- | --- | --- |
| `M_inlet` | `1.6` | Relative Mach number upstream of the blade. |
| `M_outlet` | `1.2` | Relative Mach number downstream of the blade. |
| `M_lower` | `1.1` | Mach number on the pressure-side constant-Mach arc. |
| `M_upper` | `2.0` | Mach number on the suction-side constant-Mach arc. |
| `beta_inlet` | `5.0` degrees | Relative inlet flow angle from the axial direction. |
| `r_star` | `0.01` m | Sonic-streamline radius used to scale the geometry. |
| `num_points` | `60` | Number of points per transition arc. |

The solver derives the outlet flow angle from mass-flux continuity.
The wall-arc Mach numbers and the upstream/downstream Mach numbers control
different parts of the construction. The inlet and outlet Mach numbers
differ here, so the flow is not a constant-Mach impulse example.

## Run and inspect the script

Run from the repository root after following the [source installation instructions](../getting_started.md#install-from-source-with-poetry):

```bash
poetry run python examples/rotor_design.py
```

The code below is included directly from the example. The final display block
is omitted; the script opens the figures with `plt.show()` during normal runs.

```{literalinclude} ../../../examples/rotor_design.py
:language: python
:start-at: from pathlib import Path
:end-before: Show figures during interactive runs
```

## Read the figures

```{figure} assets/rotor_design/rotor_blade.svg
:alt: Assembled rotor blade with rounded leading and trailing edges, in meters.
:width: 95%

Assembled rotor blade with rounded leading and trailing edges, in meters.
```

The filled region is the solid blade cross-section. The construction positions
the two walls and joins them using straight segments and circular edge arcs.
This is the geometry exported in `blade.csv`.

For these settings, the printed pitch is approximately 0.004753 m and the
chord is 0.012298 m, giving a solidity of approximately 2.5875. Solidity is
chord divided by pitch.

```{figure} assets/rotor_design/rotor_passage.svg
:alt: Pressure and suction passage surfaces with transition-arc junctions marked.
:width: 95%

Pressure and suction passage surfaces with transition-arc junctions marked.
```

This view shows the two flow-design surfaces before the blade closure.
The hollow circles mark the junctions between the transition regions and
constant-Mach circular arcs. The surfaces are exported in their own local
frames; they should not be concatenated directly to form the solid blade
shown above.

## Use the exported results

The script prints the absolute path to `examples/output/rotor_design/`.
It saves two SVG figures and three coordinate files, all with `x_m,y_m` headers:

| File | Contents |
| --- | --- |
| `blade.csv` | Assembled blade contour, including edge closure. |
| `pressure_local.csv` | Pressure-side surface in its local frame. |
| `suction_local.csv` | Suction-side surface in its local frame. |

Use `blade.csv` when you need the complete profile. The local surface files
are useful for examining the flow-design construction.

Try changing `r_star` to scale the blade. With the flow inputs unchanged,
the lengths scale while solidity stays the same. Increasing `num_points`
gives more points along each transition arc. Changing the Mach numbers or
inlet angle changes the underlying flow design and derived outlet angle.
