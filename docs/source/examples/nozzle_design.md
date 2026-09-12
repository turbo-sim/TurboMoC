# Design a two-phase nozzle

This example designs a symmetric nozzle for nitrogen starting inside the
liquid–vapor region. You will set the inlet state, solve the expansion,
inspect the characteristic mesh and flow properties, and export the two walls.

## Configure the expansion

| Input | Value | Meaning |
| --- | --- | --- |
| `fluid_name` | `"nitrogen"` | Fluid used by the thermodynamic model. |
| `P0` | `20e5` Pa | Stagnation pressure of 20 bar. |
| `Q0` | `0.5` | Inlet vapor mass fraction; half the mass is vapor. |
| `p_back` | `2e5` Pa | Target back pressure of 2 bar. |
| `solver` | `"conventional"` | Nozzle solver with a rounded expansion section. |

Specifying pressure and quality fixes the saturated inlet state, so no inlet
temperature is supplied. Quality is a mass fraction, not a volume fraction.
The default throat half-height is 0.01 m. The example checks convergence
before plotting or exporting the wall.

## Run and inspect the script

Run from the repository root after following the [installation instructions](../developer_guide.md#installation-for-developers):

```bash
poetry run python examples/nozzle_design.py
```

The code below is included directly from the example. The final display block
is omitted; the script opens the figures with `plt.show()` during normal runs.

```{literalinclude} ../../../examples/nozzle_design.py
:language: python
:start-at: from pathlib import Path
:end-before: Show figures during interactive runs
```

## Read the figures

```{figure} assets/nozzle_design/nozzle_contour.svg
:alt: Nozzle wall contour mirrored about the centerline; both axes are in meters.
:width: 95%

Nozzle wall contour mirrored about the centerline; both axes are in meters.
```

The wall opens downstream from the throat. The lower half is obtained by
reflecting the computed upper wall about `y = 0`. Markers identify the initial
sonic line and points in the rounded expansion section.

```{figure} assets/nozzle_design/characteristics_mesh.svg
:alt: Characteristic mesh in the upper half of the nozzle.
:width: 95%

Characteristic mesh in the upper half of the nozzle.
```

The colored lines show the characteristic segments and marching fronts used
to construct the solution. Their intersections carry the computed flow states.
This view shows how the solution extends from the initial sonic line into
the expansion and downstream wall region.

```{figure} assets/nozzle_design/axis_properties.svg
:alt: Pressure and Mach number along the nozzle centerline.
:width: 95%

Pressure and Mach number along the nozzle centerline.
```

Read pressure against the left axis in bar and Mach number against the right
axis. The pressure decrease accompanies supersonic acceleration; this
configuration gives an exit Mach number of approximately 2.11. The plotting
helper extends the last centerline state to the exit with a constant segment.

```{figure} assets/nozzle_design/wall_properties.svg
:alt: Pressure and Mach number along the computed nozzle wall.
:width: 95%

Pressure and Mach number along the computed nozzle wall.
```

This plot follows the wall instead of the centerline. Compare it with the
previous plot to see how the expansion develops at different positions across
the nozzle. Both use axial distance, rather than distance along the contour.

## Use the exported results

The script prints the absolute path to `examples/output/nozzle_design/`.
It saves four SVG figures and two CSV files:

| File | Contents |
| --- | --- |
| `wall_upper.csv` | Computed upper wall from throat to exit. |
| `wall_lower.csv` | Mirrored lower wall in the same point order. |

Both CSV headers are `x_m,y_m`; any upstream convergent section drawn by the
plotter is outside these exported `wall_final` coordinates. Rerunning the
script replaces the files in this directory.

Try changing `Q0` to another value between zero and one or changing
`p_back`, then compare the contour and property plots. Continue with the
[stator tutorial](stator_design.md) to turn the nozzle wall into a blade.
