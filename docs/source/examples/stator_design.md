# Build a stator blade from a two-phase nozzle

This example solves a cyclopentane expansion using the stator design conditions
in {cite:t}`cioffiStators2026`, then uses its wall to construct a stator
blade profile. You will see how nozzle flow design and blade geometry fit
together, and export both the source wall and the fitted blade.

## Configure the nozzle and blade

The nozzle starts at 2.513 bar with a vapor mass fraction of 0.025 and expands
toward a back pressure of 0.95 bar using the conventional solver and the curved
Sauer initial line. Its throat half-height is 0.01 m. These are the paper's
operating conditions; the current package's blade parametrization differs from
the paper's control-point construction, as discussed in
[stator theory](../theory/stator_design.md).

| Blade input | Value | Meaning |
| --- | --- | --- |
| `metal_angle_in` | `0.0` degrees | Camberline angle at the convergent-section inlet. |
| `metal_angle_out` | `70.0` degrees | Camberline angle at the throat. |
| `r_trailing` | `0.5` mm | Radius used to construct the trailing edge. |
| `n_blades` | `2` | Number of repeated profiles drawn by the plotter. |
| `show_control_points` | `False` | Hide the spline control polygon in the figure. |

`parametrize_stator_blade` takes the nozzle wall in meters and constructs
the blade in millimeters. It adds the convergent geometry and pressure side,
then fits the blade contour with a B-spline. The plot repeats this geometry
at the calculated pitch; `n_blades` changes only the display.

## Run and inspect the script

Run from the repository root after following the [installation instructions](../developer_guide.md#installation-for-developers):

```bash
poetry run python examples/stator_design.py
```

The code below is included directly from the example. The final display block
is omitted; the script opens the figures with `plt.show()` during normal runs.

```{literalinclude} ../../../examples/stator_design.py
:language: python
:start-at: from pathlib import Path
:end-before: Show figures during interactive runs
```

## Read the figures

```{figure} assets/stator_design/stator_blade.svg
:alt: Two stator profiles repeated at the design pitch, with coordinates in millimeters.
:width: 95%

Two stator profiles repeated at the design pitch, with coordinates in millimeters.
```

The spacing between the two profiles forms the passage. The fitted contour
and separate trailing-edge curve show the blade geometry assembled from
the nozzle wall. The script prints the pitch of the resulting profile.
Set `show_control_points=True` to see the points controlling the spline fit.

```{figure} assets/stator_design/nozzle_contour.svg
:alt: Source nozzle used to construct the stator, with coordinates in meters.
:width: 95%

Source nozzle used to construct the stator, with coordinates in meters.
```

This is the flow-design input to the blade parametrization. Its wall provides
the divergent-section suction-side geometry before the blade construction
transforms and supplements it.

```{figure} assets/stator_design/characteristics_mesh.svg
:alt: Characteristic solution for the source nozzle.
:width: 95%

Characteristic solution for the source nozzle.
```

The mesh belongs to the nozzle calculation. The subsequent blade
parametrization constructs geometry from that result; it does not compute
a new flow solution through the complete stator passage.

```{figure} assets/stator_design/axis_properties.svg
:alt: Centerline pressure and Mach number for the source nozzle.
:width: 95%

Centerline pressure and Mach number for the source nozzle.
```

Pressure uses the left axis in bar and Mach number uses the right axis.
These curves describe the nozzle expansion that underlies the blade design.

```{figure} assets/stator_design/wall_properties.svg
:alt: Wall pressure and Mach number for the source nozzle.
:width: 95%

Wall pressure and Mach number for the source nozzle.
```

These are the flow states along the original nozzle wall. The horizontal
coordinate is the nozzle axial position in meters, before the geometry is
transformed into the stator profile.

## Use the exported results

Results are written to `examples/output/stator_design/`, and the script
prints the absolute path. Five SVG figures accompany these coordinate files:

| File | Contents | Units |
| --- | --- | --- |
| `nozzle_wall.csv` | Source nozzle wall, from throat to exit. | m |
| `blade_curve.csv` | Fitted B-spline contour. | mm |
| `suction.csv` | Constructed suction-side points. | mm |
| `pressure.csv` | Constructed pressure-side points. | mm |
| `trailing_edge.csv` | Separate trailing-edge curve. | mm |

Keep the trailing-edge curve with the fitted contour when using the exported
profile. CSV headers explicitly identify the units; the source nozzle wall
must be scaled before combining it with blade coordinates.

Try changing `metal_angle_out` or `r_trailing` while leaving the nozzle
inputs fixed. The blade geometry and pitch change, while the source nozzle
solution stays the same.
