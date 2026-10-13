> Run the adjacent Python script without arguments. Its configuration path
> resolves relative to the script; this example writes to its adjacent `output/`.
> Package numerical callables require an explicit configuration path and contain
> no references to example YAML locations.

# 2D meshing of MoC stator blades

This example designs a planar nozzle with the method of characteristics (MoC),
parametrizes one stator blade, constructs a smooth periodic flow domain, and
creates a 2D Gmsh fluid mesh with optional inflation layers. It exports the mesh
as CGNS/ADF for downstream CFD tools and saves one styled mesh view
as PNG and SVG, with a selectable number of blades.

## Run and configure

From the repository root, install the package and its dependencies, then run:

```powershell
poetry install
poetry run python -m examples.gmsh_2D.run_mesh_generation
```

Gmsh is a regular package dependency. This example does not require the optional
`cad` or `fluent` extras: its CAD construction uses Gmsh, not CadQuery.

This example also uses the companion `barotropy` package for Matplotlib styling
and multi-format figure saving. If it is not installed, install the adjacent
checkout into the same environment:

```powershell
poetry run python -m pip install -e ../barotropy
```

Edit `examples/gmsh_2D/workflow_gmsh_stator_2D.yaml` to set the `nozzle`, `blade`, `passage`, `mesh`, and `workflow`
parameters. Lengths in the nozzle inputs are metres; blade geometry, passage
geometry, mesh sizes, and plots use millimetres. In the cascade frame, x is
pitchwise, y is axial, and flow proceeds toward decreasing y.

`main()` applies the publication plotting style and restores Matplotlib settings
when the run finishes. Construction exports use the formats, dimensions and DPI
under `workflow.geometry_figure`; mesh images have separate `mesh.matplotlib`
settings. The supplied construction defaults are PNG/SVG at 300 DPI.

The launcher resolves the YAML beside itself and writes to adjacent `output/`.
All execution controls are in the YAML; the runner needs no arguments:

| Setting | Purpose |
| --- | --- |
| `workflow.generate_mesh` | Generate the mesh; `false` stops after MoC and geometry |
| `workflow.geometry_figure.save` | Export the construction figure |
| `workflow.geometry_figure.show` | Open Matplotlib; close its window to continue |
| `workflow.geometry_figure.size` | Figure width and height in inches |
| `workflow.geometry_figure.dpi` | Construction-export resolution |
| `workflow.geometry_figure.formats` | Export extensions, e.g. `[.png, .svg, .pdf]` |
| `workflow.reporting.level` | Console and file logging level |
| `workflow.reporting.filename` | Log filename within the output folder |
| `mesh.show_gui` | Open Gmsh after generating and saving the mesh |

`mesh.show_gui: true` is sufficient to open Gmsh; there is no second runner flag.
Set it to `false` for unattended runs. It has no effect in geometry-only mode.

Programmatic runs use `turbo_moc.meshing.mesh_2d.main`, with explicit
output paths and YAML-controlled figure/viewer options. Shared implementations live in
`src/turbo_moc/`; the launcher contains no numerical code.

With the default YAML, outputs are written to the flat directory
`examples/gmsh_2D/output/`:

| Output | Contents |
| --- | --- |
| `stator_geometry.png` / `.svg` | Blade construction and flow-domain plots |
| `stator_mesh.png` / `.svg` | One mesh view with the configured blade count and optional physical-group labels |
| `stator_mesh.cgns` | Optional 2D CGNS/ADF mesh with named boundaries and periodic connectivity |
| `mesh_report.html` | Self-contained quality report with quantiles, histograms, and inflation diagnostics |
| `mesh_report.json` | Machine-readable summary |
| `mesh_cells.csv` | Per-cell metrics, original Gmsh tags, centroids, and reconstructed layer indices |
| `mesh_quality.png` / `.svg` | Exportable quality histograms |
| `mesh_generation.log` | Workflow progress, summaries, warnings and failure tracebacks |

`workflow.geometry_figure.save` controls construction exports. Mesh artifacts
require `workflow.generate_mesh: true`, with CGNS controlled by
`mesh.cgns.enabled` and the four report artifacts by `mesh.report.enabled`.
Changing these switches does not remove files left by earlier runs.

## Execution reporting

The runner reports numbered steps for configuration, the MoC design, blade
geometry and meshing. Mesh sizing, inflation and quality results appear in
aligned summaries, followed by export paths and a final completion section.
Messages are written to the console and the selected output folder's
`mesh_generation.log` as each
operation runs. The log uses UTF-8, flushes after every message and is replaced
on each run; it does not accumulate timestamped copies.

Set `workflow.reporting.level` and `workflow.reporting.filename` in the YAML. `DEBUG` includes detailed export and cleanup diagnostics;
`WARNING` shows only warnings and errors. Imported functions use the shared logger
configured by the entry point. The shared implementation is
[`turbo_moc.reporting`](../../src/turbo_moc/reporting.py), using Python's standard
`logging` module without extra dependencies.

The execution log records messages from these example scripts. MoC's native
console output is unchanged. Gmsh's native warnings and errors remain controlled
by `mesh.numerics.terminal_output` and `mesh.numerics.verbosity`; they are not redirected into this
log. There are no timing decorators, elapsed-time measurements or viewer timings.

## Nozzle and blade geometry

The shared [MoC entry point](../../src/turbo_moc/workflows/nozzle.py), `solve_nozzle`, calls
`turbo_moc.design_nozzle` afresh on every workflow run and validates its final
wall before blade construction. The default configuration uses the conventional
planar solver for Cyclopentane. No persistent MoC solution, dependency-signature
lookup or force/reuse switch is used. Solver progress is forwarded into the
workflow logger.

`parametrize_stator_blade` converts the nozzle wall to millimetres, builds the
converging suction-side construction curve, and transforms the divergent MoC
wall into the cascade frame. Reflection and pitch translation form the other
side; an upstream bridge, leading-edge closure, and circular trailing edge
complete the profile. The assembled contour is fitted with a fixed-endpoint
cubic B-spline. Gmsh receives that fitted blade spline's control points, degree,
and knots directly, with two exact quarter-circle trailing-edge arcs.

Pitch is computed from the divergent wall height, trailing-edge radius, and
outlet metal angle. The nominal divergent axial chord is the nozzle's
streamwise extent times the cosine of the outlet metal angle; the nominal
convergent axial chord is half that value. The nominal total axial chord can
therefore differ from the fitted blade's plotted axial span.

The blade-construction plot distinguishes the fitted contour, raw suction and
pressure samples, control polygon, and integrated convergent construction
curve. That construction curve forms the convergent suction surface; it is
not a mean-thickness camberline of the final fitted blade.

## Smooth flow domain and placement

`../../src/turbo_moc/geometry/passage_2d.py` constructs a reference with four joined sections:

1. A clamped cubic B-spline blends axial inlet flow to the inlet metal angle.
2. A convergent reference integrates a linear inlet-to-throat metal-angle law.
   A cubic interpolant passes through its samples and enforces the endpoint
   tangents used to join the neighboring sections.
3. A straight divergent reference follows the planar MoC symmetry-axis
   direction at the outlet metal angle, continuing to the blade's downstream
   axial extremum.
4. A clamped cubic B-spline blends the outlet metal angle back to axial flow.

Before the domain shift, the reference begins its convergent section at the
fitted contour's upstream axial extremum. Joins match tangent direction (G1);
curvature continuity is not imposed. Zero inlet metal angle gives a straight
upstream extension.

The periodic sides are this reference translated by minus/plus half pitch in
x. Straight pitch-wide caps connect their inlet and outlet endpoints. Gmsh
uses the same definition: three B-splines and one straight line per side,
with shared endpoint tags. It transfers the cubic extensions and convergent
interpolant directly rather than converting plotting samples into CAD edges.

| `passage` setting | Meaning |
| --- | --- |
| `inlet_chords` / `outlet_chords` | Cap distance from the blade's upstream/downstream axial extremum, divided by nominal total axial chord |
| `domain_shift_mode` | `manual` or `auto` |
| `domain_shift` | Manual pitchwise offset divided by pitch; positive moves the entire domain toward +x |
| `smooth.inlet_axial_tangent` | Inlet extension's axial endpoint handle length / extension axial length |
| `smooth.inlet_blade_tangent` | Inlet extension's blade-angle endpoint handle length / extension axial length |
| `smooth.outlet_blade_tangent` | Outlet extension's blade-angle endpoint handle length / extension axial length |
| `smooth.outlet_axial_tangent` | Outlet extension's axial endpoint handle length / extension axial length |
| `smooth.num_points` | Plot samples per section and convergent interpolant samples; does not prescribe mesh node counts |

Manual mode applies `domain_shift` as an absolute offset from the initial
reference. Auto mode ignores that value and searches within plus/minus half
a pitch for the rigid translation maximizing the smaller Euclidean
blade-to-periodic clearance. A coarse scan brackets a bounded scalar search.
Both modes translate the entire domain while leaving the blade fixed.

The plot reports the applied shift and the minimum clearance on each side.
Containment is checked before meshing. Clearance estimates use sampled curves;
increase sampling when inspecting very small gaps. Centering redistributes
available space and does not guarantee that a requested inflation thickness
fits between neighboring blades.

## Mesh sizing, curvature, and inflation

Gmsh meshes the smooth CAD curves in 1D before filling the fluid surface in 2D.
`mesh.numerics.curvature_elements` enables its curvature-based sizing on both the blade
and periodic boundaries. The value is the target number of elements per full
circle (2*pi radians). For local curvature kappa, the associated size is
approximately `2*pi / (curvature_elements * abs(kappa))`: tighter bends receive
smaller elements. Zero disables this curvature criterion.

The final local size combines curvature, point sizes, and an automatic core
sizing callback, bounded above by `numerics.domain_size`. Straight periodic portions
are governed by those other size controls. Plotting resolution does not fix
1D mesh resolution. See the [Gmsh sizing documentation](https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes).

`mesh.numerics`, immediately after `show_gui`, groups all Gmsh numerical
controls and the nested `inflation_layers` mapping. The sibling `matplotlib`,
`report`, and `cgns` mappings control visualization, diagnostics, and export.

| `mesh` parameter | Purpose |
| --- | --- |
| `numerics.blade_size` / `numerics.domain_size` | Tangential blade / far-field target element sizes [mm] |
| `numerics.curvature_elements` | Curvature resolution in elements per 2*pi radians |
| `numerics.algorithm` | 2D meshing algorithm; default 8 prepares triangles for recombination |
| `numerics.recombine` | Convert suitable fluid triangles to quadrilaterals |
| `numerics.recombination_algorithm` | Recombination method; default 1 is Blossom |
| `numerics.smoothing_steps` | Mesh smoothing iterations |
| `numerics.inflation_layers.enabled` | Enable quadrilateral wall layers |
| `numerics.inflation_layers.first_height` | First wall-normal layer height [mm] |
| `numerics.inflation_layers.growth_ratio` | Ratio between successive layer heights |
| `numerics.inflation_layers.thickness` | Maximum total layer thickness [mm] |
| `numerics.verbosity` / `numerics.terminal_output` | Gmsh logging level and console output |
| `show_gui` | Open the Gmsh viewer after mesh export |
| `matplotlib.num_blades` | Number of blades in the single exported view; set 1 for one passage |
| `matplotlib.annotate_boundaries` | Show physical-group labels for any blade count |
| `matplotlib.mesh_color` | Cell-edge color |
| `matplotlib.edge_color` | Blade and flow-domain boundary color |
| `matplotlib.domain_color` | Fluid-region fill color; default light grey |
| `matplotlib.dpi` | Figure, PNG, and rasterized SVG resolution; the same DPI is used throughout |
| `matplotlib.rasterize_mesh` | Rasterize dense cells/edges while retaining vector boundaries and text in SVG |

The `mesh.matplotlib` mapping controls one figure, always written as
`stator_mesh.png` and `stator_mesh.svg`. The current YAML selects three blades;
set `num_blades: 1` for one passage. Additional blades are centered,
pitch-translated copies of the actual mesh for visualization; they do not change
the CFD domain. Shared edges and periodic interfaces are drawn once.

Figure dimensions, thin line widths, Times New Roman fonts and font sizes are
fixed internally. All plot colors are fully opaque, including colors supplied
with an alpha channel. Mesh edges, boundary lines, and the fluid-region fill follow
`mesh_color`, `edge_color`, and `domain_color`; blade interiors are white. Axes, labels, annotations, and their leader lines
remain black independently of `edge_color`. Equal-axis scaling,
a complete axes box, and `$x$ coordinate (mm)` /
`$y$ coordinate (mm)` labels are always used. The interactive Gmsh viewer retains
its orange color scheme.

With annotations enabled, the view labels `blade`, `periodic_left`,
`periodic_right`, `inlet`, and `outlet` once each. Multi-blade views label a
central blade, the outer periodic sides and the combined inlet/outlet spans.
For a single blade, all labels sit outside the flow domain, with leader lines
to the corresponding boundaries and no background boxes.
Set `annotate_boundaries: false` to suppress every boundary annotation.

The mesh is exported only as CGNS; no native Gmsh mesh file is written.
`create_mesh` returns the PNG/SVG paths under `image_files["png"]` and
`image_files["svg"]`; `image_file` retains the PNG path for existing callers.
`mesh_file` contains the CGNS path when export is enabled. The runner needs no
additional plotting switches.
See [Gmsh's mesh color options](https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options).

Inflation uses Gmsh's 2D `BoundaryLayer` field. Layer heights grow from
`first_height` by `growth_ratio` until reaching the thickness limit; geometry
can restrict their extent. These settings do not specify a target y+.
Recombination may leave some triangles outside the wall layers.

### Automatic inflation-to-core transition

The independent inflation inputs are **first height, growth ratio, and maximum
total thickness**. The code calculates the nominal complete layers that fit:

```text
h_i = first_height * growth_ratio**(i - 1)
T_N = first_height * (growth_ratio**N - 1) / (growth_ratio - 1)
N   = floor(log(1 + thickness*(growth_ratio-1)/first_height) / log(growth_ratio))
```

The current inputs give 12 nominal layers, a stack thickness of 0.79161 mm,
and a last-layer height of 0.14860 mm. Gmsh receives the original three inputs;
they are not adjusted to force the last height to equal `numerics.blade_size`. Geometry
can shorten/merge local stacks, so the nominal count is not an exact layer label.

`numerics.blade_size` controls spacing **along** the blade; inflation height controls
spacing **normal** to it. Before 1D meshing, CAD curvature estimates local
tangential spacing. After 1D meshing, actual blade-edge lengths replace those
estimates for sizing the 2D core. Small trailing-edge cells therefore get a
different transition size from larger cells on flatter blade regions.

For local tangential spacing `s`, last height `h_last`, and internal area-growth
target `R = 1.2`, the core size at the end of the nominal stack is:

```text
last_layer_area = s * h_last
area_equivalent_size = sqrt(R * last_layer_area / shape_factor)
core_start_size = min(area_equivalent_size, R * h_last)
shape_factor   = 1 for recombined quads; sqrt(3)/4 for triangles
```

The area calculation is complemented by a normal-step cap: the first core
cells still share elongated inflation edges, so immediately adopting an
isotropic area-equivalent size can produce an excessive normal jump. Outside the
stack, the target size grows linearly with distance `d` from the blade:

```text
size(d) = min(domain_size,
              core_start_size + (sqrt(R)-1) * max(0, d-T_N))
```

The gradient gives roughly `R` area growth per successive regular core cell.
The distance to reach `numerics.domain_size` follows from this expression and is no
longer a user setting: **`refinement_distance` has been removed** (old YAML
entries are ignored). With `s = 0.5 mm`, the starting size is approximately
0.178 mm and the far-field distance is approximately 19.9 mm. Actual local
values vary with measured edge spacing. Without inflation, grading starts
from local blade spacing at the wall using the same internal core growth.

The callback preserves blade tangential sizing and finer Gmsh curvature/point
constraints. It wraps coordinates by one pitch and includes neighbouring
blade copies, so core targets agree at translated periodic points. The console
and report show the derived layer count, heights, core sizes, and transition
distances. No additional transition parameter is required in YAML.

`R` is an internal **sizing target**, not a guaranteed bound on generated cell
area ratios. Recombination, curvature, and truncated stacks affect real cell
areas. The quality report checks neighbouring areas and separately reports
ratios across the reconstructed inflation/core interface; that interface is
an estimate, as described below. Large outliers remain visible for inspection.

## Conformal periodic boundaries

The left boundary is the master and the right is the slave. Before mesh
generation, `gmsh.model.mesh.setPeriodic(1, right_curves, left_curves, transform)`
registers a translation of `(pitch, 0, 0)` for corresponding CAD sections.
Gmsh generates the slave 1D mesh as a periodic copy of the master. Matching
node positions and edge connectivity make the boundaries conformal after
translation; they remain separate boundaries of the single passage.
See [Gmsh's periodic API](https://gmsh.info/doc/texinfo/gmsh.html#gmsh_002fmodel_002fmesh_002fsetPeriodic).

The code checks conformity after 1D meshing and again after 2D meshing:

- All boundary nodes are covered by a bijective master/slave correspondence.
- Node coordinates satisfy the registered affine translation.
- Mapped slave edges exactly match master edge connectivity.
- Shared endpoints have consistent mappings across the four sections.

A missing or inconsistent map stops export. The final checks also require a
nonempty fluid mesh and positive scaled Jacobians. The named physical groups
are `blade`, `inlet`, `outlet`, `periodic_left`, `periodic_right`, and `fluid`.

## Mesh statistics and quality report

After meshing, `mesh_2d.compute_mesh_report(surface, boundaries, inflation=...)`
collects the active Gmsh mesh and computes geometric statistics in Python.
`write_mesh_report(report, output_dir)` writes an offline HTML report, JSON
summary, CSV cell data, and a histogram PNG. Reporting runs before Gmsh closes.
The HTML embeds its plots as SVG, so it can be opened or shared as one file.
The dictionaries returned by `create_mesh` include the summary under `report`
and the HTML filename under `report_file`.

The report includes:

- Used fluid nodes, triangle/quad counts, area, connected cell regions, and
  boundary lengths and edge counts. CAD control-point nodes are excluded.
- Equiangle skewness, centroid-based geometric orthogonality, non-orthogonality
  angles, Gmsh scaled Jacobians and signed inverse condition numbers.
- Shortest/longest edge, edge aspect ratio, cell area, equivalent size, interior
  angles, and adjacent-cell area ratios, including translated periodic neighbors.
- Inflation/core area-ratio quantiles at the reconstructed interface, and
  the automatic core-transition sizing parameters.
- Minimum, maximum, mean, standard deviation, and P01/P05/P25/P50/P75/P95/P99
  for each measure. Cell quantiles are unweighted. Separate tables distinguish
  triangles, quads, reconstructed inflation cells, and remaining cells.
- Histogram overlays, diagnostic tail counts, and the worst cells by skewness
  and orthogonality, with original Gmsh element tags and centroid coordinates.

Equiangle skewness compares corner angles with 60 degrees for triangles and
90 degrees for quads; zero is ideal. Geometric orthogonality is the smallest
cosine between an outward edge normal and the cell-centroid-to-edge-midpoint
or cell-centroid-to-neighbor-centroid vector; one is ideal. Periodic neighbor
centroids are translated into the local frame. This is an explicit geometric
measure, not a solver-specific composite orthogonal-quality definition. The
report lists definitions and references for interpreting the metrics.

Inflation cells do not have a documented public provenance label in the Gmsh
`BoundaryLayer` API. The report therefore **reconstructs an estimate** by
tracing opposite-edge quad columns from each blade mesh edge. Heights must
match the configured geometric growth within tolerance, and cumulative
thickness cannot exceed the requested limit. Fan/merged regions or matching
core quads can make membership ambiguous. Counts are explicitly labeled as
reconstructed; ordinary recombined core quads are not all counted as layers.

Inflation diagnostics show unique candidate cells, wall-edge coverage, the
layer-depth distribution and histogram, nominal full layers from the geometric
series, measured first-layer heights, achieved column thicknesses, growth
ratios, shared cells, and reasons why tracing stopped. A single layer count is
not assumed everywhere along the blade. High aspect ratios or low inverse
condition numbers can be intentional in thin wall layers. Diagnostic thresholds
are counts for inspection, not a universal CFD acceptance test. y+ is excluded
because it requires the flow solution and wall shear stress.

| `mesh.report` setting | Meaning |
| --- | --- |
| `enabled` | Generate all four report artifacts; default true |
| `histogram_bins` | Number of histogram bins; default 30 |
| `worst_cells` | Cells listed for each worst-quality ranking; default 10 |
| `inflation_tolerance` | Relative tolerance for matching reconstructed layer heights; default 0.35 |

For numerical workflows, `mesh_2d.analyze_mesh` accepts NumPy coordinates,
zero-based triangle/quad connectivity, element tags, boundary edges, and optional
periodic node mapping without requiring an active Gmsh model. `cell_data` contains the
per-cell arrays; JSON contains summary data and CSV contains the detailed arrays.

## CGNS export and CFD setup

`mesh.cgns.enabled` enables `export_fluent_cgns`. `length_scale: 0.001` converts
millimetres to metres for CGNS only. `boundary_types` assigns the desired CGNS
boundary types; it does not set pressures, temperatures, materials, or turbulence
models in the CFD solver.

`mesh_2d.export_fluent_cgns` constructs one unstructured CGNS zone with
cell and physical dimensions both equal to 2. It includes first-order
triangles/quadrilaterals, named boundary edges, parent-cell information, and
reciprocal periodic node connectivity. The NumPy-based writer in
`turbo_moc.io.cgns_adf` serializes that tree to an ADF file. Coordinates include metre units.
Unused CAD control-point nodes are excluded, and all exterior cell edges must
belong to exactly one named boundary.

| Boundary | Default CGNS type |
| --- | --- |
| `blade` | `BCWallViscous` |
| `inlet` | `BCInflowSubsonic` |
| `outlet` | `BCOutflowSubsonic` |
| `periodic_left` / `periodic_right` | `BCGeneral`, with periodic connectivity |

The periodic translations point from the current interface to its donor:
right-to-left is `(-pitch * length_scale, 0)` metres. Solver importers can differ
in how they interpret periodic metadata. Inspect the imported boundary types
and periodic pairing in the target solver. The smoke suite exercises a real mesh
and reads its CGNS export. The Fluent workflow verifies the imported vertices,
then accepts an existing periodic pair or creates the conformal translational
pair explicitly. The current refactor checks use solver test doubles and do not
establish licensed Fluent import acceptance for every release.

## Source layout and verification

| File | Responsibility |
| --- | --- |
| `run_mesh_generation.py` | Thin command-line launcher |
| `../../src/turbo_moc/meshing/mesh_2d.py` | Gmsh CAD, sizing, inflation, periodicity, CGNS, quality reports and callable `main` |
| `../../src/turbo_moc/configuration/planar.py` | YAML loading and validation |
| `../../src/turbo_moc/workflows/nozzle.py` | Fresh MoC solving, progress forwarding and final-wall validation |
| `workflow_gmsh_stator_2D.yaml` | Nozzle, blade, passage and mesh settings |
| `../../src/turbo_moc/geometry/passage_2d.py` | Blade/passage construction, splines, placement and clearance |
| `../../src/turbo_moc/visualization/geometry_mpl.py` | Blade and passage construction figures |
| `../../src/turbo_moc/visualization/mesh_mpl.py` | Mesh display controls and PNG/SVG figures |
| `../../src/turbo_moc/reporting.py` | Shared console/file handlers and workflow summaries |

The numerical modules have section comments for navigation. Geometry proceeds
from validation and curve construction to clearance and blade assembly;
plotting functions live under `turbo_moc.visualization`.
Meshing proceeds from CAD and sizing through export/reporting to `create_mesh`
and the callable case `main`. Configuration validation is imported inside
`main`, so validation can use numerical functions without a module cycle.
Only entry points configure logging handlers.

For reuse, import `build_geometry` from `turbo_moc.geometry.passage_2d` and
`plot_construction` from `turbo_moc.visualization.geometry_mpl`; import `create_mesh`,
`analyze_mesh`, `compute_mesh_report` or `main` from
`turbo_moc.meshing.mesh_2d`. The MoC solver entry point is
`turbo_moc.workflows.nozzle.solve_nozzle`, and the planar reader remains
`turbo_moc.configuration.planar.read_configuration`. Fluent imports
these owners directly.

The targeted tests exercise real small meshes, inflation, automatic placement,
periodicity, CGNS boundary/connectivity export and failed-Gmsh cleanup. Solver
entry-point tests distinguish fresh calculation from explicit active-job result
handoff. Figure rendering is mocked where it would add no numerical coverage;
Gmsh generation and CGNS export remain real.

The latest default CLI run completed a fresh MoC solve and generated 25,519
all-quad cells, with minimum scaled Jacobian 0.023 and 193 periodic pairs,
CGNS output, PNG/SVG mesh views and an HTML report. A real Fluent prepare-only
run also generated the mesh and barotropic model without launching Fluent.
See the [refactor summary](../../refactor.md) for restructuring verification.

Run just the fast mesh and Fluent automation checks:

```powershell
poetry run pytest tests/test_stator_mesh.py tests/fluent -q
```

The Fluent cases use mocks without launching Fluent; the thermodynamic fit check
uses jaxprop. The mesh runner also uses jaxprop plotting helpers and needs no
barotropy checkout. Fluent tests are excluded from the default pytest run.
For a full execution check, run the example with both viewers disabled,
then inspect `mesh_generation.log`, the CGNS mesh and the quality report.

## Shared Fluent workflow

The full workflow in [../fluent_2D/readme.md](../fluent_2D/readme.md) imports these
geometry, MoC-solving, plotting and mesh functions and reads `workflow_gmsh_stator_2D.yaml`.
It sends a separate output directory to `create_mesh`, so running Fluent writes
mesh artifacts to `examples/fluent_2D/output/gmsh/`.

Mesh-only outputs go into this example's adjacent `output/`. The Fluent
workflow owns a separate local output folder. Package callables receive
configuration and output paths explicitly; they contain no example-path registry.
