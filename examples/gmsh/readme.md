# 2D meshing of MoC stator blades

This example designs a planar nozzle with the method of characteristics (MoC),
parametrizes one stator blade, constructs a smooth periodic flow domain, and
creates a 2D Gmsh fluid mesh with optional inflation layers. It exports the mesh
as CGNS/ADF for downstream CFD tools and saves an orange PNG view of the mesh.

## Run and configure

From the repository root, install the package and its dependencies, then run:

```powershell
poetry install
poetry run python examples/gmsh/run_mesh_generation.py
```

Gmsh is a regular package dependency. This example does not require the optional
`cad` or `fluent` extras: its CAD construction uses Gmsh, not CadQuery.

Edit `stator_mesh.yaml` to set the `nozzle`, `blade`, `passage`, and `mesh`
parameters. Lengths in the nozzle inputs are metres; blade geometry, passage
geometry, mesh sizes, and plots use millimetres. In the cascade frame, x is
pitchwise, y is axial, and flow proceeds toward decreasing y.

Paths, caching, plotting, and mesh-generation switches are defined near the top
of `run_mesh_generation.py`. Paths resolve relative to the script. `GENERATE_MESH = False`
limits the run to geometry inspection. By default, `SAVE_FIGURES = True` saves
the blade construction and selected domain, while `SHOW_FIGURES = False` keeps
the Matplotlib viewer closed. Set `SHOW_FIGURES = True` to open that window;
close it to continue to meshing. Gmsh opens when `SHOW_GMSH` and `mesh.show_gui`
are both true, as in the current example. There are no plot sliders;
manual/automatic domain placement is controlled entirely through YAML.

For unattended execution, set `SHOW_GMSH = False` and `SHOW_FIGURES = False`
in the globals at the top of `run_mesh_generation.py`, then run the same command.

With the current globals and YAML, outputs are written to the flat directory
`examples/gmsh/output/`:

| Output | Contents |
| --- | --- |
| `nozzle_solution.json` | Cached MoC solution and cache signature |
| `stator_geometry.png` / `.svg` | Blade construction and flow-domain plots |
| `stator_mesh.png` | Orange view of the generated triangles/quadrilaterals |
| `stator_mesh.cgns` | Optional 2D CGNS/ADF mesh with named boundaries and periodic connectivity |
| `mesh_report.html` | Self-contained quality report with quantiles, histograms, and inflation diagnostics |
| `mesh_report.json` | Machine-readable summary |
| `mesh_cells.csv` | Per-cell metrics, original Gmsh tags, centroids, and reconstructed layer indices |
| `mesh_quality.png` | Exportable quality histograms |
| `mesh_generation.log` | Workflow progress, summaries, warnings and failure tracebacks |

The cache requires `USE_MOC_CACHE`; geometry figures require `SAVE_FIGURES`.
Mesh artifacts require `GENERATE_MESH`, with CGNS controlled by
`mesh.cgns.enabled` and the four report artifacts by `mesh.report.enabled`.
Changing these switches does not remove files left by earlier runs.

## Execution reporting

The runner reports numbered steps for configuration, the MoC design, blade
geometry and meshing. Mesh sizing, inflation and quality results appear in
aligned summaries, followed by export paths and a final completion section.
Messages are written to the console and `output/mesh_generation.log` as each
operation runs. The log uses UTF-8, flushes after every message and is replaced
on each run; it does not accumulate timestamped copies.

Set `LOG_LEVEL = "INFO"` and `LOG_FILENAME` near the start of
`run_mesh_generation.py`. `DEBUG` includes detailed export and cleanup diagnostics;
`WARNING` shows only warnings and errors. Imported functions use the shared logger
configured by the entry point. The shared implementation is
[`../workflow_logging.py`](../workflow_logging.py), using Python's standard
`logging` module without extra dependencies.

The execution log records messages from these example scripts. MoC's native
console output is unchanged. Gmsh's native warnings and errors remain controlled
by `mesh.terminal_output` and `mesh.verbosity`; they are not redirected into this
log. There are no timing decorators, elapsed-time measurements or viewer timings.

## Nozzle and blade geometry

`run_mesh_generation.py` loads a matching cached nozzle solution or calls
`turbo_moc.design_nozzle`. The default configuration uses the conventional
planar solver for Cyclopentane. The cache signature includes nozzle inputs,
solver source, and relevant dependency versions. Changes to blade, passage,
or mesh settings do not invalidate the nozzle cache. Set `FORCE_MOC_SOLVE = True`
to refresh it, or `USE_MOC_CACHE = False` to disable cache reads and writes.

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

`geometry_source.py` constructs a reference with four joined sections:

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
`mesh.curvature_elements` enables its curvature-based sizing on both the blade
and periodic boundaries. The value is the target number of elements per full
circle (2*pi radians). For local curvature kappa, the associated size is
approximately `2*pi / (curvature_elements * abs(kappa))`: tighter bends receive
smaller elements. Zero disables this curvature criterion.

The final local size combines curvature, point sizes, and an automatic core
sizing callback, bounded above by `domain_size`. Straight periodic portions
are governed by those other size controls. Plotting resolution does not fix
1D mesh resolution. See the [Gmsh sizing documentation](https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes).

| `mesh` setting | Purpose |
| --- | --- |
| `blade_size` / `domain_size` | Tangential blade / far-field target element sizes [mm] |
| `curvature_elements` | Curvature resolution in elements per 2*pi radians |
| `algorithm` | 2D meshing algorithm; default 8 prepares triangles for recombination |
| `recombine` | Convert suitable fluid triangles to quadrilaterals |
| `recombination_algorithm` | Recombination method; default 1 is Blossom |
| `smoothing_steps` | Mesh smoothing iterations |
| `inflation.enabled` | Enable quadrilateral wall layers |
| `inflation.first_height` | First wall-normal layer height [mm] |
| `inflation.growth_ratio` | Ratio between successive layer heights |
| `inflation.thickness` | Maximum total layer thickness [mm] |
| `verbosity` / `terminal_output` | Gmsh logging level and console output |
| `show_gui` | Open the Gmsh viewer after mesh export |
| `image.filename` / `image.dpi` | PNG filename and resolution; default `stator_mesh.png`, 200 dpi |

Gmsh displays both triangles and quadrilaterals in orange. The PNG uses the
actual Gmsh node coordinates and element connectivity, rendered with
Matplotlib's Agg backend so it is also saved when the viewer is disabled.
The mesh is exported only as CGNS; no native Gmsh mesh file is written.
`create_mesh` returns the PNG path under `image_file` and the CGNS path under
`mesh_file` when CGNS export is enabled.
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
they are not adjusted to force the last height to equal `blade_size`. Geometry
can shorten/merge local stacks, so the nominal count is not an exact layer label.

`blade_size` controls spacing **along** the blade; inflation height controls
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
The distance to reach `domain_size` follows from this expression and is no
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

After meshing, `meshing_source.compute_mesh_report(surface, boundaries, inflation=...)`
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

For numerical workflows, `meshing_source.analyze_mesh` accepts NumPy coordinates,
zero-based triangle/quad connectivity, element tags, boundary edges, and optional
periodic node mapping without requiring an active Gmsh model. `cell_data` contains the
per-cell arrays; JSON contains summary data and CSV contains the detailed arrays.

## CGNS export and CFD setup

`mesh.cgns.enabled` enables `export_fluent_cgns`. `length_scale: 0.001` converts
millimetres to metres for CGNS only. `boundary_types` assigns the desired CGNS
boundary types; it does not set pressures, temperatures, materials, or turbulence
models in the CFD solver.

`meshing_source.export_fluent_cgns` constructs one unstructured CGNS zone with
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
and reads its CGNS export. `../fluent/run_fluent_case.py` has been validated with Fluent 2024 R2:
CGNS imports the periodic sides as wall zones, and the workflow explicitly
creates the conformal translational pair after checking the imported vertices.

## Source layout and verification

| File | Responsibility |
| --- | --- |
| `run_mesh_generation.py` | Entry point: user switches, YAML loading, step reporting, and MoC cache signature/validation/solving |
| `stator_mesh.yaml` | Nozzle, blade, passage, and mesh settings, with parameter comments |
| `geometry_source.py` | Blade and passage construction, spline geometry, manual/automatic placement, clearance checks, and Matplotlib views |
| `meshing_source.py` | Gmsh CAD, sizing, inflation, recombination, periodic checks, Fluent CGNS export, mesh images, quality analysis, and reports |
| `../workflow_logging.py` | Shared console/file handlers, section headings, aligned summaries and failure reporting |

The two source modules have numbered section comments for navigation.
`geometry_source.py` proceeds from validation and spline construction to
clearance checks, blade assembly, and plotting. `meshing_source.py` proceeds
from CAD and sizing through export and quality analysis to the complete
`create_mesh` workflow at the end. Reusable helpers do not read YAML or manage
the nozzle cache. Only the entry point configures the logging handlers.

For reuse, import `build_geometry` and `plot_construction` from
`examples.gmsh.geometry_source`, and `create_mesh`, `analyze_mesh`, or
`compute_mesh_report` from `examples.gmsh.meshing_source`. The geometry wrappers
and `load_or_solve_nozzle` remain available from
`examples.gmsh.run_mesh_generation` for the shared Fluent workflow.

The small automation smoke suite uses two modules. It generates one
small real mesh with inflation and automatic passage placement, reads its CGNS
boundaries/connectivity, checks mesh/image export paths, and checks cache reuse and
cleanup after invalid geometry. Figure rendering is mocked to keep the suite fast;
Mesh generation and CGNS export are real. Detailed geometric checks remain in
the workflow itself; the suite avoids parameter sweeps, full quality-report
generation and separate tests of every metric.

Run just the fast mesh and Fluent automation checks:

```powershell
poetry run pytest tests/test_stator_mesh.py tests/test_stator_fluent.py -q
```

The Fluent cases use mocks without launching Fluent or importing the companion
barotropy checkout; the thermodynamic fit check uses the existing jaxprop dependency.
For a full execution check, run the example with both viewers disabled,
then inspect `mesh_generation.log`, the CGNS mesh and the quality report.

## Shared Fluent workflow

The full workflow in [../fluent/readme.md](../fluent/readme.md) imports these
geometry, caching, plotting and mesh functions and reads `stator_mesh.yaml`.
It sends a separate output directory to `create_mesh`, so running Fluent writes
mesh artifacts to `examples/fluent/output/gmsh/`.

All mesh-only outputs share the flat `output/` directory beside this README.
Set `SHOW_GMSH = False` and `SHOW_FIGURES = False` at the top of `run_mesh_generation.py`
for unattended use. There are no command-line arguments.
