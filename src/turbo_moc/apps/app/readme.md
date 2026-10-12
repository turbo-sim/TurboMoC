# TurboMoC application implementation

This directory owns the browser UI and its local job service. Shared numerical
code lives in [geometry](../../geometry) and [meshing](../../meshing).
Workflow runners live in [examples/gmsh_3D](../../../../examples/gmsh_3D/readme.md).
App launchers live under `examples/apps/`; editable YAML is under `examples/apps/cases/`.
The [theory guide](../../../../docs/internal_information/theory_guide.md) and
[roadmap](../../../../docs/internal_information/roadmap.md) live under `docs/internal_information/`.

| File | Responsibility |
| --- | --- |
| `application.py` | Dash factory, layout, controls and callbacks. |
| `workflow.py` | Stage identities, selected-result handoff, derived-geometry reuse and worker launch/poll/cancel. |
| `calculations.py` | Calculation scope, prerequisite checks and sequential row jobs with fresh dependency handoff. |
| `project_export.py` | Current-stage checks, combined CGNS packaging, reports and project download. |
| `worker.py` | Isolated numerical execution with live progress and logs. |
| `preview.py` | Smooth, capped geometry tessellation and display normals. |
| `plots.py` | Labeled 2D engineering views and quality histograms. |
| `assets/studio.js` | Browser VTK scenes, rotational/translational display copies and mapper updates. |
| `assets/mesh_edges.js` | Reused display triangles and antialiased cell-edge shader. |
| `assets/studio.css` | App layout and styling. |

The numerical owners are [geometry_2d.py](../../geometry/geometry_2d.py),
[geometry_3d.py](../../geometry/geometry_3d.py),
[meridional_channel.py](../../geometry/meridional_channel.py),
[linear_cascade.py](../../geometry/linear_cascade.py),
[moc.py](../../geometry/moc.py), [meshing_2d.py](../../meshing/meshing_2d.py),
[meshing_3d.py](../../meshing/meshing_3d.py), [quality.py](../../meshing/quality.py)
and [inspection.py](../../meshing/inspection.py).
`MeridionalChannel`, the linear-cascade coordinate adapter and
`Design.meridional_channel` express the shared section-to-physical model. The app uses the same reference/volume/export stages as the
CLI, without importing historical folders or a second numerical implementation.

The UI has two persistent workspaces, **Geometry generation** and **Mesh generation**, with
settings grouped by row. The sidebar shows every active row in a collapsible
group, with geometry or meshing settings chosen by the workspace. The selected
row supplies the quality report. All geometry and mesh diagrams
assemble available rows. Machine mode activates stator, rotor or both, while
the independent geometry type chooses linear, axial or radial mapping.
Default row labels are Stator and Rotor; workspace tabs have no numbers,
and the viewer slider reads Displayed blades. Stable YAML IDs remain unchanged.
Linear cascade is unavailable for the combined mode; mixed flow is outside
the current interface. The rotor provider uses the existing vortex-flow method;
the earlier planning-only `conventional` value migrates to `vortex` when loading
an old placeholder. It is not exposed as an alternative rotor solver.

Rotor periodic boundaries use one cubic Bezier reference guided by blade
midpoints at 10%, 50% and 90% of the streamwise chord. Endpoint handles follow
the relative inlet/outlet angles, with conservative handles when the direct
two-handle solve would fold the curve. The existing extension settings and
clearance-maximizing tangential shift remain in use; no additional YAML fields
are needed. The [construction example](../../../../examples/moc_design/rotor_flow_domain_construction.py)
plots the package geometry, its control points and extension joins.

**Apply calculations to** provides an independent checkbox for each active row.
Check both Stator and Rotor to calculate both; uncheck all rows to disable the
calculation buttons. Selection order does not change machine execution order.
The single **Compute blade design** button runs the stator MoC before the rotor
vortex method when both are selected. Each successful result is published before
the next row starts, so a coupled rotor consumes the newly computed stator exit.
**Generate 2D mesh** and **Generate 3D mesh** run the selected rows sequentially,
skipping already-current meshes. All selected rows need current blade designs;
3D generation also requires current 2D meshes for every selected row. These
dependencies are checked before a mesh queue begins. **Inspect row** selects the
quality report independently of calculation scope and the shared diagrams.

**Plot rows** independently controls geometry and mesh visibility, preserving
hidden rows' results. All diagrams, including blade-to-blade geometry and mesh,
show checked rows together. Multirow blade-to-blade plots use a common midspan
conformal chart: the first active row's leading edge defines $m'=0$, and row
offsets accumulate $\int ds/r$ through blade rows and the physical interspace.
Angular phase is included with the clockwise chart convention. These display
translations preserve row-local numerical meshes and CGNS export. Hiding a row
keeps the chart origin fixed. If the inspected row is hidden, inspection switches to the first
visible row. Meridional mesh projections include only current, generated 2D
meshes. Solid outlines mark the LE-to-TE blade extent; dashed outlines mark
actual inlet/outlet flow-domain extensions, including shared interfaces.
Pending designs show a pale prescribed blade extent without inventing a flow
domain. Calculation, visibility and inspection controls share one compact
strip, with instructions and physical interspace details available on hover.

Every queue retains its submitted project snapshot. Progress identifies the
active row and position in the queue, while the terminal retains live output
from successive rows. Completion updates that row's plots and opens the relevant
workspace. Failure stops the queue; cancellation stops its active worker and
remaining rows. Successful earlier results remain available. Overlapping polling
requests cannot launch the same successor twice.

Rotor inputs are relative-frame stagnation pressure [Pa], temperature [K],
inlet Mach number and flow angle [degrees from the meridional direction], plus
outlet and surface Mach numbers. The vortex method derives the outlet angle
from mass-flux continuity; it is not an independent input. Its rounded blade
profile uses the same conformal mapping, section mesh and span sweep as the
stator, with independent row dimensions and meshing controls. The original
vortex solver is unchanged. The Gmsh example scripts remain stator convenience
workflows; rotor design and job orchestration are exposed through this app.

**Manual inlet** accepts the user's relative-frame fluid, pressure, temperature,
Mach number and angle. **Couple to stator exit** uses the nearest preceding
active stator and its current computed exit state. The calculated values remain
visible in shaded, disabled controls. Editing the upstream design invalidates
its dependent rotor results; an unavailable or stale upstream result blocks a
coupled solve. Switching back to manual restores editable inlet inputs.

The axial interspace is the first control group for a paired axial rotor. New
or missing gap values start at 10% of the rotor midspan axial chord; existing
explicit YAML values are preserved. Adjacent active rows share one physical
interface: halfway across the minimum axial clearance, or halfway between the
supplied stator outlet and rotor inlet radii. Each row's internal domain cap is
derived from this target at every span station. The internal chord-extension
inputs are hidden; shaded readouts show the interface coordinate and physical
hub/midspan/tip extensions in millimetres. The original YAML values remain
available for single-row operation. Stator inlet and rotor outlet extensions
remain editable. See the [shared-interface design](../../../../docs/internal_information/roadmap.md#multirow-flow-domain-consistency-shared-physical-interface).

Paired domains require a positive blade gap. Axial rows must share hub radius
and blade height; radial rows must share blade height and flow direction.
Unequal endwalls need a transition geometry that is not implemented. An
inconsistent pair remains editable, with a clear message blocking generation.

The velocity triangle uses $U=2\pi n r_{in}/60$, with $n$ in RPM and radius in
metres. The radius comes from the rotor midspan inlet: hub radius plus half blade
height for axial rows, and inlet radius for radial rows. With positive swirl
defined in the blade coordinate convention,
$W_m=V\cos\alpha$ and $W_\theta=V\sin\alpha-U$.
The existing coupling service reconstructs relative stagnation conditions from
$h_{0,rel}=h+W^2/2$ at the stator exit entropy. This is the legacy local velocity
triangle: it solves neither interspace losses nor radial transport between
different stator-exit and rotor-inlet radii. The vortex method requires
supersonic relative inflow. A linear rotor uses a manual inlet because it has
no machine radius or rotational blade speed.

[configuration_project.py](../../configuration_project.py) owns the ordered rows,
stable IDs, machine-mode validation and conversion to existing flat numerical
inputs. Each row's scientific job references are retained independently when
selection changes. Geometry and mesh changes invalidate that row's affected
stages; switching the selected row is a view change. The terminal identifies
the row and stage being computed.

Editing retains the last validated project independently of currently
invalid input text. An invalid field can remain visible while the user
corrects it; it does not erase the other validated settings or publish a new
numerical configuration. If a structural row/machine change discards invalid
edits, the app reports that explicitly. Scientific jobs require a valid
current configuration rather than silently using earlier valid settings.
Dynamic axial interspace controls are also checked against the selected
row/family so obsolete controls cannot overwrite a newly selected model.

`geometry.axial_interspace_mm` on an axial rotor specifies the positive
minimum upstream TE-to-rotor LE-plane gap when an upstream row is active.
[configuration_project.py](../../configuration_project.py) resolves it through
`assembly_placement` and reports adjacent-row spacing through
`assembly_interspaces`. `shared_interfaces` supplies the common boundary and
`row_assembly` carries its target and placement into the geometry/mesh stages.
The upstream extent uses the greater hub/tip chord.
A paired rotor's derived relative position takes priority over a literal
manual display offset; the interspace is inactive for rotor-only mode.

Radial rows remain positioned by their radii. Their directed gap is positive
for clearance and negative for overlap for either inflow or outflow, with no
extra interspace input. The shared meridional view includes row envelopes,
computed rotor geometry and the interface plane/radius. Geometry and browser
mesh datasets remain in row-local coordinates; the viewer applies placement
once. Paired 3D CGNS exports apply the same axial translation and angular phase
to the exported nodes. A retained outdated mesh uses its recorded placement,
so editing the gap does not move an old mesh into an apparently updated stage.
The stage export retains independent row zones; solver interface pairing, pitch-change
handling and rotating reference frames remain downstream solver setup.
The annular 2D mesh remains a conformal reference grid, not a shared physical
2D stage mesh. Single-row export keeps its established local coordinate basis.

The header links to
[Sustainable Thermal Power, research group at DTU Construct](https://thermalpower.dtu.dk/).
The former Show all settings control is removed; settings follow their
workspace and active row groups.

The Mesh generation workspace combines 2D section/grid views and the 3D boundary/report
view. Each mesh-generation job also writes its Fluent-compatible CGNS/ADF
output; separate conversion actions are no longer needed. **Export project (.zip)**
beside the generation controls packages current, completed meshes. Temporary MSH files are internal to conversion
and are removed; MSH and internal NPZ datasets are not downloadable. Meridional/grid
captions retain the mapped first-layer-height and span-spacing estimates,
distinguishing conformal targets from physical sizes. Linear rows
support true 2D export and physical 3D extrusion, including an optional one-cell
span. The project ZIP includes 2D CGNS files only for linear cascades; axial and
radial cases contribute their physical 3D meshes.

[project_export.py](project_export.py) builds the archive from every active row,
independently of the calculation and inspection selectors. Annular export
requires a current 3D mesh for each active row. The top-level `stage.cgns` contains
one fluid zone per row in machine coordinates. For a linear cascade it contains
the current 3D extrusion when available, otherwise the true 2D mesh. If both
linear meshes are available, `stage_2d.cgns` holds the additional true 2D grid.
Different-dimensional meshes are never combined into one stage base.

The archive also contains the final `project.yaml`, `README.txt`, unchanged
individual CGNS files under `meshes/`, one consolidated `mesh_report.json`, and
one `activity.log`. [cgns_stage.py](../../io/cgns_stage.py) combines the canonical
row exports. It prefixes zone, fluid-element-section, boundary and family names
with the stable row ID, and updates periodic donor-zone references. Local
connectivity, parent indices, coordinate arrays, boundary types and periodic
transforms remain unchanged. Long names use stable suffixes to respect ADF's
32-character limit. Every row must use the same CGNS coordinate multiplier.

The stator outlet and rotor inlet remain separate faces and vertices, even at
the shared physical interface. The report records their stage zone/boundary
names. Configure their flow coupling and the rotor reference frame in the CFD
solver; packaging does not choose a mixing-plane or transient interface model.
The combined format preserves the existing CGNS 3.4/ADF export conventions.
Automated readback and real Gmsh export checks establish topology preservation.
A small two-row stage has also passed a Fluent 24.2 import and mesh check,
retaining two fluid regions and all 14 boundary patches. Fluent qualifies
their names, for example `stator_1_fluid.stator_1_outlet`, and initially treats
the patches as walls. Assign inlet/outlet, periodic and inter-row interface
types explicitly in the solver. Production-case import acceptance, periodic
pairing and CFX acceptance remain separate checks; no CFD solve was performed.
The report groups complete summary and quality data by row and mesh dimension.
The log includes each relevant computation stage once. The README records mesh
counts, boundaries, units, coordinate scaling and unavailable meshes.
Job status, source revision, row inputs and predecessor identities are checked
before packaging, so stale meshes cannot be paired with the current YAML.
View settings do not invalidate meshes. Changing a paired gap or shared
interface invalidates both affected row domains and meshes, while retaining
current MoC/vortex profile solutions. Paired assembly placement is part of the
mesh/export dependency; a solitary row's display placement remains a view
setting. Span edits invalidate 3D results while retaining a current linear
2D mesh. The ZIP covers
all active rows with a complete current stage. MoC results, job bookkeeping and internal datasets
stay outside the archive.

The dense reference mesh uses Plotly `Scattergl` for its many unique cell
edges. When its row has no current artifact, `plots.section_mesh_placeholder`
keeps an empty GL trace rather than replacing the figure with a different
renderer. This clears the visible mesh while retaining the existing scene and
its contexts. The more modest meridional grid uses SVG `Scatter`, avoiding an
unnecessary second GL scene. Twenty empty/restore cycles, including title,
axis and `uirevision` changes, retained the same two internal canvases and
exactly two GL contexts with no errors or warnings. These display changes
leave scientific meshes and exported connectivity untouched.

Every **Compute blade design** action solves the selected profiles afresh. A successful stator job writes
`moc_result.json`; downstream stages read only the explicitly selected job.
There is no MoC cache lookup or reuse policy. The fitted-template and derived
geometry stores retain rapid design iteration for that selected result. A new
MoC job changes downstream identities even when its input values are identical.

Dependencies remain managed by `pyproject.toml`: Dash/dash-vtk for the interface,
NumPy/SciPy for geometry, Gmsh for meshing, VTK/PyVista for volume inspection,
diskcache for derived-geometry reuse and PyYAML for configuration. Quality
calculations are independent array operations; inspection keeps optional VTK
filters behind a separate numerical adapter. Common logging uses the
existing [turbo_moc.reporting](../../reporting.py) unchanged, preserving its
MoC reporting/progress interfaces.

Run `turbo_moc-app` or `python -m turbo_moc.apps.app`. An existing
editable environment needs reinstallation to register the new console;
the Python module and minimal example launcher work immediately. The example
launcher delegates to the same entry point. It accepts
`--config path/to/project.yaml`, `--port 8053` and `--output-dir path/to/runtime-root`;
`create_app(config_file=None)` initializes the selected project. Rotor-only
files expose vortex design controls. YAML initialization
does not reopen sessions or adopt earlier scientific artifacts.
Workers launch as `turbo_moc.apps.app.worker`. Each job retains
`project.yaml` for the submitted machine/rows and `configuration.yaml` for
the selected flat numerical row, so project state and numerical inputs remain
reviewable separately. Job submission rejects inactive rows and unavailable
coupling dependencies before launching a worker. The package has no knowledge of example YAML locations. App initialization
uses Python defaults or a caller-supplied project file. The example launcher
chooses its adjacent `cases/` input and `output/` directory. Installed commands
write to `./output/` unless `--output-dir` overrides it. Python callers pass
`create_app(config_file, output_dir=...)` explicitly; the output root is stored
on each app, and workers recover it from their request-file location.
Browser assets come
from this directory's `assets/`. Restart after Python edits because workers
verify the submitted source revision.

The promoted package passes 112 canonical regressions and 32 migration
checks. Installed fresh axial/linear workflows use the packaged worker and
resources from outside the checkout without changing package files. A real
final-wheel Edge workflow passes 17 checks with zero exceptions, errors or
warnings, including live MoC progress, both meshes, export, meridional gating,
navigation, native sidebar edge-width edits and renderer/camera retention.
The [merging summary](../../../../docs/internal_information/merging_summary.md) records the overlapping
regression suites, final focused retest and optional dependency limits.

The 3D adapter owns a persistent React root inside each Dash viewport and
reuses installed dash-vtk components. Callback children return `no_update`;
one View and stable actor pool update data and hide unused actors without
Dash remounts. New datasets/copy counts fit; surface/width edits retain the
camera. A focused 50-edit audit also verified actual View/canvas and camera
stability without errors or warnings. No new frontend package/CDN is needed.
The [migration record](../../../../docs/internal_information/migration.md) records current and historical
checks, review artifacts, manual acceptance and remaining solver integration.


The preserved preceding app remains `turbo_moc.deprecated.app`, with the
preserved `turbo_moc-app-legacy` / `python -m turbo_moc.app` entry points. This
package migration preserves nozzle diagnostics, control-point editing,
sizing and CAD tools in that app. Vortex rotor design is now also integrated
with the current app's row and mesh workflow. Other retained capabilities and
their integration checklist are documented in the
[root merging summary](../../../../docs/internal_information/merging_summary.md).


Atomic job-status reads retry short Windows `PermissionError` sharing locks;
missing files and invalid JSON still fail explicitly. This is migration
robustness around worker publication, not a nozzle cache or a change to
scientific artifact validity.

CGNS settings remain in the sidebar under **CGNS mesh output**. Changing boundary
conditions, filename or length scaling marks generated mesh outputs outdated.
The app always produces CGNS, independently of a case's CLI export-enable flag.
2D generation retains section arrays for span extension and 3D generation retains
volume arrays/browser surfaces for inspection; neither needs a retained MSH file.
Cancelled jobs may leave internal intermediate files, but those are never exposed
as mesh downloads. Legacy app and standalone workflow output behavior is unchanged.

New rotor rows start with Gmsh's Frontal-Delaunay algorithm (`algorithm: 6`),
followed by the selected recombination step. This also handles the narrow,
curved passage at the vortex design's natural pitch; algorithm 8 produced
overlapping core cells in that linear-cascade check. The algorithm remains a
YAML/sidebar setting and saved cases retain their choice. The same canonical
section and span mesher serves both blade providers; no rotor cell-repair step
is introduced. During span mapping, similar periodic curves transform their
actual CAD boundary nodes exactly, avoiding a projection onto the coarser
plotting polyline that could fold nearby cells.
