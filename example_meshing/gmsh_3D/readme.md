# TurboMoC

Run from the repository root using the existing environment:

```powershell
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_app
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_app --config example_meshing/gmsh_3D/linear_stator.yaml --port 8053
```

Open [http://127.0.0.1:8052](http://127.0.0.1:8052). The app needs the project's mesh/mesh-viz dependencies (Gmsh, Dash, diskcache, dash-vtk and PyVista). It binds to the local machine. Stop it with Ctrl+C. Restart after changing Python sources; workers verify the source revision against the submitted snapshot.

The launcher accepts `--config` for the initial machine-project settings and
`--port` for a local port (8052 by default). Loading a YAML does not restore
session history or discover saved mesh artifacts. For example,
`--config example_meshing/gmsh_3D/rotor_only.yaml` opens rotor planning controls,
with numerical generation disabled and no substitute geometry.

This case directory keeps `run_mesh_app.py` as a small launcher, `run_mesh_generation.py` as the command-line entry point, the YAML cases and case notes. The shared theory guide and development roadmap live at the `example_meshing/` root. Shared numerical implementations live under [`../src/`](../readme.md#source-ownership-and-dependencies); workers, plots, geometry previews and browser assets live under [`../src/app/`](../src/app/readme.md). The app and CLI use the same numerical stages and do not import historical example folders. Configuration and output locations remain in this example directory.

The historical snapshots under `examples/old/` remain untouched for manual comparison. This is the consolidated development entry point; it incorporates the preserved V2 numerical foundations and the current row-based Geometry generation/Mesh generation interface. See the [migration record](../migration.md).

## Workflow

Select **Stator**, **Rotor** or **Stator + Rotor**, then choose the machine
geometry: **Linear cascade**, **Axial** or **Radial**. A combined project cannot
use Linear cascade. Radial inflow and outflow use the same mapping; the given
inlet/outlet radii determine the direction. Mixed flow is outside the current
app. Rotor controls retain planning settings for conventional or vortex
parametrization, while rotor numerical actions remain disabled until its
provider is integrated.

1. **Geometry generation:** select a row and edit its Method of characteristics,
   Meridional channel definition, Blade geometry definition and Flow domain
   definition. Compute MoC solves that row's current inputs afresh. The selected
   section and shared meridional geometry appear together, with the available
   3D machine geometry below. A linear row uses physical section coordinates;
   axial/radial rows use the conformal reference plane. The sidebar defaults
   to 380 px.
2. **Mesh generation:** generate the selected row's 2D reference. Its section mesh and
   projected meridional/extrusion grid remain empty until the reference is
   current. Span edits update the projected grid while retaining the section.
   Generate the 3D volume separately from that current reference, then inspect
   its selected boundary surfaces beside the scrollable quality report. Both
   stages live in this workspace and successful jobs navigate to it. Export
   true 2D or 3D meshes without remeshing. Display widths remain together under
   **Mesh visualization** in the left sidebar.

The header links to [Sustainable Thermal Power, research group at DTU Construct](https://thermalpower.dtu.dk/); Show all settings has been removed.

The default row labels are **Stator** and **Rotor**, and the workspace tabs
have no numeric prefixes. **Displayed blades** is the common viewer slider
label. Stable row IDs such as `stator_1` remain in YAML, jobs and outputs.

The sidebar displays collapsible groups for all active rows; the workspace
selects geometry or meshing settings. Each row retains its own settings and
scientific result references. Switching
rows changes the section, generation actions and quality report; it does not adopt the
previous row's artifacts. The shared machine display identifies the geometry
that is actually available. The rotor placeholder does not fabricate a blade,
and assembling display rows does not join their fluid domains or define a
rotor-stator solver interface.

One job runs per browser session. Jobs run in dedicated Python processes, so Gmsh and VTK state belong to the worker. The app displays a spinner, elapsed time and stage progress to the right of the action buttons. A full-width terminal is anchored to the bottom of the main pane; its collapsed header stays visible while the content scrolls. Expand it to see the live bounded solver/Gmsh log. During a native operation without a reliable completion fraction, the bar is indeterminate. MoC percentages use the existing weighted stage estimator; they are not remaining-time predictions.

Cancel job terminates the active local worker. Failed/cancelled jobs retain previous successful references. Edits during work affect the next job; completion is tagged with its original configuration. The app labels previous/outdated volumes and requires current artifacts for the next stage.

## What the views show

| View | Basis |
| --- | --- |
| Reference geometry | Selected-row fitted blade/passage at reference span. Annular rows use dimensionless conformal coordinates; linear rows use physical streamwise/pitchwise distances in mm. |
| Meridional/extrusion projection | Available annular rows in physical z-r coordinates, or the linear chord-height extrusion. Blade LE-to-TE envelopes are shaded grey; these are projections rather than blade thickness. |
| 3D blade shape | Refined geometry tessellation in physical XYZ [mm], with smooth normals and closed blade ends; its faces are display polygons. |
| 2D mesh | Actual selected-row Gmsh cells; annular conformal or linear physical coordinates are explicitly labeled. |
| Meridional/extrusion grid | Actual span stations crossed with reference periodic-boundary node stations; shown only after current 2D meshing. |
| 3D mesh boundaries | Surfaces extracted from actual volume connectivity; no independent remesh for viewing. |

The meridional mesh view is the projected grid of the future sweep, displayed only after a current 2D mesh has been generated. It does not independently mesh a meridional CFD domain or solve streamlines. This makes it consistent with the reference section rather than introducing an unrelated mesher.

For annular rows, the physical number of blades sets periodic pitch and
**Displayed blades** creates rotated visualization copies. A linear row has no
physical blade count: its fitted pitch scales with its requested chord, and
display copies are translated by that pitch. Exports contain one periodic
passage. Orange blade surfaces and grey endwalls restore the original 3D palette; 3D mesh edges are black. Both 2D meshes use orange grid lines with black blade outlines. Annular meridional plots start at zero radius, making the meridional channel's position relative to the rotation axis visible. Surface selection, edge toggles and line widths affect views only. The geometry and mesh viewers have independent controls; the 2D plot group does not repeat them.

The 3D view has a resizable mesh pane and a scrollable report on its right. The report contains counts, SICN, metric statistics, three vertically stacked histograms, and original IDs for the ten worst cells per metric. Aspect ratio uses a logarithmic axis. On narrower windows these panes stack vertically.

## Configuration and display widths

The case files use the ordered project schema:

```yaml
machine:
  design_mode: stator        # stator, rotor, stator_rotor
  geometry_type: axial      # linear, axial, radial
workflow:
  log_level: INFO
rows:
  - id: stator_1
    role: stator
    geometry:
      method_of_characteristics: {}
      meridional_channel: {}
      blade_geometry: {}
      flow_domain: {}
    mesh: {}
```

The empty dictionaries above abbreviate the complete settings in the example
files. Each row owns its geometry and meshing inputs; `workflow` is shared.
Smoothing/tangent fields sit directly in `flow_domain`, without a nested
`smooth` dictionary. Stable IDs identify row settings, jobs and artifacts;
ordered entries express machine order. Legacy flat stator cases and old
`nozzle`/`machine`/`blade`/`passage.smooth` names still load through adapters.

Use [linear_stator.yaml](linear_stator.yaml) for a physical planar cascade with
`meridional_chord` and `blade_height`, [axial_stator.yaml](axial_stator.yaml) or
[radial_stator.yaml](radial_stator.yaml) for annular stators, and
[stator_rotor.yaml](stator_rotor.yaml) or [rotor_only.yaml](rotor_only.yaml) to
inspect the rotor planning hierarchy. A rotor-only project can be edited and
saved, but has no implemented numerical provider.

In paired axial mode, the rotor geometry field `axial_interspace_mm` sets a
nonnegative minimum blade-envelope clearance. The app derives the rotor LE
plane from the upstream LE origin plus the greater hub/tip chord plus this
gap. Its default is 0 mm; it is inactive for a solitary rotor. Radial machines
use the prescribed inlet/outlet radii directly and report signed clearance or
overlap, without an additional gap input. The meridional view may show a
planned rotor channel/envelope even while rotor blade generation is pending.
Blade-envelope spacing does not join the extended flow domains or define a
solver interface; see the [spacing derivation](../theory_guide.md#111-axial-interspace-and-radial-clearance).

A linear row retains both true 2D export and 3D extrusion. Setting
`mesh.numerics.span_layers: 1` makes one volume element through the height.
Its two span boundaries and solver boundary conditions still matter; a
one-element volume is not automatically the same physical problem as a 2D
calculation. For annular rows, a 2D download is labeled as a conformal reference
mesh and does not represent the full annular flow.


`mesh.visualization.section_edge_width` controls Plotly section and meridional grid lines. `mesh.visualization.volume_edge_width` controls antialiased black cell edges in the 3D surface shader. Both default to **0.35 display pixels** in the YAML files and appear together under **Mesh visualization** in the left sidebar of the Mesh generation workspace. Changes preserve numerical artifact keys and appear in configuration snapshots.

Native WebGL lines can ignore widths below one pixel; see the [VTK maintainer discussion](https://github.com/Kitware/vtk-js/issues/673). The 3D viewer therefore uses [vtk.js shader replacements](https://github.com/Kitware/vtk-js/blob/master/Sources/Rendering/OpenGL/PolyDataMapper/api.md): per-triangle coordinates measure distance to each original cell edge in framebuffer pixels, and a coverage filter draws fractional widths. Quad triangulation diagonals are suppressed. Rendering uses the surface depth test, so hidden mesh edges stay hidden. Display triangles are prepared once per surface and reused across blade copies; this does not change the numerical mesh or exported cells. Width and edge-visibility changes update the mapper in place and preserve the camera. Canvas scaling is included when converting display pixels to framebuffer pixels.

The geometry preview uses a denser parameter-space tessellation and supplied smooth point normals. Blade root/tip caps are included with the blade selection, so hiding endwalls does not expose hollow blades. Side/cap vertices are welded before subdivision to keep their shared edges aligned. The hub is a continuous surface of revolution spanning the periodic edges, without a blade-shaped hole; the shroud retains its opening. These display surfaces leave the actual CFD boundaries unchanged. Preview refinement is independent of CFD mesh sizing and preserves general axial/radial/curved mappings.

The simplified app no longer exposes slice/worst-cell controls. Their tested backend extraction remains available in `../src/meshing/inspection.py` for a future optional inspection workspace.

## Artifact dependencies and files

For each row: MoC inputs -> template -> geometry -> 2D reference -> span sweep -> quality -> export.

Changing a row's blade/passage/mapping controls invalidates its reference and volume. Section sizing/inflation controls invalidate both meshes. Span controls invalidate only the volume. Display and CGNS filename/units/BC settings preserve numerical mesh identities.

Each job writes immutable `project.yaml` and selected-row `configuration.yaml`
snapshots, then separate outputs under `output/app/<session>/<job>/`. Reference artifacts store their topology and exact fitted definition, so the sweep consumes the section that was displayed. Successful jobs are published only after validation. Downloads include the configuration, activity log, MSH, reports and separately exported CGNS.

Every Compute MoC action calculates a new solution and saves `moc_result.json` in that job. The following stages read only that selected result. Derived-geometry and fitted-template reuse is keyed by the source, configuration and selected MoC job; it does not discover or reuse prior MoC solves. A fresh solve invalidates its downstream meshes even when the inputs are unchanged. Browser stores hold references; scientific data stays in server artifacts, apart from compact datasets needed to draw the views. Browser refresh starts a new session; project reopen/history management is future work.

To run the numerical pipeline without the app:

```powershell
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_generation
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/radial_stator.yaml
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/linear_stator.yaml
.venv/Scripts/python.exe -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/stator_rotor.yaml --row stator_1
```

Every command solves MoC afresh, then writes reference/volume meshes, quality and stage timing reports, and CGNS when enabled. `stage_timings.json` reports the new solve as `moc_s` separately from geometry/reference/volume/export durations. Native viewer and static-figure options inherited in the example YAML are not used by the staged CLI.

Read the [geometry and meshing teaching guide](../theory_guide.md) for the general conformal mapping, axial/radial derivations, mesh deformation, Gmsh connectivity and CGNS export, illustrated with generated meshes. The shorter [deformation/coordinate note](deformation_note.md) covers sparse LU, one-time passage placement, physical height estimates and quality definitions. Read [the development roadmap](../roadmap.md) for completed and remaining work.

## Performance evidence

The following historical measurements used an already converged MoC result, default cases and no native viewers/static figures. They measure only downstream geometry and meshing work; they exclude solving MoC:

| Stage | Axial | Radial |
| --- | ---: | ---: |
| Geometry / sole reference placement | 0.95 s | 0.99 s |
| Reference mesh | 2.12 s | 3.72 s |
| Sweep, validation, metrics and output | 7.10 s | 5.41 s |
| Separate audited CGNS export | 2.08 s | 1.69 s |

The historical downstream axial stages total about 12.25 s, compared with the earlier profiled V1 pipeline's 40.54 s. This is indicative rather than an identical-output benchmark: V2 adds the three volume metrics, changes span placement policy, and V1's number included profiler overhead. The V2 axial mesh had 229,075 cells versus 229,375 previously. A span-only app iteration still reuses the selected geometry and reference stages. A current CLI run or new Compute MoC action also performs a fresh solver calculation; its complete wall time is not represented by this table.

V2 performs one fit/placement search, reuses sparse LU across span, and uses SciPy's compiled spline routines while preserving legacy fitting/endpoint conventions. It retains audited Gmsh assembly, periodicity and CGNS checks. No axial/radial-specific shortcut was introduced. Native Gmsh size fields, exact curve clearance acceleration, large-volume chunking and fully lazy preview tessellation remain opportunities.

## Verification

The final implementation passes **110 tests** in 74.32 s across the eight
numerical/project/app/Fluent regression modules. Real small axial, tapered,
radial inflow/outflow, curved and linear meshes pass their topology,
periodicity, Jacobian, quality and CGNS checks. True physical 2D linear
CGNS/MSH and one-cell-span volume exports retain the correct reciprocal
translational metadata.

A real Edge browser run completed fresh axial/linear MoC and mesh generation,
both exports, live progress and cancellation. It checked row switching with
retained stator artifacts, rotor guards, axial/radial interspaces, one-element
extrusion, translated linear copies, widths, invalid-edit preservation,
mapped-height/span-spacing captions and the header. The final audit reports
zero browser exceptions, log errors or WebGL-context warnings. Persistent
React/VTK views and Plotly reference scenes retain their canvases during
updates; surface/width edits retain cameras. A fresh linear CLI selected
`stator_1` and generated 76,528 cells and 86,148 nodes with audited CGNS output.

The [migration record](../migration.md) separates current verification from
historical evidence and manual acceptance. The implemented two-dimensional
Fluent automation is separate from mesh export. Licensed Fluent execution,
3D target-reader acceptance and coupled rotor-stator setup remain future
validation/integration tasks. See the
[theory guide](../theory_guide.md#105-what-is-established-before-fluent-runs)
for periodic interpretation and portability limits.
