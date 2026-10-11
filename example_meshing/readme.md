# Consolidated meshing and Fluent examples

This is the development home for the meshing workflows and TurboMoC blade app.
It behaves as a Python package within the checkout, while remaining outside
`src/turbo_moc`. Use the existing project environment and run launchers from
the repository root.

```text
example_meshing/
|-- src/
|   |-- geometry/
|   |   |-- moc.py                  Fresh MoC solve and wall validation
|   |   |-- geometry_2d.py          Planar blade and passage construction
|   |   |-- meridional_channel.py   General hub/shroud mapping
|   |   |-- linear_cascade.py       Physical planar mapping and extrusion
|   |   `-- geometry_3d.py          Reference placement and span sections
|   |-- meshing/
|   |   |-- meshing_2d.py           2D Gmsh, reports, export and case workflow
|   |   |-- meshing_3d.py           Reference mesh, deformation, sweep and export
|   |   |-- quality.py              Explicit volume quality definitions
|   |   `-- inspection.py           Optional VTK slices and cell subsets
|   |-- fluent/
|   |   |-- barotropic_model.py     Thermodynamic model preparation
|   |   |-- workflow_2d.py          Implemented planar Fluent automation
|   |   `-- workflow_3d.py          Future-work placeholder
|   |-- app/                       UI, jobs, plots, previews and browser assets
|   |-- configuration_project.py   Ordered rows, machine modes and row adapters
|   |-- configuration.py           Per-row validation and legacy-name adaptation
|   |-- configuration_2d.py        Independent planar case validation
|   |-- paths.py                   Case/output locations
|   `-- logging.py                 Shared console/file logging
|-- gmsh_2D/                       Launcher, planar YAML and case documentation
|-- gmsh_3D/                       App/CLI launchers, project YAML and notes
|-- fluent_2D/                     Launcher, workflow YAML and documentation
|-- fluent_3D/                     Placeholder only
|-- roadmap.md                     Active blueprint and implementation checklist
|-- theory_guide.md                Geometry, mapping, meshing and export theory
`-- figures/                       Reproducible teaching plates and generator
```

## Run the workflows

```bash
python -m example_meshing.gmsh_3D.run_mesh_app
python -m example_meshing.gmsh_3D.run_mesh_app --config example_meshing/gmsh_3D/linear_stator.yaml --port 8053
python -m example_meshing.gmsh_2D.run_mesh_generation
python -m example_meshing.gmsh_3D.run_mesh_generation
python -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/radial_stator.yaml
python -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/linear_stator.yaml
python -m example_meshing.gmsh_3D.run_mesh_generation --config example_meshing/gmsh_3D/stator_rotor.yaml --row stator_1
python -m example_meshing.fluent_2D.run_fluent_case
```

The app defaults to `http://127.0.0.1:8052`. Its minimal launcher accepts
`--config` to open a machine-project YAML and `--port` for a different local
port. This initializes settings; it does not reopen browser sessions or adopt
previous numerical artifacts. A rotor-only YAML opens the planning interface
with generation disabled and no fabricated geometry. Launchers also support direct execution
by file path. CLI workflows accept `--config`; Gmsh also accepts `--output`.
Default outputs remain in the case folders under `output/`. Relative Fluent
YAML paths resolve against that YAML's directory, including `design_file`.

Every Gmsh/Fluent CLI run and every **Compute MoC** action calculates a fresh
MoC solution through `geometry.moc.solve_nozzle`. There are no persistent MoC
cache files, cache signatures or force/reuse switches. In the app, the selected
successful MoC job saves `moc_result.json` for its subsequent geometry and mesh
stages. Those stages explicitly consume that selected job; they do not search
for earlier solutions. Fitted templates and derived geometry can still be
reused while iterating the design for that job. Computing MoC again selects a
new result and invalidates downstream geometry and meshes, even for identical
input values.

Settings and workflow details are in [Gmsh 2D](gmsh_2D/readme.md),
[Gmsh 3D and TurboMoC](gmsh_3D/readme.md) and [Fluent 2D](fluent_2D/readme.md).
The [roadmap](roadmap.md) records completed work and remaining priorities.
The [theory guide](theory_guide.md) explains the conformal-to-physical mapping,
mesh deformation, volume connectivity and CGNS export, with
[code-generated figures](figures/readme.md). A shorter
[deformation/LU note](gmsh_3D/deformation_note.md) remains with the 3D cases.

The four teaching plates have been regenerated from one fresh MoC solve using
the grouped implementation. Their metadata records the exact solver inputs,
result SHA256 and provenance, dependency versions, case settings, mesh counts
and validity/periodicity checks. These illustrative resolutions do not establish
production CFD mesh independence.

## Machine projects and the app

TurboMoC uses two workspaces, **Geometry generation** and **Mesh generation**. A machine selector
chooses Stator, Rotor or Stator + Rotor; a separate geometry selector chooses
Linear cascade, Axial or Radial. Linear cascade is available only for a single
row. Mixed flow is outside the present interface. Rotor settings are planning
inputs: its geometry and meshing provider is not integrated, so its generation
actions remain disabled. The implemented stator remains usable in a combined
project.

Project YAML contains `machine`, shared `workflow`, and an ordered `rows` list.
Each row has a stable `id`, a `role`, independent `geometry` settings and
independent `mesh` settings. Row order expresses machine order; changing a
row's settings affects that row's scientific artifacts. Selecting another row
retains the first row's settings and successful jobs. Shared machine views show
available geometry, while section views, generation actions and quality reports
identify the selected row. The sidebar shows collapsible groups for every
active row; the workspace chooses geometry or mesh controls. Viewing two rows together does not create a coupled CFD mesh.

In an axial Stator + Rotor project, the rotor row's
`geometry.axial_interspace_mm` prescribes the minimum upstream trailing-edge-to-rotor leading-edge gap.
The rotor origin is derived using the greater upstream hub/tip chord; its
relative position is not a competing manual offset. Rotor-only mode leaves
this input inactive. Radial row positions follow their radii, with a signed
clearance/overlap diagnostic and no extra interspace input. The shared
meridional figure can show a planned rotor channel outline without claiming
that a rotor blade exists. These are assembly-planning/display quantities;
row meshes and exports remain independent.

The cases [axial_stator.yaml](gmsh_3D/axial_stator.yaml),
[radial_stator.yaml](gmsh_3D/radial_stator.yaml) and
[linear_stator.yaml](gmsh_3D/linear_stator.yaml) exercise the implemented stator.
[stator_rotor.yaml](gmsh_3D/stator_rotor.yaml) and
[rotor_only.yaml](gmsh_3D/rotor_only.yaml) illustrate future rotor integration.
Rotor parametrization can be recorded as `conventional` or `vortex`; neither
option currently produces rotor geometry.

All supported geometries share the same 2D blade/passage mesher. Annular rows
interpret its length-valued plane as scaled conformal coordinates. Linear
rows interpret it as physical distances, scale the source blade and pitch
together to the requested meridional chord, and extrude it through the blade
height. They need no radii or physical blade count. Both true 2D and 3D exports
remain available; `mesh.numerics.span_layers: 1` is an optional one-element
3D extrusion, not a replacement for a planar mesh. For axial/radial rows, the
exported 2D reference is a conformal construction plane, not the complete
physical annular CFD problem. See the [theory guide](theory_guide.md).

The Mesh generation workspace combines the reference section, meridional/extrusion
grid and 3D boundary viewer with its quality report. Generating 2D and 3D
meshes remains separate, and a current 2D reference is required before 3D
construction. The anchored terminal and progress controls identify the row
and stage being computed. The header links to
[Sustainable Thermal Power, research group at DTU Construct](https://thermalpower.dtu.dk/);
the former Show all settings option has been removed.

The final implementation passes **110 regression tests** in 74.32 s across
numerical, project, app and Fluent modules. A real Edge run verified fresh
axial and linear MoC/mesh generation, independent 2D/3D exports, row retention
and rotor guards, axial/radial spacing, one-element extrusion, live progress,
cancellation, invalid-edit preservation, physical-size captions and header
controls. Its final audit recorded zero browser exceptions, log errors or
WebGL-context warnings. The [migration record](migration.md) records the
verification and remaining rotor/coupled-solver work.

## Source ownership and dependencies

Edit implementations in `src/`; case folders hold launchers, configuration and
notes. Geometry, meshing, Fluent and app responsibilities have distinct package
boundaries that can later move into the installed package without depending
on historical example folders.

`MeridionalChannel` in `geometry/meridional_channel.py` defines the annular
mapping; `geometry/linear_cascade.py` supplies physical planar coordinates and
spanwise extrusion through the same section/sweep interfaces. `geometry/geometry_3d.py` fits and places one reference design, then
uses `Design.meridional_channel` for span scaling and physical mapping.
`meshing/meshing_2d.py` owns the 2D numerical functions and its callable
`main`. `meshing/meshing_3d.py` combines the deformation/topology/export
implementation with the reference, volume and export stages and its callable
`main`; there is no separate core or workflow copy.

`meshing/quality.py` stays separate because its array-based metric definitions
are reusable scientific calculations. `meshing/inspection.py` isolates the
optional PyVista/VTK filters and imports PyVista only when inspection is used.
Both serve CLI and app consumers; they belong to meshing rather than the UI.
The app owns presentation and job orchestration under `src/app/`.

The shared modules use `turbo_moc` for the MoC solver, blade fitting and ADF IO.
They do not import the historical examples or the app. Configuration readers
import numerical validation dependencies when called, avoiding circular module
initialization. `logging.py` is scoped inside `example_meshing.src` and imported
relatively; it uses Python's standard `logging` package without shadowing it.

Dependencies remain in `pyproject.toml`. The app and PyVista inspection use the
`mesh-viz` extra; Fluent support is optional. A companion barotropy checkout and
a licensed Fluent installation are needed for the complete Fluent workflow.
The three-dimensional Fluent module remains an explicit placeholder.

## Migration and compatibility

The [migration record](migration.md) maps historical owners to the current
modules. Their preserved snapshots are under `examples/old/gmsh`,
`examples/old/gmsh_3D`, `examples/old/gmsh_3D_v2` and `examples/old/fluent`.
That snapshot layout predates this refactor; the historical implementation,
launchers, YAML, assets and outputs were left untouched for manual comparison.
There are no forwarding wrappers. The historical and consolidated trees are
independent; future edits belong in `example_meshing/src/`.

`configuration_project.py` owns project normalization and the adapter that
extracts one active row for the numerical stages. Legacy flat stator YAML still
loads: the app wraps it as `stator_1`, while the existing numerical reader
returns the flat row settings. The old `nozzle`/`machine`/`blade`/`passage.smooth`
input names also remain accepted. New project rows use
`method_of_characteristics`, `meridional_channel`, `blade_geometry` and flat
`flow_domain` inside `geometry`. The independent 2D/Fluent case schema remains
available through its existing launcher. The staged launcher accepts `--row`
to select a supported row explicitly; rotor-only execution raises a clear
unsupported-provider error.

Restart the app after Python changes and create current MoC/reference/volume
jobs. Their identities include source, configuration and the selected MoC job.
Old numerical outputs remain available for comparison, but are not discovered
or adopted by the current app. Historical folders may be retired only after
manual review and a separate user request.
