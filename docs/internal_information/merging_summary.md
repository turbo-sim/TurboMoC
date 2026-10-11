# Consolidation into the TurboMoC package

The row-based geometry and meshing implementation has a single canonical home
under `src/turbo_moc`. Example folders contain launchers, editable cases,
teaching figures and documentation. The preceding app and its numerical
workflows remain available under `turbo_moc.deprecated`; compatibility imports
preserve their existing Python interfaces.

This merge promotes the current row-based app. A subsequent integration adds
the existing vortex rotor method to its geometry and meshing workflow, including
manual or stator-coupled relative inlet conditions. The legacy app's nozzle
diagnostics, control-point editor, sizing and CAD tools remain available in the
preserved app and library; their new-interface integration remains separate.

The [roadmap](roadmap.md) records the development priorities;
the [migration record](migration.md) retains the earlier
verification history; the [theory guide](theory_guide.md)
explains the numerical methods.

## Canonical implementation and dependency direction

```text
src/turbo_moc/
|-- api.py, core/, solvers/       Existing MoC implementation
|-- progress.py, reporting.py    Existing MoC progress and reporting interfaces
|-- configuration*.py           Row projects and numerical configuration
|-- paths.py                    Packaged defaults and writable runtime paths
|-- geometry/
|   |-- blade.py, spline_tool.py Shared blade fitting and spline mathematics
|   |-- moc.py                  Fresh MoC solve and wall validation
|   |-- geometry_2d.py          Planar blade and passage construction
|   |-- meridional_channel.py   General annular conformal mapping
|   |-- linear_cascade.py       Physical planar mapping and extrusion
|   `-- geometry_3d.py          Reference placement and span sections
|-- meshing/
|   |-- meshing_2d.py           2D mesh generation, reports and CLI workflow
|   |-- meshing_3d.py           Reference, deformation, sweep and export stages
|   |-- quality.py             Array-based volume quality definitions
|   `-- inspection.py          Optional VTK slices and cell subsets
|-- fluent/                    Planar workflow and explicit 3D placeholder
|-- apps/app/                  Current UI, jobs, previews and browser assets
|-- data/                      Installed default YAML resources
`-- deprecated/
    |-- app.py                 Preserved preceding interactive app
    |-- geometry/              Preserved sizing/wrapping/CAD helpers
    `-- meshing/               Preserved preceding mesh implementations
```

Geometry and meshing modules are reusable without importing an app. Both the
current CLI and current workers call the same reference/volume/export stages.
Configuration adapters expose flat row settings to these stages while keeping
the ordered machine project in the interface. The app owns job orchestration
and presentation, rather than a second meshing implementation.

Temporary example import adapters have been removed following verification.
Callers use canonical package imports directly. Browser assets have one active
owner, `turbo_moc/apps/app/assets`. Historical snapshots under `examples/old/`
were audited and retired as documented below.

## Old-to-new module map

Paths in the following table are relative to the repository root. The left
column identifies the implementation owner before package promotion.

| Previous implementation owner | Canonical owner |
| --- | --- |
| `example_meshing/src/geometry/moc.py` | `src/turbo_moc/geometry/moc.py` |
| `example_meshing/src/geometry/geometry_2d.py` | `src/turbo_moc/geometry/geometry_2d.py` |
| `example_meshing/src/geometry/meridional_channel.py` | `src/turbo_moc/geometry/meridional_channel.py` |
| `example_meshing/src/geometry/linear_cascade.py` | `src/turbo_moc/geometry/linear_cascade.py` |
| `example_meshing/src/geometry/geometry_3d.py` | `src/turbo_moc/geometry/geometry_3d.py` |
| `example_meshing/src/meshing/meshing_2d.py` | `src/turbo_moc/meshing/meshing_2d.py` |
| `example_meshing/src/meshing/meshing_3d.py` | `src/turbo_moc/meshing/meshing_3d.py` |
| `example_meshing/src/meshing/quality.py` | `src/turbo_moc/meshing/quality.py` |
| `example_meshing/src/meshing/inspection.py` | `src/turbo_moc/meshing/inspection.py` |
| `example_meshing/src/fluent/barotropic_model.py` | `src/turbo_moc/fluent/barotropic_model.py` |
| `example_meshing/src/fluent/workflow_2d.py` | `src/turbo_moc/fluent/workflow_2d.py` |
| `example_meshing/src/fluent/workflow_3d.py` | `src/turbo_moc/fluent/workflow_3d.py`, still a placeholder |
| `example_meshing/src/configuration.py` | `src/turbo_moc/configuration.py` |
| `example_meshing/src/configuration_2d.py` | `src/turbo_moc/configuration_2d.py` |
| `example_meshing/src/configuration_project.py` | `src/turbo_moc/configuration_project.py` |
| `example_meshing/src/paths.py` | `src/turbo_moc/paths.py` |
| `example_meshing/src/logging.py` | Existing `src/turbo_moc/reporting.py` reused unchanged; its API already supplies the workflow helpers |
| `example_meshing/src/app/application.py` | `src/turbo_moc/apps/app/application.py` |
| `example_meshing/src/app/workflow.py` | `src/turbo_moc/apps/app/workflow.py` |
| `example_meshing/src/app/worker.py` | `src/turbo_moc/apps/app/worker.py` |
| `example_meshing/src/app/preview.py` | `src/turbo_moc/apps/app/preview.py` |
| `example_meshing/src/app/plots.py` | `src/turbo_moc/apps/app/plots.py` |
| `example_meshing/src/app/assets/` | `src/turbo_moc/apps/app/assets/` |
| Default Gmsh/Fluent YAMLs beside example launchers | Installed copies under `src/turbo_moc/data/`; editable example cases live in `examples/configurations/` |

`geometry/blade.py`, `geometry/spline_tool.py`, the MoC API/solvers, property
plots, rotor library, coupling functions and ADF IO retain their established
owners. Package promotion does not rederive the scientific algorithms or
change nozzle solver inputs and units.

## Preserved legacy modules

| Established import | Preserved implementation owner |
| --- | --- |
| `turbo_moc.app` | `turbo_moc.deprecated.app` |
| `turbo_moc.geometry.annular` | `turbo_moc.deprecated.geometry.annular` |
| `turbo_moc.geometry.radial` | `turbo_moc.deprecated.geometry.radial` |
| `turbo_moc.geometry.meridional` | `turbo_moc.deprecated.geometry.meridional` |
| `turbo_moc.geometry.passage` | `turbo_moc.deprecated.geometry.passage` |
| `turbo_moc.geometry.sizing` | `turbo_moc.deprecated.geometry.sizing` |
| `turbo_moc.geometry.turbogrid` | `turbo_moc.deprecated.geometry.turbogrid` |
| `turbo_moc.meshing.stator_passage` | `turbo_moc.deprecated.meshing.stator_passage` |
| `turbo_moc.meshing.stator_mesh_2d` | `turbo_moc.deprecated.meshing.stator_mesh_2d` |
| `turbo_moc.meshing.stator_mesh_3d` | `turbo_moc.deprecated.meshing.stator_mesh_3d` |
| `turbo_moc.meshing.passage_volume` | `turbo_moc.deprecated.meshing.passage_volume` |
| `turbo_moc.meshing.rotor_mesh_2d` | `turbo_moc.deprecated.meshing.rotor_mesh_2d` |
| `turbo_moc.meshing.wall_spacing` | `turbo_moc.deprecated.meshing.wall_spacing` |
| `turbo_moc.meshing.stator_structured_2d` | `turbo_moc.deprecated.meshing.stator_structured_2d` |

These numerical compatibility paths refer to the same module objects as their
deprecated owners. This preserves function globals, monkeypatching and
existing callers, instead of copying functions into forwarding modules.

### Structured topology and cell repair

The structured O-H mesher remains in
`deprecated/meshing/stator_structured_2d.py`. Its block topology, knot
splitting and transfinite construction remain available to the preserved
workflow. The canonical app currently uses the consolidated unstructured
reference mesher; this merge does not add a structured-topology selector.

The guarded folded-surface repair remains in
`deprecated/meshing/stator_mesh_2d.py`, with `untangle_surface` and its existing
validity helpers (`_consistently_numbered`, `_surface_cells`, `_corner_turns`,
`_invalid_cells`, `inverted_cell_count`). The radial nonconvex-quad split
remains `_split_nonconvex_quads` in
`deprecated/meshing/stator_mesh_3d.py`. Their original workflows continue to
invoke them under their original conditions. They are absent from the
canonical `meshing_2d.py`/`meshing_3d.py` pipeline.

Keeping those helpers beside their callers preserves the legacy function
globals and repair policy. A separate `cell_repair.py` would introduce a new
compatibility boundary without making the current canonical algorithm clearer.
Future activation needs explicit contracts for boundary/periodic constraints,
quality improvement and failure handling; it must not silently change an
accepted mesh.

### Shared array utilities

Four array-only helpers have one active owner in
`turbo_moc.meshing.quality`: `cell_geometry`, `edge_topology`, `face_geometry`
and `unique_plot_segments`. The canonical meshes and preserved legacy meshes
import the same functions. They compute cell/face geometric quantities,
identify shared edges, and deduplicate plot segments without choosing CAD,
mesh topology or workflow policy. Their moved definitions preserve the
original abstract syntax; this is ownership consolidation, not a change to
the calculation.

Original definitions remain in deprecated files for comparison; active calls
use the common functions. CAD construction, topology selection, reporting
workflows and repair remain beside their respective callers where moving
them would change module-global/monkeypatch behavior. Blade fitting and
spline mathematics already share their established canonical modules.

## Interface feature matrix

“Preserved” means a feature remains in the preceding app and its underlying
library. “Deferred” means it is not yet exposed by the current row-based app;
it does not mean the underlying function was deleted.

| Capability | Current row-based app | Preserved preceding app |
| --- | --- | --- |
| Conventional/minimum-length real-fluid MoC | Fresh solve, live progress and result download | Existing solver controls and background progress |
| Nozzle characteristic, axis, wall and T-s diagnostics | Deferred | Preserved |
| Save/load an already calculated nozzle result; pin reference designs | Deferred | Preserved |
| Optional convergent inlet and wall CSV/vector plot downloads | Deferred | Preserved |
| Stator blade fitting from MoC | Implemented through current blade/passage controls | Full/semi parametrization and blade plots |
| Interactive control-point nudge/reset and blade reference overlays | Deferred | Preserved |
| Mass-flow/blade-count sizing modes | Deferred; current annular dimensions are prescribed | Preserved |
| STEP/STL blade, passage and stage CAD exports | Deferred | Preserved where supported by the legacy workflow and CAD dependencies |
| Linear, axial and radial stator mapping | Implemented, with explicit coordinate labels | Legacy axial/radial sizing/wrapping paths preserved |
| Ordered row projects, independent retained row artifacts | Implemented | Preceding phase-based state retained |
| Combined geometry/mesh workspaces and selected-row quality report | Implemented | Preceding Nozzle/Stator/Rotor/Cascade layout retained |
| Meridional geometry and projected span grid beside section mesh | Implemented | Legacy annular/section previews retained |
| Independent 2D reference and 3D generation/export | Implemented; current reference required for sweep | Preserved legacy mesh/download actions |
| One-span-cell linear extrusion | Implemented as an optional 3D setting | Not the current legacy sizing workflow |
| Structured O-H section mesh and fallback behavior | Deferred | Preserved |
| Flow-informed first-layer/endwall spacing from target $y^+$ | Deferred; current section targets are prescribed | Preserved |
| Guarded folded-surface repair/nonconvex radial-cell splitting | Inactive in canonical pipeline | Original behavior preserved |
| Orange surfaces, black 3D edges and configurable subpixel widths | Implemented | Existing viewers retained |
| Persistent VTK scenes/camera, anchored live terminal, row/stage progress | Implemented | Original background/callback presentation retained |
| Quality summary and skewness/orthogonality/aspect-ratio distributions | Implemented | Legacy quality data retained |
| Slice/worst-cell extraction | Numerical backend available; main UI remains simplified | Legacy viewers retained |
| Vortex rotor provider | Existing kernel integrated through canonical blade mapping, section and volume meshing; old planning rows migrate to vortex inputs | Existing vortex design, plots, sizing and meshing preserved |
| Stator-exit/rotor-inlet coupling | Existing local relative velocity triangle reused; calculated inlet fields visible and locked, manual mode remains available | Existing coupling tools preserved |
| Axial/radial stage assembly and positioned CAD exports | Shared mid-gap interface, derived domain caps and assembly-positioned 3D CGNS files; CAD exports deferred | Preserved |
| Fluent 2D automation | Package/script workflow available | Existing mesh export workflow retained |
| Fluent 3D automation and solver-coupled stage setup | Placeholder/future work; geometrically matching row CGNS files are available | No new solver setup introduced by this merge |

The scientific distinction between geometry and mesh remains explicit. A
geometry preview is a display tessellation of the fitted/mapped blade. A mesh
view displays the generated section or volume boundary cells. A shared stage
display assembles independently computed stator/rotor geometry and marks their
shared interface. Row meshes remain separate files and need solver-side
interface pairing. The rotor geometry adapter
maps the existing solver coordinates into the shared conformal convention;
the original vortex and stator/rotor coupling kernels remain unchanged.

Rotor inlet coupling uses the current upstream stator exit and rotor midspan
inlet radius. It is a local velocity-triangle calculation, with no interspace
loss or radial transport model. Axial interspace controls assembly placement
and the shared interface plane; radial clearance follows the supplied radii,
with a shared interface cylinder halfway between them. The same physical
target sets the stator outlet and rotor inlet at every span station, including
axial chord taper. The remaining external domain extensions stay editable.
Internal extension inputs are hidden while paired; their saved values return
in single-row mode. Read-only controls show the actual physical extensions.

The current shared-boundary model requires a positive gap, matching hub radius
and height for axial rows, and matching height and flow direction for radial
rows. Unequal endwall transitions are not synthesized. Invalid dimensions stay
visible for correction and prevent generation. Changing the gap invalidates
both affected row domains and meshes while keeping their MoC/vortex profiles.

Geometry previews and internal mesh arrays remain row-local. For a paired
project, 3D CGNS export applies the assembly translation and angular phase and
audits each internal cap against its physical target. The browser applies the
same placement exactly once; outdated meshes retain their saved placement.
Annular 2D meshes remain row-specific conformal reference grids, not physical
multirow 2D domains. Geometrically matching interface surfaces do not imply
matching nodes or blade counts. Solver interface pairing, pitch-change
treatment and rotating-frame configuration remain downstream setup.

## MoC progress and reporting

The existing `progress.py`, solver progress callbacks and public reporting
functions keep their behavior. Numerical stages forward the MoC callback
through the fresh-solve helper to worker status and live activity.
The promoted workflows reuse `turbo_moc.reporting` unchanged; its existing
API already supplies their helpers. They do not replace the nozzle's progress
machinery.

Every Compute MoC action solves anew. `moc_result.json` is the selected
successful job's output, explicitly passed to downstream stages. There is
no persistent nozzle-solution lookup or force/reuse switch. Reusing fitted
templates and derived display geometry for that selected result remains
appropriate for quick geometric iteration.

Installed default YAML and browser assets are package resources. Job outputs,
derived-geometry stores and logs belong in writable runtime directories;
they must not be written inside an installed package. Example launchers may
choose explicit case output directories.

## Entry points and resource locations

The canonical interface runs as `turbo_moc-design-app` or
`python -m turbo_moc.apps.app`, accepting `--config`, `--port` (default 8052)
and `--output-dir`. The example `examples/apps/run_app.py` delegates to this
entry point. `create_app(config_file=None)` remains available to Python callers.

The established `turbo_moc-app` / `python -m turbo_moc.app` commands retain the
preceding app, implemented under `turbo_moc.deprecated.app`.
`examples/apps/run_app_legacy.py` also retains its preceding app behavior and is
now located alongside the new launcher under `examples/apps/`. Existing editable environments need reinstallation to register
`turbo_moc-design-app`; the Python module and minimal new example launcher
work immediately. Installing `.[mesh-viz]` registers the new console entry.


Default YAML is installed under `turbo_moc/data/gmsh_2D`, `gmsh_3D` and
`fluent_2D`; editable example cases live under `examples/configurations`. App/Gmsh
outputs use `./turbo_moc_output/` or `TURBO_MOC_OUTPUT_DIR`, with `app/`,
`gmsh_2D/` and `gmsh_3D/<config-stem>/` subfolders. Explicit app/CLI output
arguments override the destination. The bundled Fluent callable default uses
runtime `fluent_2D/`; a supplied Fluent YAML retains its relative `output_dir`
and `design_file` semantics. No worker relies on a checkout path or writes
into installed source/resources.

## Merge verification and integration checklist

The completed row architecture had 110 passing regression tests and a fresh
axial/linear Edge workflow with zero browser exceptions, log errors or context
warnings before package promotion. That evidence is retained in the migration
record. Package-promotion acceptance is recorded separately below.

Package promotion passes **112 canonical regression tests** and **32
migration checks**. The final focused retest passes **52 checks** covering
migration, actual axial/tapered/radial inflow/outflow/curved meshes, linear
meshes and shared quality utilities. The final affected shared-helper/legacy
suite passes **48 checks**, with one optional CadQuery skip; it exercises
module/function identity, structured and unstructured meshes, fallback/repair,
stator/vortex-rotor axial/radial sweeps, CGNS and canonical workflows. The
preserved UI passes 12 checks. These suites overlap; the counts must not be
summed as unique coverage. Complete reports and metric arrays match archived
helpers exactly on real axial (9,404 cells) and linear (9,810 cells) grids.

The final wheel contains all ten JavaScript/CSS/YAML resources and correct
console metadata. Fresh installed axial and one-span-cell linear workflows
run outside the checkout, complete both mesh stages and exports, and leave
all 88 installed package files unchanged. SHA256 checks confirm that the 11
protected MoC/progress/reporting and rotor/coupling source files are byte-identical.

A real Edge run against the final wheel passes **17 browser checks**, with
zero exceptions, errors or warnings. It completes a fresh MoC solve with live
progress, reference and volume jobs, gated meridional display, automatic mesh
navigation and a successful CGNS download. Blade copies, edge toggles and
left-sidebar width edits update the existing renderer; a 0.15-pixel edge edit
updates the shader without remounting, and workspace changes retain the
camera/view identity.

Licensed Fluent execution and target-solver acceptance remain separate work.
The optional CadQuery skip does not establish CAD export acceptance. Fully
configured stage CFD setup remains future work; subsequent vortex rotor and
shared-interface integrations have their own focused numerical and app checks.


- [x] Canonical geometry, meshing, Fluent, project and app owners have been defined under `turbo_moc`.
- [x] Legacy app, wrapping/sizing helpers, structured mesher and original repair policies have explicit deprecated owners.
- [x] Public numerical compatibility paths preserve module identity rather than duplicate implementations.
- [x] Main app features and preserved legacy-only features are distinguished in this document.
- [x] Four array-only helpers share canonical `meshing.quality` ownership; caller-specific CAD/topology/repair policy remains local.
- [x] Package/import, legacy behavior and canonical numerical regression checks after promotion.
- [x] Wheel build, packaged resource and console metadata checks pass.
- [x] Final installed-package/runtime and app/CLI checks, including shared utilities and Windows status reads.
- [x] Real final-wheel app workflow: fresh workers, meshes/exports, live progress and renderer lifetime checks.
- [ ] User manual acceptance of the package migration and retained legacy app.
- [ ] Integrate nozzle diagnostics, reference/result persistence and convergent-wall tools into the canonical UI.
- [ ] Integrate control-point editing, sizing, CAD exports and flow-informed wall spacing after their interfaces are agreed.
- [ ] Decide whether to expose structured topology and guarded repair in the canonical workflow; retain existing legacy behavior meanwhile.
- [x] Integrate the existing vortex rotor provider, row-specific meshing and optional local stator-to-rotor inlet coupling; preserve the legacy app and scientific kernels.
- [x] Derive adjacent row domains from one shared physical interface, expose read-only effective extensions and apply assembly placement to paired 3D CGNS exports.
- [ ] Add unequal-endwall transitions, solver interface pairing, rotating frames and target-solver acceptance for full-stage CFD setup.
- [ ] Implement automated Fluent 3D setup and execution.
- [ ] Retire deprecated implementations or historical snapshots only after separate authorization.

## Examples reorganization

Design demonstrations now live in `examples/moc_design/`; their output still
goes to `examples/output/`. Both app launchers live in `examples/apps/`:
`run_app.py` opens the current interface and `run_app_legacy.py` the preceding
interface. `examples/gmsh_2D`, `gmsh_3D`, `fluent_2D` and `fluent_3D` contain
workflow runners and documentation, with no app launchers. Fluent 3D remains
an explicit placeholder. Editable YAML resides in `examples/configurations/`.
Theory, roadmap, migration notes and figures now live in `z_information/`.

Before deleting `examples/old/`, top-level functions were compared with canonical
and deprecated owners. Two unique optional Matplotlib PNG/SVG exporters were
retained in `turbo_moc.deprecated.meshing.preview`, with original function bodies.
The local unresolved-merge backup was retained in
`artifacts/examples_cleanup/gmsh/run_mesh_generation.py.orig`; the audit inventory
resides beside it. Redundant snapshots, obsolete cache wiring and generated
outputs/browser profiles were removed. No numerical algorithm changed.

## Temporary source adapters removed

The `example_meshing/src` import adapters were verified to contain no unique
implementation and removed at the user's request. Consumers now import
`turbo_moc.geometry`, `turbo_moc.meshing`, `turbo_moc.fluent`,
`turbo_moc.apps.app` and package configuration/reporting modules directly.
The historical source paths in the mapping table above identify migration
origins, not files that must remain present. Documentation and teaching figures
are preserved under `z_information/`. The legacy app is unaffected.

## Local example configuration and outputs

The `paths.py` registry and packaged `data/` YAML copies have been removed.
Each workflow example owns its runner, README and adjacent YAML. Runners need
no arguments, resolve configuration against `__file__`, and write to adjacent
`output/`. App samples live only in `examples/apps/cases/`; the app launcher
selects its local case and output. Package numerical entry points accept explicit
configuration paths, while the app has Python initialization defaults independent
of example files. Fluent paths consistently resolve relative to its YAML.
Earlier descriptions of packaged YAML/resources refer to the prior architecture.
