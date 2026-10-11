# Meshing consolidation record

The active implementation is grouped under `src/turbo_moc`, with the current
app under `apps/app/`. There is one canonical 2D meshing module and one
canonical 3D meshing module. The temporary `example_meshing/src/` adapters have been removed; callers
import the canonical package directly. Launchers live under `examples/` and
theory/migration documents under `z_information/`. The
preceding installed app and its distinct numerical workflows remain under
`turbo_moc.deprecated`. The [root merging summary](merging_summary.md) maps
their interfaces and the current/legacy feature boundary. The [roadmap](roadmap.md) tracks priorities; the
[theory guide](theory_guide.md) explains the mathematical procedure.

## Current source mapping

The historical owners were originally developed under `examples/gmsh`,
`examples/gmsh_3D`, `examples/gmsh_3D_v2` and `examples/fluent`. The source paths below identify historical owners; their redundant snapshots
were subsequently retired after the requested feature audit.

| Historical owner or responsibility | Current owner |
| --- | --- |
| `examples/old/gmsh/geometry_source.py` (historical) | `../src/turbo_moc/geometry/geometry_2d.py` |
| 2D MoC calculation | `../src/turbo_moc/geometry/moc.py`: fresh `solve_nozzle`; persistent cache logic removed |
| 2D runner YAML reader | `../src/turbo_moc/configuration_2d.py` |
| `examples/old/gmsh/meshing_source.py` (historical) and 2D runner | `../src/turbo_moc/meshing/meshing_2d.py` |
| Original 3D mapping/reference fitting | `../src/turbo_moc/geometry/meridional_channel.py` |
| Linear physical section/translation adapter | `../src/turbo_moc/geometry/linear_cascade.py` |
| V2 canonical reference design/span scaling | `../src/turbo_moc/geometry/geometry_3d.py` |
| Original 3D core, V2 stages and staged CLI | `../src/turbo_moc/meshing/meshing_3d.py` |
| Volume quality and optional VTK inspection | `../src/turbo_moc/meshing/quality.py`, `../src/turbo_moc/meshing/inspection.py` |
| Original 3D validation/current name adapter | `../src/turbo_moc/configuration.py` |
| Ordered machine projects and per-row adapters | `../src/turbo_moc/configuration_project.py` |
| Dash UI/jobs/plots/preview/assets | `../src/turbo_moc/apps/app/` |
| Fluent barotropic model preparation | `../src/turbo_moc/fluent/barotropic_model.py` |
| Implemented planar Fluent automation | `../src/turbo_moc/fluent/workflow_2d.py` |
| Future 3D Fluent automation | `../src/turbo_moc/fluent/workflow_3d.py`: placeholder |
| Shared workflow logging | `../src/turbo_moc/reporting.py` |
| Case/output locations | `../src/turbo_moc/paths.py` |

The consolidated top-level geometry, meshing, core and separate CLI-workflow
copies were removed in the earlier example consolidation. Package promotion initially retained compatibility adapters; these have
subsequently been verified and removed. There is one canonical numerical implementation.
The numerical mapping, inflation, deformation, periodicity and connectivity
algorithms are retained. `MeridionalChannel` and `Design.meridional_channel`
make the channel responsibility explicit. Unused image helpers from the old
monolithic sweep were removed; the supported mesh-surface reader and optional
PyVista viewer remain in `meshing/meshing_3d.py`.

`quality.py` remains a reusable numerical metric module. `inspection.py`
isolates optional VTK filters. Both belong to meshing rather than UI code.
The grouped packages now live in the installed package without historical
implementation dependencies. Earlier example imports forward to these owners. Configuration imports numerical
validation functions when called, avoiding circular initialization. Shared
workflow logging uses existing `turbo_moc.reporting` and Python logging while
preserving the MoC reporting/progress interfaces.

Every CLI run and Compute MoC action solves anew. No persistent MoC cache
lookup, signature validation or force/reuse option remains. The app stores
`moc_result.json` as the output of the selected successful job; subsequent
stages explicitly read that job. Derived geometry and fitted templates can
still be reused during iteration. A new MoC job invalidates downstream
artifacts even if its input values match the previous solve.

## Row-based project architecture

The accepted architecture adds an ordered project schema while retaining the
flat numerical stator API. New Gmsh/app YAML has three top-level groups:

- `machine`: `design_mode` (`stator`, `rotor`, `stator_rotor`) and
  `geometry_type` (`linear`, `axial`, `radial`).
- `workflow`: shared execution settings.
- `rows`: ordered entries with a stable `id`, `role`, row-local `geometry` and
  row-local `mesh` settings.

`geometry` contains `method_of_characteristics`, `meridional_channel`,
`blade_geometry` and `flow_domain` for an implemented stator. Rotor planning
inputs record `parametrization_type` (`conventional` or `vortex`) and
meridional settings, with no implemented rotor numerical provider. Linear
cascade is unavailable in `stator_rotor` mode. Mixed flow is outside the
current project interface.

The default axial/radial YAMLs now use this structure. New examples provide a
linear stator, combined stator/rotor planning project and rotor-only planning
project. `configuration_project.py` normalizes those files and extracts a
supported row through `row_configuration`. Existing `read_configuration`
callers receive the same flat stator settings when reading a project file;
legacy flat files can be wrapped as `stator_1`. The staged CLI accepts `--row`
for explicit selection. Attempting rotor numerical generation reports the
missing provider rather than producing substitute stator geometry.

The app replaces three workspaces with **Geometry generation** and **Mesh generation**. Each row
retains independent settings, job results and validity identities. Workers
write both a submitted `project.yaml` snapshot and the selected row's flat
`configuration.yaml`; row-prefixed progress identifies the work being done.
Inactive-row and rotor-provider guards apply at job submission as well as
in the UI. Shared
physical views assemble available rows; section plots and quality reports
follow the selected row. Paired axial rotor placement is now derived from
`geometry.axial_interspace_mm` (default 0 mm), measured from the farthest
upstream hub/tip trailing-edge plane. It is inactive for a solitary rotor.
`assembly_placement` resolves display origins without mutating mesh nodes;
`assembly_interspaces` reports minimum axial clearance or directed radial
clearance/overlap. Radial positions follow their radii and have no extra gap
input. The planned rotor meridional channel can be shown even though rotor
blade generation remains unavailable. Angular phase remains an annular display
transform; neither assembly transform creates a coupled machine mesh.

The header now links to Sustainable Thermal Power / DTU Construct, and Show
all settings is removed. Existing flat numerical row inputs and internal
stable IDs remain unchanged. Axial rotor planning YAMLs explicitly include
`axial_interspace_mm: 0.0`.

The linear mapping interprets the common reference plane as physical length,
scales source chord and pitch together, then extrudes through blade height.
It introduces translational periodicity into the sweep, quality-neighbour
mapping and CGNS metadata. Both true 2D and 3D products remain available,
including a one-element-span extrusion. Annular 2D downloads retain the
explicit label of conformal reference mesh. None of these changes remove the
independent historical planar Gmsh/Fluent entry points.

### Architecture acceptance checks

Before package promotion, verification was complete for the implemented
stator, linear mapping and rotor-planning scope:

- [x] All five project YAMLs, supported-row/legacy adapters, stable IDs and active-role guards validate.
- [x] Independent row state/jobs retain artifacts; invalid edits preserve the last validated project and block numerical submission until resolved.
- [x] Linear scaled pitch, physical mapping, translational periodicity, true 2D CGNS/MSH and one/four-span-cell 3D meshes pass coordinate/topology/Jacobian/export checks.
- [x] Axial interspace derives a taper-safe rotor origin; radial directed gaps report clearance/overlap; planned rotor outlines require no MoC and gap edits retain stator meshes.
- [x] Final broad regression suite: **110 tests passed in 74.32 s** across all eight modules (`test_stator_mesh`, `test_stator_mesh_3d`, `test_stator_fluent`, `test_gmsh_3d_v2`, `test_mesh_app_controls`, `test_meshing_consolidation`, `test_machine_project`, `test_linear_cascade`). The scoped app-control suite also passes all 13 checks.
- [x] Real Edge fresh axial/linear workflow passes MoC/progress, geometry, both mesh stages, independent exports, row/rotor guards, spacing/header controls, one-element extrusion, translated copies, cancellation, invalid-edit preservation and mapped-height/span-spacing captions.
- [x] Final browser audit: zero exceptions, log errors or WebGL-context warnings; the review server log contains no errors/tracebacks, and the temporary server was stopped cleanly.
- [x] Persistent React/VTK View and actor-pool checks retain actual objects/camera state through 50 rapid edits; Plotly reference scenes retain their two internal canvases/contexts through 20 empty/restore cycles.
- [x] Fresh linear CLI with `--row stator_1`: 76,528 cells, 86,148 nodes, minimum SICN 0.03519, maximum periodic mismatch $3.18\times10^{-14}$ mm and audited CGNS/ADF dimensions 3/3.
- [ ] User manual acceptance.

The browser's owned React root reuses installed dash-vtk components and avoids
Dash-induced VTK remounts. Dense reference meshes retain an empty `Scattergl`
scene when unavailable; meridional grids use SVG. These maintenance choices
bound renderer/context lifetimes without changing scientific mesh data.

Browser review artifacts are under `artifacts/architecture_review/`:
`browser_check.py`, `result.json`, `browser_errors.json` and screenshots.
The focused VTK audit is under `artifacts/vtk_lifecycle_review/result.json`.
These checks use real fresh MoC and Gmsh jobs; the regression run reports only
dependency deprecation warnings in addition to its passing result.

Rotor geometry, coupled machine domains/interfaces, licensed Fluent execution
and Fluent 3D automation remain outside the completed scope. Historical
retirement still requires a separate request. Earlier evidence below is
preserved for comparison and does not establish aerodynamic mesh independence.

## Promotion into the installed package

The [merging summary](merging_summary.md) provides the complete owner map
and feature matrix. Geometry, mesh stages, quality, inspection, Fluent and
project support now belong under `src/turbo_moc`; current app files belong
under `src/turbo_moc/apps/app/`. Default YAML and browser assets are packaged
resources, while logs/jobs/derived geometry use writable runtime locations.

The preceding installed app is preserved at
`turbo_moc.deprecated.app`, with `turbo_moc.app` compatibility. Six preceding
wrapping/sizing modules and seven preceding meshing implementations have
explicit deprecated owners. Their public numerical import paths alias the
same module objects, preserving function globals and existing callers.
Structured O-H meshing, guarded `untangle_surface` and radial
`_split_nonconvex_quads` keep their original legacy behavior; they are not
activated in the canonical app or mesh pipeline.

Four array-only helpers (`cell_geometry`, `edge_topology`, `face_geometry`,
`unique_plot_segments`) now share `turbo_moc.meshing.quality` ownership across
canonical and legacy meshes. Their moved definitions are AST-identical;
original deprecated definitions remain for comparison. CAD/topology/report
workflows and repair policy remain local to protect function globals and
existing monkeypatch interfaces. Existing `turbo_moc.reporting` is reused
unchanged, rather than receiving a copied logging implementation.

Windows job-status reading retries only transient `PermissionError` sharing
failures, matching atomic status publication. Missing files and invalid JSON
still report errors; the migration does not conceal malformed artifacts.

Package promotion passes 112 canonical tests and 32 migration checks. The
final 52-check focused retest covers migration, actual axial/tapered/radial
inflow/outflow/curved and linear meshes, and shared quality helpers. The final
affected shared-helper/legacy suite passes 48 checks with one optional
CadQuery skip: identity/globals, structured/unstructured/fallback/repair,
stator/vortex-rotor axial/radial sweeps, CGNS and canonical workflows. Its
complete report/metric arrays match archived helpers exactly on axial
(9,404 cells) and linear (9,810 cells) meshes. The preserved UI passes 12
checks. Counts overlap and are not summed as unique coverage.

The final wheel includes all ten JS/CSS/YAML resources and correct console
metadata. Installed fresh axial and one-cell-span linear workflows run from
outside the checkout and generate both meshes and exports, without changing
any of 88 package files. The 11 protected MoC/progress/reporting and rotor/coupling source files
retain byte-identical SHA256 values.

The final-wheel Edge workflow passes 17 checks with zero exceptions, errors
or warnings: fresh MoC/live progress, reference/volume jobs, gated meridional
display, automatic navigation, renderer/copy/edge updates, native sidebar
width editing to 0.15 pixels without remounting, retained workspace camera
identity and an HTTP-200 CGNS download. These checks establish the current
package/runtime workflow; licensed Fluent, optional CAD acceptance and real
rotor/coupled-domain integration remain outside the completed scope.

This promotion adds no new nozzle diagnostics, reference/result tools,
control-point editor, sizing, CAD or real rotor functionality to the canonical
UI. Those capabilities remain in the runnable preceding app; their future
integration is tracked explicitly. The MoC API/solvers, progress callbacks,
reporting interfaces, shared blade/spline routines and ADF IO retain their
established scientific behavior. That promotion retained historical examples; the later cleanup is documented below.

- [x] Canonical package owners and legacy/current capability boundaries documented.
- [x] Preceding app, structured topology and guarded repair policies preserved under explicit deprecated owners.
- [x] Numerical import compatibility retains module identity.
- [x] Package and legacy/canonical regression verification after promotion.
- [x] Wheel build, installed resource and console metadata checks.
- [x] Final installed-runtime and launcher verification, including shared utilities and Windows status reads.
- [x] Real fresh installed app/CLI and final-wheel browser checks after promotion.
- [ ] User manual acceptance and subsequent feature-integration priorities.

## Historical snapshots retired after audit

The later examples-reorganization request authorized removal of `examples/old/`
after checking for unique functionality. Its geometry, meshes, Fluent workflow
and app capabilities have canonical or deprecated package owners. The retired
nozzle-cache wrapper was deliberately superseded by fresh MoC solves.

Two useful Matplotlib exporters, `save_mesh_preview` and
`save_conformal_preview`, were preserved without algorithm changes in
`turbo_moc.deprecated.meshing.preview`. The local merge-conflict backup was
preserved under `artifacts/examples_cleanup/gmsh/run_mesh_generation.py.orig`.
The audit inventory is `artifacts/examples_cleanup/audit.json`. Generated
outputs and browser profiles were reproducible runtime data. The installed
legacy app and deprecated numerical implementations remain available.

Runnable examples now live under `examples/`: `moc_design/`, `apps/`,
`gmsh_2D/3D` and `fluent_2D/3D`. YAML cases live in `configurations/`;
theory remains in `z_information/`; temporary compatibility adapters have been removed.

## Verification record

The first consolidation was verified before the latest package grouping and
fresh-solve change:

- [x] 45 existing meshing/app/Fluent regressions and three import/launcher checks.
- [x] Real small meshes on axial, tapered axial, radial inflow/outflow and curved channels.
- [x] Periodicity, positive Jacobians, quality, connectivity and CGNS audits.
- [x] Planar/radial CLI runs and real Fluent mesh/model preparation.
- [x] Fresh-process imports with historical `examples.*` imports blocked.
- [x] Browser operation through MoC, reference mesh, volume and CGNS, including width controls.

These are historical checks. The subsequent source-grouping and MoC-cache
removal refactor was verified separately, before the row architecture:

- [x] New meshing modules parse with unique top-level definitions and import successfully.
- [x] Axial/radial and independent planar YAML readers validate.
- [x] Revised 2D/3D launcher help passes; obsolete force/cache arguments removed.
- [x] Original historical folders preserved; obsolete copies removed only from the consolidated source.
- [x] Source-grouping regression suite: 54 checks passed across meshing consolidation, planar/volume meshes, Fluent behavior, app controls and worker callbacks.
- [x] Regression checks verify fresh same-input solver calls, malformed-wall rejection, MoC-worker result handoff and downstream invalidation when the selected result changes.
- [x] Worker handoff, cancellation and actual Gmsh meshes on axial/tapered/radial inflow/outflow/curved channels pass, including periodicity, positive Jacobians, CGNS auditing and inspection.
- [x] Real headless Edge workflow after the refactor: fresh MoC calculation and live progress, result download, geometry, reference/meridional meshes, full volume and CGNS export; widths update in place, span changes preserve the reference, and cancellation preserves the successful volume. No browser exceptions or shader errors were observed.
- [x] Four PNG/SVG teaching plates regenerated from one fresh MoC solve, with result SHA256/provenance and actual axial/radial/tapered mesh validation recorded.
- [x] Default 3D CLI run after the refactor: fresh MoC calculation, 239,018 nodes, 211,650 hexahedra and 17,425 prisms; positive minimum Jacobian/SICN, 4,238 periodic pairs, and MSH/CGNS output.
- [x] Default 2D CLI run after the refactor: fresh MoC calculation, 25,519 all-quad cells, minimum scaled Jacobian 0.023, 193 periodic pairs, CGNS, PNG/SVG mesh plots and an HTML report.
- [x] Default Fluent 2D prepare-only run after the refactor: fresh MoC calculation, actual mesh/CGNS/report generation and a three-segment barotropic fit; maximum density/viscosity fit deviations were 0.050%/0.022%, with other reported deviations at most 0.005%. No PyFluent or licensed solver was launched.
- [ ] User manual comparison and acceptance.
- [x] Historical example snapshots subsequently retired after separate authorization and the feature audit above.

Full licensed Fluent execution and target-reader acceptance of the 3D
rotational-periodic export are separate validation tasks. Fluent 3D automation
is not implemented.

Latest end-to-end review artifacts are under `artifacts/refactor_review/`,
including browser screenshots/results and independent `gmsh_2d_cli/` and
`gmsh_3d_cli/` outputs. The default 3D CLI's 4,238 periodic pairs had a maximum
coordinate mismatch of $4.48\times10^{-14}$ mm. These runs validate execution
and topology; they do not establish aerodynamic accuracy or mesh independence.

## Manual review

For the row architecture, open the linear and combined planning YAMLs as
well as axial/radial cases. Confirm two workspaces, physical/conformal labels,
retained row inputs, disabled rotor actions and row-specific results. Generate
a linear 2D export and a one-span-cell volume separately. A shared display
should never be interpreted as a coupled fluid-domain export.


Run from the repository root:

```bash
python -m examples.apps.run_app
```

The minimal launcher delegates to `turbo_moc.apps.app.main` and accepts
`--config` to initialize a project YAML, `--port` to choose a local port and
`--output-dir` for writable runtime artifacts. Installed entry points are
`turbo_moc-design-app` / `python -m turbo_moc.apps.app`; the established
`turbo_moc-app` / `python -m turbo_moc.app` retain the preceding app. For example, use
`--config examples/apps/cases/rotor_only.yaml --port 8053` to inspect the
rotor placeholder beside another app. This loads settings without restoring
session history or adopting old artifacts.

The current app defaults to port 8052; the preceding installed app defaults
to 8050. Their launchers now live together in `examples/apps/`. Each new
Compute MoC action solves afresh; compare resulting geometry and topology
rather than assuming persistent solver reuse.

- [ ] Compare axial/radial geometry with the preceding installed app and theory
  figures, including caps, orange surfaces and the continuous hub preview.
- [ ] Generate the 2D reference; verify both grids, live terminal and automatic
  navigation. The meridional grid remains empty until the reference is current.
- [ ] Change span controls: the reference stays current and the volume becomes
  outdated. Change blade or section sizing: both meshes require regeneration.
- [ ] Compute MoC again with unchanged inputs: a new selected result invalidates
  downstream geometry and mesh identities.
- [ ] Generate/export the volume; inspect quality, surfaces, blade count and widths.
- [ ] For Fluent preparation use a copied workflow YAML with `prepare_only: true`;
  pass it with `--config` and check its relative `design_file`.

Gmsh can renumber interior nodes/cells between runs. Compare coordinates,
connectivity and quality with appropriate tolerances rather than mesh-file bytes.
Historical timing comparisons excluded a fresh MoC calculation; they are not
end-to-end timings for the current workflow.

## Examples-reorganization verification

- [x] 113 checks passed across example execution, old/new app initialization,
  launcher help from another directory, preserved preview exports, canonical
  app/mesh regressions, linear cascades, legacy volumes and package migration.
- [x] All seven relocated YAML cases validate, including the Fluent design-file link.
- [x] Documentation relative links resolve; workflow folders contain no app launcher.
- [x] Eleven protected MoC/vortex/progress/reporting modules remain byte-identical.
- [x] The audited `examples/old/` folder was removed after preserving unique helpers.

The scientific-kernel guard initially could not locate Git in the elevated
process PATH; it passed after adding the existing Git shim directory. Remaining
warnings originate from Dash, VTK and PyVista dependencies. Licensed Fluent
execution was not rerun for this directory-only reorganization.

## Temporary example-source retirement

All 21 implementation adapters in `example_meshing/src/` were verified against
existing canonical modules under `src/turbo_moc`. The remaining files contained
only package imports and logging aliases to `turbo_moc.reporting`. No unique
numerical implementation remained. Tests and the teaching-figure generator now
import the package directly; the obsolete source tree was removed. An empty `example_meshing/gmsh_3D/`
directory is locked by another process and may remain until that process closes.
Documentation and figures remain in `z_information/`, where they were moved
before this cleanup. Old `example_meshing.src.*` imports are no longer supported.
The legacy app and deprecated numerical implementations remain available.

Verification after adapter removal: all 152 affected regression checks passed
across the main run and focused retest. Both app interfaces, direct package
imports, worker/mesh paths, real meshes/exports and Fluent workflow behavior
remain covered. All seven current YAML cases validate and relative documentation
links resolve. Initial failures exposed stale paths after the user's case-file
moves and a missing test-artifact directory; these were corrected and retested.

## Local example configuration and outputs

The `paths.py` registry and packaged `data/` YAML copies have been removed.
Each workflow example owns its runner, README and adjacent YAML. Runners need
no arguments, resolve configuration against `__file__`, and write to adjacent
`output/`. App samples live only in `examples/apps/cases/`; the app launcher
selects its local case and output. Package numerical entry points accept explicit
configuration paths, while the app has Python initialization defaults independent
of example files. Fluent paths consistently resolve relative to its YAML.
Earlier descriptions of packaged YAML/resources refer to the prior architecture.

Verification of local configuration ownership: 151 affected regression checks
passed across the full run and focused retest, including both app interfaces,
linear/axial/radial meshes and export/Fluent workflow behavior. Both app launchers
initialize from an unrelated working directory. Runner tests execute each workflow
script with its numerical entry point intercepted, confirming absolute local YAML
and output paths without launching expensive jobs. All live workflow/app YAMLs
validate and relative documentation links resolve. Fluent 3D remains a placeholder;
licensed Fluent execution was not rerun. Numerical kernels remain unchanged.

- [x] Planar workflow execution, construction-figure options and reporting are
  configured under `workflow` in its YAML. `mesh.show_gui` is the sole native
  Gmsh viewer switch; there is no hidden runner override. Settings are validated
  before solving. Older planar YAML copies need the new `workflow` section.
  The Fluent workflow still owns its execution controls and uses the shared
  planar file for geometry/mesh inputs.

YAML-controlled planar workflow verification: all 27 affected checks passed
across the regression run and focused retest. Checks include real 2D meshing and
Fluent regressions, geometry-only execution, viewer enable/disable forwarding,
actual construction plotting with custom size/DPI/formats, log placement,
Matplotlib settings restoration and invalid-input rejection before MoC solving.
Native viewer calls were intercepted for automated verification; no interactive
Gmsh window was opened by the test suite.

Fluent 2D now takes construction viewing/export settings and its Gmsh switch from
its workflow YAML, including geometry size/DPI/formats and residual-plot DPI.
Gmsh 3D now honors its existing viewer/preview switches after volume export and
supports YAML log filenames and PyVista image dimensions/filenames. Runners remain
minimal and argument-free. Verification is limited to parsing, configuration
validation and intercepted workflow/viewer calls; expensive jobs and interactive
viewers are not launched for these refinements.

Current app mesh jobs now include CGNS conversion automatically. Separate export
buttons were removed; completed-job downloads offer CGNS meshes and diagnostics,
not MSH/NPZ files. Internal MSH topology is removed at worker completion. Mesh
artifact identities include CGNS output settings so changed boundary/unit/filename
settings cannot silently reuse an outdated deliverable. Standalone examples and
the deprecated app retain their existing output behavior. Verification is limited
to targeted worker/UI checks with conversion intercepted; no expensive mesh run.
