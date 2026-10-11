# Meshing consolidation record

The active implementation is grouped under `example_meshing/src/`, outside
`src/turbo_moc`. There is one 2D meshing module, one 3D meshing module and one
app. The [roadmap](roadmap.md) tracks priorities; the
[theory guide](theory_guide.md) explains the mathematical procedure.

## Current source mapping

The historical owners were originally developed under `examples/gmsh`,
`examples/gmsh_3D`, `examples/gmsh_3D_v2` and `examples/fluent`. Their current
snapshot locations are under `examples/old/`; that relocation predates this
refactor. The source paths below identify the preserved snapshots.

| Historical owner or responsibility | Current owner |
| --- | --- |
| [`examples/old/gmsh/geometry_source.py`](../examples/old/gmsh/geometry_source.py) | `src/geometry/geometry_2d.py` |
| 2D MoC calculation | `src/geometry/moc.py`: fresh `solve_nozzle`; persistent cache logic removed |
| 2D runner YAML reader | `src/configuration_2d.py` |
| [`examples/old/gmsh/meshing_source.py`](../examples/old/gmsh/meshing_source.py) and 2D runner | `src/meshing/meshing_2d.py` |
| Original 3D mapping/reference fitting | `src/geometry/meridional_channel.py` |
| Linear physical section/translation adapter | `src/geometry/linear_cascade.py` |
| V2 canonical reference design/span scaling | `src/geometry/geometry_3d.py` |
| Original 3D core, V2 stages and staged CLI | `src/meshing/meshing_3d.py` |
| Volume quality and optional VTK inspection | `src/meshing/quality.py`, `src/meshing/inspection.py` |
| Original 3D validation/current name adapter | `src/configuration.py` |
| Ordered machine projects and per-row adapters | `src/configuration_project.py` |
| Dash UI/jobs/plots/preview/assets | `src/app/` |
| Fluent barotropic model preparation | `src/fluent/barotropic_model.py` |
| Implemented planar Fluent automation | `src/fluent/workflow_2d.py` |
| Future 3D Fluent automation | `src/fluent/workflow_3d.py`: placeholder |
| Shared workflow logging | `src/logging.py` |
| Case/output locations | `src/paths.py` |

The consolidated top-level geometry, meshing, core and separate CLI-workflow
copies have been replaced by these owners, without forwarding wrappers.
The numerical mapping, inflation, deformation, periodicity and connectivity
algorithms are retained. `MeridionalChannel` and `Design.meridional_channel`
make the channel responsibility explicit. Unused image helpers from the old
monolithic sweep were removed; the supported mesh-surface reader and optional
PyVista viewer remain in `meshing/meshing_3d.py`.

`quality.py` remains a reusable numerical metric module. `inspection.py`
isolates optional VTK filters. Both belong to meshing rather than UI code.
The grouped packages can later move into the installed package without
preserving historical path dependencies. Configuration imports numerical
validation functions when called, avoiding circular initialization. Shared
logging is package-scoped and uses the standard `logging` package normally.

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

Current verification is complete for the implemented stator, linear mapping
and rotor-planning scope:

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

## Historical folders preserved

At the user's request, the snapshots in [`examples/old/gmsh`](../examples/old/gmsh/),
[`examples/old/gmsh_3D`](../examples/old/gmsh_3D/),
[`examples/old/gmsh_3D_v2`](../examples/old/gmsh_3D_v2/) and
[`examples/old/fluent`](../examples/old/fluent/) retain their sources,
launchers, configuration, assets, notes and outputs. This refactor did not
move, edit or delete those files. Their launchers reference the historical
implementation; they do not forward to the new tree. Edits are not
synchronized. Retirement requires manual review and a separate user request.

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
- [ ] Historical retirement only after separate authorization.

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
python -m example_meshing.gmsh_3D.run_mesh_app
```

The new minimal launcher accepts `--config` to initialize a project YAML and
`--port` to choose a separate local port. For example, use
`--config example_meshing/gmsh_3D/rotor_only.yaml --port 8053` to inspect the
rotor placeholder beside another app. This loads settings without restoring
session history or adopting old artifacts.

The preserved V2 launcher and new app default to port 8052, so stop one before
starting the other when comparing with the historical setup. Their generated
artifacts remain separate. Each new Compute MoC action solves afresh; compare
the resulting geometry and topology rather than assuming persistent solver
reuse.

- [ ] Compare axial/radial geometry with the preserved V2 app, including caps,
  orange surfaces and the continuous hub preview.
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
