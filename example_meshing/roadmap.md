# TurboMoC development roadmap

Updated 11 October 2026. This is the active implementation blueprint for [the consolidated meshing workflows](readme.md) and [TurboMoC](gmsh_3D/readme.md). It covers shared geometry/meshing services, the app, Gmsh 2D/3D and the existing Fluent 2D workflow. The [preserved V2 roadmap](../examples/old/gmsh_3D_v2/roadmap.md) records the preceding design and profiling context; its implementation observations refer to that snapshot.

The consolidated implementation incorporates the agreed V2 workflow, subsequent layout refinements and the accepted row-based architecture. The latest decision replaces the previous three-step tab structure with Geometry generation and Mesh generation workspaces, while preserving separate numerical stages. The main app emphasizes simple boundary viewing and reporting; advanced inspection remains a backend capability. The [theory guide](theory_guide.md) now explains the coordinate mapping, mesh deformation, volume topology and export, with [reproducible mesh figures](figures/readme.md).

The existing `example_meshing/src/` layout remains the development home. Historical snapshots under `examples/old/` are preserved for manual comparison and were not moved, edited or deleted by this refactor; acceptance or documentation changes do not authorize their removal. The [migration record](migration.md) tracks that comparison separately. Fluent 3D remains a placeholder, with future solver automation explicitly outside the completed implementation.

## Recorded decisions

| ID | Decision | Current behavior |
| --- | --- | --- |
| D1 | Originally three workspaces; superseded by A2 below. | Geometry generation and Mesh generation, with contextual row controls and separate 2D/3D generation actions. |
| D2 | One reference template, allowing annular span scaling/taper. | Fit and place once; annular sections follow the meridional mapping, while linear sections retain one physical shape. |
| D3 | Local desktop browser first. | Dash on loopback port 8052; numerical jobs in dedicated local processes. |
| D4 | Conformal targets plus physical size estimates for annular rows. | Section controls remain in q-space. Linear q-space lengths are physical lengths; annular captions report sampled mapped-height and span-spacing ranges. |
| D5 | Require an up-to-date reference mesh before volume generation. | Both the UI and job service enforce this. The sweep reads the persisted topology/definition. |
| D6 | Originally boundaries, slice and worst-cell inspection; simplified by the latest UI decision. | Main viewer exposes boundary surfaces, edges and blade-count sliders. Slice/worst-cell extraction remains tested in the backend; it is no longer an app control. |
| A1 | Retain true 2D export and 3D extrusion for linear cascades. | Independent 2D/3D products; one span element remains an optional 3D setting. Annular 2D exports are labeled conformal references. |
| A2 | Two persistent Geometry generation/Mesh generation workspaces, with settings grouped by row. | Shared physical views; selected-row section, actions and quality report; one anchored terminal. |
| A3 | Ordered rows with stable IDs and stator/rotor roles. | Each row owns geometry and mesh inputs; current providers support stators, while rotor settings are explicit placeholders. |

## Row-based architecture and current scope

A project has `machine`, shared `workflow` and an ordered `rows` collection.
Machine mode selects stator, rotor or stator + rotor; geometry type selects
linear, axial or radial mapping. Linear cascade is unavailable in combined
mode. Mixed flow is outside the present selectors and schema. Radial direction
follows the specified radii rather than a separate inflow/outflow code path.

Each row has a stable ID, a role, independent geometry/mesh settings and its
own scientific artifact references. The sidebar shows every active row in
a collapsible group; the workspace selects geometry or meshing controls. Row
selection controls section plots, generation actions and the report; available
row geometry is assembled in shared
physical views. Axial rotor interspace derives its relative LE origin from the greater
upstream hub/tip chord; radial radii supply a directed clearance/overlap
diagnostic without an extra gap input. These assembly origins and annular
angular phases do not alter row-local mesh nodes. Independent displayed row meshes are not a coupled
machine mesh.

Rotor-only and combined projects can retain conventional/vortex rotor inputs,
but their rotor numerical actions remain disabled. Stator work remains usable
in combined mode. Future integration must supply a rotor geometry provider,
then define physical row placement, interfaces, moving reference frames,
blade-count compatibility and solver setup. Do not treat a YAML hierarchy as
an implemented rotor-stator coupling.

The linear adapter scales the source profile and its pitch together to the
specified meridional chord, then extrudes through the blade height. It needs
neither radius nor physical blade count. The same 2D mesher and volume
connectivity serve linear and annular rows; periodic mapping supplies a
translation or rotation according to the geometry family. See the
[linear derivation](theory_guide.md#54-linear-cascade-and-physical-extrusion).

The current header links to
[Sustainable Thermal Power, research group at DTU Construct](https://thermalpower.dtu.dk/).
Show all settings is removed; relevant controls follow the workspace within
collapsible groups for all active rows.
The shared meridional figure also shows a planned rotor channel outline and
computed spacing while its blade provider is pending. Axial paired rotors
use `geometry.axial_interspace_mm` (default 0 mm); it is inactive for a solitary
rotor. See the [spacing derivation](theory_guide.md#111-axial-interspace-and-radial-clearance).

The reference placement decision is now explicit: keep its q-space translation fixed across span, scale the underlying centerline shape/chord, and preserve periodic pitch. This intentionally replaces V1's separate centering optimization at every span station.

## 1. Coordinate model and scientific meaning

**Implemented:** a canonical reference definition with serializable fitted blade/passage data, general hub/shroud mapping, fixed angular pitch and span scaling. Captions identify reference span, q/θ/m′/XYZ units and geometry versus mesh bases. General curved channels are exercised through the backend API. The [theory guide](theory_guide.md) derives the general conformal mapping and axial/radial limits, distinguishes prescribed pseudo-streamlines from solved flow, and explains the difference between template pitch and imposed machine pitch.

**Next:** carry the documented coordinate definitions into typed, validated interfaces and explicit units where they prevent mistakes. Expose custom hub/shroud curves without expanding the axial/radial adapters into special meshing branches. Preserve the geometric pseudo-streamline distinction.

A useful design diagnostic would compare original template pitch, imposed machine pitch and conformal chord/pitch ratio. Explain the effect of blade count without implying that a geometrically valid change automatically preserves the original MoC passage flow.

Options are a spline control-point editor, imported meridional curves, or both. An import-first workflow has a smaller implementation scope; an editor enables interactive shape design but needs curve validity and correspondence checks.

Feedback still useful: which curve import format should lead, and should wall correspondence be editable or defined by the normalized spline station?

## 2. Geometry iteration and numerical performance

**Implemented:** separate fitted-template and derived-design reuse for the explicitly selected MoC job; compiled SciPy basis/evaluation; one reference centering search; cheap per-station containment and existing cell-validity checks. Changing span or export controls does not rerun centering.

**Next:** fully lazy 3D display tessellation, a tighter preview error budget, and measured acceleration of the remaining reference clearance search. V2's initial geometry callback still prepares 3D display data, even when its collapsible group is closed. Geometry callbacks show a loading indicator, but are not dedicated numerical workers.

Compare exact segment/curve-distance acceleration against carefully controlled sampling. Replacing dense pairwise distances with nearest sampled nodes alone could change clearance semantics. Likewise, consider Gmsh-native/precomputed size fields only after matching the current inflation/core grading and periodicity.

The current LU/deformation path already uses compiled libraries; replacing it should be driven by larger-case measurements. The [numerical note](gmsh_3D/deformation_note.md) and [theory guide](theory_guide.md#7-extending-the-reference-mesh-across-span) explain the actual system, the linear-solver choice and factorization reuse.

Feedback still useful: what geometry update latency and geometric display error are acceptable?

## 3. Workspaces and control scope

**Implemented:** two persistent Geometry generation/Mesh generation workspaces; machine-mode and geometry selectors; row selection with independent row geometry, section-mesh, span and export groups; a 380 px resizable sidebar; top-level buttons with inline progress; contextual prerequisite tooltips and a warning only when a previous volume is displayed.

**Next:** a shared field schema for labels, units, bounds, tooltips and stage validation. Currently controls are generated from readable YAML paths, with stage-specific backend validation. Editing keeps the last validated project without resetting adjacent valid settings when one visible field is invalid. Structural changes explicitly report discarded invalid edits; numerical job guards require a valid current configuration. Obsolete dynamic axial-gap controls cannot overwrite a newly selected row/family. Comprehensive per-field inline validation remains future work.

Keep Geometry generation and Mesh generation as the two primary workspaces. In Geometry generation, show the selected-row section and shared meridional/extrusion projection together at equal heights; show the available 3D assembly below. In Mesh generation, retain the section/grid views above the collapsible 3D viewer and selected-row report. Annular sections use conformal coordinates; linear sections use physical distances. Use black blade outlines and grey fill, with the passage reference hidden by default. Stack paired plots on smaller screens while retaining their coordinate labels.

The report beside the 3D mesh is a good arrangement: the viewer stays available while the statistics scroll independently. Resizable panes accommodate different monitors. A future maximize-view button could help large meshes; combining all histograms into tiny cards would reduce readability.

Feedback still useful: which controls belong under an Advanced group, and should the sidebar remember collapsed groups between sessions?

## 4. Reusable reference mesh and meridional setup

**Implemented:** an independent 2D meshing action/artifact; actual reference-cell plot; side-by-side meridional grid; span layer/stretch previews without remeshing; mandatory current-reference gate.

The meridional grid uses actual periodic-boundary stations once a reference mesh exists. Before that, both mesh views remain empty with an explicit reference-mesh prerequisite. It shows the grid projected from the intended sweep, including the extended passage and LE/TE locations. It is not an independently generated meridional CFD mesh.

**Next:** link section station/η selections across both panes, label hub/shroud clearly at selected stations, expose explicit spacing tables and the chosen span law, and measure physical first-layer spacing from actual cells. Report in-surface spacing and three-dimensional wall-normal distance as distinct quantities. The displayed height range is currently sampled and in-surface. A later $y^+$ estimate also requires density, viscosity and a friction-velocity estimate or CFD result.

Options for span laws include the current symmetric tanh, one-sided clustering, independently specified endwall spacing, or imported station arrays. Keep all options behind one span-distribution contract.

Feedback still useful: should first/last endwall spacing or span cell count be the primary input?

## 5. Background jobs and user feedback

**Implemented:** row-identified dedicated workers for MoC, 2D, 3D and export (inspection is available in the backend); an always-visible terminal header anchored below the scrolling main content; stage progress beside buttons; live native/solver log; spinner; elapsed time; disabled competing actions; cancellation; immutable configuration snapshots and source-revision checks; previous-artifact retention. Workers check the submitted source fingerprint against the current files; the app does not archive or freeze a copy of the source tree.

Every Compute MoC action and CLI run solves afresh through `geometry/moc.py`, forwarding the solver callbacks. The app saves `moc_result.json` only for handoff from that selected successful job to subsequent stages. It does not discover or reuse prior MoC solutions. Weighted solver stages provide approximate percentages. Native Gmsh operations use an indeterminate bar where a precise fraction is unavailable. The terminal shows work actually performed; it does not invent shell commands for Python API calls.

**Next:** a structured event history with independent stage durations, session/job recovery after refresh or app restart, app-shutdown worker cleanup, bounded derived-geometry/output retention and global concurrency limits across multiple browser sessions.

For local desktop use, one worker per active session is adequate. A persistent job queue/pool is an option if multiple cases need concurrent computation; avoid adding distributed infrastructure before it is required. Cancel currently terminates the worker process rather than offering cooperative interruption inside native Gmsh routines.

Feedback still useful: should closing a project cancel its job, or leave it running for later inspection?

## 6. Geometry, mesh and inspection viewers

**Implemented:** refined geometry tessellation with smooth point normals and welded blade root/tip caps; orange blade surfaces, grey endwalls and black 3D mesh edges; actual mesh datasets; independent controls in each 3D view; browser-only rotated annular blade copies and translated linear copies; configurable display-only 2D/3D edge widths; PyVista/VTK processing of actual volume data for slice/worst-cell extraction; Plotly engineering views and histograms.

The architecture uses Plotly for labeled 2D plots, PyVista as a convenient Python interface for volume filters, and dash-vtk/vtk.js for browser 3D rendering. These tools serve different layers of the same visualization path.

The dense reference plot retains its Plotly GL scene with an empty
`Scattergl` trace when no current mesh is selected; meridional grids use SVG.
The browser owns a persistent React root inside each Dash viewport and reuses
installed dash-vtk components, avoiding Dash child-replacement remounts. One
View and stable actor pool update data or hide unused actors. Surface/width
changes retain cameras; new datasets/copy counts refit. Fifty rapid VTK edits
and twenty Plotly empty/restore cycles preserve their actual objects/canvases
without errors or context warnings. No new frontend package/CDN is required.
Display-only triangle expansion and retained actors still need measured
memory/level-of-detail limits for larger machines.

**Next:** save/reopen custom camera views across sessions, fully lazy geometry
previews, metric coloring with legends, clipping widgets, selected-cell picking
and display-only level of detail for larger boundaries.

Slice/worst-cell extraction is retained as a backend API rather than exposed in the main app. Any future inspection controls should be optional and explain their purpose clearly; do not restore the removed controls by default.

Feedback still useful: should the next inspection feature be metric coloring or direct cell picking?

## 7. Quality definitions and acceptance criteria

**Implemented:** actual volume equiangular face skewness, centroid-connector orthogonality including periodic neighbors transformed by the row's rotation or translation, and edge aspect ratio; summary tables; three stacked 120-bin distribution histograms; original Gmsh worst-cell IDs. Positive Jacobians/SICN and periodic checks remain mandatory.

Definitions are written in the report, numerical note and [theory guide](theory_guide.md#9-mesh-checks-and-interpretation-of-the-quality-report). They are not claimed to be identical to the target solver's quality formulas. High aspect ratio can be intentional near a wall.

**Next:** compare the metrics with a specified CFD solver, add location-aware quality acceptance thresholds, validate centroid/face conventions for nonplanar faces, and correlate worst-cell IDs with picking/highlighting and exported solver reports. Large-volume quality gathering may need chunking to reduce peak memory.

Feedback still useful: which Fluent release should be the first acceptance target, and should low-but-positive quality warn or block export?

## 8. Export, artifacts and project lifecycle

**Implemented:** export independent of remeshing; CGNS filename/BC/unit edits retain mesh keys; audited one-passage CGNS/ADF output; configuration/log/report downloads; numerical arrays saved without pickle; immutable successful artifact publication.

The 3D exporter writes CGNS 3.4 in ADF, with one combined fluid-volume element section, seven audited outward-facing boundary patches and reciprocal vertex-level periodic maps (rotation for annular rows, translation for linear rows). It reloads the file to check stored base/zone and BC data. The 2D exporter checks topology before writing but does not perform that serialized-file readback. These are implementation checks; target-reader acceptance remains to be established. In particular, periodic sides retain named BC patches alongside vertex connectivity, an importer-oriented convention discussed in [the theory guide](theory_guide.md#105-what-is-established-before-fluent-runs).

The four teaching plates have recorded case inputs, MoC-input/result provenance, selected dependency versions, mesh counts and validity/periodicity results in [figure_metadata.json](figures/figure_metadata.json). Published metadata belongs to the corresponding generated plates; the [generator](figures/generate_theory_figures.py) now obtains its wall through a fresh MoC solve. A complete source/dependency manifest for every app/CLI job remains future work.

**Next:** explicit project-format versioning and a reopen/history workflow, Fluent import acceptance fixtures, export presets, a reproducibility manifest with complete dependency versions per mesh job, project migration tests and output cleanup controls. Validate periodic interpretation and the CGNS hierarchy with the target reader and, where available, an independent conformance checker. Current app stage keys include source/configuration revisions and the selected MoC job identity. The solver has no persistent cache policy or dependency-signature lookup; dependency provenance should be added explicitly to reproducibility records rather than inferred from a cache.

Browser refresh starts a new session, while files remain on disk. This is not yet a complete save/reopen or project-history feature.

Feedback still useful: single project directory or packaged archive, and which fields must be editable in imported cases?

## 9. Package boundaries and maintainability

**Implemented:** shared services are grouped as `src/geometry/`, `src/meshing/`, `src/fluent/` and `src/app/`, outside the installed package. Geometry has explicit `MeridionalChannel` and `Design.meridional_channel` names. The 2D numerical functions and case workflow share `meshing/meshing_2d.py`; the 3D core, reusable stages and case workflow share `meshing/meshing_3d.py`. Superseded consolidated workflow/core modules are removed rather than retained as forwarding layers. `configuration_project.py`, `configuration.py`, `configuration_2d.py`, `paths.py` and package-scoped `logging.py` are the common support files. The app owns UI/jobs/rendering under `src/app/`; case folders contain thin launchers, YAML and case notes. CLI and app use the same stages, with no imports into historical examples. The 3D/app configuration names match the design groups, flow-domain smoothing is flat, and legacy cases still load. The independent 2D/Fluent workflow intentionally retains its existing `nozzle`/`blade`/`passage.smooth` schema. Comments explain placement, LU reuse, publication and browser scenes. Historical snapshots under `examples/old/` remain untouched for manual comparison.

The shared [theory guide](theory_guide.md), this roadmap and [migration record](migration.md) now live at the `example_meshing/` root. `meshing/quality.py` holds reusable array-based diagnostics; `meshing/inspection.py` isolates optional VTK filters from both numerical core and UI. Configuration readers defer their numerical validation imports to avoid initialization cycles. Fluent 2D has a named `workflow_2d.py`; `workflow_3d.py` is an explicit placeholder. The guide links mathematical operations to their source owners; the [figure generator](figures/generate_theory_figures.py) uses the same shared numerical stages.

**Next:** introduce typed reference/volume/job result objects and shared field metadata, while retaining input adapters for the existing schemas. Continue manual comparison against the preserved historical folders. Promotion of stable services into the installed `src/turbo_moc` package is an optional later architectural decision; the consolidated namespace already runs independently. Retirement of historical folders requires a separate user request.

Avoid a complete rewrite. The useful patterns remain explicit stage inputs/outputs, adapters for IO/rendering/workers, immutable snapshots, and dependency-based invalidation.

## 10. Validation and performance evidence

The existing V2/consolidation tests cover spline basis/fit/evaluation parity, fixed placement/pitch, artifact serialization, dependency keys, progress forwarding, real small meshes on axial/radial inflow/radial outflow/curved channels, periodicity/Jacobians, CGNS auditing, slices/worst cells, live logs, cancellation and app layout routes. Earlier V2 browser checks included the inspection controls before simplification; the consolidated app's browser run covered the retained workflow through CGNS export and the display widths.

Before the row-architecture extension, the grouped implementation passed 54 numerical/app/Fluent regressions. A real headless Edge run verified fresh MoC calculation with live progress and result download, geometry, both reference/meridional mesh views, volume generation, fractional edge widths, the three histograms and CGNS export. Span-only edits retained the reference; repeated MoC calculation started fresh, and cancellation preserved the successful volume. No browser exceptions or shader errors were observed. Default 2D and 3D CLI runs also completed fresh solves, actual meshing, reporting and CGNS export.

The Fluent 2D automation has behavioral tests and a new real prepare-only run covering a fresh MoC solve, 25,519 all-quad cells, 193 periodic pairs, CGNS/report generation and a three-segment barotropic fit. A licensed Fluent solve was not part of consolidation verification, and automated Fluent 3D setup is not implemented. The [migration record](migration.md) distinguishes these implementation checks from manual acceptance and target-reader validation.

The new educational meshes passed the existing section/volume validity and periodicity checks. Their reduced resolutions serve illustration rather than production mesh acceptance. Coordinate inversion, the conformal metric, mapping-Jacobian sign and rotational correspondence were also checked numerically at representative points in axial, radial, tapered axial and curved channels. These checks support the derivation in the theory guide; they do not establish aerodynamic validation or mesh independence.

The earlier default axial downstream geometry/meshing stages measured about 12.25 s including separate CGNS export but excluding the already completed MoC solve, versus the profiled V1 pipeline's 40.54 s. This comparison includes a changed placement policy, a small cell-count change and new report work; it is not a bitwise-output or unprofiled head-to-head result. See [the timing table](gmsh_3D/readme.md#performance-evidence). A fresh current CLI or Compute MoC action also performs the solver calculation; its full wall time is not represented by the historical table. The current 3D CLI records `moc_s` separately. No new production-case timing claim is made from the teaching-figure runs.

**Next:** cold/warm and memory benchmarks with repeated runs; source-stable browser regression automation; signed/tolerance comparisons of geometry and layer placement; actual target-solver import; project recovery and stale-completion tests. Keep benchmarks out of fixed wall-clock unit-test assertions.

The final row-architecture implementation passes **110 tests in 74.32 s**
across eight numerical/project/app/Fluent modules. A real Edge run verifies
fresh axial/linear MoC and mesh generation, independent exports, row retention
and rotor guards, spacing/header controls, one-element extrusion, translated
copies, live progress, cancellation, invalid-edit preservation and restored
physical-size captions. Its final audit reports zero browser exceptions, log
errors or WebGL-context warnings. The review server log is clear and the
server was stopped cleanly.

A fresh linear CLI selected `stator_1` and produced 76,528 cells and 86,148
nodes, minimum SICN 0.03519 and maximum periodic mismatch
$3.18\times10^{-14}$ mm. Its audited CGNS/ADF base dimensions are 3/3.
Review artifacts are under `artifacts/architecture_review/`; the focused
VTK lifecycle record is `artifacts/vtk_lifecycle_review/result.json`.
The [migration record](migration.md) separates these checks from historical
performance evidence and remaining solver/manual acceptance.

## Follow-up implementation order

1. Finish field metadata and inline stage validation; preserve the camera when displayed actor sets change.
2. Add project reopen/job recovery and worker/output lifecycle controls for local use.
3. Establish Fluent acceptance fixtures for planar and rotational-periodic imports, including boundary interpretation and quality-report comparisons.
4. Add custom meridional-curve import, linked 2D/span selection and measured physical layer spacing; add optional metric coloring where it helps inspection.
5. Profile larger cases, then address native sizing fields, exact clearance acceleration or memory chunking where evidence supports it.

Keep the current consolidated source location during these changes. Full Fluent 3D automation, a meridional flow solver, advanced inspection and installed-package promotion can be scoped as separate extensions after their requirements are agreed.

## Earlier implementation checklist

- [x] D1: three workspaces and contextual persistent sidebar.
- [x] D2: one conformal template with general span scaling/taper.
- [x] D3: local browser app with isolated numerical workers.
- [x] D4: conformal inputs and sampled mapped physical height estimates.
- [x] D5: current reference-mesh prerequisite, enforced twice.
- [x] D6 revised: simple boundary-view UI; axis-aligned slice/worst-cell backend retained.
- [x] Explicit 2D and 3D meshing buttons.
- [x] Meridional geometry and side-by-side reference/meridional grids.
- [x] Actual reference boundary stations used for the meridional grid.
- [x] Span-only changes preserve the reference artifact.
- [x] Busy indicator, elapsed time, stage progress and live solver/Gmsh log.
- [x] MoC progress forwarded through the shared fresh-solve entry point.
- [x] Cancellation/failure preserve previous successful artifacts.
- [x] 3D viewer and independent scrollable quality report.
- [x] Only skewness, orthogonality and edge aspect ratio histograms.
- [x] Displayed blade-count and surface/edge controls.
- [x] One reference fit/centering search, retained translation and pitch.
- [x] Compiled spline operations with parity checks.
- [x] LU/mapping/metric equations documented.
- [x] Independent audited CGNS export.
- [x] Numerical checks on axial, radial inflow/outflow and curved channels.
- [x] Browser workflow through export and span-only invalidation; slice/worst-cell extraction retained and checked separately from the simplified UI.
- [x] Separate submission/polling callbacks with a regression check for retained job handles.
- [x] TurboMoC branding and renamed three-step workspaces.
- [x] Sidebar adjusted to 380 px; design groups reordered and machine-type selector moved inside the Meridional channel definition group.
- [x] 3D/app YAML/source design names aligned; flat flow-domain smoothing with legacy-file compatibility.
- [x] Paired, equally sized θ–m′/meridional geometry; grey fill and black outlines; reference line hidden by default.
- [x] Full-width collapsible 3D geometry group with independent display controls.
- [x] Always-visible collapsed terminal anchored at the main pane bottom.
- [x] Progress beside buttons; redundant global artifact-status line removed.
- [x] Simplified 3D controls, full-machine blade-count sliders, no unused 3D controls in 2D.
- [x] Display-only 2D/3D widths and narrower quality-histogram bins.
- [x] Both widths grouped in the left sidebar, with 0.35 px defaults in both YAML cases.
- [x] Antialiased, depth-tested 3D cell edges with working subpixel widths and hidden quad diagonals; width/toggle edits preserve the camera.
- [x] Browser pixel checks at 0.1/0.35/1.8 px; 2D stroke patches and no redraw on 3D display edits.
- [x] Orange 3D blade surfaces restored; 3D mesh edges black.
- [x] Orange grid lines in both 2D meshes; both meridional plots start at zero radius.
- [x] Refined geometry preview with smooth normals and welded root/tip caps.
- [x] Continuous surface-of-revolution hub preview between periodic edges, without a blade cutout.
- [x] Closed preview-shell checks for axial, radial and curved channels.
- [x] Meridional mesh view hidden until an up-to-date 2D mesh exists.
- [x] Successful 2D/3D jobs automatically open their result workspace; failures/cancellations do not navigate.
- [x] Minimal app launcher; app implementation/assets under `example_meshing/src/app/` and shared numerical services under `example_meshing/src/`.
- [x] One consolidated 3D staged workflow, also used by the app, with no historical example imports.
- [x] Consolidated Gmsh 2D and Fluent 2D launchers, YAML and documentation.
- [x] Fluent 3D placeholder only; originals retained for manual verification.
- [x] Previous consolidation: numerical regressions, import isolation, CLI meshes/CGNS and real Fluent preparation verified; historical evidence retained in the migration record.
- [x] General conformal-to-3D theory guide, axial/radial derivations, graph-harmonic deformation/LU and export hierarchy documented.
- [x] Four code-generated teaching plates in PNG/SVG with generator, exact configurations and provenance/validation metadata.
- [x] Numerical spot checks of coordinate inversion, metric, Jacobian and rotational mapping, plus recorded sweep counts.
- [x] Active roadmap and shared theory guide located at the `example_meshing/` root, with updated documentation links.
- [x] Geometry, meshing, Fluent and app services grouped in explicit subpackages.
- [x] One dimensional meshing module per dimension, including callable CLI workflows; separate 3D core/workflow copies removed.
- [x] Explicit `MeridionalChannel` and `Design.meridional_channel` names.
- [x] `geometry/moc.py` always calculates a fresh MoC result; persistent MoC cache/signature/force controls removed.
- [x] Selected successful MoC-job handoff retained separately from fitted-template/derived-geometry reuse.
- [x] Common logging renamed to package-scoped `logging.py`; configuration imports avoid numerical cycles.
- [x] Meshing import/parse checks, axial/radial/planar YAML validation and revised CLI help pass.
- [x] Latest source organization: 54 numerical/app/Fluent regressions pass, including fresh repeated solves, malformed-wall rejection, selected MoC-job handoff and invalidation.
- [x] Current actual small Gmsh meshes validate axial/tapered/radial inflow/outflow/curved periodicity, positive Jacobians, CGNS and inspection.
- [x] Four PNG/SVG teaching plates regenerated from one fresh MoC solve, with result SHA256/provenance and actual axial/radial/tapered mesh validation recorded.
- [x] Real fresh-solve Edge browser workflow through reference/meridional/volume meshing and export after the refactor, with live progress, responsive widths and cancellation checked.
- [x] Real fresh-solve default 2D/3D CLI runs after the refactor, including actual CGNS and reports.
- [x] Real Fluent 2D prepare-only run after the refactor, with fresh MoC, mesh/export and barotropic-model preparation; no licensed solver launch.

- [ ] Fully lazy geometry tessellation with a measured display-error and latency budget.
- [x] Owned persistent React root/VTK View and actor pool pass a 50-edit focused lifecycle audit; surface/width edits retain cameras and new datasets/copy counts refit.
- [ ] Saved custom camera views across sessions and cell picking/metric coloring.
- [ ] Full inline field validation/units schema and per-stage validation badges.
- [ ] Typed reference/volume/job result interfaces and shared field metadata.
- [ ] Custom hub/shroud curve import and linked section/span inspection in the browser.
- [ ] Measured physical layer spacing/3D wall-normal distance and explicit sampled-size tables.
- [ ] Project reopen/history, job recovery and automatic shutdown/retention.
- [ ] Full per-mesh dependency/source reproducibility manifest and project migration tests.
- [ ] Target Fluent import/periodicity/quality acceptance and independent CGNS conformance assessment.
- [ ] Native sizing-field equivalence experiments and large-volume memory work.
- [ ] User's manual comparison and acceptance of the consolidated implementation; preserve originals.

The following are optional extensions, with scope to be agreed before implementation:

- [ ] An advanced inspection workspace; retain the simple main boundary viewer.
- [ ] Interactive hub/shroud curve editor beyond import support.
- [ ] Worker-based geometry previews if latency measurements justify process isolation.
- [ ] Continuous-span collision certification or automatic placement repair beyond sampled containment checks.
- [ ] Flow-informed $y^+$ estimates, using stated fluid/friction assumptions or CFD data.
- [ ] An independently solved/meshed meridional CFD domain.
- [ ] Automated Fluent 3D setup and execution beyond mesh import acceptance.
- [ ] Promotion of stable services into the installed package.
- [ ] Retirement of historical examples only on a separate user request.


## Row-architecture implementation and verification checklist

- [x] Ordered project schema, stable row IDs, active-row validation and legacy flat-stator adaptation.
- [x] Geometry generation/Mesh generation workspaces; collapsible settings for all active rows; independent retained artifacts.
- [x] Shared physical machine views and selected-row section/actions/report.
- [x] Stator/rotor/stator+rotor modes, paired-linear restriction and unsupported-rotor controls.
- [x] Linear chord/height mapping with scaled source pitch and no radius/physical blade-count input.
- [x] Translational linear periodicity in volume assembly, quality and reciprocal CGNS metadata.
- [x] Independent true physical 2D CGNS/MSH and 3D extrusion, including positive-Jacobian one-cell-span meshes.
- [x] Row-identified worker snapshots/progress/publication guards and fresh MoC result handoff.
- [x] Axial/radial project YAMLs, linear/rotor planning examples and real CLI row selection.
- [x] Axial taper-safe interspace, radial directed-gap report and planned rotor channel outline; gap edits retain stator meshes.
- [x] Linked Sustainable Thermal Power / DTU Construct credit; Show all settings removed; requested labels retained.
- [x] Last-validated-project retention, visible invalid-edit preservation, structural-change feedback and stale dynamic-control/job guards.
- [x] Mapped physical first-layer-height and span-spacing captions.
- [x] Persistent React/VTK views and Plotly scenes pass rapid canvas/context/camera checks.
- [x] Final 110-test broad suite and real fresh axial/linear Edge workflow pass, including both exports and cancellation; zero browser exceptions/log errors/context warnings.
- [ ] User manual acceptance of the new workflow.

Rotor numerical integration, coupled machine meshing/export, multistage
interface physics and automated Fluent 3D execution remain future work.
