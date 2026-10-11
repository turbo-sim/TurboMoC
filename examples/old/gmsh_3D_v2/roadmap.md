# V2 implementation roadmap

Updated 10 October 2026. This is the active implementation blueprint for [gmsh_3D_v2](readme.md). The [original assessment](../../app_design_roadmap.md) records the V1 source review, options and profiling evidence; its “current implementation” observations refer to that review snapshot.

V2 implements the agreed workflow and the TurboMoC layout refinements. The main app now emphasizes simple boundary viewing and reporting; advanced inspection remains a backend capability. Remaining items below should guide subsequent changes rather than trigger a broad rewrite.

## Recorded decisions

| ID | Decision | V2 behavior |
| --- | --- | --- |
| D1 | Three workspaces with a persistent contextual sidebar. | Geometric design, Two-dimensional meshing, Three-dimensional meshing; relevant groups appear by workspace, with Show all settings available. Values stay mounted across workspace changes. |
| D2 | One conformal template, allowing current span scaling/taper. | Fit once, scale through the general hub/shroud mapping; optimize reference passage placement once. |
| D3 | Local desktop browser first. | Dash on loopback port 8052; numerical jobs in dedicated local processes. |
| D4 | Conformal targets plus physical size estimates. | Section sizing/inflation controls remain in q-space; the meridional caption reports sampled mapped-height and span-spacing ranges. |
| D5 | Require an up-to-date reference mesh before volume generation. | Both the UI and job service enforce this. The sweep reads the persisted topology/definition. |
| D6 | Originally boundaries, slice and worst-cell inspection; simplified by the latest UI decision. | Main viewer exposes boundary surfaces, edges and blade-count sliders. Slice/worst-cell extraction remains tested in the backend; it is no longer an app control. |

The reference placement decision is now explicit: keep its q-space translation fixed across span, scale the underlying centerline shape/chord, and preserve periodic pitch. This intentionally replaces V1's separate centering optimization at every span station.

## 1. Coordinate model and scientific meaning

**Implemented:** a canonical reference definition with serializable fitted blade/passage data, general hub/shroud mapping, fixed angular pitch and span scaling. Captions identify reference span, q/θ/m′/XYZ units and geometry versus mesh bases. General curved channels are exercised through the backend API.

**Next:** make the coordinate definitions a package-level contract, add units/coordinate types where they prevent mistakes, and expose custom hub/shroud curves without expanding the axial/radial adapters into special meshing branches. Preserve the geometric pseudo-streamline distinction.

Options are a spline control-point editor, imported meridional curves, or both. An import-first workflow has a smaller implementation scope; an editor enables interactive shape design but needs curve validity and correspondence checks.

Feedback still useful: which curve import format should lead, and should wall correspondence be editable or defined by the normalized spline station?

## 2. Geometry iteration and numerical performance

**Implemented:** separate fit/template and full-design caches; compiled SciPy basis/evaluation; one reference centering search; cheap per-station containment and existing cell-validity checks. Changing span or export controls does not rerun centering.

**Next:** fully lazy 3D display tessellation, a tighter preview error budget, and measured acceleration of the remaining reference clearance search. V2's initial geometry callback still prepares 3D display data, even when its collapsible group is closed. Geometry callbacks show a loading indicator, but are not dedicated numerical workers.

Compare exact segment/curve-distance acceleration against carefully controlled sampling. Replacing dense pairwise distances with nearest sampled nodes alone could change clearance semantics. Likewise, consider Gmsh-native/precomputed size fields only after matching the current inflation/core grading and periodicity.

The current LU/deformation path already uses compiled libraries; replacing it should be driven by larger-case measurements. The [numerical note](deformation_note.md) explains the actual system and reuse.

Feedback still useful: what geometry update latency and geometric display error are acceptable?

## 3. Workspaces and control scope

**Implemented:** three workflow workspaces; one persistent sidebar with explicit geometry, section-mesh, span and export groups; optional all-settings view; a 380 px resizable sidebar; top-level buttons with inline progress; contextual prerequisite tooltips and a warning only when a previous volume is displayed.

**Next:** a shared field schema for labels, units, bounds, tooltips and stage validation. Currently controls are generated from readable YAML paths, with stage-specific backend validation. Invalid input messages do not yet provide a comprehensive inline validation experience.

Keep the three primary workspaces. In Geometric design, show θ–m′ and meridional plots together at equal heights in one collapsible group; show the full-width 3D shape below in its own group. Use black blade outlines and grey fill. The passage reference/camberline is hidden by default. On smaller screens stack paired plots; preserve visible labels rather than hiding scientific basis information.

The report beside the 3D mesh is a good arrangement: the viewer stays available while the statistics scroll independently. Resizable panes accommodate different monitors. A future maximize-view button could help large meshes; combining all histograms into tiny cards would reduce readability.

Feedback still useful: which controls belong under an Advanced group, and should the sidebar remember collapsed groups between sessions?

## 4. Reusable reference mesh and meridional setup

**Implemented:** an independent 2D meshing action/artifact; actual reference-cell plot; side-by-side meridional grid; span layer/stretch previews without remeshing; mandatory current-reference gate.

The meridional grid uses actual periodic-boundary stations once a reference mesh exists. Before that, both mesh views remain empty with an explicit reference-mesh prerequisite. It shows the grid projected from the intended sweep, including the extended passage and LE/TE locations. It is not an independently generated meridional CFD mesh.

**Next:** link section station/η selections across both panes, label hub/shroud clearly at selected stations, expose explicit spacing tables and the chosen span law, and measure physical first-layer heights from actual cells. The displayed height range is currently sampled and in-surface, not a 3D blade-normal measurement.

Options for span laws include the current symmetric tanh, one-sided clustering, independently specified endwall spacing, or imported station arrays. Keep all options behind one span-distribution contract.

Feedback still useful: should first/last endwall spacing or span cell count be the primary input?

## 5. Background jobs and user feedback

**Implemented:** dedicated workers for MoC, 2D, 3D and export (inspection is available in the backend); an always-visible terminal header anchored below the scrolling main content; stage progress beside buttons; live native/solver log; spinner; elapsed time; disabled competing actions; cancellation; source/configuration snapshots; previous-artifact retention.

MoC callbacks now pass through the shared cache helper. Weighted solver stages provide approximate percentages. Native Gmsh operations use an indeterminate bar where a precise fraction is unavailable. The terminal shows work actually performed; it does not invent shell commands for Python API calls.

**Next:** a structured event history with independent stage durations, session/job recovery after refresh or app restart, app-shutdown worker cleanup, bounded cache/output retention and global concurrency limits across multiple browser sessions.

For local desktop use, one worker per active session is adequate. A persistent job queue/pool is an option if multiple cases need concurrent computation; avoid adding distributed infrastructure before it is required. Cancel currently terminates the worker process rather than offering cooperative interruption inside native Gmsh routines.

Feedback still useful: should closing a project cancel its job, or leave it running for later inspection?

## 6. Geometry, mesh and inspection viewers

**Implemented:** refined geometry tessellation with smooth point normals and welded blade root/tip caps; orange blade surfaces, grey endwalls and black 3D mesh edges; actual mesh datasets; independent controls in each 3D view; browser-only rotated blade copies from one blade to the full annulus; configurable display-only 2D/3D edge widths; PyVista/VTK processing of actual volume data for slice/worst-cell extraction; Plotly engineering views and histograms.

The architecture uses Plotly for labeled 2D plots, PyVista as a convenient Python interface for volume filters, and dash-vtk/vtk.js for browser 3D rendering. These tools serve different layers of the same visualization path.

**Next:** preserve camera state when actor sets change, fully lazy geometry previews, metric coloring with legends, clipping widgets, selected-cell picking and display-only level of detail for larger boundaries. V2 remounts dash-vtk views when actor sets change to avoid existing listener problems; the camera consequently refits. Width and edge-visibility updates change the surface shader in place and preserve the camera. Hidden views fit when first shown. The screen-space edge shader supports fractional widths, hides quad triangulation diagonals and uses the surface depth test. The display-only triangle expansion costs additional browser memory; large boundary datasets still need measured level-of-detail limits.

Slice/worst-cell extraction is retained as a backend API rather than exposed in the main app. Any future inspection controls should be optional and explain their purpose clearly; do not restore the removed controls by default.

Feedback still useful: should the next inspection feature be metric coloring or direct cell picking?

## 7. Quality definitions and acceptance criteria

**Implemented:** actual volume equiangular face skewness, centroid-connector orthogonality including rotated periodic neighbors, and edge aspect ratio; summary tables; three stacked 120-bin distribution histograms; original Gmsh worst-cell IDs. Positive Jacobians/SICN and periodic checks remain mandatory.

Definitions are written in the report and numerical note. They are not claimed to be identical to the target solver's quality formulas. High aspect ratio can be intentional near a wall.

**Next:** compare the metrics with a specified CFD solver, add location-aware quality acceptance thresholds, validate centroid/face conventions for nonplanar faces, and correlate worst-cell IDs with picking/highlighting and exported solver reports. Large-volume quality gathering may need chunking to reduce peak memory.

Feedback still useful: which solver's import/quality report is the first acceptance target, and should low-but-positive quality warn or block export?

## 8. Export, artifacts and project lifecycle

**Implemented:** export independent of remeshing; CGNS filename/BC/unit edits retain mesh keys; audited one-passage CGNS/ADF output; configuration/log/report downloads; numerical arrays saved without pickle; immutable successful artifact publication.

**Next:** a versioned project schema and reopen workflow, export presets/solver integration, a reproducibility manifest with complete dependency versions per mesh job, project migration tests and output cleanup controls. The nozzle cache already validates source/dependency versions; numerical stage keys currently include source/configuration revisions.

Browser refresh starts a new session, while files remain on disk. This is not yet a complete save/reopen or project-history feature.

Feedback still useful: single project directory or packaged archive, and which fields must be editable in imported cases?

## 9. Package boundaries and maintainability

**Implemented:** a minimal top-level app launcher with implementation/worker/numerical/rendering modules and assets together in `app/`; canonical configuration names match the design groups, flow-domain smoothing fields are flat, and a compatibility adapter loads legacy case files; the V2 app factory has no module-level app creation; numerical stages, job orchestration, plotting, quality and inspection live in separate modules. Existing Gmsh assembly/deformation/export logic is reused rather than copied. Comments explain placement, LU reuse, data publication and browser scene choices.

**Next:** gradually extract shared example functions into package services, introduce typed reference/volume/job result objects and one validation schema, and keep compatibility wrappers for the current examples. V2 still imports reusable functions from V1/2D examples; that is an incremental migration, not the final package architecture.

Avoid a complete rewrite. The useful patterns remain explicit stage inputs/outputs, adapters for IO/rendering/workers, immutable snapshots, and dependency-based invalidation.

## 10. Validation and performance evidence

V2 tests cover spline basis/fit/evaluation parity, fixed placement/pitch, artifact serialization, dependency keys, progress forwarding, real small meshes on axial/radial inflow/radial outflow/curved channels, periodicity/Jacobians, CGNS auditing, slices/worst cells, live logs, cancellation and app layout routes.

The default axial numerical stages measured about 12.25 s including separate CGNS export, versus the earlier profiled V1 pipeline's 40.54 s. This comparison includes a changed placement policy, a small cell-count change and new report work; it is not a bitwise-output or unprofiled head-to-head result. See [the timing table](readme.md#performance-evidence).

**Next:** cold/warm and memory benchmarks with repeated runs; source-stable browser regression automation; signed/tolerance comparisons of geometry and layer placement; actual target-solver import; project recovery and stale-completion tests. Keep benchmarks out of fixed wall-clock unit-test assertions.

## Follow-up implementation order

1. Finish field metadata/validation, session recovery and camera persistence.
2. Add custom meridional-curve import and linked 2D/span selection.
3. Add metric coloring, physical layer measurements and target-solver acceptance checks.
4. Profile larger cases; then address native sizing fields, exact clearance acceleration or memory chunking where evidence supports it.
5. Move stable shared numerical/workflow services into the package.

## Implementation checklist

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
- [x] MoC progress forwarded through the shared cache helper.
- [x] Cancellation/failure preserve previous successful artifacts.
- [x] 3D viewer and independent scrollable quality report.
- [x] Only skewness, orthogonality and edge aspect ratio histograms.
- [x] Displayed blade-count and surface/edge controls.
- [x] One reference fit/centering search, retained translation and pitch.
- [x] Compiled spline operations with parity checks.
- [x] LU/mapping/metric equations documented.
- [x] Independent audited CGNS export.
- [x] Numerical checks on axial, radial inflow/outflow and curved channels.
- [x] Browser workflow through export, slice/worst-cell views and span-only invalidation.
- [x] Separate submission/polling callbacks with a regression check for retained job handles.
- [x] TurboMoC branding and renamed three-step workspaces.
- [x] Sidebar adjusted to 380 px; design groups reordered and machine-type selector moved inside the channel group.
- [x] YAML/source design names aligned; flat flow-domain smoothing with legacy-file compatibility.
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
- [x] Minimal app launcher; implementation and assets grouped under `app/`.
- [ ] Fully lazy geometry tessellation and worker-based geometry previews.
- [ ] Camera persistence across actor changes and cell picking/metric coloring.
- [ ] Optional advanced inspection workspace, if requested; keep the main boundary viewer simple.
- [ ] Full inline field validation/units schema and per-stage validation badges.
- [ ] Custom hub/shroud curve editor/import in the browser.
- [ ] Continuous-span collision proofs or automatic placement repair.
- [ ] Exact physical 3D wall-normal heights/y+ and full sampled-size tables.
- [ ] Independently solved/meshed meridional CFD domain, if later needed.
- [ ] Project reopen/history, job recovery and automatic shutdown/retention.
- [ ] Target-solver import/quality acceptance integration.
- [ ] Native sizing-field equivalence experiments and large-volume memory work.
- [ ] Shared example services migrated into the package.

