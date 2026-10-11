# TurboMoC development roadmap

Updated 11 October 2026. This is the active implementation blueprint for [the consolidated meshing workflows](readme.md) and [TurboMoC](../examples/gmsh_3D/readme.md). It covers shared geometry/meshing services, the app, Gmsh 2D/3D and the existing Fluent 2D workflow. The migration record distinguishes earlier design and profiling observations from the current implementation.

The consolidated implementation incorporates the agreed V2 workflow, subsequent layout refinements and the accepted row-based architecture. The latest decision replaces the previous three-step tab structure with Geometry generation and Mesh generation workspaces, while preserving separate numerical stages. The main app emphasizes simple boundary viewing and reporting; advanced inspection remains a backend capability. The [theory guide](theory_guide.md) now explains the coordinate mapping, mesh deformation, volume topology and export, with [reproducible mesh figures](figures/readme.md).

The canonical implementation is now under `src/turbo_moc/`, with the current app under `apps/app/`. `examples/` contains launchers and editable cases; `z_information/` retains theory and teaching figures; obsolete compatibility adapters have been removed. The [merging summary](merging_summary.md) records the full module map and legacy/current feature matrix. Historical snapshots were retired after the requested feature audit; two optional Matplotlib exporters were retained under `turbo_moc.deprecated.meshing.preview`. The [migration record](migration.md) tracks that comparison separately. Fluent 3D remains a placeholder, with future solver automation explicitly outside the completed implementation.

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
| A2 | Two persistent Geometry generation/Mesh generation workspaces, with settings grouped by row. | Shared physical views; independent calculation scope and inspected-row section/report; one anchored terminal. |
| A3 | Ordered rows with stable IDs and stator/rotor roles. | Each row owns geometry and mesh inputs; MoC stators and vortex rotors share the canonical mapping and mesh stages. |

## Row-based architecture and current scope

A project has `machine`, shared `workflow` and an ordered `rows` collection.
Machine mode selects stator, rotor or stator + rotor; geometry type selects
linear, axial or radial mapping. Linear cascade is unavailable in combined
mode. Mixed flow is outside the present selectors and schema. Radial direction
follows the specified radii rather than a separate inflow/outflow code path.

Each row has a stable ID, a role, independent geometry/mesh settings and its
own scientific artifact references. The sidebar shows every active row in
a collapsible group; the workspace selects geometry or meshing controls. Row
inspection controls section plots and the report, while calculation checkboxes
select rows for sequential jobs and independent visibility checkboxes select
the rows assembled in shared
physical views. Axial rotor interspace derives its relative LE origin from the greater
upstream hub/tip chord; radial radii supply a directed clearance/overlap
diagnostic without an extra gap input. Adjacent domains meet at a shared
physical plane or cylinder. Browser/NPZ meshes remain row-local; paired 3D
CGNS exports apply the assembly origin and angular phase. Combined stage files
retain independent zones that require interface pairing and rotating-frame
setup in the solver.

Rotor-only and combined projects now support the existing vortex-flow rotor
provider. Manual relative-frame inlet inputs remain independent; optional
stator-exit coupling derives the inlet Mach number, flow angle and stagnation
state from the current upstream result and rotor inlet blade speed. Coupled
values remain visible but locked. This reuses the legacy local velocity
triangle without changing its scientific solver; it does not model interspace
losses or radial transport. Physical interface geometry is implemented;
solver coupling, rotating frames, blade-count compatibility and a configured
stage CFD case remain future work.

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
The shared meridional figure shows the row channel outlines and computed
spacing, while the shared blade view assembles available stator and rotor
results. Axial paired rotors
use `geometry.axial_interspace_mm`; new or missing values start at 10% of the
rotor midspan axial chord, $0.1(c_{R,h}+c_{R,t})/2$. Explicit saved values are
preserved, including zero, although a paired shared interface requires a
strictly positive gap before generation. This is an initial value in mm, not a persistent
chord-ratio constraint. The control appears first in paired axial rotor
settings and is inactive for a solitary rotor. See the
[spacing derivation](theory_guide.md#111-axial-interspace-and-radial-clearance).

### Multirow flow-domain consistency: shared physical interface

**Implemented following the user's choice:** one physical interface per
adjacent active row pair, fixed halfway through the blade gap. The upstream
stator inlet and downstream rotor outlet remain independently specified.
The two internal caps derive from the common interface. Shaded read-only
controls show its coordinate and hub/midspan/tip physical extensions; the
meridional view marks the shared plane or radius. Internal YAML chord fractions
remain stored for single-row operation, but are inactive while the interface
is present. There is no competing manual internal-extension mode.

For axial rows, measure the clearance $g$ from the farthest stator trailing edge
and use a planar interface:

$$
z_{R,LE}=z_{S,TE,max}+g,\qquad z_I=z_{S,TE,max}+f g,\qquad f=0.5.
$$

For radial rows, the supplied blade radii already set the physical gap; an
interface at radius $r_I=(1-f)r_{S,TE}+f r_{R,LE}$ works for inflow and outflow.
Locate this interface in each row's conformal chart using
$m'_I=s\ln(r_I/r_{LE})$, where $s$ is the row's radial flow direction. Equal
conformal extension fractions are not equal physical radial distances.

Domain caps target the physical interface at **every span station**. Blade
scaling is independent of the cap targets; the existing mesh deformation
transfers the reference topology to each adjusted section. This keeps one
axial plane even when the upstream blade chord varies across span.

Generation requires positive clearance and compatible endwalls. Axial rows
must have the same hub radius and blade height; radial rows must have the same
blade height and radial flow direction. The UI retains inconsistent edits and
explains the mismatch. Unequal endwall transitions remain future work.

Paired 3D CGNS export applies the assembly translation/rotation, checks every
internal boundary node against its physical target and records the interface
pair. Browser and saved computational grids remain local to each row, with
the recorded transform applied once for display/export. Geometric coincidence
does not require identical interface nodes or blade counts: nonconformal
pairing, pitch-change treatment and rotating-frame setup remain solver tasks.
The annular 2D grid is still a row-specific conformal reference, not a
physically assembled 2D stage.

Changing the gap or interface geometry invalidates both affected row domains
and meshes while retaining valid MoC/vortex profile solutions. Shared-interface
setup is independent of the optional stator-exit velocity-triangle coupling.

The deformation collar reduces only its extra protected core buffer when
space is limited. Shared-cap nodes can slide tangentially while their normal
coordinate and periodic corners remain fixed. A real coupled case with a
2 mm axial gap, unchanged 1 mm inflation target and 25 span layers generated
and exported both rows, with interface mismatch below $1.1\times10^{-14}$ mm.
Its minimum SICN was 0.0211 for the stator and 0.00208 for the rotor: positive
cells alone do not establish acceptable CFD quality. Small gaps and thick
inflation stacks still require user tuning; the app does not silently reduce
the requested layer thickness. See the updated [deformation note](deformation_note.md).

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

The current LU/deformation path already uses compiled libraries; replacing it should be driven by larger-case measurements. The [numerical note](deformation_note.md) and [theory guide](theory_guide.md#7-extending-the-reference-mesh-across-span) explain the actual system, the linear-solver choice and factorization reuse.

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

Every Compute blade design action solves the selected MoC/vortex profiles afresh; stator CLI runs also solve afresh through `geometry/moc.py`, forwarding the solver callbacks. The app saves `moc_result.json` only for handoff from that selected successful job to subsequent stages. It does not discover or reuse prior MoC solutions. Selected rows execute sequentially, publishing each result before resolving the next row's dependencies. Weighted solver stages provide approximate percentages across the row queue. Native Gmsh operations use an indeterminate bar where a precise fraction is unavailable. The terminal accumulates work actually performed; it does not invent shell commands for Python API calls.

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

**Implemented:** generation automatically produces audited one-passage CGNS/ADF output. CGNS filename/BC/unit edits invalidate the app deliverable keys; shared-interface or assembly-placement edits invalidate the affected domains and meshes. The reusable backend can export an existing mesh without remeshing. Project ZIP downloads require a complete current active stage and place combined `stage.cgns` at the top level, alongside settings, logs, quality reports and interface metadata. Row-specific originals remain under `meshes/`. The combined file retains independent row zones, local topology and separate interface sides; node/element IDs are not welded or renumbered. Numerical arrays are saved without pickle and successful artifacts are published atomically.

The 3D exporter writes CGNS 3.4 in ADF, with one combined fluid-volume element section, seven audited outward-facing boundary patches and reciprocal vertex-level periodic maps (rotation for annular rows, translation for linear rows). It reloads the file to check stored base/zone and BC data. The 2D exporter checks topology before writing but does not perform that serialized-file readback. These are implementation checks; target-reader acceptance remains to be established. In particular, periodic sides retain named BC patches alongside vertex connectivity, an importer-oriented convention discussed in [the theory guide](theory_guide.md#105-what-is-established-before-fluent-runs).

The four teaching plates have recorded case inputs, MoC-input/result provenance, selected dependency versions, mesh counts and validity/periodicity results in [figure_metadata.json](figures/figure_metadata.json). Published metadata belongs to the corresponding generated plates; the [generator](figures/generate_theory_figures.py) now obtains its wall through a fresh MoC solve. A complete source/dependency manifest for every app/CLI job remains future work.

**Next:** explicit project-format versioning and a reopen/history workflow, Fluent import acceptance fixtures, export presets, a reproducibility manifest with complete dependency versions per mesh job, project migration tests and output cleanup controls. Validate periodic interpretation and the CGNS hierarchy with the target reader and, where available, an independent conformance checker. Current app stage keys include source/configuration revisions and the selected MoC job identity. The solver has no persistent cache policy or dependency-signature lookup; dependency provenance should be added explicitly to reproducibility records rather than inferred from a cache.

Browser refresh starts a new session, while files remain on disk. This is not yet a complete save/reopen or project-history feature.

Feedback still useful: which fields must be editable in imported cases, and what session/history information should accompany the current project archive?

## 9. Package boundaries and maintainability

**Implemented:** shared services have canonical owners in `turbo_moc.geometry`, `turbo_moc.meshing`, `turbo_moc.fluent` and `turbo_moc.apps.app`. Geometry retains explicit `MeridionalChannel` and `Design.meridional_channel` names. Each dimension has one meshing implementation including its callable workflow; `quality.py` and optional `inspection.py` remain reusable numerical modules. Package-level configuration/paths support the numerical stages; workflow reporting reuses existing `reporting.py` unchanged, without replacing MoC progress. Four array-only helpers in `meshing.quality` serve canonical and legacy meshes; CAD/topology/report/repair policy remains local. App presentation, workers and browser assets live together under `apps/app/`. Cases and thin launchers live under `examples/`, with YAML in `configurations/` and app launchers in `apps/`. Earlier example imports adapt to these owners. Legacy app/wrapping/sizing/structured/repair implementations remain runnable under `turbo_moc.deprecated`, with original numerical import paths preserved as module-identity aliases. Superseded historical example snapshots were retired after audit; unique optional exporters remain archived. See the [merging summary](merging_summary.md) for the feature boundary.

The shared [theory guide](theory_guide.md), this roadmap and [migration record](migration.md) now live under `z_information/`. `meshing/quality.py` holds reusable array-based diagnostics; `meshing/inspection.py` isolates optional VTK filters from both numerical core and UI. Configuration readers defer their numerical validation imports to avoid initialization cycles. Fluent 2D has a named `workflow_2d.py`; `workflow_3d.py` is an explicit placeholder. The guide links mathematical operations to their source owners; the [figure generator](figures/generate_theory_figures.py) uses the same shared numerical stages.

**Next:** introduce typed reference/volume/job result objects and shared field metadata, while retaining input adapters for the existing schemas. Continue manual comparison against the preserved historical folders. Complete package-promotion verification and integrate legacy-only features deliberately, using the documented feature matrix rather than adding overlapping implementations. Retirement of historical folders requires a separate user request.

Avoid a complete rewrite. The useful patterns remain explicit stage inputs/outputs, adapters for IO/rendering/workers, immutable snapshots, and dependency-based invalidation.

## 10. Validation and performance evidence

The existing V2/consolidation tests cover spline basis/fit/evaluation parity, fixed placement/pitch, artifact serialization, dependency keys, progress forwarding, real small meshes on axial/radial inflow/radial outflow/curved channels, periodicity/Jacobians, CGNS auditing, slices/worst cells, live logs, cancellation and app layout routes. Earlier V2 browser checks included the inspection controls before simplification; the consolidated app's browser run covered the retained workflow through CGNS export and the display widths.

Before the row-architecture extension, the grouped implementation passed 54 numerical/app/Fluent regressions. A real headless Edge run verified fresh MoC calculation with live progress and result download, geometry, both reference/meridional mesh views, volume generation, fractional edge widths, the three histograms and CGNS export. Span-only edits retained the reference; repeated MoC calculation started fresh, and cancellation preserved the successful volume. No browser exceptions or shader errors were observed. Default 2D and 3D CLI runs also completed fresh solves, actual meshing, reporting and CGNS export.

The Fluent 2D automation has behavioral tests and a new real prepare-only run covering a fresh MoC solve, 25,519 all-quad cells, 193 periodic pairs, CGNS/report generation and a three-segment barotropic fit. A licensed Fluent solve was not part of consolidation verification, and automated Fluent 3D setup is not implemented. The [migration record](migration.md) distinguishes these implementation checks from manual acceptance and target-reader validation.

The new educational meshes passed the existing section/volume validity and periodicity checks. Their reduced resolutions serve illustration rather than production mesh acceptance. Coordinate inversion, the conformal metric, mapping-Jacobian sign and rotational correspondence were also checked numerically at representative points in axial, radial, tapered axial and curved channels. These checks support the derivation in the theory guide; they do not establish aerodynamic validation or mesh independence.

The earlier default axial downstream geometry/meshing stages measured about 12.25 s including separate CGNS export but excluding the already completed MoC solve, versus the profiled V1 pipeline's 40.54 s. This comparison includes a changed placement policy, a small cell-count change and new report work; it is not a bitwise-output or unprofiled head-to-head result. See [the timing table](../examples/gmsh_3D/readme.md#performance-evidence). A fresh current CLI or Compute MoC action also performs the solver calculation; its full wall time is not represented by the historical table. The current 3D CLI records `moc_s` separately. No new production-case timing claim is made from the teaching-figure runs.

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
- [x] Minimal app launcher; canonical app implementation/assets under `src/turbo_moc/apps/app/` and shared numerical services under `src/turbo_moc/`; earlier example imports retained.
- [x] One consolidated 3D staged workflow, also used by the app, with no historical example imports.
- [x] Consolidated Gmsh 2D and Fluent 2D launchers, YAML and documentation.
- [x] Fluent 3D placeholder only; originals retained for manual verification.
- [x] Previous consolidation: numerical regressions, import isolation, CLI meshes/CGNS and real Fluent preparation verified; historical evidence retained in the migration record.
- [x] General conformal-to-3D theory guide, axial/radial derivations, graph-harmonic deformation/LU and export hierarchy documented.
- [x] Four code-generated teaching plates in PNG/SVG with generator, exact configurations and provenance/validation metadata.
- [x] Numerical spot checks of coordinate inversion, metric, Jacobian and rotational mapping, plus recorded sweep counts.
- [x] Active roadmap and shared theory guide located under `z_information/`, with updated documentation links.
- [x] Geometry, meshing, Fluent and app services grouped in explicit subpackages.
- [x] One dimensional meshing module per dimension, including callable CLI workflows; separate 3D core/workflow copies removed.
- [x] Explicit `MeridionalChannel` and `Design.meridional_channel` names.
- [x] `geometry/moc.py` always calculates a fresh MoC result; persistent MoC cache/signature/force controls removed.
- [x] Selected successful MoC-job handoff retained separately from fitted-template/derived-geometry reuse.
- [x] Historical example logging was package-scoped; promoted workflow helpers now use `turbo_moc.reporting` while preserving MoC reporting/progress behavior.
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
- [x] Canonical geometry/meshing/Fluent/app services promoted into the installed package; legacy implementations and import compatibility preserved.
- [ ] Retirement of historical examples only on a separate user request.


## Row-architecture implementation and verification checklist

- [x] Ordered project schema, stable row IDs, active-row validation and legacy flat-stator adaptation.
- [x] Geometry generation/Mesh generation workspaces; collapsible settings for all active rows; independent retained artifacts.
- [x] Shared physical machine views and independently inspected-row section/report.
- [x] Stator/rotor/stator+rotor modes, paired-linear restriction and vortex rotor controls.
- [x] Linear chord/height mapping with scaled source pitch and no radius/physical blade-count input.
- [x] Translational linear periodicity in volume assembly, quality and reciprocal CGNS metadata.
- [x] Independent true physical 2D CGNS/MSH and 3D extrusion, including positive-Jacobian one-cell-span meshes.
- [x] Row-identified worker snapshots/progress/publication guards and fresh MoC result handoff.
- [x] Axial/radial project YAMLs, linear and executable rotor app examples; stator CLI row selection.
- [x] Axial taper-safe interspace, radial directed-gap report and rotor geometry assembly; gap edits retain profile solutions and invalidate both affected row domains/meshes.
- [x] Shared mid-gap axial plane or radial cylinder with derived per-span caps and visible read-only physical extensions.
- [x] Positive-gap/endwall compatibility checks, meridional interface marker and assembly placement in paired 3D CGNS exports.
- [x] Manual or stator-coupled relative rotor inlet, visible locked computed inputs, and upstream-result dependency guards.
- [x] Linked Sustainable Thermal Power / DTU Construct credit; Show all settings removed; requested labels retained.
- [x] Last-validated-project retention, visible invalid-edit preservation, structural-change feedback and stale dynamic-control/job guards.
- [x] Mapped physical first-layer-height and span-spacing captions.
- [x] Persistent React/VTK views and Plotly scenes pass rapid canvas/context/camera checks.
- [x] Final 110-test broad suite and real fresh axial/linear Edge workflow pass, including both exports and cancellation; zero browser exceptions/log errors/context warnings.
- [ ] User manual acceptance of the new workflow.

Solver-coupled stage setup, endwall transition geometry, interspace-flow models,
multistage interface physics and automated Fluent 3D execution remain future work.


## Package-promotion acceptance checklist

This promotion changes source ownership, resources and launch paths. The
110-test/browser evidence above establishes the preceding row architecture;
post-promotion verification is tracked independently here and in the
[merging summary](merging_summary.md).

- [x] One canonical implementation under `src/turbo_moc`, with app files grouped under `apps/app/`.
- [x] Ordered row/project APIs, stator mappings and current app features retain their architectural contracts.
- [x] Preceding app and six wrapping/sizing helpers have explicit deprecated owners.
- [x] Seven legacy meshing modules, structured O-H topology and original cell-repair policies are preserved; they remain inactive in the canonical pipeline.
- [x] Existing MoC solver/progress interfaces and unchanged reporting implementation are preserved.
- [x] Array-only cell/face/edge geometry and plot-segment helpers share canonical ownership.
- [x] Package imports, canonical/legacy regressions and 32 migration checks; protected MoC/progress/reporting files remain byte-identical.
- [x] Built-wheel resource and console metadata checks.
- [x] Final wheel resources/console metadata and installed axial/linear jobs outside the checkout; package files remain unchanged.
- [x] Final-wheel Edge workflow passes 17 checks with zero exceptions/errors/warnings, including workers, progress, meshes/export, width/copy updates and retained renderer/camera.
- [ ] Canonical UI integration of legacy nozzle diagnostics, result/reference tools, control-point editing, sizing, CAD and flow-informed wall spacing.
- [x] Existing vortex rotor provider integrated into canonical geometry, mesh and app stages; legacy scientific kernels retained unchanged.
- [ ] User manual acceptance; future retirement of deprecated package implementations requires a separate request.


Promotion verification includes 112 canonical regressions, the final
52-check focused migration/numerical retest, and a final 48-check affected
shared-helper/legacy suite with one optional CadQuery skip. Archived-helper
report/metric arrays match exactly on real axial/linear grids. Counts overlap and are not added together.
The [merging summary](merging_summary.md) records the exact scope. Licensed
Fluent and optional CAD acceptance are separate from these package checks.

- [x] Reorganized design scripts, old/new app launchers and workflow runners under `examples/`.
- [x] Centralized editable YAML under `examples/configurations/`; kept theory and compatibility adapters in `z_information/`.
- [x] Audited historical examples and preserved optional preview exporters and the local merge-conflict backup before retirement.

- [x] Verified and removed temporary example-source adapters; tests and figure generation import canonical package modules directly.

## Local example configuration and outputs

The `paths.py` registry and packaged `data/` YAML copies have been removed.
Each workflow example owns its runner, README and adjacent YAML. Runners need
no arguments, resolve configuration against `__file__`, and write to adjacent
`output/`. App samples live only in `examples/apps/cases/`; the app launcher
selects its local case and output. Package numerical entry points accept explicit
configuration paths, while the app has Python initialization defaults independent
of example files. Fluent paths consistently resolve relative to its YAML.
Earlier descriptions of packaged YAML/resources refer to the prior architecture.

- [x] Planar workflow execution, construction-figure options and reporting are
  configured under `workflow` in its YAML. `mesh.show_gui` is the sole native
  Gmsh viewer switch; there is no hidden runner override. Settings are validated
  before solving. Older planar YAML copies need the new `workflow` section.
  The Fluent workflow still owns its execution controls and uses the shared
  planar file for geometry/mesh inputs.

## Sequential multirow calculations and stage export

The calculation checkboxes select any subset of active rows without changing
the machine design mode or the row being inspected. Compute blade design runs
fresh profile calculations in row order, handing the new stator result to a
coupled rotor before its worker starts. The two mesh buttons apply the same
scope, validate all selected prerequisites first and skip current meshes.
Each queue holds an immutable project snapshot, uses isolated per-row workers,
publishes each success immediately and stops after failure or cancellation.
The status identifies the active row and queue position; terminal logs accumulate.

Export covers every active row regardless of calculation scope. Annular stages
require current volumes for all rows. The ZIP's top-level `stage.cgns` preserves
distinct fluid zones, row-qualified names, reciprocal periodic donor references
and both sides of each shared physical interface. Linear cases retain true 2D
and 3D export; if both are current, an additional `stage_2d.cgns` is supplied.
CGNS coordinate multipliers must agree across rows. The stage manifest maps
row IDs to exported zone and boundary names. No solver flow-interface model or
rotating-frame assignment is inferred from packaging.

- [x] Independent per-row calculation and plot-visibility checkboxes, with Inspect row for the section/report.
- [x] Compact row controls; static calculation guidance and interspace details appear on hover.
- [x] Shared meridional mesh projection of visible rows with current 2D meshes; row-local blade-to-blade inspection.
- [x] Solid meridional blade extents and dashed actual inlet/outlet flow-domain extensions; zero radius/span origin retained.
- [x] Sequential design/2D/3D row jobs with current dependency handoff and prerequisite gates.
- [x] Live cumulative logs, queue progress, per-row publication and failure/cancellation handling.
- [x] Complete-stage export gate and combined, dimension-consistent CGNS with distinct row zones.
- [x] Focused Dash callback checks and actual small Gmsh stage exports verify topology and periodicity preservation.
- [x] Small real stage imported in Fluent 24.2: two fluid regions, 14 boundary patches and successful mesh check.
- [ ] Production Fluent/CFX acceptance, periodic/interface setup and rotating-frame validation.
- [ ] Durable session reopening and queue recovery after app-server restart.
