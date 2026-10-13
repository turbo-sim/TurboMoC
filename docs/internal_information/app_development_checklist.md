# Organized development TODOs and exploration prompts


## 0. Quick thoughts
- [x] Ensure that the fluent automation workflow is an strict superset of the geometry/meshing workflow, such that the yaml file of the geometry/meshing could be embedded into a broaded CFD simulation/optimization framework.
- [ ] Decide whether to expose structured topology and guarded repair in the canonical workflow; retain existing legacy behavior meanwhile.
- [ ] Create Fluent workflow for stator+rotor 3D cases.


## pACKAGE Refactoring suggestions from chatGPT

- Split `core/thermo_characteristics.py` into fluid-state management and MoC
  characteristic equations, retaining one shared fluid-manager cache.
- Separate CAD exporters from stator parametrization. This would remove the
  remaining shared-geometry dependency on legacy wrapping.
- Consider separating meridional wall definitions, coordinate mapping, and
  section construction when their interfaces are established.
- Separate mesh generation, spanwise deformation, volume assembly, and CGNS
  export from the large mesh modules in independently verified steps.
- Audit duplicate legacy/current geometry and mesh helpers before consolidating
  them. Identical source can still depend on different module globals.
- Consolidate visualization styling only after comparing actual images and
  interactive behavior. Keep app callbacks and live Fluent operations with their
  respective applications.

## 1. App design controls and diagnostics

- [x] **A01 — Add a nozzle design mode.** Expose standalone nozzle design in the app's design-mode selector, with appropriate inputs, geometry and flow diagnostics, and saved-case support.
- [ ] **A02 — Enable stator control-point editing.** Allow users to move the stator's control points and inspect the resulting blade geometry. Define which points are editable and how edits preserve geometric constraints and invalidate dependent results.
- [ ] **A03 — Evaluate a pinned reference design.** **Decision:** determine whether retaining a previous geometry as a reference overlay materially helps design comparison. If useful, define what is pinned, how it is displayed, and when it is cleared.
- [x] **A04 — Restore symmetric and asymmetric blading options.** Identify the intended blade families and the previous behavior, then expose the relevant symmetry choice in the current workflow.
- [x] **A05 — Add a collapsible design-statistics panel.** Place a summary at the bottom of the geometry page with performance estimates from the stator MoC and rotor vortex methods. Include a table of velocity vectors and thermodynamic properties at each flow station; evaluate velocity-triangle plots and other useful distributions. Show units, reference frames, assumptions, and whether results are current.

## 2. Blade parametrization and meridional geometry

- [ ] **B01 — Develop a subsonic rotor parametrization.** Support a camberline-plus-thickness representation for subsonic rotor inlets, including cases such as the EMPOWER turbine. The source proposes implementing the parametrization in TurboDash first and importing it into the turbine-design workflow.
- [ ] **B02 — Add a streamline curvature method for rotor blade design.** Implement a meridional throughflow solver based on the streamline curvature method: radial equilibrium along quasi-orthogonals, iterated streamline positions and curvature, and real-gas properties along each streamline. Use it to design the rotor along the span: spanwise velocity triangles, blade angles and loading, including flared or curved endwalls (B04). Its stream-surface positions and stream-sheet thicknesses can also provide the $r(m)$ and $b(m)$ that the stream-sheet MoC requires (§5 and §14 of `docs/notes/moc_cartesian_and_stream_surface_notes.md`). **Decision:** define where it applies (subsonic or transonic meridional flow; supersonic relative flow inside the blade row is outside the classical method), which loss and deviation models to use, how it couples with the vortex blade (supersonic) and the B01 parametrization (subsonic), and whether an existing implementation (e.g. TurboDash) can be reused.
- [ ] **B03 — Add a conventional constant-width rotor option.** Provide an alternative to vortex-designed blades so that conventional constant-width blading can be compared with the vortex method in the 2027 Turbo Expo paper. Define the parameters held fixed for a fair comparison.
- [ ] **B04 — Extend flaring and hub/shroud shape options.** Support axial and radial blades and linear cascades with varying channel height. Include straight-line flaring, a logarithmic shroud profile where appropriate, and smooth B-spline flaring. **Decision:** investigate whether interpolating B-splines of degree 1, 2, and 3 can provide one coherent interface.
- [ ] **B05 — Add pitch-based scaling to impose a fixed blade count.** The current app imposes the meridional chord and derives the natural blade count $N^*=2\pi/(k_d C')$, rounded up (`apps/app/blade_counts.py`). Add the alternative used by the legacy app: impose $N$, set the reference pitch $2\pi R_{\rm ref}/N$ equal to the scaled design pitch, and derive the chord from the solver's pitch-to-chord ratio $k_d$, so the designed passage is reproduced exactly with the chosen count. Expose this as a per-row sizing selector ("Chord" / "Blade count"). For axial rows, keep the hub/tip chord ratio, or use chord $\propto r$ for exact similarity along the span. For radial rows, the chosen $N$ fixes the conformal chord, i.e. the radius ratio $r_{\rm out}/r_{\rm in}$ (as `consistent_radial_wrap` does in the legacy app). Mass flow follows from the annulus in either mode. See §12.4–12.7 of `docs/notes/moc_cartesian_and_stream_surface_notes.md`.

## 3. Flow definitions, operating conditions, and solver validation

- [x] **C01 — Warn when rotor inlet angles become invalid at high rotational speed.** When excessive speed produces a negative rotor inlet relative angle (`beta_in`), show an explicit warning instead of allowing a silent solver failure, following the existing Mach-number constraint warning pattern. Marked complete in the source; verify current coverage if this area is changed.
- [ ] **C02 — Reconcile backpressure, mass flow, and rotational-speed definitions.** Document how mass flow is estimated and how angular speed, specific speed, and blade velocity ratio are calculated. **Decision:** determine whether an explicit final rotor-exit static pressure should define the stage expansion and its performance parameters. Evaluate the existing use of backpressure and whether an inlet Mach number plus frame transformations can determine the required pressure under stated assumptions. Distinguish stator backpressure, rotor-inlet pressure, and final rotor-exit pressure.
- [ ] **C03 — Offer alternative stator exit-condition inputs.** **Decision:** assess an exit-target selector with “Backpressure” and “Isentropic Mach number” modes.  Define the thermodynamic reference state and absolute/relative frame, and ensure only one exit target is active at a time.
- [ ] **C03 — Offer alternative rotor exit-condition inputs.** **Decision:** assess an exit-target selector with “Backpressure” and “Isentropic Mach number” modes.  Define the thermodynamic reference state and absolute/relative frame, and ensure only one exit target is active at a time.

## 4. Mesh sizing, wall layers, and quality inspection

- [ ] **D01 — Improve spanwise mesh discretization.** Allow independent hub and tip target sizes, inflation thicknesses, and expansion ratios, with a defined transition through the span. Coordinate these controls with the wall-layer settings in D03.
- [ ] **D02 — Define mesh sizing in mapped and physical coordinates.** Decide whether users specify physical lengths or sizes in the reference `(theta, m')` plane, and show the equivalent interpretation. Apply the policy consistently to blade and domain sizes and to blade, hub, and shroud first-cell heights. Derive and verify the local scaling from the coordinate map: the source notes `dm' = dx/r` and `dtheta = dy/r`, whose coordinate definitions and applicability must be checked. The implemented approach should cover axial, radial and mixed flow blades.
- [ ] **D03 — Configure blade, hub, and shroud inflation independently.** For each wall family, specify total layer thickness, growth ratio, and first-cell height. Calculate the resulting number of cells analytically and display it in a read-only field. Define integer rounding and whether the last layer is adjusted to reach the requested thickness; report the thickness actually obtained.
- [ ] **D04 — Evaluate a wall-resolution estimator based on y+.** **Decision:** determine whether estimated y+ or a first-cell height for a target y+ would help mesh setup. State the flow properties, wall-shear correlation, characteristic length, and limitations of the estimate.
- [x] **D05 — Inspect the worst-quality mesh cells.** Implemented in the right-hand **Configure display** panel under **Display mode**: worst-N and threshold filters for skewness, orthogonality, and edge aspect ratio; a fixed green-to-red legend with automatic rounded aspect-ratio limits; transparent context from the selected boundary surfaces; and one complete spanwise cell layer at a time. Inspection uses the saved volume, preserves the camera, and leaves numerical calculations and exports unchanged. See the [app guide](../../src/turbo_moc/app/readme.md).

## 5. CAD export, visualization, and CFD post-processing

- [ ] **E01 — Generate and export STEP geometry.** Support a single flow-domain export and a full disk-and-blade assembly suitable for downstream manufacturing or 3D-print preparation. Specify units, solid validity, and whether the assembly is exported as separate parts or fused geometry; distinguish the fluid domain from the printable solid.
- [ ] **E02 — Harmonize 3D visualization in scripts and the app.** Use a consistent PyVista/VTK data and plotting approach, with matching geometry, mesh boundaries, scalar conventions, and camera orientation where practical. Account for the difference between desktop and browser rendering.
- [ ] **E03 — Produce CFD reports from Fluent and CFX results.** Build independent post-processing scripts that read supported case/data files or documented exported datasets and generate a simulation report, potentially using PyVista. The original mention of a “tab” is superseded by its explicit instruction to keep this work outside the app. Define supported formats and report contents before implementation.

## 6. Configuration and workflow consistency

- [ ] **F01 — Separate advanced numerical settings.** Move less frequently changed MoC controls, such as numerical point counts, into an advanced app section. **Decision:** assess a corresponding nested YAML dictionary while preserving defaults and compatibility with existing cases.
- [ ] **F02 — Use descriptive configuration names.** Prepare a naming table for stator MoC and rotor vortex settings, replacing terse mathematical abbreviations with clear English names. Distinguish readable UI labels, such as “Exit Mach number” and “Backpressure,” from machine-readable YAML keys, such as `exit_mach_number` and `backpressure`. Include units, frame conventions, and a migration strategy for existing keys.
- [x] **F03 — Audit YAML consistency across entry points.** Compare app-loaded cases with automated 2D workflow configurations. Identify differences in keys, nesting, defaults, units, validation, and solver behavior, distinguishing intentional workflow-specific settings from accidental incompatibilities.
- [ ] **F04 — Express nozzle and blade curvature inputs relative to the throat half-height.** After uniform scaling only scale-free ratios reach the machine, so $y_t$ itself should not matter. In the solver, however, the kernel-arc radius $\rho_d$, the Sauer throat curvature $\rho_t$ and the blade's `r_trailing` are absolute lengths. Changing $y_t$ alone changes the shape through $\rho_d/y_t$, $\rho_t/y_t$ and $r_{\rm TE}/y_t$: for a nitrogen test case, doubling $y_t$ changed pitch/chord by 7 % and the natural blade count from 29.1 to 27.2, with the same exit Mach number. Expose them as multiples of $y_t$ (e.g. `rho_t_ratio`, `rho_d_ratio`, `r_trailing_ratio`) in `design_nozzle`, `parametrize_stator_blade`, the YAML schema and both apps, so $y_t$ becomes a pure length scale and the inputs read as shape parameters. Convert existing absolute values on load. See §12.7 of `docs/notes/moc_cartesian_and_stream_surface_notes.md`.

## 7. Codebase maintenance

- [ ] **G01 — Audit deprecated code before retirement.** Compare the old and current codebases and inventory everything under `deprecated/`. Identify unique capabilities, active imports, compatibility entry points, and required migrations. Remove code only after confirming that useful behavior is retained or deliberately retired and that supported callers no longer depend on it.



# Harness prompts: explore and demonstrate before implementation

For each task, copy the shared instructions below followed by the matching feature prompt into the harness. Each combined prompt defines an exploration task, not authorization to implement the production feature. C01 is a verification prompt because the source marks it complete.

### Shared instructions for every exploration

```text
Work in the method_of_characteristics repository.
Explore the feature described below before changing
production behavior. Inspect the relevant implementation, existing tests,
example cases, and internal documentation. Verify what is already supported
in the current app, scripted workflows, and deprecated code; documentation
may describe different development stages. Reuse existing capabilities.

Create a small, isolated, reproducible demonstration under a clearly named
exploration directory, using existing small cases where possible. Show the
current behavior and a proposed behavior or preview with identical inputs.
Choose evidence suited to the feature: an image for a visual choice, a short
script with numerical checks for an algorithm, or a generated comparison
report for an audit. A mockup must be labelled as a mockup; it is not proof
that the feature works. Do not integrate the prototype into production code.

Run the demonstration when dependencies and inputs are available. Record
the exact command, inputs, units, assumptions, checks, and generated artifact
paths in a short Markdown report. Include the key evidence in your response,
with links to the script and outputs. State what passed, failed, or could not
be run; never substitute invented results for missing data or dependencies.
Explain what the evidence establishes and what remains unverified.

Finish with a recommended scope, affected modules/configuration fields,
compatibility implications, and proportionate acceptance checks for later
implementation. Identify unresolved decisions that need Roberto's input.
Stop after the demonstration and report, before production implementation.
```

### A01 — Nozzle design mode

```text
Explore adding standalone nozzle design to the app's design-mode selector.
Locate the existing nozzle solver, examples, and any legacy nozzle interface.
Run one small nozzle case and save its wall/characteristic plot and a summary
of inlet conditions, exit state, and convergence. Produce a labelled UI
preview showing the selector, essential controls, and diagnostic outputs.
Explain how the case would save/load and which app results become stale when
inputs change. Use the computed plot as evidence for the proposed interface.
```

### A02 — Stator control-point editing

```text
Explore editing stator control points. Reuse the existing parametrization
and identify constrained versus movable points. Write a small script that
moves one eligible point and overlays the original and modified profiles,
including the control polygon. Check closure, throat/clearance, edge geometry,
and other constraints actually imposed by the current model. Include one
inadmissible edit and its expected diagnostic. Report which geometry, mesh,
and flow results would need invalidation after a valid edit.
```

### A03 — Pinned reference design

```text
Evaluate whether a pinned reference helps geometry comparison. Generate a
before/after overlay for one meaningful design change with the same camera,
scale, and clear labels. Preview pin, clear, and reference-status controls.
Identify the minimum snapshot data and whether references survive saving or
switching rows. Explain what the overlay reveals that existing views do not,
and recommend whether this feature merits implementation.
```

### A04 — Symmetric and asymmetric blading

```text
Locate the previous symmetry controls and determine which stator/rotor
families they apply to and what symmetry means geometrically. Generate one
example of each supported mode using matched operating/design inputs where
possible. Save a side-by-side profile image and numerical checks of the
relevant symmetry relation, closure, and imposed constraints. Report how to
restore the choice without implying unsupported solver capabilities.
```

### A05 — Design-statistics panel

```text
Explore a collapsible geometry-page summary using an existing stator/rotor
case. Create a standalone report with station-by-station thermodynamic and
velocity quantities, units, absolute/relative frames, and velocity triangles.
Trace each value to its solver output or explicit calculation; flag unavailable
quantities instead of filling them in. Check applicable mass/energy balances
and frame transformations. Add a labelled panel preview and explain how it
would distinguish estimated, incomplete, and stale results. Coordinate the
physical definitions with C02/C03 rather than inventing new conventions.
```

### B01 — Subsonic rotor parametrization

```text
Compare the available camberline-plus-thickness parametrizations in this
repository and TurboDash, including Roberto's more advanced approach if it
is accessible. Confirm the destination intended by the source's “turboboc”
name and the proposed TurboDash-first integration order. Prototype one
representative subsonic profile; use EMPOWER inputs only if documented data
is available. Plot camberline, thickness, and the resulting surfaces, and
check closure, edge treatment, continuity, and self-intersection. Separate
geometric feasibility from aerodynamic validation. Report missing inputs,
ownership decisions, and the smallest reusable integration interface.
```

### B02 — Streamline curvature method for rotor design

```text
Explore a streamline curvature throughflow method for rotor blade design.
Inventory any existing throughflow or radial-equilibrium code in this
repository and in TurboDash. Write a small standalone solver for one axial
rotor with straight endwalls: quasi-orthogonals, radial equilibrium,
iterated streamline positions and curvature, real-gas properties along each
streamline. Check mass conservation per stream tube and convergence of the
streamline positions, and compare with a free-vortex analytical solution.
Plot the meridional streamlines and the spanwise velocity triangles. Report
the range of validity (subsonic/transonic meridional flow, supersonic relative
flow), the loss and deviation models needed, and how its stream sheets could
supply r(m) and b(m) to the stream-sheet MoC.
```

### B03 — Constant-width conventional rotor

```text
Explore a conventional constant-width rotor option for comparison with the
vortex method. First define precisely what “constant width” means here.
Generate matched example profiles and an overlay, plus a table of the inlet/
outlet conditions, turning, chord, pitch, throat, thickness, and quantities
that cannot be matched simultaneously. Check geometric validity. Explain
which comparison is fair for the paper and which performance conclusions
would require CFD or other evidence beyond this geometric demonstration.
```

### B04 — Flaring and hub/shroud shapes

```text
Explore prescribed flaring for axial, radial, and linear-cascade geometries.
Use a small script to compare straight-line, logarithmic where applicable,
and interpolating B-spline degree-1/2/3 channel shapes. Plot the meridional
curves and representative 3D blades/channels. Check endpoint heights,
positive passage height, requested smoothness, and mapping validity; state
which spline degrees can represent which shapes and whether logarithmic
profiles would need approximation. Separately write a feasibility note on
hub-to-shroud MoC design: governing assumptions, available solver support,
and a minimal numerical experiment. Do not conflate a prescribed wall curve
with a flow-designed wall. Recommend a coherent parametrization interface.
```

### B05 — Pitch-based scaling with a fixed blade count

```text
Explore a "Blade count" sizing mode beside the current "Chord" mode. Trace how
the stator MoC template and the rotor vortex blade are scaled today
(blade_counts.py, meridional_channel.build_section) and how the legacy app
sized from the pitch. For one axial stator, one axial rotor and one radial
stator, compute the chord (or radius ratio) implied by a chosen N, then compare
pitch/design, throat/design and section scale at hub, midspan and tip against
the current chord-imposed, rounded-up count. Include chord proportional to r.
Report which inputs become derived or read-only, how existing YAML files load,
and how the mode interacts with the stage interspace and shared interfaces.
```

### C01 — High-speed inlet-angle warning (verification)

```text
Verify the warning already marked complete in the source TODO. Sweep rotor
speed for one documented coupled case across the inlet-angle validity limit.
Record speed, inlet relative angle, applicable Mach constraints, warning
text, and solver outcome. Demonstrate that a valid case still runs and that
an invalid case produces a visible diagnostic rather than a silent crash.
Save a compact results table and, if available, a screenshot of the warning.
Inspect the sign convention instead of assuming all negative angles are
invalid. Report any coverage gap before proposing a production change.
```

### C02 — Pressure, mass flow, and speed definitions

```text
Audit the current formulas for mass flow, angular speed, specific speed, and
blade velocity ratio. Trace their inputs through app cases and scripts and
document units, station definitions, reference frames, and expansion basis.
Write a small script that independently reproduces the current calculations
for one stator/rotor case and compares alternative pressure references.
Check applicable mass/energy balances and dimensional consistency. Determine
whether rotor-inlet Mach number and frame transformations actually suffice
to recover final rotor-exit pressure, listing any extra information needed.
Produce a numerical comparison report and recommend whether final rotor-exit
static pressure should become an explicit input, with compatibility effects.
```

### C03 — Backpressure or exit isentropic Mach number

```text
Identify the solver meant by “rotor MoC” and explore alternative exit targets
only for solvers where the thermodynamics and governing assumptions support
them. Define the isentropic reference state and absolute/relative frame.
For one documented fluid/state, write a small script converting a valid
backpressure to an isentropic Mach target and back, then compare resulting
exit states and geometry when the solver supports both. Report numerical
agreement and tolerances, nonunique or invalid states, and unavailable solver
paths. Preview a mutually exclusive target selector and explain validation
and saved-case behavior before recommending implementation.
```

### D01 — Spanwise discretization

```text
Explore independent hub and tip refinement using one small passage mesh.
Compare current spacing with a prototype using different endwall target
sizes, layer thicknesses, and growth ratios. Save a spanwise-spacing plot,
an endwall mesh image, and a table of cell counts and quality measures.
Check positive cell volumes/Jacobians, requested versus achieved spacing,
periodicity, and the transition between the endwall regions. Include a case
where the requested regions overlap and explain the expected diagnostic.
Coordinate the control definitions with D02/D03.
```

### D02 — Mapped versus physical mesh sizes

```text
Derive the local length/metric transformation for the repository's actual
(theta, m') map, clarifying the source expressions dm'=dx/r and dtheta=dy/r.
Cover axial, radial, and mixed-flow examples and verify the derivation with
finite differences. In a small script, transform representative reference
mesh edges into physical space and compare predicted and measured lengths,
including wall-normal first-cell heights at the blade, hub, and shroud.
Produce an error table and plots showing where one scale factor is inadequate.
Recommend a user-facing sizing policy and explain how it affects blade/core
sizes, endwall spacing, units, and existing YAML values.
```

### D03 — Independent inflation layers

```text
Explore separate blade, hub, and shroud layer controls. Derive the cell-count
calculation from total thickness, first-cell height, and growth ratio,
including the growth-ratio-one limit. Write a short script checking the
formula against explicitly summed layer heights and showing the chosen
integer rounding, adjusted final layer if any, and achieved thickness.
Cover disabled layers, invalid inputs, and thickness below the first-cell
height. Add a read-only count-field preview and, if feasible, a small mesh
image verifying wall-normal spacing and behavior at blade/endwall corners.
```

### D04 — y+ estimation

```text
Inspect existing wall-spacing utilities before proposing a y+ estimator.
Write a small script that estimates y+ from first-cell wall distance and
estimates wall distance from target y+, using an explicitly documented
wall-shear correlation and flow-property assumptions. Distinguish the cell
height from its centroid-to-wall distance. Check units and round-trip
consistency, and report sensitivity to Reynolds number, roughness if relevant,
and uncertain properties. Include a table/plot and a control preview that
labels the result as an estimate. Assess limits for compressible, real-fluid,
and two-phase cases; do not claim CFD validation without simulation evidence.
```

### D05 — Worst-cell inspection

```text
Inspect existing quality and worst-cell extraction APIs and documented app
UI decisions. On one small real mesh, independently verify the ranking of
cells by skewness and orthogonality, including which metric direction is
worse. Render only the worst N cells in geometric context, with a labelled
green-to-red scale and original cell IDs. Verify selected count, ties, and
N beyond the mesh size, and link locations to the quality report. Preview
an optional inspection control and explain whether it would help users tune
geometry or mesh settings without cluttering the main workflow.
```

### E01 — STEP domain and disk/blade assembly

```text
Inventory current and legacy CAD exporters. Generate a minimal STEP fluid
domain and a small disk/blade assembly with existing APIs or an isolated
prototype. Reopen the exported files with an available CAD reader and report
units, part/solid counts, bounding boxes, valid solids, and expected volumes.
Save a rendered image of each export and distinguish separate-part and fused
assembly options. Identify gaps such as blade-root joining or watertightness;
explain what must be checked before calling the solid printable. If CAD
dependencies are unavailable, provide the runnable script and mark export
and reimport validation as unperformed.
```

### E02 — Consistent 3D visualization

```text
Compare the script and app rendering paths for the same geometry/mesh.
Build a small shared PyVista/VTK dataset demonstration and capture matched
views with the same surfaces, scalar range, units, edges, and camera intent.
Check point/cell counts, bounds, boundary selection, and scalar association
numerically. Report differences required by desktop versus browser rendering
and propose shared data/style helpers only where they reduce duplication.
Use images to support the visual comparison and data checks to establish
that both views represent the same object.
```

### E03 — Standalone Fluent/CFX post-processing

```text
Explore an independent report script, outside the app. Inventory available
Fluent and CFX readers and the actual case/data or exported sample datasets;
document formats, optional dependencies, and licensing needs separately.
Process a small available dataset into a report containing domain/boundary
views, representative pressure/velocity fields, mass-flow balance, and
supported station/performance quantities. Verify field names, units, frame
conventions, averaging, and conservation integrals. Save the report and its
figures with an exact reproduction command. If either solver's real data is
missing, demonstrate the report pipeline on clearly labelled synthetic data
or a supported export and state that native reader support is unverified.
```

### F01 — Advanced numerical settings

```text
Inventory MoC controls and distinguish design inputs from numerical settings.
Preview an advanced section and propose a corresponding YAML structure.
Use a short script to compare effective solver inputs for an existing case
and its proposed migrated form, including missing/default values and save/
load round trips. Produce a compatibility report and a small UI image.
Explain which settings stay visible and how to avoid changing numerical
results merely by moving controls or nesting keys. Coordinate with F02/F03.
```

### F02 — Descriptive variable names

```text
Produce a naming table for stator MoC and rotor vortex settings: current key,
proposed YAML key, UI label, meaning, units/frame, and affected consumers.
Verify meanings from code rather than guessing from symbols. Distinguish
user-visible labels from Python/YAML identifiers. Write a small migration
demonstration for one case and show equal effective solver inputs before
and after renaming. Report collisions, old/new key precedence, aliases or
deprecation needs, and cases that cannot be renamed automatically. Do not
perform a repository-wide rename during this exploration.
```

### F03 — YAML consistency audit

```text
Compare app cases with the automated 2D Gmsh and Fluent workflow schemas and
the scripts that consume them. Run a small audit script that reports keys,
nesting, defaults, units, validation, and effective solver/meshing inputs.
Include one equivalent design loaded through both paths where supported.
Classify differences as intentional workflow extensions, aliases, missing
coverage, or behavioral mismatches; cite the consuming code. Save a Markdown
comparison report and a minimal reproducer for each material mismatch.
Recommend a shared configuration contract without assuming identical files
are appropriate for workflows with different responsibilities.
```

### F04 — Curvature inputs relative to the throat half-height

```text
Trace how y_t, rho_t, rho_d and r_trailing enter the MoC solvers and the
stator parametrization. For one small case, run (i) the reference inputs,
(ii) y_t doubled with the other lengths unchanged and (iii) all four doubled.
Report exit Mach number, nozzle length/y_t, pitch/chord, throat/pitch and the
natural blade count, and confirm that (iii) reproduces (i) after scaling.
Propose ratio-based keys, their defaults, UI labels and the conversion of
existing absolute YAML values, and list every consumer that must change.
```

### G01 — Deprecated-code audit

```text
Audit deprecated code without deleting it. Inventory modules, entry points,
imports, tests, and unique features; map each feature to its current owner
or mark it as lacking a replacement. Check dynamic imports and configuration/
launcher references in addition to static imports. Run focused comparisons
for capabilities proposed for retirement and record evidence of equivalent
behavior or intentional differences. Produce a Markdown migration matrix
classifying code as still required, needing migration, or potentially
removable, with uncertainty and downstream compatibility risks. Finish with
a proposed retirement sequence and verification requirements. This prompt
authorizes the audit and evidence only, not code deletion.
```
