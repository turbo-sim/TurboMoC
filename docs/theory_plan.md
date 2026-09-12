# Theory pages: proposed structure and editorial questions

Prepared on 12 September 2026. This is a planning document for the future pages under `docs/source/theory/`. It is intentionally outside the Sphinx source tree. No theory pages have been drafted yet.

## How to use this document

Edit the outline directly and replace the `Your answer:` placeholders with your preferences. Short answers are sufficient; “use the recommendation” is also useful. The proposed defaults are suggestions for discussion. Your answers will guide the first draft in a subsequent prompt.

For a quick review, start with questions **1–6**, which determine the objectives, readership, scope, and mathematical depth. The remaining questions settle the presentation and supporting material.

## 1. Proposed objectives

The theory pages should explain how the physical assumptions and thermodynamic properties determine the characteristic field and, ultimately, the nozzle or blade geometry. A reader should finish with enough understanding to select an appropriate formulation, interpret a computed design, and recognize the limits of the model.

The proposed learning objectives are to:

1. Understand the assumptions behind the steady, inviscid, irrotational, isentropic flow model and the homogeneous equilibrium model (HEM).
2. Explain how equilibrium sound speed affects Mach number, characteristic directions, and nozzle geometry in single-phase and two-phase expansions.
3. Understand the governing equations, characteristic compatibility relations, and boundary conditions used to construct a nozzle with uniform exit flow.
4. Distinguish smooth sonic transitions from flashing transitions at which the flow jumps from subsonic to supersonic conditions, and understand the consequences for initialization.
5. Explain wet-to-dry characteristic crossings without confusing a sound-speed discontinuity with an entropy-producing shock.
6. Understand how the MoC contour is incorporated into a stator passage and which additional geometric choices are required.
7. Interpret published verification and performance results within their assumptions and distinguish them from validation of the present package.

**Question 1 — What is the main purpose of these pages?**

Recommendation: an explanatory reference for researchers and engineers using the package, with enough derivation to understand the method. Other possible emphases are a teaching chapter for newcomers or a mathematical reference detailed enough to support an independent implementation. Please rank these purposes and amend the objectives above.

Your answer:

> [Write here.]

**Question 2 — Who is the primary reader, and what can we assume they know?**

Recommendation: assume undergraduate thermodynamics, fluid mechanics, and basic calculus; introduce characteristics, equilibrium mixture acoustics, and non-ideal expansion behaviour explicitly. Do readers already know compressible flow, partial differential equations, and numerical methods?

Your answer:

> [Write here.]

## 2. Basis in the supplied papers

Both PDFs were read in full, including the methods, results, conclusions, and references. Selected equation and figure pages were also inspected visually to check the extracted text. Page numbers below refer to the supplied PDFs.

| Source | Material relevant to the theory pages | Proposed use |
| --- | --- | --- |
| **Paper A:** Cioffi, Parisi, Agromayor, and Haglind, *Extension of the method of characteristics for the design of supersonic stators in two-phase turbines*. The supplied PDF is marked “Accepted version”, Journal of Turbomachinery, 1 September 2026; 14 pages. | Section 2.1, pp. 3–4: governing equations, HEM closure, equilibrium sound speed, Sauer initialization, kernel and turning regions. Section 2.2, pp. 3–4: stator parametrization. Sections 2.3–3.3, pp. 4–7: CFD/barotropic assumptions and verification. Section 4, pp. 7–11: inlet-condition sensitivity, design and off-design performance, geometric sensitivity. | Main source for the baseline method, stator construction, and style of technical exposition. |
| **Paper B:** Cioffi, Parisi, Agromayor, and Haglind, *Non-ideal compressible flow dynamics of two-phase supersonic expansions and their implications for nozzle design*. Supplied NICFD 2026 manuscript; 6 pages. | Section 2, pp. 2–3: MM wet-to-dry and nitrogen flashing expansions, non-monotonic Mach number, saturation crossings, maximum mass flux. Section 3, pp. 3–4: characteristic crossings and choice of initial line. Section 4, pp. 4–5: comparisons with inviscid barotropic CFD. Section 5, pp. 5–6: scope and non-equilibrium limitations. | Main source for explaining phase-boundary effects and extending the baseline construction to wet-to-dry and flashing flows. |

The papers are complementary: Paper A establishes the equilibrium MoC and stator design framework; Paper B explains how to handle expansions that cross a saturation boundary. The documentation should combine their material in physical and mathematical dependency order, rather than repeat two separate research narratives.

Repository context was checked in the public nozzle API, thermodynamic and characteristic core, conventional and minimum-length solvers, stator geometry functions, rotor module, existing documentation, and rotor notes. This was a scope and consistency review, not a numerical audit or reproduction of the papers' results.

### Points that need particular care in the draft

- **Choking and sonic conditions:** Paper B identifies choking through the maximum isentropic mass flux. For sufficiently subcooled flashing cases, this maximum occurs at the saturation crossing and does not correspond to a smooth state with `M = 1`. The low-subcooling case behaves differently. The explanation should retain both possibilities.
- **Mach number and pressure:** Paper B shows that a given Mach number can correspond to more than one pressure along an expansion. A pressure target and the selected expansion branch therefore deserve explicit discussion. The API recommends pressure as the design target and requires it for the minimum-length solver.
- **Discontinuities and shocks:** In the wet-to-dry example, pressure evolves smoothly while sound speed and Mach number jump. The analogy with weak-shock characteristic handling describes a construction technique; it does not establish that the phase crossing is a physical shock.
- **Evidence:** Paper A compares the CFD model with single-phase MDM experiments, verifies MoC against single-phase and two-phase CFD, and assesses stator performance using viscous CFD. These support different claims. Neither paper establishes experimental validation of the two-phase nozzle designs under non-equilibrium conditions.
- **Barotropic CFD:** Properties are prescribed along an isentrope, but viscous momentum equations can still produce dissipation. The pages should explain this approximation without equating an isentropic property table with an everywhere-isentropic CFD flow.
- **Research results and software:** The supplied papers do not give a dedicated treatment of the package's minimum-length solver or vortex-flow rotor method. Their verification results should not be transferred automatically to those methods or to the present code revision.

## 3. Proposed page structure

The recommended core comprises an overview and seven subject pages. The filenames below are proposals, not files created by this review. A first draft of roughly **4,500–6,500 words**, excluding equations, captions, and optional appendices, would allow a substantial explanation while keeping each page readable.

| Order | Proposed file | Reader's main question | Principal basis |
| --- | --- | --- | --- |
| 0 | `index.md` | What does this theory cover, and where should I start? | Both papers and package scope |
| 1 | `thermodynamic_model.md` | What fluid model closes the flow equations? | A §2.1; B §2 |
| 2 | `isentropic_expansions.md` | How do real-fluid properties affect acceleration and choking? | A §4.1; B §2 |
| 3 | `method_of_characteristics.md` | How are the flow equations solved along characteristics? | A §2.1; current planar formulation |
| 4 | `nozzle_design.md` | How are the initial line, characteristic net, and nozzle wall constructed? | A §2.1; B §3; solver implementations |
| 5 | `saturation_crossings.md` | What changes when an expansion crosses the saturation dome? | B §§2–4 |
| 6 | `stator_design.md` | How does a nozzle contour become a stator blade and passage? | A §2.2 and §4.2; geometry implementation |
| 7 | `verification_and_limits.md` | What evidence supports the method, and what can it predict? | A §§2.3–4.2; B §§4–5 |

### 0. Overview — `index.md`

Introduce the preliminary design problem: prescribed stagnation conditions and an exit target, with a wall shape that produces an adapted expansion and uniform exit flow under the model assumptions. Explain why one-dimensional area relations alone do not determine the internal wave pattern or the wall contour.

State the scope and prerequisites, give a compact map of the pages, and link to the runnable examples and API reference. Keep the broader motivation from ORC, refrigeration, and liquefaction applications to one short paragraph.

### 1. Thermodynamic model — `thermodynamic_model.md`

Proposed subsections:

1. Single-phase states and equilibrium liquid–vapor mixtures; equation of state and independent state variables.
2. HEM assumptions: common velocity, pressure, and temperature, together with phase equilibrium; implications of neglecting slip and finite-rate phase change.
3. Vapor mass fraction, void fraction, and mixture properties. Derive the relation between quality, void fraction, and density so that mass and volume weighting are unambiguous.
4. Equilibrium sound speed: introduce `a² = (∂p/∂ρ)s`, then the explicit two-phase expression used in Paper A, Eq. (11). Explain the saturation derivatives and why additional equilibrium constraints alter acoustic response.
5. Stagnation-to-static closure through constant entropy and total enthalpy; roles of CoolProp/HEOS and the package's thermodynamic interface.

Show the defining sound-speed expression in the main text. Reserve a full derivation from phase relaxation theory for an optional appendix. Include viscosity only where needed to explain the CFD comparison.

### 2. Isentropic expansions — `isentropic_expansions.md`

Proposed subsections:

1. Recover velocity and Mach number from an isentrope: `V²/2 = h0 − h` and `M = V/a`.
2. Mass flux `G = ρV`, its maximum, and the smooth sonic condition. A short derivation can show why the usual `M = 1` result requires differentiability.
3. Single-phase non-ideal expansions: Mach number need not increase monotonically as pressure decreases.
4. Expansions within the two-phase region, wet-to-dry expansions, and flashing expansions; locate them on temperature–entropy diagrams.
5. How changes in inlet quality and temperature affect normalized nozzle length and area ratio, keeping the sensitivity findings tied to Paper A's fluids and operating range.
6. Why pressure is a useful design target when Mach number does not uniquely specify an outlet state.

A perfect-gas comparison would establish familiar limiting behaviour without making constant heat-capacity-ratio relations the basis for the real-fluid theory.

### 3. Method of characteristics — `method_of_characteristics.md`

Proposed subsections:

1. Steady planar continuity, momentum/energy closure, and irrotationality; assumptions needed to obtain the velocity equations.
2. Hyperbolicity in supersonic flow and the two characteristic directions, `dy/dx = tan(θ ± μ)`.
3. Compatibility relations in velocity components, followed by their planar angle–velocity form.
4. Generalized Prandtl–Meyer integral, `dν = sqrt(M² − 1) dV/V`, with its reference state and applicable supersonic branch specified. Connect this to the invariants used by the current code.
5. Interior-point, symmetry-axis, and wall boundary constructions; wall tangency and characteristic intersections.

Include an annotated characteristic-cell diagram and enough algebra to connect the governing equations to the compatibility equations. Define the characteristic signs on the diagram; do not rely on “left-running” and “right-running” without a coordinate convention.

### 4. Nozzle design — `nozzle_design.md`

Proposed subsections:

1. Design inputs, dimensional scaling, throat half-height, and the desired exit state.
2. Critical-state identification and the initial line: smooth Sauer construction versus a flat initial front. Explain the physical applicability of each before discussing software defaults.
3. Rounded-arc conventional nozzle: initial-value region, expansion kernel, and stopping condition.
4. Turning region: characteristic construction, mass conservation, wall recovery, and alignment toward uniform exit flow.
5. Minimum-length construction, if selected in Question 3: centred expansion fan, generalized Prandtl–Meyer turning, and reflected characteristics. State the assumptions behind the half-turning relation and distinguish the ideal sharp-corner construction from a finite curved initial front.
6. Numerical resolution and useful checks: outlet pressure and angle, mass conservation, contour regularity, and sensitivity to discretization. Explain the principles here; put parameter recipes and code listings in Examples.

Include a labelled nozzle diagram identifying each construction region. The details of phase crossings are introduced here and developed on the next page.

### 5. Saturation crossings — `saturation_crossings.md`

Proposed subsections:

1. Sound-speed changes at liquid and vapor saturation boundaries and their effects on the Mach angle.
2. Wet-to-dry expansion: supersonic states on both sides of the crossing, characteristic crossover, the folded solution surface described in Paper B, and the treatment of the wall contour.
3. Low-subcooling flashing: crossing into the mixture while still subsonic, followed by a smooth sonic transition.
4. Stronger-subcooling flashing: choking at the saturation boundary, a subsonic-to-supersonic jump, and a flat initial front downstream of the jump.
5. Consequences and limitations: the short constant-state region caused by flat-front initialization in the reported flashing example; the requirement for supersonic characteristic marching; limits of equilibrium modelling.

Use paired pressure and Mach-number plots to make the distinction between a saturation discontinuity and a shock explicit. Treat the cases in Paper B as examples, not an exhaustive classification of all fluids and inlet states.

### 6. Stator design — `stator_design.md`

Proposed subsections:

1. From a symmetric nozzle passage to a periodic stator cascade: flow passage versus solid blade.
2. Coordinate systems, metal angles, passage widths, and pitch; explain Paper A's pitch relation with a geometry sketch.
3. Fitting the MoC contour with a cubic B-spline and preserving the intended wall shape.
4. Convergent pressure and suction surfaces, tangent connections, leading and trailing edges, and blade closure.
5. Fully bladed and semi-bladed flow regions; distinguish these physical regions from the names of the package's two geometry variants.
6. Design implications: adapted expansion and exit uniformity; effects of back pressure, inlet quality, and finite trailing-edge thickness on downstream flow.

Describe each implemented geometry variant only to the extent selected below. Paper A's dimensions, control-point placements, and fitting tolerance are case-study choices unless the current implementation demonstrably enforces them.

### 7. Verification and limits — `verification_and_limits.md`

Proposed subsections:

1. What was compared: single-phase experiments versus CFD; inviscid CFD versus MoC; viscous CFD of complete stators.
2. A short explanation of barotropic property modelling and its consistency with the MoC thermodynamic closure.
3. Representative evidence from both papers, stating the fluid, conditions, comparison quantity, and error definition.
4. Interpretation of design and off-design behaviour: shocks, boundary layers, wakes, exit-angle deviation, and efficiency are assessed by the cited CFD studies.
5. Limits of extrapolation: non-equilibrium phase change, slip, viscous effects, three-dimensional effects, and operating conditions beyond the reported cases.

Any quoted efficiency or error is a result for a stated case. For example, Paper A's 97.30% stator efficiency is a viscous CFD result for its design case, not an efficiency prediction returned by the inviscid MoC solver. Published comparisons are also not a substitute for verifying the current software revision.

### Optional extensions

| Proposed addition | Contents | Recommendation for the first draft |
| --- | --- | --- |
| `rotor_design.md` | Vortex-flow construction, transition and circular arcs, relative-frame assumptions, inlet/outlet continuity, blade pitch and closure, and the real-fluid generalization. | Defer unless complete coverage of the package is a primary objective. It requires separate primary-source reading beyond the supplied papers. |
| `derivations.md` | Full characteristic derivation, Sauer expansion, equilibrium acoustic derivation, and possibly the fundamental derivative of gas dynamics. | Add only the derivations needed for the requested mathematical depth. |
| `notation.md` | Shared symbols, units, characteristic signs, coordinate systems, and mapping to API names. | Use a compact table initially; promote it to a separate page if the rotor is included or the notation becomes extensive. |

**Question 3 — Which methods should the first draft cover?**

Please edit the final column. “Brief” means a conceptual subsection; “full” means equations and construction details; “defer” means omit from the first draft.

| Topic | Recommendation | Your choice |
| --- | --- | --- |
| Planar conventional nozzle MoC | Full | [Write here.] |
| Single-phase and equilibrium two-phase closure | Full | [Write here.] |
| Wet-to-dry and flashing extensions | Full | [Write here.] |
| Minimum-length nozzle method | Brief comparison; full treatment if equally important to users | [Write here.] |
| Stator parametrization | Full physical construction | [Write here.] |
| Both stator geometry variants | Brief comparison, with detailed usage in Examples | [Write here.] |
| Vortex-flow rotor method | Defer | [Write here.] |
| Axisymmetric characteristic equations | Brief scope note; no claim of a verified axisymmetric solver | [Write here.] |
| Radial mapping, annulus geometry, and 3D CAD | Defer to geometry/examples documentation | [Write here.] |

**Question 4 — Does this page sequence and length fit your intended documentation?**

Recommendation: retain the proposed sequence and allow short cross-references between pages. For a smaller guide, merge thermodynamics with isentropic expansions and merge saturation crossings with nozzle design. Please indicate preferred merges, missing topics, or a different length target.

Your answer:

> [Write here.]

**Question 5 — What mathematical detail should appear in the main text?**

Recommendation: show the governing equations, essential derivation steps, and physical interpretation; move lengthy algebra to optional appendices. In particular, should readers be able to reproduce the characteristic unit processes from the text alone?

| Topic | Recommended depth | Your preference |
| --- | --- | --- |
| Mixture density, quality, and void fraction | Short derivation | [Write here.] |
| HEM sound speed | Explicit equation and explanation; full derivation optional | [Write here.] |
| Choking and maximum mass flux | Short derivation, including the smoothness assumption | [Write here.] |
| Characteristics and compatibility | Intermediate derivation from the governing equations | [Write here.] |
| Generalized Prandtl–Meyer relation | Derivation and interpretation | [Write here.] |
| Sauer initialization | Explain assumptions and give working relations; long derivation optional | [Write here.] |
| Wall construction and characteristic marching | Equations plus a diagram | [Write here.] |
| B-splines and geometric continuity | Explain the construction; avoid a general spline textbook treatment | [Write here.] |
| Fundamental derivative and nonclassical gas dynamics | Defer unless central to the intended teaching objectives | [Write here.] |

**Question 6 — Should the pages primarily describe the papers' method, the current implementation, or both?**

Recommendation: explain the physical method using the papers, then add short, clearly identified implementation notes where the package differs. This matters particularly for the flat-front default, generalized Prandtl–Meyer invariants, minimum-length solver, and stator parametrization.

Your answer:

> [Write here.]

## 4. Proposed writing style

The useful common pattern in the papers is **physical problem → assumptions → mathematical formulation → construction or comparison → interpretation**. Paper A's methods section is the strongest model for equation-led exposition; Paper B is the stronger model for explaining regimes and their design consequences.

For the documentation, I propose to:

- Use connected scientific prose, with each paragraph developing one physical or mathematical point.
- Introduce the purpose of an equation before displaying it, define its symbols immediately, and explain its consequence afterward.
- State assumptions where they first become relevant and qualify conclusions by the evidence that supports them.
- Preserve the papers' technical vocabulary and causal explanations while correcting typographical errors and simplifying long sentences.
- Use figures to support an argument, with captions identifying the conditions, quantities, and intended comparison.
- Use present tense for equations and method descriptions; attribute case-specific findings to the papers.
- Keep novelty claims, broad literature surveys, and energy-system motivation brief so the pages remain useful as a reference.
- Use numbered equations and citations through the existing MyST/Sphinx setup, with links to Examples and the API where practical usage belongs.

**Question 7 — Which paper or passages best represent the writing style you want?**

Recommendation: Paper A §2.1 for mathematical exposition and Paper B §§2–3 for physical explanations. Please identify any preferred sections, captions, or other papers, and any wording habits you would like to avoid.

Your answer:

> [Write here.]

**Question 8 — What voice and language conventions should we use?**

Recommendation: neutral scientific voice (“the method”, “the flow”), with “we” used sparingly for a derivation; British English consistently, including “vapour”, “behaviour”, and “parametrization”. The supplied papers mix conventions, so this is an editorial choice rather than a strict transcription of their style. Would you prefer the authors' “we”, direct address to the reader, or a different spelling convention?

Your answer:

> [Write here.]

**Question 9 — How closely should we retain the papers' notation?**

Recommendation: retain familiar symbols but resolve collisions: `V` for speed and `u, v` for components; `μ` for Mach angle and `α` for void fraction; `Q` for vapour quality; `s` for entropy and a distinct pitch symbol such as `t`. Choose either `M` or `Ma` consistently. Give a small mapping to API names and explicitly distinguish throat half-height from full passage width, radians from degrees, and metres from millimetres.

Your answer:

> [Write here.]

## 5. Figures, examples, and evidence

The following figures would do most of the explanatory work. These are figure proposals, not a request to reproduce every result in the papers.

| Proposed figure | Purpose | Starting material |
| --- | --- | --- |
| Mixture sound speed versus quality | Explain the acoustic consequences of equilibrium assumptions and saturation limits | A Fig. 1 |
| Expansion paths with pressure, Mach number, and mass flux | Distinguish within-dome, wet-to-dry, smooth-sonic flashing, and discontinuous flashing cases | A Figs. 7–8; B Figs. 1–2 |
| Characteristic cell and complete nozzle construction | Define signs, boundary processes, kernel, and turning region | New schematic; existing mesh figure as supporting material |
| Saturation crossing and initial-front comparison | Explain characteristic crossover and flat versus Sauer initialization | B Fig. 3 |
| Stator passage and blade construction | Define coordinate systems, pitch, widths, control points, and bladed regions | A Fig. 2; existing stator image |
| A compact MoC–CFD comparison | Show the meaning and limits of verification evidence | A Figs. 5–6 or B Figs. 4–6 |

**Question 10 — How should figures be produced?**

Recommendation: create clean schematics for the theory and regenerate scientific plots from the underlying data/scripts where available, using the papers' plotting conventions. For the first draft, would you prefer finished figures, selected attributed figures from the PDFs, or detailed placeholders? If original plotting scripts or data already exist elsewhere, please note their paths.

Your answer:

> [Write here.]

**Question 11 — Which cases should anchor the explanations?**

Recommendation: cyclopentane for expansion within the two-phase region and stator design, MM for wet-to-dry flow, and nitrogen for flashing. Use these cases consistently across diagrams and link to runnable examples rather than repeat long scripts in Theory. Should a perfect-gas worked example be included first, and do you want numerical worked calculations or only illustrative cases?

Your answer:

> [Write here.]

**Question 12 — How much CFD, verification, and off-design discussion belongs here?**

Recommendation: a concise evidence table, a short barotropic-model explanation, and selected physical interpretations of off-design behaviour. Leave detailed Fluent setup, mesh studies, complete performance tables, and turbulence-model equations in the papers. Would you prefer a fuller validation chapter, or only a short limits section with citations?

Your answer:

> [Write here.]

## 6. Source and implementation questions to resolve

The following items should be reconciled before finalizing the corresponding equations or claims. They do not require changing code as part of this documentation task.

| Item | Observation | Proposed treatment |
| --- | --- | --- |
| Paper A, Eq. (1), p. 3 | The displayed coefficient of `v_y` is `(u² − a²)`, whereas the velocity equation is expected to contain `(v² − a²)` there. The repeated `u` is present in the rendered PDF, not just the extracted text. | Re-derive from continuity and irrotationality and check the primary reference before writing the corrected equation. |
| Paper A, text below Eq. (2), p. 3 | The stated planar/axisymmetric values of `δ` appear reversed relative to the geometric source term. The code identifies planar flow with `δ = 0`. | Use a consistent derivation and state the convention explicitly. Keep this separate from whether the package supports axisymmetric marching. |
| Paper A, Fig. 2, p. 4 | The figure legend and nearby prose assign different colours to the turning and kernel-extension control points. | Define each region by geometry and labels; reconcile the colours if reusing the figure. |
| Sauer reference | Paper A lists 1992; Paper B lists 1947. NASA identifies NACA-TM-1147 as published on 30 June 1947. | Use the verified report metadata. See the [NASA catalogue record](https://ntrs.nasa.gov/citations/20050019415). |
| Publication identity of Paper A | Its supplied cover identifies a Journal of Turbomachinery accepted version, while Paper B cites an IMECE proceedings version. | Obtain the preferred bibliographic record and cite that version consistently. |
| Initial line | The public API defaults to a flat front; the classical Sauer line is optional. A flat-front option can also be used for a smooth-sonic case. | Separate the mathematical applicability of the constructions from current software defaults and numerical regularization. |
| Planar versus axisymmetric support | The API exposes `delta_flow`, but core routines use planar invariants and the mass-flow calculation fixes `δ = 0`. | Do not infer verified axisymmetric support from the argument name alone. |
| Wet-to-dry wall treatment | Paper B states that the characteristic fold is corrected during wall post-processing, but does not provide a complete algorithm. | Explain only what the paper establishes until the intended wall-treatment procedure and its current implementation are located and checked. |
| Stator parametrization | Paper A's construction and fixed fitting tolerance should not automatically be equated with the present geometry functions and their control-point settings. | Map the common geometry and identify implementation differences before stating guarantees. |
| Rotor notes | The notes say edge rounding is not implemented, while the current rotor module includes edge-closure routines. | Treat the notes as background, and verify any rotor description against the current code and primary sources. |

The limited online check also confirmed the bibliographic identity of Flåtten and Lund (2011), the primary relaxation-model source used by both supplied papers. Its institutional record describes the model hierarchy and analytical wave velocities: [SINTEF publication record](https://www.sintef.no/en/publications/publication/862799/). The full acoustic derivation has not been independently audited in this planning step.

The existing `docs/source/references/bibliography.bib` contains some relevant background sources, including the Spinelli experiments, but does not yet contain the two supplied papers or several central method references. The documentation configuration supports MyST citations and a shared bibliography; final records can be added during drafting after the intended versions are established.

**Question 13 — How should source corrections and differences from the package be presented?**

Recommendation: use checked equations and consistent notation in the main text, retain correction notes in the draft review material, and show implementation notes only where they affect how a reader selects or interprets a design. Please add known errata, missing derivations, or the location of the wet-to-dry wall-treatment algorithm.

Your answer:

> [Write here.]

**Question 14 — Which references and manuscript versions should we cite?**

Please provide the preferred citation/DOI or Zotero record for the two supplied papers if available. Are there additional group papers, theses, classical references, or unpublished method notes that should be central to the draft? If the rotor or a full minimum-length derivation is included, I propose reading the relevant primary sources before drafting those sections.

Your answer:

> [Write here.]

## 7. Instructions for the subsequent draft

After you edit this file and request the draft, the proposed workflow is to read your answers, settle the page scope, check the required equations and additional sources, and write the selected MyST pages. The draft will link each major method or result to its source, distinguish published evidence from present implementation behaviour, and include the figure treatment you select above. Code changes or new physical-model development are outside this proposed documentation scope.

**Question 15 — What should a successful first draft contain, and what should take priority?**

Recommendation: complete prose and essential equations for the selected core pages, source citations, consistent notation, and either figures or explicit figure placeholders according to Question 10. Please list any must-have topics, exclusions, preferred review order, or other constraints that are not captured above.

Your answer:

> [Write here.]
