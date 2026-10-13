# Mesh quality diagnosis and refinement roadmap

**Case:** app baseline, `examples/apps/cases/axial_stator.yaml`, stator only.  
**Investigation date:** 13 October 2026.  
**Scope:** source inspection, fresh geometry/mesh generation, numerical inspection and controlled meshing experiments. No application code or baseline configuration was changed. No finite-volume CFD solution was run.

## Recommendation

**Yes—introduce a quality safeguard for quad recombination, then use local refinement where it is needed. Do not make general refinement the first remedy for this defect.**

The suspected cell is indeed a nearly triangular quadrilateral: three consecutive vertices lie on the curved periodic boundary, producing a **173.08° corner**. Its four edges have ordinary lengths; it is not a collapsed-edge or zero-area cell. This defect already exists in the 2D reference mesh and is carried through all 25 spanwise layers.

The most economical tested remedy was raising Gmsh's `Mesh.RecombineMinimumQuality` from its default **0.01 to 0.1**. For this baseline, it replaced the offending quad with two triangles and reduced maximum 3D equiangular skewness from **0.9231 to 0.6447**, at a cost of **25 additional volume cells**, or **0.011%**. A threshold of 0.2 gave the same result. These are candidate settings for subsequent implementation and regression testing, not changes made by this investigation.

The baseline is not demonstrably unusable for FVM. All volume Jacobians are positive, periodic matching passes, and the app's minimum orthogonality is **0.7651**. There is a localized, avoidable shape-quality outlier, rather than evidence that the entire mesh is unsuitable. Its actual effect on convergence and predicted losses still needs a CFD comparison.

## What was reproduced

The investigation called the same geometry and meshing functions used by the app: fresh MoC solve → stator fit → app blade-count sizing → reference mesh → 3D sweep. It did not use a screenshot-derived mesh or an older cached result.

The YAML contains 31 blades, but the app's successful design calculation suggests and applies **28 blades**, from a natural count of 27.35687. The reproduction included that step. It generated exactly the counts shown in the supplied screenshots:

| Baseline quantity | Measured value |
| --- | ---: |
| Reference nodes | 9,463 |
| Reference cells | 9,450 |
| Reference quads / triangles | 8,735 / 715 |
| Spanwise cell layers | 25 |
| Volume cells | 236,250 |
| Hexahedra / triangular prisms | 218,375 / 17,875 |
| Maximum 2D / 3D skewness | 0.923099 / 0.923098 |
| 99th percentile 3D skewness | 0.436482 |
| Volume cells with skewness > 0.85 | 25, all from one reference quad |
| Volume cells with skewness > 0.95 | 0 |
| Minimum app orthogonality | 0.765072 |
| Minimum Gmsh signed inverse condition number (`minSICN`) | 0.060027 |
| Maximum periodic node mismatch | 4.49 × 10⁻¹⁴ mm |

The relevant settings were algorithm 8, Blossom recombination (algorithm 1), five smoothing steps, blade size 0.25 mm, domain size 1 mm, and 50 curvature elements. Inflation settings were first height 0.03 mm, growth ratio 1.2 and requested thickness 1 mm. The generated profile specifies 11 layers; actual local layer behavior remains a separate mesh diagnostic. Span stretch was 1.5. Runtime versions: Gmsh 4.15.2, Python 3.13.2, NumPy 2.5.3.

**Reading the screenshot:** “Worst 10,000” always selects 10,000 cells, regardless of whether they are poor. Here its cutoff is approximately **0.325**. Only 25 cells exceed 0.7. To isolate this particular defect in the app, select the mesh quality filter, metric **Skewness**, selection **Threshold**, and a threshold of **0.85**. That should show the single column of 25 cells for this reproduced baseline.

## Root cause: a boundary corner inside a recombined quad

![Baseline section and close-up of the offending periodic-boundary quad](artifacts/mesh_diagnosis/baseline_diagnosis.png)

*Figures are plotted from the generated mesh arrays. Blue lines identify periodic boundaries. Coordinates are those of the annular conformal reference plane, expressed in mm, not global Cartesian coordinates.*

The worst quad has these measured properties:

| Property | Value |
| --- | --- |
| Reference-array node indices, zero based | 418, 419, 420, 5186 |
| Boundary contact | Two consecutive edges on `periodic_left` |
| Corner angles, in node order | 57.38°, **173.08°**, 36.90°, 92.64° |
| Edge lengths | 0.404, 0.430, 0.673, 0.461 mm |
| Edge-length aspect ratio | 1.666 |
| Area | 0.1653 mm² |
| Centroid in reference coordinates, pitchwise / meridional | −12.7636 / −1.3796 mm |

Nodes 418–419–420 follow the smooth convergent part of the periodic curve. The middle vertex is within that spline, not at a junction between CAD segments. Recombination has retained a four-sided cell whose boundary-facing corner is almost straight. A roughly triangular region has been represented as a quad with an extra boundary vertex.

For a quad, the app calculates equiangular skewness as

\[
S=\max\left(\frac{\theta_{\max}-90^\circ}{90^\circ},\frac{90^\circ-\theta_{\min}}{90^\circ}\right).
\]

The 173.08° corner therefore gives `S = 0.9231`. This angular degeneracy does **not** mean that the cell's area or volume is approaching zero.

The source explains why this is allowed:

- `mesh_2d.configure_recombination()` enables recombination but does not set a minimum recombination quality. The installed Gmsh default is 0.01; node repositioning and five topology-optimization passes are already enabled.
- Periodic CAD is constructed with smooth splines and a translated counterpart. Matching nodes and edges are explicitly checked. A correct periodic bijection does not guarantee good shapes of the cells beside it.
- Reference and volume validation reject invalid geometry/topology, but do not impose a maximum equiangular-skewness requirement.
- `mesh_3d.assemble_volume()` sweeps each reference quad into one hex per layer. The worst column's skewness remains between 0.9230972 and 0.9230975. The 3D mapping is therefore not the main cause of this outlier.

Gmsh documents a dedicated minimum-quality control for recombination. Its quality scale must not be assumed to equal `1 − app_skewness`; the app's angle metric and the resulting 3D mesh must still be checked independently. [Gmsh manual, mesh options](https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eRecombineMinimumQuality)

For tracing the result in the generated baseline, the worst volume element has **Gmsh ID 237645**, in layer 25. The same column's element in the screenshot's layer 21 has ID **202705**. IDs apply to these generated artifacts and can change after remeshing.

## Controlled experiments

All cases used the same fitted blade, periodic geometry, blade count, inflation parameters and 25 spanwise layers. Each mesh was regenerated and swept. All ten runs passed the existing Jacobian, boundary and periodicity checks.

`S` below is the app's equiangular skewness. `O` is its minimum face-normal/centroid-connector cosine, including transformed periodic neighbors; higher is better. Neither is a claim about a particular CFD solver's identically named diagnostic.

| Experiment | 2D cells | 3D cells | Max 2D S | Max 3D S | 3D cells S > 0.85 | Min 3D O |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9,450 | 236,250 | 0.9231 | 0.9231 | 25 | 0.7651 |
| Recombination minimum quality **0.1** | 9,451 | 236,275 | 0.6166 | 0.6447 | 0 | 0.7651 |
| Recombination minimum quality **0.2** | 9,451 | 236,275 | 0.6166 | 0.6447 | 0 | 0.7651 |
| Refine periodic edges only, near the bend | 9,477 | 236,925 | 0.9763 | 0.9763 | 150 | 0.4745 |
| Refine a smooth local patch, boundary + interior | 9,979 | 249,475 | 0.5766 | 0.6010 | 0 | 0.7667 |
| Global targets × 0.7 | 12,895 | 322,375 | 0.9480 | 0.9480 | 25 | 0.7235 |
| Global targets × 0.5 | 18,339 | 458,475 | 0.9874 | 0.9873 | 50 | 0.7472 |
| Global targets × 0.5 **and quality guard 0.1** | 18,341 | 458,525 | 0.6281 | 0.6298 | 0 | 0.7472 |
| Disable core recombination | 13,154 | 328,850 | 0.4682 | 0.4880 | 0 | 0.7460 |
| Algorithm 6, with baseline recombination | 10,517 | 262,925 | 0.6566 | 0.7002 | 0 | 0.6267 |

Experiment definitions and limitations:

- **Global targets:** blade and domain sizes were multiplied by the stated factor; curvature elements increased to 71 and 100 respectively. Inflation heights/thickness and span layers were held fixed. These are mesh-sizing sensitivity experiments, not a uniformly refined CFD grid family.
- **Local patch:** cap the existing size field at 0.2 mm within 2 mm of the bad boundary vertex, then blend smoothly to 1 mm at radius 5 mm. Repeat the patch exactly every pitch. Combine it with the existing sizing callback by taking the minimum. The periodic-only experiment uses the same cap but applies it only to periodic curve entities.
- **No recombination:** the boundary-layer field still requests quads. This produces a mixed mesh with a triangular core, not an entirely triangular section. The current core-grading formula also depends on the recombination flag, so this experiment is not a pure topology-only comparison.
- **Quality guard:** matching the baseline and 0.2 meshes by node position showed one removed quad and two added triangles, with unchanged boundary nodes and edges. Gmsh also moved 249 interior nodes by up to 0.0347 mm during optimization. The result is not a strictly isolated split with all interior coordinates frozen.
- Mesh quality is not monotonic in resolution: the 99th percentile improved under global refinement, while the worst cells became worse. The finer unguarded mesh had `minSICN = 0.00652`, despite still passing positivity checks.

![Comparison of tested remedies at the original defective bend](artifacts/mesh_diagnosis/remedy_comparison.png)

*Each panel shows the same reference-coordinate window. Its title reports the maximum over the entire reference mesh; after remeshing, the global worst cell may be elsewhere.*

### Why finer periodic discretization can make this worse

Refinement improves representation of the curve, but does not prevent a quad from retaining three consecutive boundary vertices. If those vertices span a small curve-turning angle δ, the quad's middle angle is approximately `180° − δ`, so its skewness is approximately `1 − δ/90°`. Smaller boundary segments reduce δ and can bring that angle **closer to 180°**.

This explains why a boundary-size change needs a topology-quality safeguard and compatible interior spacing. Merely increasing the number of periodic nodes is not a robust remedy. Likewise, `flow_domain.num_points` controls geometry sampling/interpolation; it is not the number of final Gmsh periodic edges. Increasing it is not a direct mesh-quality control.

## Is the nearly triangular quad a problem for FVM?

**It is worth removing, but this particular cell does not establish a CFD failure.**

Finite-volume conservation balances fluxes across shared faces. A control volume can remain conservative with a nearly straight polygon corner, provided the geometry is valid and each shared/periodic flux is treated consistently. A high equiangular-skewness score is not, by itself, proof of loss of conservation. [CFD Direct: the finite-volume concept](https://doc.cfd.direct/notes/cfd-general-principles/the-finite-volume-concept)

Accuracy and robustness also depend on face interpolation, gradient reconstruction and alignment of face normals with neighboring cell-center connectors. Non-orthogonality introduces correction terms; their accuracy and stability depend on the numerical scheme. This is why face-based quality and the actual solver diagnostics matter alongside corner angles. [CFD Direct: surface-normal gradient](https://doc.cfd.direct/notes/cfd-general-principles/surface-normal-gradient), [gradient discretisation](https://doc.cfd.direct/notes/cfd-general-principles/gradient-discretisation)

For the identified column, the app's minimum orthogonality is **0.7922**; for the single worst-skewness volume cell it is **0.8910**. Thus the angular outlier is not simultaneously an extreme non-orthogonality outlier. The whole mesh's minimum cosine, 0.7651, corresponds to about 40.1° using this app's definition. The report uses arithmetic vertex centers; a solver using geometric volume/face centroids can report different values.

Ansys also warns that high skewness can reduce accuracy and destabilize solutions. Its often quoted maximum of 0.95 is presented for triangular/tetrahedral meshes and is not a universal acceptance threshold for this mixed hex/prism mesh or this app's metric. [Ansys Fluent: mesh requirements and considerations](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/flu_ug/flu_ug_GridQuality.html)

A good triangle swept into a prism is a reasonable alternative to a poor quad swept into a hex. Here the pipeline already contains and handles both cell types. High aspect ratio in a well-aligned inflation layer is a different issue and should not automatically trigger isotropic refinement or splitting.

## Recommended implementation sequence

### 1. Add a recombination safeguard and report the tail

Start by testing an explicit `Mesh.RecombineMinimumQuality = 0.1` in the existing recombination setup, with a configurable value and documented meaning. It removed the baseline outlier and both >0.95 outliers in the half-size experiment, while retaining nearly all quads.

Retain separate checks of the **actual completed mesh**: maximum and percentile skewness, counts above selected limits, minimum signed Jacobian, minimum orthogonality, periodic conformity, and boundary-layer integrity. Recheck after spanwise deformation and mapping. Do not turn successful Gmsh generation into a claim of CFD readiness.

Suggested initial review policy—not a universal CFD standard:

- Keep invalid/inverted cells, nonmanifold topology and failed periodic matching as hard failures.
- Flag cells with app skewness above 0.85 for review; treat values above 0.95 as a stronger repair signal.
- Show the number of distinct reference cells as well as the number of swept volume cells. In this case “1 reference cell / 25 volume cells” is more informative than just “25 poor cells.”
- Preserve cell IDs and classify flagged cells by boundary adjacency, core/inflation location and spanwise layer.

Before adopting a default, cover the baseline stator, rotor, stator–rotor assembly, linear cascades, and tapered annular cases. Verify mixed-element CGNS export and the intended CFD import path. Export/import was not rerun in this investigation.

### 2. Use selective topology repair if the guard is insufficient

Reject or split near-flat quads, particularly those with two consecutive edges on the same smooth periodic curve. Prefer existing Gmsh recombination controls before adding custom connectivity edits.

If a custom repair is eventually needed, **evaluate both diagonals**. For the exact frozen baseline quad:

| Split | Resulting triangle skewnesses |
| --- | --- |
| Join boundary vertices 418–420 | **0.9442**, 0.4409: creates a thin boundary sliver |
| Join middle boundary vertex 419 to interior vertex 5186 | **0.0923**, **0.3850**: good alternative |

Blindly splitting every poor quad can therefore make matters worse. Accept a split only when both children have positive area, acceptable shape, compatible neighbors and acceptable swept volumes. Preserve all outer edges and periodic node pairs. Split the reference topology consistently through the entire span, not a single volume cell in isolation. Treat inflation cells separately to preserve their layer structure.

### 3. Add periodic-compatible local refinement

The tested smooth patch reduced maximum 3D skewness to 0.6010 at **5.60%** more cells. This is useful when a region also needs better geometric or flow resolution, or when a simple recombination guard cannot repair its surroundings.

For an automated quality-driven loop:

1. Mesh and inspect both the reference section and the swept volume.
2. Map flagged volume cells back to reference cells, cluster nearby problems and distinguish topology defects from insufficient resolution.
3. Repair avoidable topology first; add refinement patches for remaining clusters.
4. Repeat each size patch at translated periodic locations, including its influence inside the fluid region. Use a smooth transition into existing core grading.
5. Remesh, sweep and compare quality, inflation behavior, periodic matching and cell cost. Keep the previous acceptable candidate when a trial is worse.
6. Stop after a small configured number of passes, on insufficient improvement, or at a cell-count limit; report any unresolved locations.

A smooth point-centered size field is appropriate; a hard size jump or simply embedding extra isolated points is not equivalent. The existing sizing callback is installed before meshing and refreshed after the 1D mesh. Any additional sizing mechanism must survive that refresh and combine with existing wall/core constraints. Use the tested patch as an initial experiment, not as a hard-coded coordinate-specific production fix.

For broader periodic-curve resolution, consider arc-length spacing constrained by curvature or chord error on the master curve, with the slave copied exactly. Retain the quality guard: resolving the curve more accurately does not prevent the near-flat-quad topology.

### 4. Reserve broader topology changes for persistent cases

Disabling core recombination removed the outlier but increased volume cells by **39.2%**. Switching to algorithm 6 increased cells by **11.3%** and worsened the minimum orthogonality in this test. Neither is the first choice for this baseline.

If many cases exhibit boundary-hugging quads, investigate a structured strip along the periodic curve or a block layout that controls boundary-adjacent cell alignment. This is substantially more work: it must accommodate curvature, interfaces, inflation clearances and annular deformation. Changing the periodic curve itself may redistribute the problem and should preserve pitch symmetry, clearance and a fair comparison of the flow domain.

### 5. Perform a separate CFD grid-convergence study

The experiments above establish mesh-quality behavior, not solution independence. After selecting a robust meshing policy, solve at least three systematically refined grids with consistent geometry, physics, boundary conditions and discretization schemes. Control iterative error and compare mass flow, outlet angle, pressure distribution, total-pressure loss and any relevant shock position. Include wall-normal resolution and spanwise resolution explicitly, rather than assuming that smaller core cells address them.

Use observed convergence and, where applicable, a Grid Convergence Index. Three grids help estimate order and check the asymptotic range; shocks and non-monotonic convergence require care. First compare the baseline and inexpensive quality-guard mesh to determine whether removing this particular outlier changes the CFD result or convergence history. [NASA: examining spatial grid convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)

## Source map and reproducibility

| Source | Relevant responsibility |
| --- | --- |
| [App baseline launcher](examples/apps/run_app.py) | Selects the stator-only baseline YAML |
| [Blade-count sizing](src/turbo_moc/app/blade_counts.py) | Natural pitch and suggested integer count |
| [Passage geometry](src/turbo_moc/geometry/passage_2d.py) | Smooth periodic reference and spline sampling |
| [2D mesher](src/turbo_moc/meshing/mesh_2d.py) | Periodic CAD, sizing, inflation, recombination and conformity |
| [3D mesher](src/turbo_moc/meshing/mesh_3d.py) | Reference generation, deformation, hex/prism sweep and validation |
| [Quality definitions](src/turbo_moc/meshing/quality.py) | Actual angle-based skewness and face-based orthogonality |

Local investigation artifacts are under **`artifacts/mesh_diagnosis/`**, which is ignored by Git. The document contains the essential numerical findings independently; the linked figures and raw evidence require that local artifact directory to be retained or packaged separately when sharing.

- `run_study.py`: standalone experimental harness; changes settings/hooks only inside its process.
- `inspect_results.py`: array inspection, cell tracing, topology comparison and plotting.
- `provenance.json`: runtime versions, Gmsh defaults, sizing result and source/configuration SHA-256 hashes.
- `effective_config.json`, `design.json`: effective baseline configuration and fitted geometry reused across experiments.
- `comparison.json`, `anatomy.json`: tabulated results and exact offending-cell details.
- Each experiment directory: `reference_mesh.msh`, `reference_mesh.npz`, `stator_mesh_3d.msh`, `volume_data.npz`, quality/validation summaries and `diagnosis.json`.

From the repository root, the baseline can be regenerated with:

```powershell
.venv/Scripts/python.exe artifacts/mesh_diagnosis/run_study.py baseline
```

Then run the same command with `quality_0.1`, `quality_0.2`, `periodic_only`, `local_patch`, `refine_0.7`, `refine_0.5`, `guarded_refine`, `no_recombine` or `algorithm6`. Geometry is reused from the baseline artifacts. To rebuild the comparisons and figures:

```powershell
.venv/Scripts/python.exe artifacts/mesh_diagnosis/inspect_results.py
```

Source/configuration hashes recorded for the meshing and geometry modules and the baseline YAML were unchanged at the end of the experiments. All remedies in this document remain recommendations; no production implementation was made.
