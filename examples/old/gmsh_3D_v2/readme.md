# TurboMoC

Run from the repository root using the existing environment:

```powershell
.venv/Scripts/python.exe -m examples.gmsh_3D_v2.run_mesh_app
```

Open [http://127.0.0.1:8052](http://127.0.0.1:8052). The app needs the project's mesh/mesh-viz dependencies (Gmsh, Dash, diskcache, dash-vtk and PyVista). It binds to the local machine. Stop it with Ctrl+C. Restart after changing Python sources; workers verify the source revision against the submitted snapshot.

The top level keeps `run_mesh_app.py` as a small launcher, `run_mesh_generation.py` as the command-line entry point, the YAML cases, and the theory/roadmap notes. The implementation lives under [`app/`](app/readme.md), including numerical stage adapters, workers, plots, geometry previews and browser assets. Configuration and output locations remain in this example directory.

The original examples remain available. Two shared improvements also benefit them: compiled SciPy B-spline fitting/evaluation and forwarding MoC progress callbacks through the nozzle cache helper.

## Workflow

1. **Geometric design:** edit Method of characteristics, Meridional channel definition, Blade geometry definition and Flow domain definition, in that order. Choose axial/radial within the channel group; switching families preserves other inputs. Compute MoC once for its current inputs. The reference θ–m′ section and meridional geometry appear together in one collapsible group; the full-width 3D shape appears below, with its own display controls. The sidebar defaults to 380 px.
2. **Two-dimensional meshing:** generate the reference mesh explicitly. Both mesh views stay empty until a current reference mesh exists; the meridional grid then uses actual reference boundary stations. Successful generation opens this workspace automatically. Span edits update the grid without regenerating that section mesh. The sidebar's **Mesh visualization** group contains both display widths; its 2D width adjusts these two plots together.
3. **Three-dimensional meshing:** generate the volume from the current reference artifact; successful generation opens this workspace automatically. View selected boundary surfaces and optionally mesh edges beside the independently scrollable report. Displayed blades is a slider from one blade to the full machine count. Export CGNS separately.

One job runs per browser session. Jobs run in dedicated Python processes, so Gmsh and VTK state belong to the worker. The app displays a spinner, elapsed time and stage progress to the right of the action buttons. A full-width terminal is anchored to the bottom of the main pane; its collapsed header stays visible while the content scrolls. Expand it to see the live bounded solver/Gmsh log. During a native operation without a reliable completion fraction, the bar is indeterminate. MoC percentages use the existing weighted stage estimator; they are not remaining-time predictions.

Cancel job terminates the active local worker. Failed/cancelled jobs retain previous successful references. Edits during work affect the next job; completion is tagged with its original configuration. The app labels previous/outdated volumes and requires current artifacts for the next stage.

## What the views show

| View | Basis |
| --- | --- |
| Reference geometry | Fitted blade/passage at η = 0.5, plotted as θ [rad] and −m′ [dimensionless]. |
| Meridional channel | General hub/shroud curves in z–r [mm], with the blade LE-to-TE envelope shaded grey. This is a meridional projection, not blade thickness. |
| 3D blade shape | Refined geometry tessellation in physical XYZ [mm], with smooth normals and closed blade ends; its faces are display polygons. |
| 2D mesh | Actual Gmsh reference cells in θ–m′ coordinates. |
| Meridional grid | Actual span stations crossed with reference periodic-boundary node stations; shown only after current 2D meshing. |
| 3D mesh boundaries | Surfaces extracted from actual volume connectivity; no independent remesh for viewing. |

The meridional mesh view is the projected grid of the future sweep, displayed only after a current 2D mesh has been generated. It does not independently mesh a meridional CFD domain or solve streamlines. This makes it consistent with the reference section rather than introducing an unrelated mesher.

The physical number of blades sets periodic pitch. **Displayed blades** only creates rotated visualization copies. Exports contain one periodic passage. Orange blade surfaces and grey endwalls restore the original 3D palette; 3D mesh edges are black. Both 2D meshes use orange grid lines with black blade outlines. Both meridional plots start at zero radius, making the channel's position relative to the rotation axis visible. Surface selection, edge toggles and line widths affect views only. The geometry and mesh viewers have independent controls; the 2D workspace does not repeat them.

The 3D view has a resizable mesh pane and a scrollable report on its right. The report contains counts, SICN, metric statistics, three vertically stacked histograms, and original IDs for the ten worst cells per metric. Aspect ratio uses a logarithmic axis. On narrower windows these panes stack vertically.

## Configuration and display widths

V2 case files use `method_of_characteristics`, `meridional_channel`, `blade_geometry` and `flow_domain`. The smoothing/tangent fields sit directly in `flow_domain`; there is no nested `smooth` dictionary. Legacy `nozzle`/`machine`/`blade`/`passage.smooth` files still load through the compatibility adapter; job snapshots use the current names.

`mesh.visualization.section_edge_width` controls Plotly section and meridional grid lines. `mesh.visualization.volume_edge_width` controls antialiased black cell edges in the 3D surface shader. Both default to **0.35 display pixels** in the YAML files and appear together under **Mesh visualization** in the left sidebar of either meshing workspace (or **Show all settings**). Changes preserve numerical artifact keys and appear in configuration snapshots.

Native WebGL lines can ignore widths below one pixel; see the [VTK maintainer discussion](https://github.com/Kitware/vtk-js/issues/673). The 3D viewer therefore uses [vtk.js shader replacements](https://github.com/Kitware/vtk-js/blob/master/Sources/Rendering/OpenGL/PolyDataMapper/api.md): per-triangle coordinates measure distance to each original cell edge in framebuffer pixels, and a coverage filter draws fractional widths. Quad triangulation diagonals are suppressed. Rendering uses the surface depth test, so hidden mesh edges stay hidden. Display triangles are prepared once per surface and reused across blade copies; this does not change the numerical mesh or exported cells. Width and edge-visibility changes update the mapper in place and preserve the camera. Canvas scaling is included when converting display pixels to framebuffer pixels.

The geometry preview uses a denser parameter-space tessellation and supplied smooth point normals. Blade root/tip caps are included with the blade selection, so hiding endwalls does not expose hollow blades. Side/cap vertices are welded before subdivision to keep their shared edges aligned. The hub is a continuous surface of revolution spanning the periodic edges, without a blade-shaped hole; the shroud retains its opening. These display surfaces leave the actual CFD boundaries unchanged. Preview refinement is independent of CFD mesh sizing and preserves general axial/radial/curved mappings.

The simplified app no longer exposes slice/worst-cell controls. Their tested backend extraction remains available in `app/inspection_source.py` for a future optional inspection workspace.

## Artifact dependencies and files

Nozzle inputs → template → geometry → 2D reference → span sweep → quality → export.

Changing blade/passage/machine controls invalidates the reference and volume. Section sizing/inflation controls invalidate both meshes. Span controls invalidate only the volume. Display and CGNS filename/units/BC settings preserve numerical mesh identities.

Each job writes an immutable configuration snapshot and separate outputs under `output/app/<session>/<job>/`. Reference artifacts store their topology and exact fitted definition, so the sweep consumes the section that was displayed. Successful jobs are published only after validation. Downloads include the configuration, activity log, MSH, reports and separately exported CGNS.

The local geometry cache uses source/configuration keys, with separate fitted-template reuse. Browser stores hold references; scientific data stays in server artifacts, apart from compact datasets needed to draw the views. Browser refresh starts a new session; project reopen/history management is future work.

To run the numerical pipeline without the app:

```powershell
.venv/Scripts/python.exe -m examples.gmsh_3D_v2.run_mesh_generation
.venv/Scripts/python.exe -m examples.gmsh_3D_v2.run_mesh_generation --config examples/gmsh_3D_v2/radial_stator.yaml
```

These commands write reference/volume meshes, quality and timing reports, and CGNS when enabled. Native viewer and static-figure options inherited in the example YAML are not used by V2.

Read [the deformation/coordinate note](deformation_note.md) for the sparse LU equations, one-time passage placement, physical height estimates and precise quality definitions. Read [the revised implementation roadmap](roadmap.md) for completed and remaining work.

## Performance evidence

Measured in this workspace with the existing converged nozzle, default cases and no native viewers/static figures:

| Stage | Axial | Radial |
| --- | ---: | ---: |
| Geometry / sole reference placement | 0.95 s | 0.99 s |
| Reference mesh | 2.12 s | 3.72 s |
| Sweep, validation, metrics and output | 7.10 s | 5.41 s |
| Separate audited CGNS export | 2.08 s | 1.69 s |

Axial stages total about 12.25 s, compared with the earlier profiled V1 pipeline's 40.54 s. This is indicative rather than an identical-output benchmark: V2 adds the three volume metrics, changes span placement policy, and V1's number included profiler overhead. The V2 axial mesh had 229,075 cells versus 229,375 previously. A span-only iteration reuses the geometry and reference stages.

V2 performs one fit/placement search, reuses sparse LU across span, and uses SciPy's compiled spline routines while preserving legacy fitting/endpoint conventions. It retains audited Gmsh assembly, periodicity and CGNS checks. No axial/radial-specific shortcut was introduced. Native Gmsh size fields, exact curve clearance acceleration, large-volume chunking and fully lazy preview tessellation remain opportunities.

## Verification

35 targeted tests passed across V2, the shared 2D workflow/cache, existing 3D mapping/export, and V1 input/key checks. Real V2 meshes were checked on axial, axial taper, radial inflow/outflow and curved channels. The tests cover compiled spline parity, reference placement, periodicity, Jacobians, CGNS boundary auditing, quality data, inspection, live logging, cancellation and the separate submission/polling callbacks.

Earlier headless Edge runs covered the full pipeline through CGNS export and the backend inspection views: 380 px sidebar, smooth normals delivered to geometry actors, orange blades/black mesh edges, capped previews, empty pre-mesh meridional view, automatic result navigation, anchored terminal, independent display controls, all 31 blades, span-only invalidation and preservation of design inputs when switching channel families. Closed preview shells were also tested on axial, radial and curved channels. The radial command-line entry point previously completed its reference, volume, report and export stages.

The latest Edge check generated a reference and volume mesh, verified both sidebar width controls and their YAML defaults, and checked that both 2D grids respond without remeshing. Actual screenshots at 0.1, 0.35 and 1.8 px showed increasing 3D edge coverage; actor property values alone were not used as evidence. Width and edge-toggle edits preserved the view and camera. The clean browser session reported no browser exceptions or shader errors. A callback regression also verifies that 3D display edits leave the 2D plots alone, and 2D width edits patch strokes without rereading mesh arrays or rebuilding plots.

