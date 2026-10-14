> **App YAML compatibility:** see [Reuse an app project](../../workflow_configuration.md) for the shared schema, copy/paste instructions, and 2D/3D restrictions.

# MoC stator volume mesh and barotropic Fluent simulation

This example generates a fresh 3D stator passage with the existing
[Gmsh 3D workflow](../../gmsh_3D/readme.md), builds the same barotropic material as
[Fluent 2D](../../fluent_2D/readme.md), and configures a 3D Fluent simulation.
It supports one selected stationary stator row with axial, radial or linear
geometry. Rotors and coupled multirow simulations are outside this workflow.

From the repository root:

```powershell
poetry install --extras fluent
poetry run python -m pip install -e ../barotropy
poetry run python -m examples.fluent_3D.stator.run_fluent_case
```

Fluent must be installed and licensed. The YAML selects Fluent 2024 R2, double
precision, four processors and a hidden GUI for contour rendering. The small
runner resolves its adjacent YAML independently of the working directory.
All paths inside that YAML resolve relative to it.

## Configuration and execution

Edit `workflow_fluent_stator_3D.yaml`. Its `machine`, `workflow`, and `rows` blocks
are a complete app project; replace them directly with an app download. Keep the
Fluent-only fields. `row_id: stator_1` selects its stator. Geometry, mesh settings,
inlet conditions and outlet pressure come from that row. The Fluent workflow
uses the existing volume mesher and barotropic model without changing their
algorithms or thermodynamic closure. Geometry and material preparation share
one freshly solved MoC result, including its actual inlet quality.

| Setting | Effect |
| --- | --- |
| `output_dir: output` | Write this example's artifacts under its adjacent `output/` |
| `prepare_only: true` | Generate mesh and barotropic expressions without launching Fluent; overrides `run_case` |
| `run_case: false` | Save the configured case without initialization or iterations |
| `run_case: true` | Initialize, execute the staged solve, save case/data and export enabled contours |
| `geometry_figure` | Save/show the blade and passage in the 2D reference construction plane |
| `show_gmsh` | Open the actual volume mesh in Gmsh; close the viewer to continue |
| `show_pyvista` / `save_mesh_figure` | Show/save an actual 3D mesh surface preview |
| `fluent.solution_strategy` | The same configurable first-/second-order stages as the 2D example |
| `fluent.residual_equations` | Continuity, all three velocity components, k and omega tolerances |
| `fluent.boundaries` | Names for inlet, outlet, blade, hub, shroud and both periodic sides |
| `fluent.wall_yplus` | Include final wall-resolution diagnostics for blade, hub and shroud |
| `fluent.view` | Fluent camera view for the contour exports; default `isometric` |
| `fluent.periodic_instancing` | Display repeated passages using the geometry's periodic transformation |
| `reporting` | Console/file log level and filename inside the output directory |

The Fluent workflow owns its viewer switches; the mesh-only example's enabled
viewers do not open automatically. Gmsh, PyVista and Dash VTK are installed by
default. Fluent automation is enabled by the `fluent` extra shown above.

Preview appearance and its filename come from the selected row's `mesh.pyvista`
settings in the design YAML. The `geometry_figure` reference-plane drawing does
not show the actual volume mesh.

## Fluent setup and results

The workflow imports metre-scaled CGNS coordinates, checks the mesh, and applies
pressure inlet/outlet conditions and stationary no-slip blade, hub and shroud
walls. Axial and radial passages use rotational periodicity about their geometry
axis; linear passages use translational periodicity. Imported periodic vertices
must match the geometry transformation before the pair is accepted.

Physics and numerical controls match the 2D workflow: steady pressure-based
coupled flow, SST k-omega, energy disabled, pressure-dependent barotropic density
and viscosity, and one hybrid initialization followed by the configured stages.
The equilibrium isentropic material remains the existing barotropic model;
adding the third spatial dimension does not add energy or phase-relaxation
transport. The workflow leaves the supplied mesh resolution unchanged.

Each stage may stop at the residual tolerances and then advances to the next
stage. Final checks include pressure/property extrema, inlet/outlet mass balance,
all configured residuals, and optional wall y-plus diagnostics. Wall diagnostics
help assess the supplied grid; they do not refine it. A valid but unconverged run
saves its final iterate with `converged: false`; solver errors or invalid property
fields prevent the final case/data save. Pressures outside the barotropic fit
range produce a warning and are recorded in the summary.

Pressure and barotropic Mach contours show an internal midspan section in the
actual 3D coordinates: a cylinder for axial geometry and a midspan plane for
radial or linear geometry. Periodic instances are display copies; the simulation
still contains one passage. The Mach contour uses
`VelocityMagnitude / barotropic_speed_of_sound`, as in the 2D example.

| Output location | Contents |
| --- | --- |
| `output/workflow.log` | Workflow progress, final results and failures |
| `output/gmsh/` | Reference/volume meshes, CGNS export and mesh reports; enabled geometry/mesh previews |
| `output/barotropy/` | Property expressions, inlet state, fit validation and enabled fit figures |
| `output/fluent/` | `stator_fluent_3d.cas.h5`, matching `.dat.h5` after solving, transcript, setup/run summaries, residual CSV/PNG and enabled pressure/Mach PNGs |

Preparation-only runs produce no Fluent artifacts. Case-only runs produce a
setup case, setup summary and transcript. A solved run replaces that setup case
with the final case and matching data. Stable filenames are overwritten on repeat
runs, and the transcript is reset for each session.

Implementation lives in
[`workflow_3d.py`](../../../src/turbo_moc/fluent/workflow_3d.py), with shared Fluent
execution in [`common.py`](../../../src/turbo_moc/fluent/common.py). Automated checks
exercise configuration and mocked solver behavior. A live Fluent 2024 R2 axial
case with 229,075 cells verified CGNS import, rotational periodicity of
`-360/31` degrees, a staged 10-iteration execution, final reports including wall
y-plus, both pressure/Mach PNGs, and final case/data output. This short run did
not converge and does not establish mesh independence or aerodynamic accuracy.
Radial and linear Fluent setup currently have unit-test coverage only. The
mesher and barotropic physics were unchanged. Before relying on a result,
inspect the saved case, transcript, run summary, mesh quality, wall y-plus and
contours.

## Offline PyVista postprocessing

After solving, run the adjacent plotting script from the repository root:

```powershell
poetry install --extras fluent
# Save PNGs and open interactive windows; close each window to advance:
poetry run python examples/fluent_3D/stator/plot_fluent_case.py
# Export only, without opening windows:
poetry run python examples/fluent_3D/stator/plot_fluent_case.py --no-show
```

The script reads the matching `stator_fluent_3d.cas.h5` and `.dat.h5` files plus
`setup_summary.json`. Paths resolve beside the script. PyVista's Fluent CFF reader
loads the volume solution; PyFluent's offline HDF5 reader provides the blade/hub
faces and their saved pressure values. No Fluent session or license is needed.

Five PNGs at 400 DPI are saved under `output/pyvista/`, beside `barotropy/`, `fluent/`, and
`gmsh/`: `blade_hub_pressure.png`, `midspan_flow.png`,
`three_blades_hub_surface.png`, `three_blades_hub_mesh.png`, and
`single_blade_views.png`.
Each image is exported before its interactive window opens for inspection, so
closing the window does not interrupt saving. Individual images are 3200 by
2000 pixels (8 by 5 inches); the four-view image is 4800 by 3200 pixels
(12 by 8 inches). The PNG metadata records 400 DPI.
The midspan cut is cylindrical for axial geometry and planar for radial/linear
geometry. It uses saved Mach when available and otherwise velocity magnitude;
the barotropic Mach expression is normally evaluated in Fluent and is not a
stored solution field. Cell values are averaged to nodes for the midspan contour.
The midspan view also includes the blade and hub surfaces. Cameras align with
the blade chord and use an oblique view above the hub, with the same orientation
for the pressure, flow, surface and mesh images. Black lines trace surface
outlines and sharp joins; only the mesh image includes individual cell edges.
The three-passage views use the saved periodic transformation. The mesh image
shows actual blade/hub boundary cells, including their quad/triangle topology.

The single-blade PNG contains four camera views in one 2-by-2 figure. Complete
blade geometry views reconstruct the hub and shroud caps from the two saved
blade end rings, including cylindrical curvature for axial cases. These caps
are used for geometry display; pressure plots use the saved fluid-facing wall
values, and the mesh view uses the original boundary mesh.
