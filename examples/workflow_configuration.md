# Reuse an app project in Gmsh and Fluent workflows

All four workflow examples use the app project structure: `machine`, `workflow`, and `rows`. The Gmsh YAML files are complete app projects. The Fluent YAML files contain the same project plus automation, barotropic-model, and Fluent settings. Geometry and mesh parameters have the same names, nesting, units, validation, and meaning as in the app.

## Copy settings from the app

1. Download the project YAML from the app (or extract `project.yaml` from its ZIP export).
2. For Gmsh, replace the entire adjacent workflow YAML with that downloaded project.
3. For Fluent, replace the complete `machine`, `workflow`, and `rows` blocks with those from the download. Keep the other top-level blocks from the Fluent example, including `row_id`, `output_dir`, `geometry_figure`, `reporting`, `barotropic`, and `fluent`, plus the execution/viewer switches. Do not add a `design_file` alongside an embedded project.
4. Set `row_id` to an active stator row for Fluent. The Gmsh callables select the first active stator by default and also accept `row_id=`.
5. Run the adjacent Python launcher. Paths in Fluent settings resolve relative to that YAML; Gmsh launchers write to their adjacent `output/` directories.

No renaming to `nozzle`, `blade`, or `passage`, flattening, or manual transfer of fluid properties is needed. The working fluid stays at `machine.fluid_name`; geometry and mesh stay under each `rows` entry. The selected row's physical shared-interface placement is retained when using a stator?rotor app project. These automation runners solve a stationary stator only, not a coupled stator?rotor CFD stage.

## Choose the appropriate workflow

| Example | App project used by the supplied YAML | Output |
| --- | --- | --- |
| `gmsh_2D` | `linear_stator.yaml` | Scaled 2D reference passage and CGNS |
| `gmsh_3D` | `axial_stator.yaml` | Reference mesh, swept physical volume and CGNS |
| `fluent_2D` | Same project as Gmsh 2D | Physical linear-cascade mesh, barotropic model and optional Fluent run |
| `fluent_3D` | Same project as Gmsh 3D | Physical volume mesh, barotropic model and optional Fluent run |

Fluent 2D accepts linear cascades only. Gmsh 2D can also construct an axial or radial row's conformal reference section, but that chart is not a physical Cartesian annular passage; use Fluent 3D for those geometries. Planar workflow coordinates are pitchwise x and negative-streamwise y, in millimetres before CGNS scaling. App-exported linear CGNS uses a rotated physical coordinate convention; lengths and geometry are equivalent.

The 2D workflow now respects the app's meridional chord scaling and pitch. Compared with the historical standalone planar example, this can change the physical blade size and cell count. Span settings remain in the YAML but are unused by 2D meshing. Hub/shroud boundary types remain stored in the project and are omitted only when writing a 2D mesh. The configured CGNS filename is respected, even if its name contains `3d`; choose a suitable filename in the app or YAML if desired.

## Execution settings

Gmsh uses `workflow.save_figures`, `workflow.show_gmsh`, `workflow.show_pyvista` (3D only), and `workflow.log_level`. For unattended Gmsh runs set viewer switches to `false`. Optional Gmsh 2D overrides under `workflow` are `generate_mesh`, `geometry_figure` (`save`, `show`, `size`, `dpi`, `formats`), and `reporting` (`level`, `filename`). Without overrides it generates a mesh, uses `save_figures` for the construction plot, keeps Matplotlib noninteractive, and logs to `mesh_generation.log`. Per-row `mesh.matplotlib` controls mesh images separately.

Fluent's top-level `geometry_figure`, `show_gmsh`, `show_pyvista`/`save_mesh_figure` (3D), `reporting`, `prepare_only`, and `run_case` control its execution independently of the app's `workflow` preferences. `prepare_only: true` generates the mesh and barotropic model without launching Fluent; it still requires the barotropy dependency. The row's CGNS export must be enabled with `length_scale: 0.001` for metre coordinates.

## Older configurations

Historical standalone planar YAML and the separate Fluent `design_file` format remain readable. In the latter form omit the embedded `machine`, `workflow`, and `rows` blocks, and point `design_file` to a Gmsh YAML or downloaded app project. Relative paths retain their original YAML-relative meaning. New example files use the inline form so each is self-contained.
