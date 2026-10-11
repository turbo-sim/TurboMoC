# Gmsh 3D workflow

Edit `workflow_gmsh_stator_3D.yaml` and run `run_mesh_generation.py` without
arguments. The runner resolves its YAML relative to itself and writes to its
adjacent `output/` folder, regardless of the working directory.

```powershell
python examples/gmsh_3D/run_mesh_generation.py
```

The example calculates a fresh MoC wall, defines a conformal blade section,
generates its 2D reference mesh, deforms that mesh across span and maps it into
physical 3D coordinates. It audits volume connectivity, periodicity and quality
before exporting MSH and CGNS. Console progress is also saved in
`output/mesh_generation.log`. Geometry uses millimetres; CGNS length scaling
is specified by the YAML. The numerical callable accepts explicit configuration,
output and optional row ID, without assuming an examples-directory layout.

[Theory](../../z_information/theory_guide.md),
[deformation mathematics](../../z_information/deformation_note.md) and
[roadmap](../../z_information/roadmap.md) explain the pipeline. App runners and
sample app cases are separate under [apps/](../apps/readme.md).

## Workflow display and reporting

All viewer/export switches are read from this example's YAML:

| Setting | Effect |
| --- | --- |
| `workflow.show_gmsh` | Open the saved 3D mesh in Gmsh after export |
| `workflow.show_pyvista` | Open the configured surface view in PyVista |
| `workflow.save_figures` | Save the PyVista surface preview without requiring an interactive window |
| `workflow.log_level` / `.log_filename` | Console/file logging level and filename within `output/` |
| `mesh.pyvista.window_size` | Viewer and preview dimensions in pixels |
| `mesh.pyvista.figure_filename` | PNG preview filename within `output/` |
| `mesh.pyvista.surfaces` / `.show_mesh_edges` | Visible boundary patches and mesh-edge display |

The sample enables both native viewers; close Gmsh to proceed to PyVista.
For unattended runs, set both `show_*` switches to `false`. Set `save_figures`
to `false` too if no PyVista preview is wanted. PyVista is installed by default.
The source keeps filename/window
fallbacks for earlier 3D project files; current example controls are explicit.
