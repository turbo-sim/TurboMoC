# Adaptive-size 2D Gmsh mesh

This example uses the same fitted blade and flow-domain inputs as
`examples/gmsh_2D`, but calls `turbo_moc.meshing.mesh_2d.create_mesh`.
Set `domain_min_size` and `domain_size` under `mesh.numerics`; they are the
lower and upper global element-size bounds in millimetres. `blade_size`
continues to control the blade CAD point size and local blade resolution.
Gmsh adapts the local size between those bounds using curvature, wall spacing
and inflation settings.

Run from the repository root:

```powershell
poetry run python examples/gmsh_2D_adaptive/run_mesh_generation.py
```
