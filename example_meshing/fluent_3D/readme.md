# Fluent 3D

Reserved for a future automated 3D Fluent workflow. No solver setup or
execution is implemented here. [workflow_3d.py](../src/fluent/workflow_3d.py)
is an import-safe source placeholder; it starts no solver and loads no optional
Fluent dependencies.

The current 3D Gmsh/CGNS pipeline is implemented in
[meshing_3d.py](../src/meshing/meshing_3d.py), with case files and launchers under
[gmsh_3D/](../gmsh_3D/readme.md). Fluent automation is available in
[workflow_2d.py](../src/fluent/workflow_2d.py), launched from
[fluent_2D/](../fluent_2D/readme.md). That workflow calculates a fresh MoC result
once per invocation and shares it with its geometry and barotropic model.
