"""Placeholder for a future three-dimensional Fluent automation workflow.

No solver workflow is implemented here. The current Fluent implementation is
workflow_2d.py; three-dimensional Gmsh generation and CGNS export are available
independently in ../meshing/meshing_3d.py.

A future workflow should configure the exported hub/shroud walls and rotational
periodic boundaries, validate its barotropic model, and control a three-dimensional
Fluent solve. Importing this placeholder loads no optional solver dependencies and
starts no external process.
"""
