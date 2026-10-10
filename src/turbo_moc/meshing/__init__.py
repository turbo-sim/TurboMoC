"""Gmsh meshing of turbomachinery blade passages.

Submodules (imported on demand, so ``import turbo_moc`` never needs gmsh):

* ``stator_passage`` -- stator blade contour and smooth periodic flow domain.
* ``stator_mesh_2d`` -- 2D blade-to-blade Gmsh mesh with inflation layers,
  translational periodicity, CGNS export and quality report.
"""
