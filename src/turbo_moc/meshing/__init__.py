"""Gmsh meshing of turbomachinery blade passages.

Submodules are imported on demand so importing turbo_moc needs no Gmsh setup.
The canonical pipeline is in meshing_2d and meshing_3d, with independent
quality and inspection helpers. Historical stator/rotor mesh module names
are compatibility aliases to turbo_moc.deprecated.meshing; structured
meshing and cell repair remain confined to those legacy workflows.
"""
