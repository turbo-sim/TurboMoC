"""Moved to ``turbo_moc.meshing.stator_mesh_2d``.

This name is kept as an alias of the SAME module object, so existing
importers (examples/gmsh_3D, examples/fluent, tests) keep working unchanged,
including monkeypatching of module attributes (tests/test_stator_mesh.py
patches ``save_mesh_image``, which create_mesh looks up in this module).
"""

import sys

from turbo_moc.meshing import stator_mesh_2d as _impl

sys.modules[__name__] = _impl
