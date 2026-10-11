"""Moved to ``turbo_moc.meshing.stator_passage``.

This name is kept as an alias of the SAME module object, so existing
importers (examples/gmsh_3D, examples/fluent, tests) keep working unchanged,
including monkeypatching of module attributes.
"""

import sys

from turbo_moc.meshing import stator_passage as _impl

sys.modules[__name__] = _impl
