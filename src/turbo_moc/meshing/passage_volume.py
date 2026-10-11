"""Compatibility alias for the preserved legacy meshing implementation.

New applications use ``turbo_moc.meshing.meshing_2d`` and ``meshing_3d``.
This alias retains the same module object, including private helpers and
monkeypatches, for the legacy app and existing importers.
"""

import sys

from turbo_moc.deprecated.meshing import passage_volume as _implementation

sys.modules[__name__] = _implementation
