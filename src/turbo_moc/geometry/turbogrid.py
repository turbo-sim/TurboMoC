"""Compatibility entry point for the preserved legacy implementation.

Alias the module itself so private helpers and monkeypatched globals keep
the same identity as the implementation used by legacy callbacks.
"""
from importlib import import_module
import sys

_implementation = import_module("turbo_moc.deprecated.geometry.turbogrid")

sys.modules[__name__] = _implementation
