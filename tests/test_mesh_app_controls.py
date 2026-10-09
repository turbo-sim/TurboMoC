"""Focused browser-input and snapshot checks; no MoC solve or mesh generation."""

from copy import deepcopy

import pytest

pytest.importorskip("dash_vtk")
from examples.gmsh_3D import run_mesh_app as app


def test_browser_numbers_allow_fractional_geometry_but_require_integer_counts():
    config = app.workflow.read_configuration(app.CASES["axial_stator"])
    # JavaScript sends an integral float such as 1.0 back to Python as 1.
    config["mesh"]["numerics"]["inflation_layers"]["thickness"] = 1
    identifiers = [
        {"path": "mesh.numerics.inflation_layers.thickness"},
        {"path": "machine.num_blades"},
    ]
    edited = app.edited_configuration(config, [0.3, 29.0], identifiers)
    assert edited["mesh"]["numerics"]["inflation_layers"]["thickness"] == 0.3
    assert type(edited["machine"]["num_blades"]) is int
    app.workflow.validate_configuration(edited)
    with pytest.raises(ValueError, match="requires an integer"):
        app.edited_configuration(config, [0.3, 29.5], identifiers)


def test_view_changes_preserve_snapshots_but_numerics_and_geometry_invalidate_them():
    config = app.workflow.read_configuration(app.CASES["axial_stator"])
    original = app.config_state(config)
    edited = deepcopy(config)
    edited["mesh"]["pyvista"]["surfaces"]["hub"] = False
    edited["mesh"]["pyvista"]["show_mesh_edges"] = False
    assert app.config_state(edited)["mesh_key"] == original["mesh_key"]
    edited["mesh"]["numerics"]["span_layers"] += 1
    state = app.config_state(edited)
    assert state["geometry_key"] == original["geometry_key"]
    assert state["mesh_key"] != original["mesh_key"]
    edited["machine"]["num_blades"] += 1
    assert app.config_state(edited)["geometry_key"] != original["geometry_key"]
