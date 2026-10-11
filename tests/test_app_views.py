"""Row visibility, assembly coordinates and mesh/geometry preview integrity."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

from turbo_moc.geometry import geometry_3d as geo
from turbo_moc.apps.app import workflow as flow
from turbo_moc import configuration_project as projects
from app_helpers import row_state

CASE_DIR = Path(__file__).resolve().parents[1] / "examples/apps/cases"
CASES = {n: CASE_DIR / f"{n}.yaml" for n in ("axial_stator", "radial_stator")}


@pytest.mark.parametrize("family", ["axial", "radial", "curved"])
def test_preview_blade_is_closed_and_has_smooth_normals(family, small_design):
    from turbo_moc.apps.app.preview import preview_geometry
    nozzle, config = small_design
    definition = None
    if family == "radial":
        config["meridional_channel"] = flow.read_configuration(
            CASES["radial_stator"])["meridional_channel"]
    elif family == "curved":
        knots = [0,0,0,0,1,1,1,1]
        definition = geo.GeometryDefinition(
            BSpline(knots, [[100,0],[100,8],[101,15],[102,20]], 3),
            BSpline(knots, [[110,0],[110,8],[111,16],[112,22]], 3), 12)
    design = geo.build_design(nozzle, config, definition=definition)
    data = preview_geometry(design)
    blade = data["surfaces"]["blade"]
    xyz = np.asarray(blade["points"]).reshape(-1, 3)
    normals = np.asarray(blade["normals"]).reshape(-1, 3)
    assert np.isfinite(xyz).all() and normals.shape == xyz.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    # Sharp-edge normal splitting duplicates coordinates. Weld those duplicates
    # for a topology check: every edge belongs to exactly two triangles.
    welded, inverse = np.unique(xyz, axis=0, return_inverse=True)
    triangles = inverse[np.asarray(blade["polys"]).reshape(-1, 4)[:, 1:]]
    edges = np.concatenate([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]])
    unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2), "Blade preview has an uncapped or mismatched edge."
    assert len(welded) - len(unique_edges) + len(triangles) == 2

    # The supporting hub is a single revolution sector without a blade hole.
    # Its Euler characteristic is one (a disk), rather than zero (an annulus).
    hub = data["surfaces"]["hub"]
    xyz = np.asarray(hub["points"]).reshape(-1, 3)
    welded, inverse = np.unique(xyz, axis=0, return_inverse=True)
    triangles = inverse[np.asarray(hub["polys"]).reshape(-1, 4)[:, 1:]]
    edges = np.concatenate([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]])
    unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.isfinite(xyz).all() and np.all(np.isin(counts, [1, 2]))
    assert len(welded) - len(unique_edges) + len(triangles) == 1
    boundary = unique_edges[counts == 1]
    _, degree = np.unique(boundary, return_counts=True)
    assert np.all(degree == 2), "Hub boundary contains an open or branched seam."


def test_meridional_mesh_is_empty_until_reference_is_current(small_design, monkeypatch):
    from turbo_moc.apps.app import create_app
    nozzle, config = small_design
    design = geo.build_design(nozzle, config)
    monkeypatch.setattr(flow, "get_design", lambda *args: design)
    app = create_app()
    key = next(k for k in app.callback_map if k.startswith("..reference-plot.figure"))
    callback = app.callback_map[key]["callback"].__wrapped__
    state = row_state(config)
    for reference in (None, {"key": "outdated"}):
        section, grid, caption, _ = callback(state, {"stator_1": {"reference": reference}}, None, {})
        # The section keeps an empty GL trace to reuse its browser scene. No
        # coordinates or legend may appear before a current mesh exists.
        assert all(len(trace.x) == len(trace.y) == 0 and not trace.showlegend
                   for trace in section.data)
        assert not grid.data
        assert "up-to-date 2D mesh" in caption


def test_volume_display_edits_preserve_2d_plots(small_design, monkeypatch):
    from dash import Patch, no_update
    from turbo_moc.apps.app import application
    nozzle, config = small_design
    design = geo.build_design(nozzle, config)
    monkeypatch.setattr(flow, "get_design", lambda *args: design)
    # Reference stations are enough for this callback regression; no numerical
    # job is needed to check which visualization edits replace a plot.
    section = {"xy": design.curves["periodic_left"], "cells": {2: np.array([[0, 1, 2]])},
               "boundaries": {"periodic_left": np.arange(len(design.curves["periodic_left"]))}}
    reads = []
    def artifact(*args):
        reads.append(args)
        return section, design
    monkeypatch.setattr(application, "reference_artifact", artifact)
    app = application.create_app()
    key = next(k for k in app.callback_map if k.startswith("..reference-plot.figure"))
    callback = app.callback_map[key]["callback"].__wrapped__
    state = row_state(config)
    reference = {"session": "a"*32, "job": "b"*32, "key": state["reference"]}
    registry = {"stator_1": {"reference": reference}}
    first = callback(state, registry, None, {})
    config["mesh"]["visualization"]["volume_edge_width"] = 1.2
    state = row_state(config)
    assert all(value is no_update for value in callback(
        state, registry, None, first[0].to_plotly_json()))
    assert len(reads) == 1
    config["mesh"]["visualization"]["section_edge_width"] = 0.65
    state = row_state(config)
    changed = callback(state, registry, None, first[0].to_plotly_json())
    assert len(reads) == 1
    assert all(isinstance(patch, Patch) for patch in changed[:2])
    assert all(operation["params"]["value"] == 0.65 for patch in changed[:2]
               for operation in patch.to_plotly_json()["operations"]
               if operation["location"][-1] == "width")
    config["mesh"]["numerics"]["span_layers"] += 1
    state = row_state(config)
    changed = callback(state, registry, None, first[0].to_plotly_json())
    assert len(reads) == 2 and changed[0].data[0].line.width == 0.65


def test_row_visibility_filters_geometry_and_shared_mesh_projection(small_design, monkeypatch):
    from turbo_moc.apps.app import application
    nozzle, config = small_design
    design = geo.build_design(nozzle, config)
    design.blade.update(natural_pitch=design.blade["pitch"], metal_angle_out=0.)
    state = row_state(config)
    state["project"] = projects.set_design_mode(state["project"], "stator_rotor")
    rows = projects.active_rows(state["project"])
    monkeypatch.setattr(application, "ready_rows", lambda *_: [
        (row, config, {}, {"geometry": row["id"]}) for row in rows])
    monkeypatch.setattr(flow, "get_design", lambda *_: design)
    monkeypatch.setattr(application, "cached_preview", lambda *_: {"surfaces": {}})
    monkeypatch.setattr(flow, "resolved_configuration", lambda *_: config)
    section = {"xy": design.curves["periodic_left"], "cells": {2: np.array([[0, 1, 2]])},
        "boundaries": {"periodic_left": np.arange(len(design.curves["periodic_left"]))}}
    monkeypatch.setattr(application, "reference_artifact", lambda *_: (section, design))
    app = application.create_app()
    callback = lambda prefix: next(value["callback"].__wrapped__
        for key, value in app.callback_map.items() if key.startswith(prefix))
    geometry = callback("..geometry-data.data")
    shown = geometry(state, {}, ["rotor_1"], None)
    assert len(shown[0]["rows"]) == 2  # Browser actor visibility retains all row artifacts.
    assert shown[2].data and all(trace.legendgroup == "Rotor" for trace in shown[2].data)
    assert all("Stator" not in trace.name for trace in shown[3].data)
    blade = next(trace for trace in shown[3].data if trace.name == "Rotor")
    assert blade.line.dash in (None, "solid")
    domains = [trace for trace in shown[3].data if "flow domain" in trace.name]
    assert len(domains) == 2 and all(trace.line.dash == "dash" for trace in domains)
    both = geometry(state, {}, ["stator_1", "rotor_1"], None)
    assert {trace.legendgroup for trace in both[2].data} == {"Stator", "Rotor"}
    assert both[0]["key"] == shown[0]["key"]  # Visibility cannot reset the 3D camera.
    assert not geometry(state, {}, [], None)[3].data
    reference = {"session": "a"*32, "job": "b"*32, "key": flow.keys(config)["reference"]}
    registry = {row["id"]: {"reference": reference} for row in rows}
    mesh_view = callback("..reference-plot.figure")
    shown = mesh_view(state, registry, ["rotor_1"], {})
    assert shown[0].data and all(trace.legendgroup == "Rotor" for trace in shown[0].data)
    assert shown[1].data and all(trace.legendgroup == "rotor_1" for trace in shown[1].data)
    both = mesh_view(state, registry, ["stator_1", "rotor_1"], {})
    assert {trace.legendgroup for trace in both[0].data} == {"Stator", "Rotor"}
    widths = both[0].layout.meta["section_traces"]
    assert widths == {"stator_1": 0, "rotor_1": 2}
    config["mesh"]["visualization"]["section_edge_width"] = .65
    patched = mesh_view(state, registry, ["stator_1", "rotor_1"], both[0].to_plotly_json())
    operations = patched[0].to_plotly_json()["operations"]
    assert {operation["location"][1] for operation in operations
        if operation["location"][-1] == "width"} == {0, 2}
    assert {trace.legendgroup for trace in both[1].data} == {"stator_1", "rotor_1"}
    assert both[1].layout.yaxis.range[0] == 0
    assert not mesh_view(state, registry, [], {})[1].data


@pytest.mark.parametrize("family,radii", [("axial", None),
    ("radial", (100., 80., 78., 60.)), ("radial", (80., 100., 102., 120.))])
def test_shared_section_chart_preserves_interspace_and_interface(small_design, family, radii):
    from turbo_moc.apps.app import plots
    nozzle, base = small_design
    project = projects.set_design_mode(projects.default_project(family), "stator_rotor")
    stator, rotor = project["rows"]
    rotor["placement"]["angular_phase_deg"] = 18.
    if family == "axial":
        stator["geometry"]["meridional_channel"].update(hub_chord=20., tip_chord=22.)
        rotor["geometry"]["axial_interspace_mm"] = 3.
        expected = -25. / 105.
    else:
        stator["geometry"]["meridional_channel"].update(inlet_radius=radii[0], outlet_radius=radii[1])
        rotor["geometry"]["meridional_channel"].update(inlet_radius=radii[2], outlet_radius=radii[3])
        expected = -abs(np.log(radii[2] / radii[0]))
    views, designs = [], {}
    # Synthetic profiles isolate assembly coordinates from either flow solver.
    for row in (stator, rotor):
        config = deepcopy(base)
        config["meridional_channel"] = deepcopy(row["geometry"]["meridional_channel"])
        config["meridional_channel"]["num_blades"] = 12
        config["assembly"] = projects.row_assembly(project, row["id"])
        design = geo.build_design(nozzle, config)
        designs[row["id"]] = design
        views.append((row["id"], row["role"].title(), design, None, .35))
    offsets = plots.section_offsets(project, {row_id: design.meridional_channel for row_id, design in designs.items()})
    np.testing.assert_allclose(offsets["rotor_1"], [-np.pi / 10, expected], atol=1e-12)
    upstream = designs["stator_1"].curves["outlet"] / designs["stator_1"].meridional_channel.reference_radius
    downstream = designs["rotor_1"].curves["inlet"] / designs["rotor_1"].meridional_channel.reference_radius
    np.testing.assert_allclose(upstream[:, 1] + offsets["stator_1"][1],
        downstream[:, 1] + offsets["rotor_1"][1], atol=1e-12)
    both, _ = plots.assembly_section_figure(project, views)
    only_rotor, _ = plots.assembly_section_figure(project, views[1:])
    for together, alone in zip(both.data[5:], only_rotor.data):
        np.testing.assert_allclose(together.x, alone.x)
        np.testing.assert_allclose(together.y, alone.y)
