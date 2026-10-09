"""Small workflow checks: one real mesh, cache reuse, and failed-mesh cleanup."""

import copy
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import gmsh
import numpy as np
import pytest

from turbo_moc.io.cgns_adf import read_adf


@pytest.fixture(scope="module")
def mesh_example():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        runner = importlib.import_module("examples.gmsh.run_mesh_generation")
        meshing = importlib.import_module("examples.gmsh.meshing_source")
        config = runner.read_configuration()
        # Exercise real blade fitting and automatic passage placement without
        # solving a thermodynamic MoC case. Keep the test mesh and plots small.
        config["blade"].update(
            r_trailing=0.1, num_baseline_points=100, num_curve_points=100
        )
        config["passage"]["smooth"]["num_points"] = 40
        config["passage"].update(inlet_chords=0.5, outlet_chords=0.5)
        x = np.linspace(0, 0.01, 80)
        nozzle = {
            "converged": True,
            "wall_final": {
                "x": x.tolist(),
                "y": (0.002 + 0.001 * (x / x[-1]) ** 2).tolist(),
            },
        }
        blade, curves = runner.geo.build_geometry(nozzle, config)
        settings = copy.deepcopy(config["mesh"])
        settings["show_gui"] = False
        settings["numerics"].update(
            terminal_output=False,
            blade_size=0.4,
            domain_size=2,
            curvature_elements=16,
            smoothing_steps=1,
        )
        settings["numerics"]["inflation_layers"].update(
            enabled=True, first_height=0.01, growth_ratio=1.5, thickness=0.2
        )
        settings["numerics"]["recombine"] = True
        settings["matplotlib"].update(dpi=40, num_blades=3, annotate_boundaries=True)
        # Full quality-report generation belongs to the example execution check.
        settings["report"]["enabled"] = False
        yield runner, meshing, blade, curves, settings


def test_mesh_workflow_exports_a_usable_periodic_mesh(
    mesh_example, monkeypatch, tmp_path
):
    from matplotlib.figure import Figure

    # Keep mesh generation/export real, but skip slow figure drawing.
    # Visual output is checked by running the examples, not by this smoke suite.
    plotted = {}

    def save_figure(figure, filename, **options):
        ax = figure.axes[0]
        plotted[Path(filename).name] = {
            "labels": {text.get_text() for text in ax.texts},
            "cells": len(ax.collections[0].get_paths()),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "boxed": all(spine.get_visible() for spine in ax.spines.values()),
            "opaque": all(collection.get_alpha() == 1 for collection in ax.collections),
            "font": ax.xaxis.label.get_fontfamily(),
        }
        Path(filename).write_bytes(b"mock figure")

    monkeypatch.setattr(Figure, "savefig", save_figure)
    _, meshing, blade, curves, settings = mesh_example
    save_requested_image = meshing.save_mesh_image

    def check_other_views(filename, surface, **kwargs):
        result = save_requested_image(filename, surface, **kwargs)
        for count, annotate in ((1, True), (2, True), (3, False)):
            view_file = tmp_path / f"check_{count}_{annotate}.png"
            options = kwargs["settings"] | {
                "num_blades": count,
                "annotate_boundaries": annotate,
            }
            save_requested_image(view_file, surface, **(kwargs | {"settings": options}))
            view = plotted[view_file.name]
            assert bool(view["labels"]) == annotate
            if annotate:
                assert view["labels"] == set(settings["cgns"]["boundary_types"])
        return result

    monkeypatch.setattr(meshing, "save_mesh_image", check_other_views)
    result = meshing.create_mesh(blade, curves, settings, tmp_path, show_gui=False)

    assert sum(result["cells"].values()) > 0
    assert result["minimum_scaled_jacobian"] > 0
    assert result["periodic_node_pairs"] > 0
    assert result["quadrilateral_fraction"] > 0
    # Inspect the exported file at the workflow level, leaving detailed topology
    # validation to the checks already performed by create_mesh/export_fluent_cgns.
    base = read_adf(result["mesh_file"]).get(label="CGNSBase_t")
    np.testing.assert_array_equal(base.data, [2, 2])
    zone = base.get(label="Zone_t")
    assert int(zone.data[0, 1]) == sum(result["cells"].values())
    boundaries = zone.get(label="ZoneBC_t").find_all("BC_t")
    assert {bc.name: bc.text for bc in boundaries} == settings["cgns"]["boundary_types"]
    assert (
        len(zone.get(label="ZoneGridConnectivity_t").find_all("GridConnectivity_t"))
        == 2
    )
    for filename in (
        Path(result["mesh_file"]).name,
        Path(result["image_file"]).name,
    ):
        assert (tmp_path / filename).stat().st_size > 0
    images = result["image_files"]
    assert set(images) == {"png", "svg"}
    for path in images.values():
        assert Path(path).stat().st_size > 0
        view = plotted[Path(path).name]
        assert view["cells"] == 3 * sum(result["cells"].values())
        assert view["xlabel"] == r"$x$ coordinate (mm)"
        assert view["ylabel"] == r"$y$ coordinate (mm)"
        assert view["boxed"] and view["opaque"]
        assert view["font"] == ["serif"]
        assert view["labels"] == set(settings["cgns"]["boundary_types"])
    assert not gmsh.isInitialized()


def test_nozzle_cache_reuses_matching_inputs_and_preserves_good_results(
    mesh_example, monkeypatch, tmp_path
):
    runner, *_ = mesh_example
    result = {"converged": True, "wall_final": {"x": [0, 0.05], "y": [0.01, 0.015]}}
    solve = Mock(return_value=result)
    monkeypatch.setattr(runner.turbo_moc, "design_nozzle", solve)
    cache = tmp_path / "nozzle.json"
    runner.load_or_solve_nozzle({"n": 15}, cache)
    assert runner.load_or_solve_nozzle({"n": 15}, cache) == result
    assert solve.call_count == 1
    runner.load_or_solve_nozzle({"n": 20}, cache)
    assert solve.call_count == 2
    previous = cache.read_text()
    solve.return_value = {"converged": False}
    with pytest.raises(RuntimeError, match="did not converge"):
        runner.load_or_solve_nozzle({"n": 20}, cache, force_solve=True)
    assert cache.read_text() == previous
    assert json.loads(previous)["result"] == result


def test_invalid_periodic_geometry_closes_gmsh(mesh_example, tmp_path):
    _, meshing, blade, curves, settings = mesh_example
    invalid = copy.deepcopy(curves)
    invalid["periodic_right"][1, 0] += 1
    with pytest.raises(ValueError, match="pitch-translated"):
        meshing.create_mesh(blade, invalid, settings, tmp_path, show_gui=False)
    assert not gmsh.isInitialized()
