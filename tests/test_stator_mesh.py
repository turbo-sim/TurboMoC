"""Small workflow checks: a real mesh, fresh MoC solves and failed-mesh cleanup."""

import copy
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import gmsh
import numpy as np
import pytest

from turbo_moc.io.cgns_adf import read_adf


@pytest.fixture(autouse=True)
def without_barotropy(monkeypatch):
    """Mesh workflows must work without the optional Fluent companion package."""
    monkeypatch.setitem(sys.modules, "barotropy", None)


@pytest.fixture(scope="module")
def mesh_example():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        runner = importlib.import_module("turbo_moc.meshing.meshing_2d")
        meshing = importlib.import_module("turbo_moc.meshing.meshing_2d")
        from turbo_moc.configuration_2d import read_configuration
        from turbo_moc.geometry import geometry_2d
        config = read_configuration(Path(__file__).resolve().parents[1] / "examples/gmsh_2D/workflow_gmsh_stator_2D.yaml")
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
        blade, curves = geometry_2d.build_geometry(nozzle, config)
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


def test_explicit_moc_calls_always_solve_and_do_not_write_files(monkeypatch, tmp_path):
    from turbo_moc.geometry import moc
    result = {"converged": True, "wall_final": {"x": [0, 0.05], "y": [0.01, 0.015]}}
    solve = Mock(return_value=result)
    monkeypatch.setattr(moc.turbo_moc, "design_nozzle", solve)
    monkeypatch.chdir(tmp_path)
    assert moc.solve_nozzle({"n": 15}) is result
    progress = Mock()
    assert moc.solve_nozzle({"n": 15}, progress_callback=progress) is result
    assert solve.call_args.kwargs["progress_callback"] is progress
    assert solve.call_count == 2
    assert not list(tmp_path.iterdir())
    solve.return_value = {"converged": False}
    with pytest.raises(RuntimeError, match="did not converge"):
        moc.solve_nozzle({"n": 15})


@pytest.mark.parametrize("wall", [
    {"x": [0, float("nan")], "y": [0, 1]},
    {"x": [0, 1], "y": [0]},
    {"x": [0], "y": [1]},
])
def test_moc_rejects_unusable_converged_walls(monkeypatch, wall):
    from turbo_moc.geometry import moc
    monkeypatch.setattr(moc.turbo_moc, "design_nozzle",
                        Mock(return_value={"converged": True, "wall_final": wall}))
    with pytest.raises(RuntimeError, match="did not converge"):
        moc.solve_nozzle({})


def test_invalid_periodic_geometry_closes_gmsh(mesh_example, tmp_path):
    _, meshing, blade, curves, settings = mesh_example
    invalid = copy.deepcopy(curves)
    invalid["periodic_right"][1, 0] += 1
    with pytest.raises(ValueError, match="pitch-translated"):
        meshing.create_mesh(blade, invalid, settings, tmp_path, show_gui=False)
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("generate_mesh,show_gui,save_figure,show_figure", [
    (True, True, True, False),
    (True, False, False, True),
    (False, True, False, False),
])
def test_planar_workflow_obeys_yaml_controls(
        mesh_example, monkeypatch, tmp_path, generate_mesh, show_gui,
        save_figure, show_figure):
    import yaml
    import jaxprop as jxp
    import matplotlib.pyplot as plt
    from turbo_moc.geometry import geometry_2d, moc

    runner, _, blade, curves, _ = mesh_example
    source = Path(__file__).resolve().parents[1] / "examples/gmsh_2D/workflow_gmsh_stator_2D.yaml"
    config = yaml.safe_load(source.read_text())
    config["workflow"] = {
        "generate_mesh": generate_mesh,
        "geometry_figure": {"save": save_figure, "show": show_figure,
                            "size": [9, 5], "dpi": 123, "formats": [".svg"]},
        "reporting": {"level": "DEBUG", "filename": "custom.log"},
    }
    config["mesh"]["show_gui"] = show_gui
    case = tmp_path / "case.yaml"
    case.write_text(yaml.safe_dump(config))
    solve = Mock(return_value={})
    mesh = Mock()
    viewer = Mock()
    exports = []

    def save(figure, filename, *, formats, dpi):
        exports.append((tuple(figure.get_size_inches()), dpi,
                        formats, Path(filename).name))

    monkeypatch.setattr(moc, "solve_nozzle", solve)
    monkeypatch.setattr(geometry_2d, "build_geometry", lambda *args: (blade, curves))
    plot = Mock(wraps=geometry_2d.plot_construction)
    monkeypatch.setattr(geometry_2d, "plot_construction", plot)
    monkeypatch.setattr(runner, "create_mesh", mesh)
    monkeypatch.setattr(jxp, "savefig_in_formats", save)
    monkeypatch.setattr(plt, "show", viewer)
    dpi_before = plt.rcParams["savefig.dpi"]
    runner.main(case, tmp_path / "output")
    solve.assert_called_once()
    assert mesh.call_count == int(generate_mesh)
    if generate_mesh:
        assert mesh.call_args.kwargs["show_gui"] is show_gui
    assert plot.call_count == int(save_figure or show_figure)
    assert viewer.call_count == int(show_figure)
    assert exports == ([((9.0, 5.0), 123, [".svg"], "stator_geometry")] if save_figure else [])
    assert (tmp_path / "output/custom.log").is_file()
    assert plt.rcParams["savefig.dpi"] == dpi_before


@pytest.mark.parametrize("section,key,value", [
    (None, "generate_mesh", "false"),
    ("geometry_figure", "size", [0, 8]),
    ("geometry_figure", "dpi", True),
    ("geometry_figure", "formats", [".unsupported"]),
    ("reporting", "level", "VERBOSE"),
    ("reporting", "filename", "../outside.log"),
])
def test_invalid_workflow_settings_fail_before_moc(monkeypatch, tmp_path, section, key, value):
    import yaml
    from turbo_moc.geometry import moc
    from turbo_moc.meshing import meshing_2d

    source = Path(__file__).resolve().parents[1] / "examples/gmsh_2D/workflow_gmsh_stator_2D.yaml"
    config = yaml.safe_load(source.read_text())
    settings = config["workflow"] if section is None else config["workflow"][section]
    settings[key] = value
    case = tmp_path / "invalid.yaml"
    case.write_text(yaml.safe_dump(config))
    solve = Mock()
    monkeypatch.setattr(moc, "solve_nozzle", solve)
    with pytest.raises(ValueError, match="workflow"):
        meshing_2d.main(case, tmp_path / "output")
    solve.assert_not_called()
