"""Opt-in checks for 3D Fluent cuts, periodic transforms and graphics controls."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from turbo_moc.fluent import graphics_3d as graphics


@pytest.fixture
def options():
    return {
        "pressure_contour": {
            "enabled": True,
            "filename": "pressure.png",
            "width": 640,
            "height": 480,
        },
        "mach_contour": {
            "enabled": True,
            "filename": "mach.png",
            "width": 640,
            "height": 480,
        },
        "periodic_instancing": {"enabled": True, "repeats": 3},
        "view": "isometric",
    }


@pytest.fixture
def geometry():
    return {
        "kind": "axial",
        "axis_origin": [0, 0, 0],
        "axis_direction": [0, 0, 1],
        "angle_deg": -360 / 31,
        "section": {"type": "cylinder", "radius_m": 0.105},
    }


@pytest.fixture
def solver(monkeypatch):
    # The optional PyFluent dependency is not needed to exercise validation or
    # the settings calls. A real Fluent import/render is a separate acceptance.
    monkeypatch.setitem(
        sys.modules,
        "ansys.fluent.core",
        SimpleNamespace(
            SurfaceFieldDataRequest=SimpleNamespace,
            SurfaceDataType=SimpleNamespace(
                Vertices="vertices", FacesConnectivity="faces"
            ),
        ),
    )
    solver = MagicMock()
    solver.settings.setup.cell_zone_conditions.fluid.get_object_names.return_value = [
        "imported-fluid"
    ]
    solver.fields.field_data.get_field_data.return_value = {
        graphics.SECTION_NAME: SimpleNamespace(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            connectivity=[np.array([0, 1, 2])],
        )
    }
    return solver


def test_axial_cut_and_copies_use_physical_cylinder_and_signed_rotation(
    solver, options, geometry
):
    summary = graphics.prepare_contour_graphics(solver, options, geometry)
    results = solver.settings.results
    results.surfaces.quadric_surface.__setitem__.assert_called_once_with(
        graphics.SECTION_NAME,
        {"attribute": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], "value": 0.105**2},
    )
    results.graphics.periodic_instancing.__setitem__.assert_called_once_with(
        "imported-fluid",
        {
            "periodic_type": "rotational",
            "rotation": [0, 0, 1],
            "axis_origin": [0, 0, 0],
            "rotation_angle": pytest.approx(-2 * np.pi / 31),
            "repeats": 3,
            "surfaces": [graphics.SECTION_NAME],
        },
    )
    assert summary == {"surface": graphics.SECTION_NAME, "vertices": 3, "faces": 1}
    contours = results.graphics.contour.__setitem__.call_args_list
    assert [call.args[1]["field"] for call in contours] == [
        "pressure",
        "expr:barotropic_mach_number",
    ]
    assert all(
        call.args[1]["surfaces_list"] == [graphics.SECTION_NAME] for call in contours
    )


def test_display_rotation_matches_one_physical_blade_pitch(options, geometry):
    angle = graphics._instance_settings(options, geometry)["rotation_angle"]
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    radius, axial = 0.105, 0.02
    left = np.array([radius, 0, axial])
    # The annular mesh has 31 blades and clockwise left-to-right pairing.
    right = np.array(
        [radius * np.cos(2 * np.pi / 31), -radius * np.sin(2 * np.pi / 31), axial]
    )
    assert np.allclose(rotation @ left, right, rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ["radial", "linear"])
def test_planar_midspan_cut_and_geometry_specific_copies(
    solver, options, geometry, kind
):
    geometry.update(
        kind=kind,
        translation_m=[0, 0.02, 0],
        section={"type": "plane", "point_m": [0, 0, 0.005], "normal": [0, 0, 2]},
    )
    graphics.prepare_contour_graphics(solver, options, geometry)
    results = solver.settings.results
    results.surfaces.plane_slice.__setitem__.assert_called_once_with(
        graphics.SECTION_NAME, {"normal": [0, 0, 1], "distance_from_origin": 0.005}
    )
    transform = results.graphics.periodic_instancing.__setitem__.call_args.args[1]
    if kind == "linear":
        assert transform["periodic_type"] == "translational"
        assert transform["translation"] == [0, 0.02, 0]
        assert "rotation_angle" not in transform
    else:
        assert transform["periodic_type"] == "rotational"


def test_all_disabled_contours_never_access_solver_or_create_output(options, tmp_path):
    options["pressure_contour"]["enabled"] = False
    options["mach_contour"]["enabled"] = False
    destination = tmp_path / "unused"
    assert graphics.prepare_contour_graphics(object(), options, {}) is None
    assert graphics.export_contours(object(), options, destination) == []
    assert not destination.exists()


@pytest.mark.parametrize(
    "section",
    [
        {"type": "cylinder", "radius_m": 0},
        {"type": "cylinder", "radius_m": float("nan")},
        {"type": "plane", "point_m": [0, 0, 0], "normal": [0, 0, 0]},
        {"type": "plane", "point_m": [0, float("inf"), 0], "normal": [0, 0, 1]},
        {"type": "unsupported"},
    ],
)
def test_invalid_sections_are_rejected_before_solver_access(options, geometry, section):
    geometry["section"] = section
    with pytest.raises(ValueError):
        graphics.prepare_contour_graphics(object(), options, geometry)


@pytest.mark.parametrize(
    "bad_data",
    [
        SimpleNamespace(vertices=np.empty((0, 3)), connectivity=[]),
        SimpleNamespace(vertices=np.ones((3, 2)), connectivity=[np.array([0, 1, 2])]),
        SimpleNamespace(
            vertices=np.full((3, 3), np.nan), connectivity=[np.array([0, 1, 2])]
        ),
        SimpleNamespace(vertices=np.ones((3, 3)), connectivity=None),
    ],
)
def test_empty_or_invalid_cut_cannot_produce_blank_contour(
    solver, options, geometry, bad_data
):
    solver.fields.field_data.get_field_data.return_value = {
        graphics.SECTION_NAME: bad_data
    }
    with pytest.raises(RuntimeError, match="no valid surface faces"):
        graphics.prepare_contour_graphics(solver, options, geometry)
    solver.settings.results.graphics.contour.__setitem__.assert_not_called()


def test_solver_section_errors_propagate(solver, options, geometry):
    solver.fields.field_data.get_field_data.side_effect = RuntimeError(
        "surface creation failed"
    )
    with pytest.raises(RuntimeError, match="surface creation failed"):
        graphics.prepare_contour_graphics(solver, options, geometry)


def test_exports_both_images_with_periodic_copies_and_requested_view(
    solver, options, tmp_path
):
    results = solver.settings.results.graphics
    results.picture.save_picture.side_effect = lambda file_name: Path(
        file_name
    ).write_bytes(b"png")
    images = graphics.export_contours(solver, options, tmp_path)
    assert [image.name for image in images] == ["pressure.png", "mach.png"]
    assert all(image.stat().st_size > 0 for image in images)
    assert results.periodic_instancing["imported-fluid"].display.call_count == 2
    assert results.views.restore_view.call_args.kwargs == {"view_name": "isometric"}
    assert results.views.auto_scale.call_count == 2


def test_disabled_image_and_instances_are_not_exported(solver, options, tmp_path):
    options["mach_contour"]["enabled"] = False
    options["periodic_instancing"]["enabled"] = False
    picture = solver.settings.results.graphics.picture
    picture.save_picture.side_effect = lambda file_name: Path(file_name).write_bytes(
        b"png"
    )
    images = graphics.export_contours(solver, options, tmp_path)
    assert [image.name for image in images] == ["pressure.png"]
    assert not (tmp_path / "mach.png").exists()
    solver.settings.results.graphics.periodic_instancing.__getitem__.assert_not_called()


def test_export_failure_is_reported(solver, options, tmp_path):
    with pytest.raises(RuntimeError, match="did not produce"):
        graphics.export_contours(solver, options, tmp_path)


@pytest.mark.parametrize("zones", [[], ["fluid-a", "fluid-b"]])
def test_periodic_graphics_require_one_imported_fluid_zone(
    solver, options, geometry, zones
):
    solver.settings.setup.cell_zone_conditions.fluid.get_object_names.return_value = (
        zones
    )
    with pytest.raises(RuntimeError, match="exactly one fluid cell zone"):
        graphics.prepare_contour_graphics(solver, options, geometry)
    solver.settings.results.surfaces.quadric_surface.__setitem__.assert_not_called()


def test_cylinder_equation_supports_offset_axis(geometry):
    geometry = deepcopy(geometry)
    geometry.update(axis_origin=[0.3, -0.1, 2], axis_direction=[0, 0, 5])
    _, settings = graphics._section_settings(geometry)
    radius = geometry["section"]["radius_m"]
    theta = np.linspace(0, 2 * np.pi, 20)
    x, y, z = 0.3 + radius * np.cos(theta), -0.1 + radius * np.sin(theta), theta
    monomials = np.array(
        [x * x, y * y, z * z, x * y, y * z, z * x, x, y, z, np.ones_like(theta)]
    )
    assert np.allclose(np.asarray(settings["attribute"]) @ monomials, settings["value"])
