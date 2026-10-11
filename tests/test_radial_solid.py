"""Checks for the radial-wrap CAD solid (turbo_moc.geometry.blade's
_build_radial_blade_solid / export_radial_blade_step / _stl) and for the
conformal map's exact inlet/outlet radius identity the Phase 4 radial
cascade's gap readout relies on."""

import numpy as np
import pytest

from turbo_moc.geometry.blade import parametrize_stator_blade
from turbo_moc.geometry.radial import wrap_blade_radial, wrap_curve_radial, wrap_rotor_blade_radial

cq = pytest.importorskip("cadquery")

from turbo_moc.geometry.blade import (  # noqa: E402  (needs cadquery present)
    _build_radial_blade_solid,
    export_radial_blade_step,
    export_radial_blade_stl,
)

DEPTH = 5.0


def _shoelace(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _stator_blade():
    wall_x = np.linspace(0.0, 10.0, 50)
    return parametrize_stator_blade(wall_x, 1.0 + 0.1 * wall_x, metal_angle_in=0.0,
                                    metal_angle_out=70.0, r_trailing=0.1143)


def _rotor_blade():
    # Closed, rotor-convention contour: chordwise along x, pitchwise along y.
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    return {"blade": {"x": (2.0 * np.cos(t)).tolist(), "y": (0.3 * np.sin(t)).tolist()}}


def _assert_closed_solid(solid):
    assert solid.isValid()
    assert solid.Shells() and all(shell.Closed() for shell in solid.Shells())


def test_stator_radial_solid_volume_is_section_area_times_depth():
    blade = _stator_blade()
    r1, r2 = 150.0, 300.0
    _, solid = _build_radial_blade_solid(cq, blade, r1, r2, DEPTH, source="stator")

    wrapped = wrap_blade_radial(blade, r1, r2, n_blades=1)
    x = wrapped["blades"][0]["x"] + wrapped["trailing_edges"][0]["x"]
    y = wrapped["blades"][0]["y"] + wrapped["trailing_edges"][0]["y"]

    _assert_closed_solid(solid)
    assert np.isclose(solid.Volume(), _shoelace(x, y) * DEPTH, rtol=1e-8)


def test_rotor_radial_solid_volume_is_section_area_times_depth():
    rotor = _rotor_blade()
    r1, r2 = 100.0, 200.0
    _, solid = _build_radial_blade_solid(cq, rotor, r1, r2, DEPTH, source="rotor")

    wrapped = wrap_rotor_blade_radial(rotor, r1, r2, n_blades=1)
    x, y = wrapped["blades"][0]["x"], wrapped["blades"][0]["y"]

    _assert_closed_solid(solid)
    assert np.isclose(solid.Volume(), _shoelace(x, y) * DEPTH, rtol=1e-8)


def test_rotor_radial_wrap_is_independent_of_blade_scale():
    """Why the app has no rotor 'scale' input in radial mode: the map
    normalises the contour by its own chordwise extent, so a nondimensional
    (r*) blade and any uniformly rescaled copy wrap to the same geometry."""
    rotor = _rotor_blade()
    big = {"blade": {k: [v * 437.0 for v in rotor["blade"][k]] for k in ("x", "y")}}
    a = wrap_rotor_blade_radial(rotor, 100.0, 200.0, n_blades=5)["blades"]
    b = wrap_rotor_blade_radial(big, 100.0, 200.0, n_blades=5)["blades"]
    for ca, cb in zip(a, b):
        assert np.allclose(ca["x"], cb["x"], rtol=0, atol=1e-9)
        assert np.allclose(ca["y"], cb["y"], rtol=0, atol=1e-9)


def test_radial_solid_spans_exactly_z_offset_to_z_offset_plus_depth():
    z_offset = 12.5
    _, solid = _build_radial_blade_solid(cq, _rotor_blade(), 100.0, 200.0, DEPTH,
                                          source="rotor", z_offset=z_offset)
    bb = solid.BoundingBox()
    assert np.isclose(bb.zmin, z_offset)
    assert np.isclose(bb.zmax, z_offset + DEPTH)


def test_radial_solid_rejects_unknown_source():
    with pytest.raises(ValueError):
        _build_radial_blade_solid(cq, _rotor_blade(), 100.0, 200.0, DEPTH, source="diffuser")


def test_radial_step_and_stl_exports_write_nonempty_files(tmp_path):
    face, solid, stl = tmp_path / "face.step", tmp_path / "solid.step", tmp_path / "blade.stl"
    export_radial_blade_step(_stator_blade(), 150.0, 300.0, DEPTH, face, solid, source="stator")
    export_radial_blade_stl(_rotor_blade(), 100.0, 200.0, DEPTH, stl, source="rotor")
    for path in (face, solid, stl):
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.parametrize("chordwise_axis", ["x", "y"])
@pytest.mark.parametrize("r1, r2", [(100.0, 200.0), (300.0, 150.0)])
def test_conformal_map_sends_chord_max_to_r1_and_chord_min_to_r2(chordwise_axis, r1, r2):
    """The Phase 4 radial cascade's gap readout reads r1/r2 straight off
    each row's inputs instead of evaluating the map -- valid only because
    _reference_point_and_scale anchors x1 at the chord maximum and sets
    c_axial = min - max, making r == r1 / r2 EXACT at the two extremes."""
    t = np.linspace(0.0, 2.0 * np.pi, 150)
    a, b = 3.0 * np.cos(t) + 0.7, 0.4 * np.sin(t)
    x, y = (a, b) if chordwise_axis == "x" else (b, a)
    chord = x if chordwise_axis == "x" else y

    copies, _ = wrap_curve_radial(x, y, r1, r2, 1, chordwise_axis=chordwise_axis)
    radius = np.hypot(copies[0]["x"], copies[0]["y"])

    assert np.isclose(radius[np.argmax(chord)], r1)
    assert np.isclose(radius[np.argmin(chord)], r2)
