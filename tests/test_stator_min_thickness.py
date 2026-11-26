"""Stator option blade_geometry.min_thickness_ratio: the subsonic body is at least ratio x TE thickness."""
import numpy as np
import pytest

from turbo_moc.geometry.stator_blade import (convergent_thickness, editable_control_points,
                                             parametrize_stator)


def _inputs(config, **changes):
    inputs = {k: v for k, v in config["blade_geometry"].items() if k not in ("control_point_offsets",)}
    inputs.update(n_cp=30, num_baseline_points=400, num_curve_points=400)  # As real stators
    inputs.update(changes)
    return inputs


def test_thickening_reaches_target_and_keeps_the_moc_walls(small_design):
    nozzle, config = small_design
    wall = nozzle["wall_final"]
    base = parametrize_stator(wall["x"], wall["y"], **_inputs(config, min_thickness_ratio=0.0))
    _, thickness = convergent_thickness(base)  # Full subsonic range
    te = 2 * base["r_trailing"]
    ratio = 1.3 * thickness.min() / te  # Ask for 30% more than the baseline minimum
    thick = parametrize_stator(wall["x"], wall["y"], **_inputs(config, min_thickness_ratio=ratio))
    moved = {int(i) for i in thick["min_thickness_displacements"]}
    reach = max(np.asarray(base["control_points"])[i, 0] for i in moved)
    _, after = convergent_thickness(thick, end=reach)  # The stations the subsonic points control
    assert after.min() >= ratio * te * (1 - 2e-3)
    assert moved and moved <= set(editable_control_points(base))
    # Points downstream of the throats (the supersonic MoC walls) are not touched
    before, now = np.asarray(base["control_points"]), np.asarray(thick["control_points"])
    still = [i for i in range(len(before)) if i not in moved]
    assert np.allclose(before[still], now[still])
    limit = max(base["throat_point"]["x"], base["pressure_throat_point"]["x"])
    assert all(before[i, 0] < limit for i in moved)


def test_off_by_default_and_shown_for_new_rows(small_design):
    from turbo_moc.configuration import project as projects
    from turbo_moc.configuration.row import normalize
    _, config = small_design
    config["blade_geometry"].pop("min_thickness_ratio", None)
    assert normalize(config)["blade_geometry"]["min_thickness_ratio"] == 0.0
    new = projects.default_project("axial")
    assert projects.get_row(new, "stator_1")["geometry"]["blade_geometry"]["min_thickness_ratio"] == 0.0


def test_user_edits_are_relative_to_the_thickened_baseline(small_design):
    nozzle, config = small_design
    wall = nozzle["wall_final"]
    thick = parametrize_stator(wall["x"], wall["y"], **_inputs(config, min_thickness_ratio=5.0))
    index = editable_control_points(thick)[0]
    edited = parametrize_stator(wall["x"], wall["y"], **_inputs(
        config, min_thickness_ratio=5.0, control_point_displacements={index: [0.0, 0.01]}))
    delta = np.asarray(edited["control_points"]) - np.asarray(thick["control_points"])
    assert np.count_nonzero(np.abs(delta).sum(axis=1) > 1e-12) == 1


def test_unreachable_ratio_is_a_readable_error(small_design):
    nozzle, config = small_design
    wall = nozzle["wall_final"]
    with pytest.raises(ValueError, match="cannot be reached|self-intersect"):
        parametrize_stator(wall["x"], wall["y"], **_inputs(config, min_thickness_ratio=200.0))


def test_invalid_ratio(small_design):
    nozzle, config = small_design
    wall = nozzle["wall_final"]
    with pytest.raises(ValueError, match="min_thickness_ratio"):
        parametrize_stator(wall["x"], wall["y"], **_inputs(config, min_thickness_ratio=-1.0))
