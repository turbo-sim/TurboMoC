import numpy as np

from turbo_moc.app import workflow as flow
from turbo_moc.geometry import blade_row as geo
from turbo_moc.geometry.stator_blade import parametrize_stator


def exported(monkeypatch, nozzle, config):
    wall = nozzle["wall_final"]
    template = parametrize_stator(wall["x"], wall["y"], **config["blade_geometry"])
    design = geo.build_design(nozzle, config, template=template)
    monkeypatch.setattr(flow, "get_stator_template", lambda *args, **kwargs: template)
    monkeypatch.setattr(flow, "get_design", lambda *args, **kwargs: design)
    text = flow.stator_control_points_csv(config, None)
    rows = [line.split(",") for line in text.splitlines() if not line.startswith("#")][1:]
    return text, template, design, np.array([[float(value) for value in row] for row in rows])


def test_export_matches_the_design_control_points(monkeypatch, small_design):
    nozzle, config = small_design
    text, template, design, table = exported(monkeypatch, nozzle, config)
    assert np.allclose(table[:, 1:3], template["control_points"])
    assert np.allclose(table[:, 3:5], design.blade["control_points"])
    assert f"degree {len(template['knots']) - len(template['control_points']) - 1}" in text
    assert not table[:, 6].any()


def test_export_includes_applied_editor_offsets(monkeypatch, small_design):
    nozzle, config = small_design
    _, _, _, base = exported(monkeypatch, nozzle, config)
    config["blade_geometry"]["control_point_offsets"] = {6: 0.01}
    text, template, design, table = exported(monkeypatch, nozzle, config)
    moved = np.linalg.norm(table[:, 1:3] - base[:, 1:3], axis=1) > 1e-12
    assert np.nonzero(moved)[0].tolist() == [6]
    assert table[6, 6] == 0.01 and "1 moved by the control-point editor" in text
    assert np.allclose(table[:, 3:5], design.blade["control_points"])

