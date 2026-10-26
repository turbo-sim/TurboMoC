import json

import numpy as np
import pytest

from turbo_moc.app import display_panel
from turbo_moc.visualization import style


def test_app_palette_is_the_shared_palette():
    assert display_panel.COLORS == style.SURFACE_COLORS
    row = {"role": "rotor", "mesh": {"matplotlib": {"num_blades": 1},
                                     "pyvista": {"surfaces": {"blade": True}, "show_mesh_edges": False}},
           "geometry": {"meridional_channel": {"type": "axial", "num_blades": 10}}}
    assert display_panel.defaults(row)["colors"]["blade"] == style.ROTOR_BLADE_COLOR


def test_copy_transform_matches_the_app_actor_transform():
    first = style.copy_transform(0, is_linear=False, num_blades=4)
    assert np.allclose(first[:3, :3] @ [1, 0, 0], [-1, 0, 0])
    second = style.copy_transform(1, is_linear=False, num_blades=4,
                                  placement={"angular_phase_deg": 0, "streamwise_offset_mm": 5})
    assert np.allclose(second[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert second[2, 3] == 5
    linear = style.copy_transform(2, is_linear=True, pitch_mm=3.5)
    assert np.allclose(linear[:3, :3], np.eye(3)) and linear[1, 3] == 7


def test_quality_colors_follow_the_app_legend():
    colors = style.quality_rgb([0.0, 0.5, 1.0, np.nan])
    assert colors.tolist() == [[35, 155, 86], [244, 208, 63], [192, 57, 43], list(style.QUALITY_MISSING)]


def test_desktop_viewer_reads_the_app_surface_data(tmp_path):
    pytest.importorskip("pyvista")
    from turbo_moc.visualization import mesh_pyvista
    surface = {"points": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], "polys": [4, 0, 1, 2, 3]}
    (tmp_path / "browser_surfaces.json").write_text(json.dumps({"blade": surface}), encoding="utf-8")
    mesh = mesh_pyvista.load_surfaces(tmp_path, ("blade",))["blade"]
    assert (mesh.n_points, mesh.n_cells) == (4, 1)
    with pytest.raises(ValueError, match="hub"):
        mesh_pyvista.load_surfaces(tmp_path, ("blade", "hub"))
