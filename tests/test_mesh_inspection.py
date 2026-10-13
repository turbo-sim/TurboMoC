"""Saved-volume inspection preserves cell identity, quality and sweep layers."""
import json
import os

import numpy as np
import pytest
from scipy.interpolate import BSpline

from turbo_moc.meshing import inspection


@pytest.fixture(autouse=True)
def clear_inspection_caches():
    inspection._load_artifact.cache_clear()
    inspection._inspect_cached.cache_clear()
    yield
    inspection._load_artifact.cache_clear()
    inspection._inspect_cached.cache_clear()


@pytest.fixture
def mixed_volume(tmp_path):
    # Cells are deliberately not in layer or ID order, and type blocks differ.
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [2, 1.]])
    stations = [0, .2, 1]
    xyz = np.concatenate([np.column_stack((xy + [eta / 3, 0], np.full(7, eta)))
                          for eta in stations])
    arrays = {"xyz": xyz,
        "cells_5": np.array([[7, 8, 9, 10, 14, 15, 16, 17], [0, 1, 2, 3, 7, 8, 9, 10]]),
        "cells_6": np.array([[4, 5, 6, 11, 12, 13], [11, 12, 13, 18, 19, 20]]),
        "tags_5": np.array([40, 10]), "tags_6": np.array([30, 20]),
        "skewness": np.array([.9, .2, .9, .4]),
        "orthogonality": np.array([.2, .8, .2, .9]),
        "aspect_ratio": np.array([100., 1., 10., 2.])}
    np.savez_compressed(tmp_path / "volume_data.npz", **arrays)
    (tmp_path / "mesh_summary.json").write_text(json.dumps({
        "section_nodes": 7, "span_stations": stations}), encoding="utf-8")
    return tmp_path, arrays


def face_data(result):
    data = result["surfaces"]["inspection"]
    position, faces = 0, 0
    while position < len(data["polys"]):
        position += data["polys"][position] + 1
        faces += 1
    assert position == len(data["polys"])
    assert len(data["cell_ids"]) == len(data["cell_values"]) == len(data["color_values"]) == faces
    return data


@pytest.mark.parametrize("metric", ["skewness", "orthogonality", "aspect_ratio"])
def test_worst_cells_rank_real_ids_with_deterministic_ties(mixed_volume, metric):
    folder, arrays = mixed_volume
    result = inspection.inspect_mesh(folder, "worst", metric=metric, n=2)
    expected = [40, 30] if metric == "aspect_ratio" else [30, 40]
    assert [item["cell_id"] for item in result["details"]] == expected
    assert result["selected_count"] == 2
    data = face_data(result)
    assert set(data["cell_ids"]) == set(expected)
    lookup = dict(zip([40, 10, 30, 20], arrays[metric]))
    assert data["cell_values"] == pytest.approx([lookup[i] for i in data["cell_ids"]])
    # The returned dictionary is usable by the JSON transport without ndarray coercion.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("metric,threshold,expected", [
    ("skewness", .9, {30, 40}), ("orthogonality", .2, {30, 40}),
    ("aspect_ratio", 10, {30, 40}), ("skewness", 1, set()),
    ("orthogonality", 0, set()), ("aspect_ratio", 101, set())])
def test_threshold_direction_equality_and_empty_selection(mixed_volume, metric, threshold, expected):
    folder, _ = mixed_volume
    result = inspection.inspect_mesh(folder, "threshold", metric=metric, threshold=threshold)
    assert result["selected_count"] == len(expected)
    assert set(face_data(result)["cell_ids"]) == expected
    if not expected:
        assert "No cells match" in result["note"]


def test_layers_follow_connectivity_not_cell_order_or_cartesian_planes(mixed_volume):
    folder, _ = mixed_volume
    info = inspection.inspection_info(folder)
    assert info["total_count"] == 4
    assert info["layer_count"] == 2
    assert info["span_stations"] == [0, .2, 1]
    first = inspection.inspect_mesh(folder, "layer", layer=1)
    second = inspection.inspect_mesh(folder, "layer", layer=2)
    assert set(face_data(first)["cell_ids"]) == {10, 30}
    assert set(face_data(second)["cell_ids"]) == {20, 40}
    assert first["span_range"] == [0, .2]
    assert second["span_range"] == [.2, 1]
    assert first["layer"] == 1 and second["layer"] == 2


def test_color_directions_and_aspect_scale_remain_pinned_to_whole_grid(mixed_volume):
    folder, _ = mixed_volume
    skew_result = inspection.inspect_mesh(folder, "worst", n=4)
    assert skew_result["color_range"] == [0, 1]
    skew = face_data(skew_result)
    assert skew["color_values"] == pytest.approx(skew["cell_values"])
    orth_result = inspection.inspect_mesh(folder, "worst", metric="orthogonality", n=4)
    assert orth_result["color_range"] == [0, 1]
    orth = face_data(orth_result)
    assert orth["color_values"] == pytest.approx(1 - np.array(orth["cell_values"]))
    aspect = inspection.inspect_mesh(folder, "layer", metric="aspect_ratio", layer=1)
    assert aspect["color_range"] == [1, 200]
    assert aspect["logarithmic"]
    data = face_data(aspect)
    assert data["color_values"] == pytest.approx(np.log(data["cell_values"]) / np.log(200))
    worst = inspection.inspect_mesh(folder, "worst", metric="aspect_ratio", n=1)
    assert worst["color_range"] == aspect["color_range"]
    assert face_data(worst)["cell_values"] == pytest.approx([100] * len(face_data(worst)["cell_ids"]))
    assert max(face_data(worst)["color_values"]) == pytest.approx(np.log(100) / np.log(200))
    explicit = face_data(inspection.inspect_mesh(folder, "layer", metric="aspect_ratio",
                                                layer=1, aspect_max=10))
    assert explicit["color_values"] == pytest.approx(np.log(explicit["cell_values"]) / np.log(10))


@pytest.mark.parametrize("maximum,expected", [
    (1, 2), (1.0001, 2), (2, 5), (4.99, 5), (5, 10), (10, 20),
    (37, 50), (50, 100), (99.9, 100), (100, 200), (1000, 2000),
    (np.nextafter(100., 0), 100), (np.nextafter(100., np.inf), 200),
])
def test_automatic_aspect_color_ceiling_is_round_and_strictly_above_maximum(maximum, expected):
    ceiling = inspection._aspect_color_ceiling(maximum)
    assert ceiling == expected
    assert ceiling > maximum


def test_unrepresentable_rounded_aspect_limit_fails_explicitly():
    with pytest.raises(ValueError, match="finite rounded color-scale maximum"):
        inspection._aspect_color_ceiling(np.finfo(float).max)


def test_legacy_slice_and_default_worst_remain_available_without_span_summary(mixed_volume):
    folder, _ = mixed_volume
    (folder / "mesh_summary.json").unlink()
    assert inspection.inspect_mesh(folder, "off") is None
    assert inspection.inspect_mesh(folder, "worst")["selected_count"] == 4
    sliced = inspection.inspect_mesh(folder, "slice", "Z", .1)
    assert set(face_data(sliced)["cell_ids"]) == {10, 30}
    assert "Physical Z" in sliced["note"]
    assert inspection.inspection_info(folder)["layer_count"] == 0
    with pytest.raises(ValueError, match="Span layers unavailable"):
        inspection.inspect_mesh(folder, "layer")


@pytest.mark.parametrize("options", [
    {"mode": "worst", "n": 0}, {"mode": "worst", "n": 1.5},
    {"mode": "worst", "n": True}, {"mode": "worst", "n": float("inf")},
    {"mode": "threshold", "threshold": float("nan")},
    {"mode": "threshold", "threshold": 1.1},
    {"mode": "threshold", "threshold": .9, "metric": "aspect_ratio"},
    {"mode": "layer", "layer": 0}, {"mode": "layer", "layer": 3},
    {"mode": "slice", "fraction": -1}, {"mode": "slice", "axis": "R"},
    {"mode": "worst", "metric": "unknown"},
    {"mode": "worst", "metric": "aspect_ratio", "aspect_max": 1},
    {"mode": "worst", "metric": "aspect_ratio", "aspect_max": float("inf")},
])
def test_invalid_controls_fail_explicitly(mixed_volume, options):
    with pytest.raises(ValueError):
        inspection.inspect_mesh(mixed_volume[0], **options)


def test_invalid_saved_values_and_span_topology_are_not_silently_rendered(mixed_volume):
    folder, arrays = mixed_volume
    arrays["skewness"][0] = np.nan
    np.savez_compressed(folder / "volume_data.npz", **arrays)
    with pytest.raises(ValueError, match="skewness"):
        inspection.inspection_info(folder)
    arrays["skewness"][0] = .9
    arrays["cells_5"][0, 0] = 0  # Cell now spans three saved stations.
    np.savez_compressed(folder / "volume_data.npz", **arrays)
    info = inspection.inspection_info(folder)
    assert info["layer_count"] == 0
    assert "consecutive" in info["layer_error"]
    assert inspection.inspect_mesh(folder, "worst", n=1)["selected_count"] == 1
    with pytest.raises(ValueError, match="consecutive"):
        inspection.inspect_mesh(folder, "layer")


def test_render_limit_reports_true_count_without_sampling(mixed_volume, monkeypatch):
    monkeypatch.setattr(inspection, "RENDER_CELL_LIMIT", 1)
    result = inspection.inspect_mesh(mixed_volume[0], "worst", n=3)
    assert result["selected_count"] == 3
    assert result["render_limited"]
    assert result["render_limit"] == 1
    assert not face_data(result)["points"]
    assert len(result["details"]) == 3
    assert "no cells were sampled" in result["note"]


def test_cache_is_bounded_and_invalidates_overwritten_artifacts(mixed_volume, monkeypatch):
    folder, arrays = mixed_volume
    calls = []
    original = inspection.np.load
    def tracked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(inspection.np, "load", tracked)
    first = inspection.inspect_mesh(folder, "worst", n=1)
    assert inspection.inspect_mesh(folder, "worst", n=1) is first
    inspection.inspect_mesh(folder, "worst", n=2)
    inspection.inspection_info(folder)
    assert len(calls) == 1
    copy = inspection.load_grid(folder)
    copy.cell_data["skewness"][:] = 0
    assert inspection.inspect_mesh(folder, "worst", n=1)["details"][0]["value"] == .9
    arrays["skewness"][1] = 1
    path = folder / "volume_data.npz"
    previous = path.stat().st_mtime_ns
    np.savez_compressed(path, **arrays)
    os.utime(path, ns=(previous + 1_000_000, previous + 1_000_000))
    changed = inspection.inspect_mesh(folder, "worst", n=1)
    assert changed["details"] == [{"cell_id": 10, "value": 1}]
    assert len(calls) == 2
    assert inspection._load_artifact.cache_info().maxsize == 2
    assert inspection._inspect_cached.cache_info().maxsize == 2


@pytest.mark.parametrize("geometry", ["axial", "radial", "curved", "linear"])
def test_actual_generated_mesh_layers_cover_all_original_cells(small_design, tmp_path, geometry):
    from turbo_moc.configuration.row import read_configuration
    from turbo_moc.geometry import blade_row
    from turbo_moc.meshing import mesh_3d

    nozzle, config = small_design
    definition = None
    if geometry == "radial":
        config["meridional_channel"] = read_configuration(
            "examples/apps/cases/radial_stator.yaml")["meridional_channel"]
    elif geometry == "linear":
        config["meridional_channel"] = {"type": "linear", "meridional_chord": 20., "blade_height": 10.}
    elif geometry == "curved":
        knots = [0, 0, 0, 0, 1, 1, 1, 1]
        definition = blade_row.GeometryDefinition(
            BSpline(knots, [[100, 0], [100, 8], [101, 15], [102, 20]], 3),
            BSpline(knots, [[110, 0], [110, 8], [111, 16], [112, 22]], 3), 12)
    design = blade_row.build_design(nozzle, config, definition=definition)
    reference = mesh_3d.reference_mesh(design, config, tmp_path)
    section, design = mesh_3d.load_section(tmp_path)
    mesh_3d.volume_mesh(section, design, config, tmp_path)
    with np.load(tmp_path / "volume_data.npz") as saved:
        all_ids = set(np.concatenate([saved[k] for k in ("tags_5", "tags_6") if k in saved]))
    info = inspection.inspection_info(tmp_path)
    assert info["layer_count"] == 3
    seen = set()
    for layer in range(1, 4):
        result = inspection.inspect_mesh(tmp_path, "layer", layer=layer)
        ids = set(face_data(result)["cell_ids"])
        assert len(ids) == result["selected_count"] == reference["section_cells"]
        assert not ids & seen
        seen.update(ids)
    assert seen == all_ids
    report = json.loads((tmp_path / "mesh_report.json").read_text())
    for metric in inspection.METRIC_LABELS:
        result = inspection.inspect_mesh(tmp_path, "worst", metric=metric)
        assert result["details"][0]["value"] == pytest.approx(report["worst_cells"][metric][0]["value"])
