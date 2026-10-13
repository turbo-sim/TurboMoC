"""Inspection controls use immutable artifacts and never launch numerical jobs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from turbo_moc.app import application, inspection_panel as panel
from app_helpers import components


@pytest.fixture
def app():
    return application.create_app()


def request(**options):
    return {"key": "selection-1", "row_id": "rotor_1", "label": "Rotor", "job": "b" * 32,
        "reference": {"session": "a" * 32, "job": "b" * 32}, "status": "Outdated", "context": True,
        "options": {"mode": "worst", "metric": "skewness", "n": 12, **options}}


def result(metric="skewness"):
    return {"surfaces": {"inspection": {"points": [], "polys": []}}, "mode": "worst", "note": "Selection note",
        "metric": metric, "metric_label": panel.METRICS[metric], "definition": "Original volume quality",
        "color_range": [1, 100] if metric == "aspect_ratio" else [0, 1], "logarithmic": metric == "aspect_ratio",
        "selected_count": 12, "total_count": 1000, "details": [{"cell_id": 17, "value": .91}],
        "details_truncated": False}


def callback(app, name):
    return next(item["callback"].__wrapped__ for key, item in app.callback_map.items() if name in key)


def test_controls_are_available_without_modifying_configuration(app):
    layout = components(app.server.test_client().get("/_dash-layout").get_json())
    assert all("mesh-inspection-" + name in layout for name in ("settings", "request", "result", "summary"))
    request_callback = app.callback_map["mesh-inspection-request.data"]
    assert {item["id"] for item in request_callback["inputs"]} == {
        "mesh-inspection-settings", "mesh-display-row", "volume-data", "plotted-rows"}
    extract_callback = app.callback_map["mesh-inspection-result.data"]
    assert extract_callback["inputs"] == [{"id": "mesh-inspection-request", "property": "data"}]
    controls = str(layout["mesh-display-editors"])
    assert "Display mode" in controls and "Boundary surfaces" in controls
    assert "Mesh quality filter" in controls and "Spanwise layer" in controls
    assert "inspection-tab" not in controls and "inspection-aspect_max" not in controls
    for pane in ("geometry", "mesh"):
        assert f"{pane}-configure-display" not in layout
        assert f"{pane}-close-display" not in layout


def test_inspection_preferences_are_independent_per_row(app):
    saved = {"stator_1": deepcopy(panel.DEFAULTS), "rotor_1": deepcopy(panel.DEFAULTS)}
    original = deepcopy(saved)
    names = list(panel.DEFAULTS)
    values = [["filter" if name == "mode" else 42 if name == "n" else saved["rotor_1"][name]] for name in names]
    ids = [[panel.identifier(name, "rotor_1")] for name in names]
    changed = app.callback_map["mesh-inspection-settings.data"]["callback"].__wrapped__(*values, *ids, saved)
    assert changed["rotor_1"]["mode"] == "filter" and changed["rotor_1"]["n"] == 42
    assert changed["stator_1"] == saved["stator_1"] and saved == original


def test_saved_outdated_mesh_can_be_inspected_through_dash(app, monkeypatch, tmp_path):
    app.server.config["TURBO_MOC_OUTPUT_DIR"] = tmp_path
    payload = result()
    before = deepcopy(payload)
    calls = []
    def inspect(directory, **options):
        calls.append((directory, options))
        return payload
    monkeypatch.setattr(panel.inspection, "inspect_mesh", inspect)
    monkeypatch.setattr(panel.flow, "start_job", lambda *args, **kwargs: pytest.fail("Inspection launched a job"))
    selected = request()
    http = app.server.test_client().post("/_dash-update-component", json={
        "output": "mesh-inspection-result.data", "outputs": {"id": "mesh-inspection-result", "property": "data"},
        "inputs": [{"id": "mesh-inspection-request", "property": "data", "value": selected}],
        "state": [], "changedPropIds": ["mesh-inspection-request.data"]})
    assert http.status_code == 200, http.get_data(as_text=True)
    response = http.get_json()["response"]["mesh-inspection-result"]["data"]
    assert calls == [(tmp_path / ("a" * 32) / ("b" * 32), selected["options"])]
    assert response["status"] == "Outdated" and response["key"] == selected["key"]
    assert response["job"] == selected["job"] and response["row_id"] == "rotor_1"
    assert payload == before, "Adding app metadata must not mutate the shared extraction cache"


def test_unavailable_hidden_and_invalid_requests_clear_previous_results(monkeypatch):
    monkeypatch.setattr(panel.inspection, "inspect_mesh", lambda *args, **kwargs: pytest.fail("Unexpected extraction"))
    assert panel.extract(None) is None
    hidden = {**request(), "error": "The inspected row is hidden."}
    assert panel.extract(hidden)["error"] == hidden["error"]
    invalid = request()
    invalid["reference"]["session"] = "../outside"
    assert "Invalid job reference" in panel.extract(invalid)["error"]
    assert "Choose a quality filter" in panel.extract(request(mode="unsupported"))["error"]


def test_extraction_errors_are_visible_and_keep_request_identity(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Span layer must be between 1 and 6.")
    monkeypatch.setattr(panel.inspection, "inspect_mesh", fail)
    response = panel.extract(request(mode="layer", layer=7))
    assert response["key"] == "selection-1" and response["row_id"] == "rotor_1"
    assert "Span layer must be" in response["note"]
    assert "surfaces" not in response


@pytest.mark.parametrize("metric,expected", [("skewness", ["0", "0.5", "1"]),
    ("orthogonality", ["1", "0.5", "0"]), ("aspect_ratio", ["1", "10", "100"])])
def test_legend_uses_fixed_metric_direction_and_logarithmic_ticks(metric, expected):
    payload = {**result(metric), **{key: value for key, value in request().items() if key != "options"}}
    children = panel.summary(payload, request())
    legend = children[0].children[1]
    assert [item.children for item in legend.children[2].children] == expected
    text = str(children)
    assert "Outdated mesh" in text and "12 of 1,000 volume cells selected" in text
    assert "Metric definition and selected cell IDs" not in text and "Original cell ID" not in text
    assert "Updating mesh inspection" in str(panel.summary({**payload, "key": "old"}, request()))
    assert panel.summary(payload, None) == []


def test_layer_controls_use_saved_nonuniform_stations_and_clamp(app, monkeypatch):
    change = callback(app, '"type":"inspection-layer"}.max')
    data = {"rows": [{"id": "rotor_1", "span_stations": [0, .1, .4, 1]}]}
    item = panel.identifier("layer", "rotor_1")
    monkeypatch.setattr(panel, "ctx", SimpleNamespace(triggered_id=panel.identifier("next", "rotor_1")))
    maximum, value, marks, caption, previous_disabled, next_disabled, slider_disabled = change(data, 1, 0, 1, item)
    assert maximum == 3 and value == 2 and "0.1–0.4" in caption
    assert not previous_disabled and not next_disabled and not slider_disabled
    monkeypatch.setattr(panel, "ctx", SimpleNamespace(triggered_id="volume-data"))
    assert change(data, 99, 0, 1, item)[1] == 3
    assert change(data, 99, 0, 1, item)[-2] is True
    missing = change(None, 2, 0, 1, item)
    assert "Generate a 3D mesh" in missing[3]
    assert missing[1] == 2 and missing[-3:] == (True, True, True)
    assert change(data, missing[1], 0, 1, item)[1] == 2


def test_mode_controls_show_only_relevant_fields(app):
    fields = callback(app, '"type":"inspection-active-fields"')
    result = fields("layer", "worst", "aspect_ratio")
    assert result[:4] == ({"display": "none"}, {}, {"display": "none"}, {})
    assert result[6:] == ("Edge aspect ratio ≥", 1, None)
    result = fields("filter", "threshold", "orthogonality")
    assert result[:6] == ({"display": "none"}, {}, {}, {"display": "none"}, {"display": "none"}, {})
    assert result[6:] == ("Orthogonality ≤", 0, 1)
    assert fields("surface", "worst", "skewness")[:2] == ({}, {"display": "none"})
