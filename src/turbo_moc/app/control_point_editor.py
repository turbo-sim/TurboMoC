"""Bridge to the installed nurbspy viewer; all geometry evaluation stays in Python.

Only this module knows about nurbspy's prototype model/transport API. No viewer
code is copied, and no per-browser geometry is stored in shared Python objects.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from dash import Input, Output
from flask import jsonify, request, send_from_directory

from turbo_moc.geometry.stator_blade import (
    apply_control_point_displacements, apply_control_point_offsets, blade_chord,
    blade_self_intersects, editable_control_points,
    _with_control_points,
)
from turbo_moc.geometry.passage_2d import blade_contour

EDIT_FIELDS = ("control_point_offsets", "control_point_displacements")


@lru_cache(maxsize=1)
def nurbspy_api():
    """Use the declared PyPI dependency through normal Python imports."""
    from nurbspy.numpy import NurbsCurve
    from nurbspy.visualization import GeometryEditor
    import nurbspy
    return NurbsCurve, GeometryEditor, Path(nurbspy.__file__).parent / "web_assets"


def geometry_editor(blade, points=None, *, display=None):
    curve_type, editor_type, _ = nurbspy_api()
    points = np.asarray(blade["control_points"] if points is None else points, dtype=float)
    curve = curve_type(control_points=points.T, weights=np.ones(len(points)),
                       degree=3, knots=np.asarray(blade["knots"], dtype=float))
    return editor_type(curve, id="stator-cp", title="Stator blade — drag control points",
                       quality="fine", preview_quality="full",
                       display={"view": "xy", "axes": True, "grid": True,
                                "geometry_color": "#233f50", **(display or {})})


def initial_blade(session):
    blade = session["blade"]
    edits = session["edits"]
    if edits.get("control_point_displacements"):
        return apply_control_point_displacements(blade, edits["control_point_displacements"])
    return apply_control_point_offsets(blade, edits.get("control_point_offsets"))


def make_session(blade, inputs, identity):
    session = {"blade": blade, "identity": identity,
               "edits": {name: deepcopy(inputs[name]) for name in EDIT_FIELDS if name in inputs},
               "editable": editable_control_points(blade)}
    editor = geometry_editor(blade, initial_blade(session)["control_points"])
    session["scene"] = editor.evaluate_scene()
    context_display = {"control_points": False, "control_polygon": False, "geometry_color": "#9aa7ad"}
    baseline = geometry_editor(blade, display=context_display).evaluate_scene()
    curve_type, editor_type, _ = nurbspy_api()
    edge = np.array([blade["trailing_edge"]["x"], blade["trailing_edge"]["y"]])
    trailing_edge = editor_type(curve_type(control_points=edge, degree=1), id="stator-edge",
                                quality="fine", preview_quality="full",
                                display={"view": "xy", **context_display}).evaluate_scene()
    # Drawing-only reference vertices. The browser reuses nurbspy's line
    # primitives and the edited curve buffer for periodic copies; no extra
    # editors, sampling implementations, or control points are needed.
    session["references"] = {"baseline": baseline["points"], "trailing_edge": trailing_edge["points"]}
    session["pitch"] = float(blade["pitch"])
    return session


def use_design_passage(session, design):
    """Express the production midspan pitch in the native editing plane.

    Annular geometry holds its physical pitch/chord fixed when the fitted
    contour is rescaled. Linear cascades scale the natural pitch with the blade.
    Reuse the production mapping instead of reimplementing blade-count formulas.
    """
    mapping = design.meridional_channel
    if not mapping.is_linear:
        session["pitch_to_chord"] = float(mapping.pitch / mapping.reference_chord)
        session["pitch"] = passage_pitch(session, initial_blade(session))
    return session


def passage_pitch(session, blade):
    if "pitch_to_chord" in session:
        return float(session["pitch_to_chord"] * np.ptp(blade_contour(blade)[:, 1]))
    return float(blade["pitch"])


def points_from_model(session, model):
    """Apply the host's planar, locked-point, and fixed B-spline contract."""
    editor = geometry_editor(session["blade"])
    geometry = editor.geometry_from_state(model)
    points = np.asarray(geometry.P, dtype=float)
    if np.any(points[2]) or not np.array_equal(geometry.W, np.ones(points.shape[1])):
        raise ValueError("The stator must stay in XY with unit weights.")
    baseline = np.asarray(session["blade"]["control_points"], dtype=float)
    locked = sorted(set(range(len(baseline))) - set(editable_control_points(session["blade"])))
    if not np.array_equal(points[:2, locked].T, baseline[locked]):
        raise ValueError("The first two and last two control points are locked.")
    return points[:2].T


def edits_from_model(session, model):
    """Preserve no-op legacy cases; store only intentionally changed XY offsets."""
    points = points_from_model(session, model)
    starting_blade = initial_blade(session)
    original = np.asarray(starting_blade["control_points"], dtype=float)
    if np.array_equal(points, original):
        if blade_self_intersects(starting_blade):
            raise ValueError("The edited blade intersects itself; reduce the displacements before applying.")
        return deepcopy(session["edits"])
    baseline = np.asarray(session["blade"]["control_points"], dtype=float)
    scale = blade_chord(session["blade"])
    delta = (points - baseline) / scale
    # Only roundoff-sized differences become zero, never float32 display data.
    delta[np.abs(delta) < 1e-14] = 0.0
    vectors = {i: delta[i].tolist() for i in editable_control_points(session["blade"]) if np.any(delta[i])}
    for key, value in (session["edits"].get("control_point_displacements") or {}).items():
        index = int(key)
        if np.array_equal(points[index], original[index]):
            vectors[index] = deepcopy(value)
    edited = apply_control_point_displacements(session["blade"], vectors)
    if blade_self_intersects(edited):
        raise ValueError("The edited blade intersects itself; reduce the displacements before applying.")
    return {"control_point_displacements": vectors}


def register(app):
    """Serve nurbspy assets and a stateless, per-request model evaluator."""
    prefix = app.config.routes_pathname_prefix.rstrip("/")

    @app.server.route(prefix + "/_turbomoc_nurbspy/<path:filename>")
    def nurbspy_asset(filename):
        return send_from_directory(nurbspy_api()[2], filename)

    @app.server.route(prefix + "/_turbomoc_cp/evaluate", methods=["POST"])
    def evaluate():
        try:
            payload = request.get_json()
            session, packet = payload["session"], payload["packet"]
            editor = geometry_editor(session["blade"], initial_blade(session)["control_points"])
            model = editor.initial_state
            if packet.get("action") != "edit":
                raise ValueError("Expected an edit, including explicit baseline coordinates for Reset.")
            model.update(control_points=packet["control_points"], weights=packet["weights"])
            points = points_from_model(session, model)
            result = editor.render_packet(packet)
            # Use the same sampled production contour at both display qualities
            # so the passage spacing cannot jitter between preview/full meshes.
            blade = _with_control_points(session["blade"], points)
            if result["quality"] == "full" and blade_self_intersects(blade):
                raise ValueError("The edited blade intersects itself; the edit was cancelled.")
            response = jsonify(**result, pitch=passage_pitch(session, blade))
            response.headers["Cache-Control"] = "no-store"
            return response
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            return jsonify(error=str(exc)), 422

    app.clientside_callback(
        """async function(session, reset, close, style) {
            const trigger = dash_clientside.callback_context.triggered_id;
            const module = await import(MODULE);
            if (style && style.display === 'none') return module.close();
            return module.update(session, trigger, ENDPOINT, ASSETS);
        }""".replace("MODULE", repr(app.get_asset_url("control_points.mjs")))
              .replace("ENDPOINT", repr(app.get_relative_path("/_turbomoc_cp/evaluate")))
              .replace("ASSETS", repr(app.get_relative_path("/_turbomoc_nurbspy/"))),
        Output("cp-ready", "data"), Input("cp-session", "data"), Input("cp-reset", "n_clicks"),
        Input("cp-cancel", "n_clicks"), Input("cp-editor", "style"), prevent_initial_call=True,
    )
    app.clientside_callback(
        """async function(clicks) {
            if (!clicks) return dash_clientside.no_update;
            const module = await import(MODULE);
            return module.capture();
        }""".replace("MODULE", repr(app.get_asset_url("control_points.mjs"))),
        Output("cp-commit", "data"), Input("cp-apply", "n_clicks"), prevent_initial_call=True,
    )
