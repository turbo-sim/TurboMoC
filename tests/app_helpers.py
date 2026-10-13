"""Shared Dash request helpers; no test cases or production dependencies."""

def call(app, prefix, changed, values):
    key = next(key for key in app.callback_map if key.startswith(prefix))
    callback = app.callback_map[key]
    outputs = callback["output"]
    if not isinstance(outputs, list):
        outputs = [outputs]
    payload = {
        "output": key,
        "outputs": [{"id": o.component_id, "property": o.component_property} for o in outputs],
        "changedPropIds": [changed],
    }
    for field in ("inputs", "state"):
        payload[field] = [{**item, "value": values.get((item["id"], item["property"]),
                          [] if item["id"].startswith("{") else None)} for item in callback[field]]
    response = app.server.test_client().post("/_dash-update-component", json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"]


def components(value):
    """Index rendered Dash components without depending on their nesting."""
    result = {}
    if isinstance(value, dict):
        props = value.get("props", {})
        if isinstance(props.get("id"), str):
            result[props["id"]] = props
        for item in value.values():
            result.update(components(item))
    elif isinstance(value, list):
        for item in value:
            result.update(components(item))
    return result


def row_state(config, registry=None, row_id="stator_1"):
    from turbo_moc.configuration import project as projects
    from turbo_moc.app import workflow as flow

    project = projects.normalize_project(config)
    project["rows"][0]["id"] = row_id
    reference = (registry or {}).get(row_id, {}).get("nozzle")
    return {"project": project, "row_id": row_id, "role": "stator", "supported": True,
            "valid": True, "config": config, **flow.keys(config, reference)}
