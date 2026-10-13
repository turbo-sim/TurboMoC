/* Inspection requests are display state; responses belong to one saved mesh
 * and one selection. Never draw a late response over newer controls. */
window.turbomocInspection = {
  request: function(preferences, rowId, data, shown) {
    const settings = (preferences || {})[rowId] || {};
    if (!settings.mode || settings.mode === "surface") return null;
    const row = ((data || {}).rows || []).find(item => item.id === rowId);
    const options = {mode: settings.mode === "filter" ? settings.method : "layer",
      metric: settings.metric || "skewness"};
    if (options.mode === "worst") options.n = settings.n;
    if (options.mode === "threshold") options.threshold = settings.threshold;
    if (options.mode === "layer") options.layer = settings.layer;
    const request = {row_id: rowId, job: row && row.job,
      reference: row && {session: row.session, job: row.job},
      label: row ? row.label : rowId, status: row && row.status,
      context: (settings.context || []).some(value => value === "surfaces" || value === "blade"), options};
    if (!row) request.error = "Generate a 3D mesh for this row to inspect its cells.";
    else if (shown != null && !shown.includes(rowId))
      request.error = "The inspected row is hidden. Select it under Show to inspect its cells.";
    request.key = JSON.stringify(request);
    return request;
  },
  active: function(result, request) {
    if (!request || request.error) return null;
    if (result && result.key === request.key && !result.error) return result;
    // Hide the previous subset immediately while retaining the saved mesh's
    // camera and (optionally) its selected boundary surfaces as context.
    return {row_id: request.row_id, job: request.job, key: request.key + "|pending",
      mode: "pending", context: request.context,
      surfaces: {inspection: {points: [], polys: [], cell_values: [], cell_ids: [], color_values: []}},
      note: result && result.key === request.key ? result.note : "Updating mesh inspection…"};
  }
};
