/* Requests belong to immutable artifacts and the current row selection. */
window.turbomocInspection = {
  request: function(preferences, selected, data, shown) {
    const ids = (Array.isArray(selected) ? selected : [selected]).filter(id => id !== "all" && id);
    if (!ids.length) return null;
    const settings = (preferences || {})[ids[0]] || {};
    if (!settings.mode || settings.mode === "surface") return null;
    const datasets = ((data || {}).rows || []);
    const source = datasets.find(row => row.id === ids[0]);
    const stations = (source || {}).span_stations || [];
    const layer = Math.max(1, Math.min(stations.length - 1, Number(settings.layer) || 1));
    const position = stations.length > 1 ? (stations[layer - 1] + stations[layer]) / 2 : null;
    const rows = ids.filter(id => shown == null || shown.includes(id)).map(rowId => {
      const row = datasets.find(item => item.id === rowId);
      const options = {mode: settings.mode === "filter" ? settings.method : "layer",
        metric: settings.metric || "skewness"};
      if (options.mode === "worst") options.n = settings.n;
      if (options.mode === "threshold") options.threshold = settings.threshold;
      if (options.mode === "layer") {
        const target = (row || {}).span_stations || [];
        options.layer = position == null || target.length < 2 ? settings.layer :
          Math.max(1, Math.min(target.length - 1, target.findIndex(value => value > position)));
      }
      const request = {row_id: rowId, job: row && row.job,
        reference: row && {session: row.session, job: row.job},
        label: row ? row.label : rowId, status: row && row.status,
        context: settings.context || ["blade", "inlet", "outlet", "periodic_left", "periodic_right", "hub", "shroud"], options};
      if (!row) request.error = "Generate a 3D mesh for " + request.label + " to inspect its cells.";
      request.key = JSON.stringify(request);
      return request;
    });
    if (!rows.length) return null;
    return {rows, key: JSON.stringify(rows)};
  },
  active: function(result, request) {
    if (!request) return null;
    if (request.rows) {
      const current = result && result.key === request.key ? result : null;
      return {key: request.key, rows: request.rows.map(row =>
        window.turbomocInspection.active(current && current.rows.find(item => item.row_id === row.row_id), row))
        .filter(Boolean)};
    }
    if (request.error) return null;
    if (result && result.key === request.key && !result.error) return result;
    // Clear stale cells immediately, keeping only the requested context.
    return {row_id: request.row_id, job: request.job, key: request.key + "|pending",
      mode: "pending", context: request.context,
      surfaces: {inspection: {points: [], polys: [], cell_values: [], cell_ids: [], color_values: []}},
      note: result && result.key === request.key ? result.note : "Updating mesh inspection..."};
  }
};
