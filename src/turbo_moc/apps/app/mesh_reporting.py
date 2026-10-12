"""Reports for saved passage meshes, independent of viewer selections."""
from functools import lru_cache

import numpy as np
from dash import dcc, html

from . import workflow as flow, plots

METRICS = ("skewness", "orthogonality", "aspect_ratio")
METRIC_LABELS = {"skewness": "Skewness", "orthogonality": "Orthogonality", "aspect_ratio": "Edge aspect ratio"}
SICN_LABEL = "Minimum signed inverse condition number (SICN)"
SICN_TIP = ("Gmsh's minimum signed inverse condition number measures cell shape quality. "
    "Values near zero indicate poor or degenerate shapes; negative values indicate inversion. "
    "Thin intentional inflation cells can have low positive values.")
TOTAL_TIP = ("One generated passage per row. Displayed blade copies do not affect statistics. "
    "Means, percentiles and histograms pool the actual cell samples.")
NODE_TIP = "Node counts are summed per row; nodes on shared interfaces are not merged."


def metric_table(metrics, definitions):
    headings = [html.Th("Metric"), *[html.Th(label, title=tip) for label, tip in (
        ("Min", "Minimum cell value"), ("Mean", "Mean over all included cells"),
        ("Max", "Maximum cell value"), ("P95", "95th percentile of the included cell values"))]]
    return html.Table([html.Thead(html.Tr(headings)), html.Tbody([
        html.Tr([html.Td(METRIC_LABELS[name], title=definitions[name]),
            *[html.Td(f"{metric[key]:.4g}") for key in ("min", "mean", "max", "p95")]])
        for name, metric in metrics.items()])], className="report-table")


def histogram_row(metrics, definitions):
    return html.Div([html.Div([
        html.H4(METRIC_LABELS[name], title=definitions[name], className="metric-title"),
        dcc.Graph(figure=plots.histogram(name, metrics[name]), config={"displayModeBar": False, "displaylogo": False}),
    ], className="metric-plot", title=definitions[name]) for name in METRICS], className="metric-plots")


@lru_cache(maxsize=8)
def metric_arrays(filename):
    # Job artifacts are immutable; cache just the three scalar arrays, not xyz
    # or connectivity. The absolute filename also distinguishes app outputs.
    with np.load(filename, allow_pickle=False) as data:
        return {name: data[name] for name in METRICS}


def mesh_status(project, row, registry):
    """Check inputs and the selected predecessor chain, including recomputes."""
    refs = (registry or {}).get(row["id"], {})
    volume = refs.get("volume")
    try:
        config = flow.resolved_configuration(project, row["id"], registry)
        keys = flow.keys(config, refs.get("nozzle"))
        if not all(flow.current(refs.get(stage), keys[stage])
                   for stage in ("nozzle", "reference", "volume")):
            return "Outdated"
        request = flow.read_json(flow.directory(volume) / "request.json")
        if request.get("source_revision", flow.SOURCE_REVISION) != flow.SOURCE_REVISION:
            return "Outdated"
        for stage in ("nozzle", "reference"):
            old, new = request.get("references", {}).get(stage), refs.get(stage)
            if [(old or {}).get(k) for k in ("session", "job")] != [
                    (new or {}).get(k) for k in ("session", "job")]:
                return "Outdated"
        return "Current"
    except (OSError, ValueError, RuntimeError, KeyError):
        return "Outdated"


def aggregate(entries):
    """Pool cell samples for exact means, percentiles and common histogram bins."""
    metrics = {}
    for name in METRICS:
        values = np.concatenate([entry["arrays"][name] for entry in entries])
        edges = (np.geomspace(1, max(1.01, float(values.max())), 121)
                 if name == "aspect_ratio" else np.linspace(0, 1, 121))
        counts, edges = np.histogram(values, bins=edges)
        metrics[name] = {"min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "p95": float(np.quantile(values, .95)),
            "samples": len(values), "counts": counts.tolist(), "bin_edges": edges.tolist()}
    return metrics


def total_panel(rows, entries, issues):
    available = {entry["id"]: entry for entry in entries}
    current = sum(entry["status"] == "Current" for entry in entries)
    result = [html.P(f"{len(entries)} of {len(rows)} row meshes available; {current} current.",
        title=TOTAL_TIP, className="report-availability")]
    if issues or current < len(entries):
        notes = [f"{label}: {message}" for label, message in issues]
        notes += [f"{entry['label']}: outdated mesh included; regenerate for the current settings."
                  for entry in entries if entry["status"] != "Current"]
        result.append(html.Div(" ".join(notes), className="message"))
    if not entries:
        return [*result, html.P("Generate a 3D mesh to see quality statistics.")]
    pooled = aggregate(entries)
    def values(entry):
        summary, arrays = entry["summary"], entry["arrays"]
        return [sum(summary["cells"].values()), summary["volume_nodes"],
            summary["quality"]["min_sicn"], arrays["skewness"].max(), arrays["orthogonality"].min(),
            arrays["aspect_ratio"].max(), int(np.count_nonzero(arrays["skewness"] > .8)),
            int(np.count_nonzero(arrays["orthogonality"] < .2))]
    by_id = {entry["id"]: values(entry) for entry in entries}
    individual = list(by_id.values())
    total = [sum(v[0] for v in individual), sum(v[1] for v in individual),
        min(v[2] for v in individual), pooled["skewness"]["max"],
        pooled["orthogonality"]["min"], pooled["aspect_ratio"]["max"],
        sum(v[6] for v in individual), sum(v[7] for v in individual)]
    headings = ["Metric", *[row["label"] for row in rows], "Combined"]
    labels = ["Volume cells", "Volume nodes", SICN_LABEL, "Maximum skewness",
        "Minimum orthogonality", "Maximum edge aspect ratio", "Cells with skewness > 0.8",
        "Cells with orthogonality < 0.2"]
    def formatted(value, index):
        return f"{int(value):,}" if index in (0, 1, 6, 7) else f"{value:.4g}"
    comparison = [html.Tr([html.Td("Status"), *[html.Td(
        available[row["id"]]["status"] if row["id"] in available else "Unavailable") for row in rows],
        html.Td("Complete" if current == len(rows) else "Partial / outdated")])]
    for index, label in enumerate(labels):
        comparison.append(html.Tr([html.Td(label, title=NODE_TIP if index == 1 else SICN_TIP if index == 2 else None), *[html.Td(
            formatted(by_id[row["id"]][index], index) if row["id"] in available else "—")
            for row in rows], html.Td(formatted(total[index], index))]))
    result.append(html.Table([html.Thead(html.Tr([html.Th(v, title=TOTAL_TIP if v == "Combined" else None)
        for v in headings])), html.Tbody(comparison)], className="report-table"))
    definitions = entries[0]["report"]["definitions"]
    result += [metric_table(pooled, definitions), histogram_row(pooled, definitions)]
    return result
