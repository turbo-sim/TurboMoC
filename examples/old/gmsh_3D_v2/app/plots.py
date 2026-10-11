"""Labeled engineering plots; display controls never change numerical artifacts."""
import numpy as np
import plotly.graph_objects as go
from .geometry_source import section_data, meridional_grid

BLADE_FILL = "#b8bec2"
EDGE = "#171c1f"
MESH_EDGE = "orange"


def figure(title, x, y):
    result = go.Figure()
    result.update_layout(title=title, xaxis_title=x, yaxis_title=y,
        template="plotly_white", margin=dict(l=55, r=20, t=55, b=65),
        uirevision=title, legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
    result.update_yaxes(scaleanchor="x", scaleratio=1)
    return result


def section_figure(design):
    result = figure("Blade-to-blade geometry · η = 0.5",
                    "θ [rad, clockwise]", "−m′ [dimensionless]")
    data = section_data(design)
    # The passage reference defines periodic walls; it is not a blade camberline.
    # Keep it out of the default plot to avoid implying a physical blade feature.
    for name in ("periodic_left", "periodic_right", "inlet", "outlet", "blade"):
        points = np.asarray(data[name])
        result.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="lines",
            name=name.replace("_", " "), line=dict(
                width=1.5 if name == "blade" else 1,
                color=EDGE if name == "blade" else "#72858e"),
            fill="toself" if name == "blade" else None, fillcolor=BLADE_FILL))
    return result


def section_mesh_figure(section, design, edge_width=0.35):
    result = figure("2D reference mesh · η = 0.5", "θ [rad, clockwise]", "−m′ [dimensionless]")
    xy = section["xy"] / design.channel.reference_radius
    # Unique edges prevent drawing every shared interior edge twice.
    edges = np.concatenate([np.stack((c, np.roll(c, -1, axis=1)), axis=-1).reshape(-1, 2)
                            for c in section["cells"].values()])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    segments = np.full((len(edges), 3, 2), np.nan)
    segments[:, :2] = xy[edges]
    result.add_trace(go.Scattergl(x=segments[:, :, 0].ravel(), y=segments[:, :, 1].ravel(),
        mode="lines", line=dict(color=MESH_EDGE, width=edge_width), name="cell edges"))
    contour = np.asarray(section_data(design)["blade"])
    result.add_trace(go.Scatter(x=contour[:, 0], y=contour[:, 1], mode="lines",
        line=dict(color=EDGE, width=1.2), fill="toself", fillcolor=BLADE_FILL, name="blade"))
    return result


def meridional_figure(design, numerics, section=None, *, geometry_only=False, edge_width=0.35):
    result = figure("Meridional geometry" if geometry_only else "Meridional grid projection",
                    "z [mm]", "r [mm]")
    data = meridional_grid(design, numerics, section)
    rz = np.asarray(data["rz"])
    # Shade the LE-to-TE envelope across the span. This is a projection, not
    # physical blade thickness or an independently solved meridional flow.
    station = np.linspace(0, 1, 90)
    hub = design.channel.meridional_points(station, 0)
    shroud = design.channel.meridional_points(station, 1)[::-1]
    outline = np.vstack((hub, shroud, hub[:1]))
    result.add_trace(go.Scatter(x=outline[:, 1], y=outline[:, 0], mode="lines",
        line=dict(color=EDGE, width=1.5), fill="toself", fillcolor=BLADE_FILL,
        name="Blade meridional envelope"))
    rows = rz[[0, -1]] if geometry_only else rz
    for i, row in enumerate(rows):
        result.add_trace(go.Scattergl(x=row[:, 1], y=row[:, 0], mode="lines",
            line=dict(color=EDGE if geometry_only else MESH_EDGE,
                      width=1 if geometry_only else edge_width),
            name="Hub" if i == 0 else "Shroud" if i == len(rows)-1 else "Span station",
            showlegend=i in (0, len(rows)-1)))
    if not geometry_only:
        lines = np.full((rz.shape[1], rz.shape[0] + 1, 2), np.nan)
        lines[:, :-1] = rz.transpose(1, 0, 2)
        result.add_trace(go.Scattergl(x=lines[:, :, 1].ravel(), y=lines[:, :, 0].ravel(),
            mode="lines", line=dict(color=MESH_EDGE, width=edge_width), showlegend=False))
    # Show the rotation axis in both views. Constrain the plot domain rather
    # than expanding this range when Plotly enforces equal physical scales.
    radius_max = max(float(np.max(rz[..., 0])), float(np.max(outline[:, 0])))
    result.update_yaxes(range=[0, 1.05 * radius_max], autorange=False,
                       minallowed=0, constrain="domain")
    return result, data


def histogram(name, metric):
    bins = np.asarray(metric["bin_edges"])
    result = go.Figure(go.Bar(x=(bins[:-1] + bins[1:]) / 2,
        y=metric["counts"], width=0.94 * np.diff(bins),
        marker=dict(color="#376a7d", line=dict(width=0))))
    result.update_layout(title=name.replace("_", " ").title(), height=225,
        template="plotly_white", margin=dict(l=45, r=15, t=40, b=35), yaxis_title="Cells")
    if name == "aspect_ratio":
        result.update_xaxes(type="log")
    return result

