"""Labeled engineering plots; display controls never change numerical artifacts."""
import numpy as np
import plotly.graph_objects as go
from ..geometry.geometry_3d import section_data, meridional_grid

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


def _section_axes(design, *, mesh=False):
    """Keep the numerical chart orientation explicit in both section viewers."""
    if design.meridional_channel.is_linear:
        title = "Physical linear-cascade mesh" if mesh else "Linear-cascade geometry"
        return title + " · η = 0.5", "Pitchwise distance [mm]", "−Streamwise distance [mm]"
    title = "Conformal reference mesh" if mesh else "Conformal blade-to-blade geometry"
    return title + " · η = 0.5", "θ [rad, clockwise]", "−m′ [dimensionless]"


def section_figure(design):
    result = figure(*_section_axes(design))
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
    result = figure(*_section_axes(design, mesh=True))
    mapping = design.meridional_channel
    xy = section["xy"] if mapping.is_linear else section["xy"] / mapping.reference_radius
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


def section_mesh_placeholder(title):
    """Clear the mesh without repeatedly replacing its browser GL canvases."""
    result = figure(title, "", "")
    # Keep the dense mesh renderer present when a row has no current artifact.
    # A completely empty figure makes Plotly discard its GL scene; restoring
    # the mesh then allocates new contexts that browsers may retain until GC.
    # This empty trace draws nothing and keeps the existing scene reusable.
    result.add_trace(go.Scattergl(x=[], y=[], mode="lines", showlegend=False))
    return result


def _meridional_axes(design):
    if design.meridional_channel.is_linear:
        return "Streamwise distance [mm]", "Span height [mm]"
    return "z [mm]", "r [mm]"


def _meridional_envelope(design):
    """LE-to-TE projection of the blade across span, not its physical thickness."""
    station = np.linspace(0, 1, 90)
    start = design.meridional_channel.meridional_points(station, 0)
    end = design.meridional_channel.meridional_points(station, 1)[::-1]
    return np.vstack((start, end, start[:1]))


def _zero_height_origin(result, height_max):
    # Constrain the plot domain so equal physical scales cannot move the lower
    # limit away from the radius/span origin when the figure is resized.
    result.update_yaxes(range=[0, max(1e-9, 1.05 * height_max)], autorange=False,
                       minallowed=0, constrain="domain")


def meridional_figure(design, numerics, section=None, *, geometry_only=False, edge_width=0.35):
    title = "Meridional geometry" if geometry_only else "Meridional grid projection"
    if design.meridional_channel.is_linear:
        title = "Streamwise/span geometry" if geometry_only else "Streamwise/span grid projection"
    result = figure(title, *_meridional_axes(design))
    data = meridional_grid(design, numerics, section)
    rz = np.asarray(data["rz"])
    # Shade the LE-to-TE envelope across the span. This is a projection, not
    # physical blade thickness or an independently solved meridional flow.
    outline = _meridional_envelope(design)
    result.add_trace(go.Scatter(x=outline[:, 1], y=outline[:, 0], mode="lines",
        line=dict(color=EDGE, width=1.5), fill="toself", fillcolor=BLADE_FILL,
        name="Blade meridional envelope"))
    rows = rz[[0, -1]] if geometry_only else rz
    wall_names = ("Span start", "Span end") if design.meridional_channel.is_linear else ("Hub", "Shroud")
    for i, row in enumerate(rows):
        label = wall_names[0] if i == 0 else wall_names[1] if i == len(rows) - 1 else "Span station"
        # Span projections have only a modest number of points. SVG lines stay
        # crisp at fractional widths and leave GPU contexts for the dense
        # blade-to-blade mesh and the two three-dimensional viewers.
        result.add_trace(go.Scatter(x=row[:, 1], y=row[:, 0], mode="lines",
            line=dict(color=EDGE if geometry_only else MESH_EDGE,
                      width=1 if geometry_only else edge_width),
            name=label,
            showlegend=i in (0, len(rows)-1)))
    if not geometry_only:
        lines = np.full((rz.shape[1], rz.shape[0] + 1, 2), np.nan)
        lines[:, :-1] = rz.transpose(1, 0, 2)
        result.add_trace(go.Scatter(x=lines[:, :, 1].ravel(), y=lines[:, :, 0].ravel(),
            mode="lines", line=dict(color=MESH_EDGE, width=edge_width), showlegend=False))
    height_max = max(float(np.max(rz[..., 0])), float(np.max(outline[:, 0])))
    _zero_height_origin(result, height_max)
    return result, data


def assembly_meridional_figure(views, planned_rows=()):
    """Show independently generated rows in a shared physical machine view.

    Each entry is (row label, Design, placement). Axial rows use an axial
    translation; linear rows use a streamwise translation. Radial rows are
    already placed by their radii, so a streamwise offset has no effect here.
    Angular phase is invisible in the meridional projection.
    """
    views, planned_rows = list(views), list(planned_rows)
    linear = (views and views[0][1].meridional_channel.is_linear) or (
        planned_rows and planned_rows[0][1]["type"] == "linear")
    axes = ("Streamwise distance [mm]", "Span height [mm]") if linear else ("z [mm]", "r [mm]")
    title = "Machine meridional geometry"
    if linear:
        title = "Linear-cascade streamwise/span geometry"
    result = figure(title, *axes)
    height_max = 0.0
    for label, design, placement in views:
        outline = _meridional_envelope(design)
        mapping = design.meridional_channel
        is_radial = (not mapping.is_linear and all(
            np.allclose(curve.c[:, 1], curve.c[0, 1], rtol=0, atol=1e-12)
            for curve in (mapping.definition.hub, mapping.definition.shroud)
        ))
        offset = 0.0 if is_radial else float((placement or {}).get("streamwise_offset_mm", 0.0))
        result.add_trace(go.Scatter(
            x=outline[:, 1] + offset, y=outline[:, 0], mode="lines", name=label,
            line=dict(color=EDGE, width=1.5), fill="toself", fillcolor=BLADE_FILL,
            hovertemplate=f"{label}<br>%{{x:.3g}} mm, %{{y:.3g}} mm<extra></extra>",
        ))
        height_max = max(height_max, float(outline[:, 0].max()))
    # A pending row has channel dimensions, but no blade solution. Show only
    # that prescribed envelope with a dashed outline; never fabricate a blade.
    from ..geometry.meridional_channel import geometry_from_machine
    for label, dimensions, placement in planned_rows:
        if dimensions["type"] == "linear":
            chord, height = dimensions["meridional_chord"], dimensions["blade_height"]
            outline = np.array([[0, 0], [0, chord], [height, chord], [height, 0], [0, 0]])
        else:
            definition = geometry_from_machine(dimensions)
            stations = np.linspace(0, 1, 90)
            hub, tip = definition.hub(stations), definition.shroud(stations)
            outline = np.vstack((hub, tip[::-1], hub[:1]))
        offset = float(placement.get("streamwise_offset_mm", 0))
        result.add_trace(go.Scatter(x=outline[:, 1] + offset, y=outline[:, 0], mode="lines",
            name=f"{label} meridional channel (planning)",
            line=dict(color=EDGE, width=1.3, dash="dash"), fill="toself", fillcolor="rgba(184,190,194,0.15)"))
        height_max = max(height_max, float(outline[:, 0].max()))
    _zero_height_origin(result, height_max)
    return result


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

