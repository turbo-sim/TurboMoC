"""Labeled engineering plots; display controls never change numerical artifacts."""
import numpy as np
import plotly.graph_objects as go
from ...geometry.geometry_3d import section_data, meridional_grid

BLADE_FILL = "#b8bec2"
EDGE = "#171c1f"
MESH_EDGE = "orange"
APP_FONT = dict(family="Segoe UI, Arial, sans-serif", color="#233c46", size=12)


def figure(title, x, y):
    # Expandable groups provide visible titles; retain this identity for zoom.
    result = go.Figure()
    result.update_layout(xaxis_title=x, yaxis_title=y,
        template="plotly_white", margin=dict(l=55, r=20, t=48, b=45),
        font=APP_FONT, uirevision=title, showlegend=False)
    axes = dict(showline=True, mirror="allticks", linecolor="#526d79", linewidth=1,
        ticks="inside", ticklen=5, tickwidth=1, tickcolor="#526d79",
        tickformat=".2f", exponentformat="none", showexponent="none",
        gridcolor="#e4ebee", zeroline=False, title_font={**APP_FONT, "size": 13})
    result.update_xaxes(**axes)
    result.update_yaxes(**axes)
    result.update_yaxes(scaleanchor="x", scaleratio=1)
    return result


def _section_axes(design, *, mesh=False):
    """Keep the numerical chart orientation explicit in both section viewers."""
    if design.meridional_channel.is_linear:
        title = "Physical linear-cascade mesh" if mesh else "Linear-cascade geometry"
        return title + " · η = 0.5", "Pitchwise distance [mm]", "−Streamwise distance [mm]"
    title = "Conformal reference mesh" if mesh else "Conformal blade-to-blade geometry"
    return title + " · η = 0.5", "θ [rad, clockwise]", "−m′ [dimensionless]"


def section_figure(design, *, offset=None, label=None):
    result = figure(*_section_axes(design))
    data = section_data(design)
    # The passage reference defines periodic walls; it is not a blade camberline.
    # Keep it out of the default plot to avoid implying a physical blade feature.
    for name in ("periodic_left", "periodic_right", "inlet", "outlet", "blade"):
        points = np.asarray(data[name])
        if offset is not None:
            points = points + offset
        result.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="lines",
            name=f"{label}: {name.replace('_', ' ')}" if label else name.replace("_", " "),
            legendgroup=label, line=dict(
                width=1.5 if name == "blade" else 1,
                color=EDGE if name == "blade" else "#72858e"),
            fill="toself" if name == "blade" else None, fillcolor=BLADE_FILL))
    return result


def section_mesh_figure(section, design, edge_width=0.35, *, offset=None, label=None):
    result = figure(*_section_axes(design, mesh=True))
    mapping = design.meridional_channel
    xy = section["xy"] if mapping.is_linear else section["xy"] / mapping.reference_radius
    if offset is not None:
        xy = xy + offset
    # Unique edges prevent drawing every shared interior edge twice.
    edges = np.concatenate([np.stack((c, np.roll(c, -1, axis=1)), axis=-1).reshape(-1, 2)
                            for c in section["cells"].values()])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    segments = np.full((len(edges), 3, 2), np.nan)
    segments[:, :2] = xy[edges]
    result.add_trace(go.Scattergl(x=segments[:, :, 0].ravel(), y=segments[:, :, 1].ravel(),
        mode="lines", line=dict(color=MESH_EDGE, width=edge_width),
        name=f"{label}: cell edges" if label else "cell edges", legendgroup=label))
    contour = np.asarray(section_data(design)["blade"])
    if offset is not None:
        contour = contour + offset
    result.add_trace(go.Scatter(x=contour[:, 0], y=contour[:, 1], mode="lines",
        line=dict(color=EDGE, width=1.2), fill="toself", fillcolor=BLADE_FILL,
        name=f"{label}: blade" if label else "blade", legendgroup=label))
    return result


def section_offsets(project, mappings=None):
    """Put row-local charts into one midspan m'-theta chart for display.

    The first active row's leading edge defines m'=0, even when hidden. Across
    rows, accumulate integral(ds/r) along their midspan curves and across the
    straight physical interspace. For a bridge of length L with linear radius,
    the integral is L*log(r2/r1)/(r2-r1), or L/r1 at constant radius. This covers
    axial, radial inflow and radial outflow without arbitrary plot offsets.
    Angular phase is counterclockwise in XYZ and clockwise in this chart.
    Numerical reference meshes and exported coordinates remain untouched.
    """
    from ... import configuration_project as projects
    from ...geometry.meridional_channel import MeridionalChannel, geometry_from_machine
    rows = projects.active_rows(project)
    if len(rows) == 1:
        return {rows[0]["id"]: np.zeros(2)}
    offsets, previous_end, previous_chord, origin = {}, None, 0.0, 0.0
    for row in rows:
        mapping = (mappings or {}).get(row["id"])
        if mapping is None:
            mapping = MeridionalChannel(geometry_from_machine(row["geometry"]["meridional_channel"]))
        placement = projects.assembly_placement(project, row["id"])
        ends = mapping.meridional_points(np.array([0., 1.]), .5).copy()
        ends[:, 1] += _mapping_offset(mapping, placement)
        if previous_end is not None:
            length = np.linalg.norm(ends[0] - previous_end)
            r1, r2 = previous_end[0], ends[0, 0]
            dr = r2 - r1
            bridge = length / r1 if abs(dr) <= 1e-12 * r1 else length * np.log1p(dr / r1) / dr
            origin += previous_chord + bridge
        offsets[row["id"]] = np.array([-np.deg2rad(placement.get("angular_phase_deg", 0)), -origin])
        previous_end, previous_chord = ends[1], mapping.prime_chord(.5)
    return offsets


def assembly_section_figure(project, views, *, mesh=False):
    """Plot checked rows together; return per-row mesh trace indices for patches.

    Entries contain (id, label, Design, section-or-None, edge width). A single
    row keeps its original axes; multirow plots share the assembly chart.
    """
    views = list(views)
    if not views:
        result = section_mesh_placeholder("Blade-to-blade mesh") if mesh else figure(
            "Blade-to-blade geometry", "θ [rad, clockwise]", "−m′ [dimensionless]")
        return result, {}
    offsets = section_offsets(project, {row_id: design.meridional_channel
        for row_id, _, design, _, _ in views})
    result, indices = None, {}
    for row_id, label, design, section, width in views:
        local = (section_mesh_figure(section, design, width, offset=offsets[row_id], label=label) if mesh else
            section_figure(design, offset=offsets[row_id], label=label))
        if result is None:
            result = figure(*_section_axes(design, mesh=mesh))
        if mesh:
            indices[row_id] = len(result.data)
        result.add_traces(local.data)
    return result, indices


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


def meridional_offset(design, placement):
    """Radial rows are positioned by radius; other rows translate streamwise."""
    return _mapping_offset(design.meridional_channel, placement)


def _mapping_offset(mapping, placement):
    radial = not mapping.is_linear and all(
        np.allclose(curve.c[:, 1], curve.c[0, 1], rtol=0, atol=1e-12)
        for curve in (mapping.definition.hub, mapping.definition.shroud))
    return 0.0 if radial else float((placement or {}).get("streamwise_offset_mm", 0))


def _flow_domain_extensions(result, design, *, label="", offset=0):
    """Map actual inlet/outlet caps and their endwall extensions into (r,z).

    Use the same span-dependent sections as the mesh sweep, including shared
    physical interfaces. Sampling these curves never generates a mesh or
    repeats the reference blade-placement optimisation.
    """
    mapping = design.meridional_channel
    height_max = 0.0
    for name, blade_station in (("inlet", 0.0), ("outlet", 1.0)):
        cap, walls = [], []
        for eta in np.linspace(0, 1, 17):
            _, curves = design.section(float(eta))
            q = curves[name][0]
            station = (-q[1] / mapping.reference_chord if mapping.is_linear else
                float(mapping.parameter_from_prime(-q[1] / mapping.reference_radius, float(eta))))
            cap.append(mapping.meridional_points(station, float(eta)))
            if eta in (0, 1):
                walls.append(mapping.meridional_points(np.linspace(blade_station, station, 25), float(eta)))
        # An open U joins the two blade edges to the domain cap. Leaving the
        # blade-side edge open keeps the solid blade outline visually distinct.
        outline = np.vstack((walls[0], np.asarray(cap)[1:], walls[1][-2::-1]))
        result.add_trace(go.Scatter(x=outline[:, 1] + offset, y=outline[:, 0], mode="lines",
            name=f"{label + ' ' if label else ''}{name} flow domain",
            line=dict(color="#72858e", width=1.2, dash="dash")))
        height_max = max(height_max, float(outline[:, 0].max()))
    return height_max


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
    height_max = max(float(np.max(rz[..., 0])), float(np.max(outline[:, 0])),
                     _flow_domain_extensions(result, design))
    _zero_height_origin(result, height_max)
    return result, data


def assembly_meridional_figure(views, planned_rows=(), *, interfaces=()):
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
    envelopes = {}
    for label, design, placement in views:
        outline = _meridional_envelope(design)
        offset = meridional_offset(design, placement)
        result.add_trace(go.Scatter(
            x=outline[:, 1] + offset, y=outline[:, 0], mode="lines", name=label,
            line=dict(color=EDGE, width=1.5), fill="toself", fillcolor=BLADE_FILL,
            hovertemplate=f"{label}<br>%{{x:.3g}} mm, %{{y:.3g}} mm<extra></extra>",
        ))
        envelopes[label] = outline
        height_max = max(height_max, float(outline[:, 0].max()),
                         _flow_domain_extensions(result, design, label=label, offset=offset))
    # A pending row has channel dimensions, but no blade solution. Show only
    # the prescribed LE-to-TE extent, not a computed blade surface or domain.
    from ...geometry.meridional_channel import geometry_from_machine
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
            line=dict(color=EDGE, width=1.3), fill="toself", fillcolor="rgba(184,190,194,0.15)"))
        envelopes[label] = outline
        height_max = max(height_max, float(outline[:, 0].max()))
    for interface in interfaces:
        # Both channel walls must match before an interface record is emitted.
        # Its extent can therefore use either neighboring meridional envelope.
        upstream_id = interface["upstream_id"]
        label = {"stator_1": "Stator", "rotor_1": "Rotor"}.get(
            upstream_id, upstream_id.replace("_", " ").title())
        outline = envelopes.get(label)
        if outline is None:
            continue
        coordinate = interface["coordinate_mm"]
        axial = interface["geometry_type"] == "axial"
        x = [coordinate, coordinate] if axial else [outline[:, 1].min(), outline[:, 1].max()]
        y = [outline[:, 0].min(), outline[:, 0].max()] if axial else [coordinate, coordinate]
        result.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Shared row interface",
            line=dict(color="#376a7d", width=1.5, dash="dashdot"),
            hovertemplate=f"Shared interface<br>{'z' if axial else 'r'} = {coordinate:.4g} mm<extra></extra>"))
    _zero_height_origin(result, height_max)
    return result


def assembly_meridional_mesh_figure(entries):
    """Combine only generated row grids in physical assembly coordinates.

    Return trace indices by row so width-only edits can update SVG styles
    without transferring or rebuilding numerical mesh data.
    """
    result, indices, details, height_max = None, {}, {}, 0.0
    for row_id, label, design, numerics, section, placement, width in entries:
        local, data = meridional_figure(design, numerics, section, edge_width=width)
        if result is None:
            result = figure("Machine meridional grid projection", *_meridional_axes(design))
        offset = meridional_offset(design, placement)
        indices[row_id] = []
        for index, trace in enumerate(local.data):
            if 1 <= index <= numerics["span_layers"] + 2:
                indices[row_id].append(len(result.data))
            trace.x = np.asarray(trace.x) + offset
            trace.name = f"{label}: {trace.name}" if trace.name else label
            trace.legendgroup = row_id
            result.add_trace(trace)
        height_max = max(height_max, local.layout.yaxis.range[1] / 1.05)
        details[row_id] = data
    if result is None:
        result = figure("Generate a 2D mesh for a plotted row", "z [mm]", "r [mm]")
    _zero_height_origin(result, height_max)
    return result, indices, details


def histogram(name, metric):
    bins = np.asarray(metric["bin_edges"])
    result = go.Figure(go.Bar(x=(bins[:-1] + bins[1:]) / 2,
        y=metric["counts"], width=0.94 * np.diff(bins),
        marker=dict(color="#376a7d", line=dict(width=0))))
    result.update_layout(height=245,
        template="plotly_white", font=APP_FONT,
        margin=dict(l=48, r=15, t=15, b=42), yaxis_title="Cells", bargap=0)
    axes = dict(showline=True, mirror="allticks", linecolor="#526d79", linewidth=1,
        ticks="inside", ticklen=5, tickcolor="#526d79", gridcolor="#e4ebee", zeroline=False,
        title_font=APP_FONT)
    result.update_xaxes(**axes, tickformat=".2f")
    result.update_yaxes(**axes, tickformat=",.0f", rangemode="tozero")
    if name == "aspect_ratio":
        result.update_xaxes(type="log")
    return result

