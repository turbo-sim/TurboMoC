"""Matplotlib mesh figures and Gmsh display configuration."""

from pathlib import Path

import gmsh
import numpy as np

from turbo_moc.meshing.quality import unique_plot_segments
from turbo_moc.reporting import display_path, get_logger

logger = get_logger("gmsh.meshing")
MESH_ORANGE = (255, 140, 0)


def configure_mesh_display():
    """Use orange for both 2D element types, independent of saved GUI colors."""
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)
    for name in ("Triangles", "Quadrangles", "Lines"):
        gmsh.option.setColor(f"Mesh.Color.{name}", *MESH_ORANGE)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)


# Fixed publication styling; colors and view controls are configurable below.
MESH_LINEWIDTH = 0.12
BOUNDARY_LINEWIDTH = 0.9
MESH_FONT_SIZE = 11.0


def mesh_figure_options(settings):
    """Validate the small set of user controls for one exported mesh view."""
    from matplotlib.colors import is_color_like

    defaults = {
        "num_blades": 3,
        "annotate_boundaries": True,
        "mesh_color": "orange",
        "edge_color": "black",
        "domain_color": "lightgrey",
        "dpi": 300,
        "rasterize_mesh": True,
    }
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError("Unknown option or invalid mapping in mesh.matplotlib.")
    options = defaults | settings
    if type(options["num_blades"]) is not int or options["num_blades"] < 1:
        raise ValueError("mesh.matplotlib.num_blades must be a positive integer.")
    if type(options["dpi"]) is not int or options["dpi"] <= 0:
        raise ValueError("mesh.matplotlib.dpi must be a positive integer.")
    for key in ("annotate_boundaries", "rasterize_mesh"):
        if type(options[key]) is not bool:
            raise ValueError(f"mesh.matplotlib.{key} must be true or false.")
    for key in ("mesh_color", "edge_color", "domain_color"):
        if not is_color_like(options[key]):
            raise ValueError(f"mesh.matplotlib.{key} must be a Matplotlib color.")
    return options


def collect_mesh_plot_data(fluid_surface, boundaries=None):
    """Read actual cells and named 1D physical groups; omit unused CAD nodes."""
    kinds, _, arrays = gmsh.model.mesh.getElements(2, fluid_surface)
    if not len(kinds) or any(kind not in (2, 3) for kind in kinds):
        raise ValueError("Mesh images require first-order triangles/quadrilaterals.")
    used = np.unique(np.concatenate(arrays))
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    xyz = dict(zip(tags, np.asarray(coordinates).reshape(-1, 3)))
    local = {tag: index for index, tag in enumerate(used)}
    xy = np.array([xyz[tag][:2] for tag in used])
    cells, edges = [], []
    for kind, nodes in zip(kinds, arrays):
        group = np.array([local[tag] for tag in nodes], dtype=int).reshape(
            -1, 3 if kind == 2 else 4
        )
        cells.extend(group)
        edges.append(
            np.stack((group, np.roll(group, -1, axis=1)), axis=-1).reshape(-1, 2)
        )
    edges = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
    if boundaries is None:
        boundaries = {
            gmsh.model.getPhysicalName(
                dim, tag
            ): gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups(1)
        }
    boundary_edges = {}
    for name, curves in boundaries.items():
        lines = []
        for curve in curves:
            types, _, nodes = gmsh.model.mesh.getElements(1, curve)
            if any(kind != 1 for kind in types):
                raise ValueError("Mesh images require first-order boundary lines.")
            lines.extend(
                [local[tag] for tag in edge]
                for array in nodes
                for edge in array.reshape(-1, 2)
            )
        if lines:
            boundary_edges[name] = np.asarray(lines, dtype=int)
    # Physical boundaries use their own line color instead of a second mesh edge.
    physical = {
        tuple(sorted(edge)) for lines in boundary_edges.values() for edge in lines
    }
    edges = np.asarray(
        [edge for edge in edges if tuple(edge) not in physical], dtype=int
    ).reshape(-1, 2)
    return {"xy": xy, "cells": cells, "edges": edges, "boundaries": boundary_edges}




def annotate_mesh_boundaries(ax, data, options, shifts):
    """Label outer passage boundaries and a central blade for any blade count."""
    # A cascade labels each group once. Its periodic sides are the outermost
    # copies; inlet/outlet span all copies and the blade label uses the middle.
    count = len(shifts)
    nodes_per_copy = len(data["xy"])
    xy = np.vstack([data["xy"] + shift for shift in shifts])
    groups = {}
    for name, edges in data["boundaries"].items():
        copies = {
            "periodic_left": [0],
            "periodic_right": [count - 1],
            "blade": [count // 2],
        }.get(name, range(count))
        groups[name] = np.concatenate(
            [edges + copy * nodes_per_copy for copy in copies]
        )
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    width, height = hi - lo
    for name, edges in groups.items():
        midpoints = xy[edges].mean(axis=1)
        vertical_align = "center"
        if name in ("inlet", "outlet"):
            anchor = midpoints[np.argmin(abs(midpoints[:, 0] - midpoints[:, 0].mean()))]
            label = (
                anchor[0],
                anchor[1] + (0.045 if name == "inlet" else -0.045) * height,
            )
            align = "center"
            if count == 1:
                # Use the domain extent, rather than the local boundary height,
                # so text stays outside even when the inlet/outlet is curved.
                label = (
                    anchor[0],
                    hi[1] + 0.045 * height
                    if name == "inlet"
                    else lo[1] - 0.045 * height,
                )
                vertical_align = "bottom" if name == "inlet" else "top"
        elif name.startswith("periodic_"):
            left = name == "periodic_left"
            y = lo[1] + (0.64 if left else 0.34) * height
            anchor = midpoints[np.argmin(abs(midpoints[:, 1] - y))]
            label = (
                lo[0] - 0.06 * width if left else hi[0] + 0.06 * width,
                anchor[1] + 0.025 * height,
            )
            align = "right" if left else "left"
        elif name == "blade":
            # A short leader above the leading edge avoids crossing other blades.
            anchor = midpoints[np.argmax(midpoints[:, 1])]
            label = (anchor[0], anchor[1] + 0.13 * height)
            align = "center"
            if count == 1:
                # Point to the upper blade surface near the leading edge.
                # Place text locally above/right, in the open space beside the
                # curved passage, rather than beyond the entire domain width.
                blade_width = np.ptp(midpoints[:, 0])
                target_x = midpoints[:, 0].min() + 0.30 * blade_width
                distance = abs(midpoints[:, 0] - target_x)
                near_target = midpoints[
                    distance <= distance.min() + 0.02 * blade_width
                ]
                anchor = near_target[np.argmax(near_target[:, 1])]
                label = tuple(anchor + np.array([0.22 * width, 0.20 * height]))
                align = "left"
        else:
            continue
        ax.annotate(
            name,
            xy=anchor,
            xytext=label,
            ha=align,
            va=vertical_align,
            fontsize=MESH_FONT_SIZE * 0.85,
            color="black",
            bbox=(
                None
                if count == 1
                else {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
            ),
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "lw": 0.65,
                "shrinkA": 3,
                "shrinkB": 2,
            },
            zorder=5,
        )


def plot_mesh_figure(data, *, pitch, options):
    """Draw the selected number of pitch-translated blades in one boxed figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    blade_count = options["num_blades"]
    if blade_count > 1 and (pitch is None or not np.isfinite(pitch) or pitch <= 0):
        raise ValueError(
            "A positive blade pitch is required for a cascade mesh figure."
        )
    # Force full opacity even when a user supplies an RGBA color or hex alpha.
    mesh_color = to_rgba(options["mesh_color"], alpha=1)
    edge_color = to_rgba(options["edge_color"], alpha=1)
    domain_color = to_rgba(options["domain_color"], alpha=1)
    xy = data["xy"]
    offsets = (np.arange(blade_count) - (blade_count - 1) / 2) * (pitch or 0)
    shifts = [np.array([offset, 0.0]) for offset in offsets]
    fig = Figure(
        figsize=(8.5 if blade_count == 1 else 10.0, 5.5),
        layout="constrained",
        dpi=options["dpi"],
        facecolor="white",
    )
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_facecolor("white")
    ax.add_collection(
        PolyCollection(
            [xy[cell] + shift for shift in shifts for cell in data["cells"]],
            facecolors=domain_color,
            edgecolors="none",
            antialiased=False,
            alpha=1,
            rasterized=options["rasterize_mesh"],
            zorder=1,
        )
    )
    mesh_segments = unique_plot_segments(
        np.concatenate([xy[data["edges"]] + shift for shift in shifts])
    )
    ax.add_collection(
        LineCollection(
            mesh_segments,
            colors=mesh_color,
            linewidths=MESH_LINEWIDTH,
            alpha=1,
            rasterized=options["rasterize_mesh"],
            zorder=2,
        )
    )
    if data["boundaries"]:
        lines = np.concatenate(list(data["boundaries"].values()))
        boundary_segments = unique_plot_segments(
            np.concatenate([xy[lines] + shift for shift in shifts])
        )
        ax.add_collection(
            LineCollection(
                boundary_segments,
                colors=edge_color,
                linewidths=BOUNDARY_LINEWIDTH,
                alpha=1,
                zorder=3,
            )
        )
    ax.autoscale_view()
    annotate = options["annotate_boundaries"]
    ax.margins(
        x=(0.30 if blade_count == 1 else 0.18) if annotate else 0.035,
        y=0.085 if annotate else 0.035,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ coordinate (mm)", fontsize=MESH_FONT_SIZE, color="black")
    ax.set_ylabel(r"$y$ coordinate (mm)", fontsize=MESH_FONT_SIZE, color="black")
    ax.tick_params(
        direction="in", labelsize=MESH_FONT_SIZE * 0.9, width=0.7, colors="black"
    )
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(False)
    if annotate:
        annotate_mesh_boundaries(ax, data, options, shifts)
    return fig, ax


def save_mesh_image(
    filename, fluid_surface, *, boundaries=None, pitch=None, settings=None
):
    """Export one Times-styled mesh view to PNG and SVG without opening a GUI."""
    from matplotlib import rc_context

    options = mesh_figure_options(settings or {})
    data = collect_mesh_plot_data(fluid_surface, boundaries)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    with rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": MESH_FONT_SIZE,
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "svg.fonttype": "none",
        }
    ):
        fig, _ = plot_mesh_figure(data, pitch=pitch, options=options)
        try:
            for format in ("png", "svg"):
                path = filename.with_suffix(f".{format}")
                fig.savefig(
                    path,
                    # Also sets the resolution of rasterized collections in SVG.
                    dpi=options["dpi"],
                    facecolor="white",
                    bbox_inches="tight",
                    pad_inches=0.08,
                )
                paths[format] = str(path)
                logger.info(
                    "  Mesh figure (%d blades): %s",
                    options["num_blades"],
                    display_path(path),
                )
        finally:
            fig.clear()
    return paths
