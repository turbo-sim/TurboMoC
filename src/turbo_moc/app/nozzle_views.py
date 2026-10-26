"""Nozzle-mode figures, mesh scene and quality report."""
import json

import dash_vtk
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from turbo_moc.app import mesh_reporting, plots
from turbo_moc.app import workflow as flow
from turbo_moc.geometry import nozzle as geometry

MM = 1e3
SURFACE_COLORS = {"wall": "#a6b3b8", "inlet": "#c94f4f", "outlet": "#c94f4f", "side": "#4d9966"}
DEFAULT_SURFACES = ("wall", "outlet")


DOMAIN = "#72858e"


def section_figure(shape):
    x, y = shape["x"] * MM, shape["y"] * MM
    result = plots.figure("Nozzle cross section", "Axial distance [mm]",
                          "Radius [mm]" if shape["axisymmetric"] else "Height [mm]")
    result.add_trace(go.Scatter(x=np.concatenate([x, x[::-1], x[:1]]), y=np.concatenate([y, -y[::-1], y[:1]]),
                                mode="lines", name="nozzle", fill="toself", fillcolor=plots.BLADE_FILL,
                                line=dict(color=plots.EDGE, width=1.5)))
    for index, name in ((0, "inlet"), (len(x) - 1, "outlet")):
        result.add_trace(go.Scatter(x=[x[index], x[index]], y=[-y[index], y[index]], mode="lines", name=name,
                                    line=dict(color=DOMAIN, width=1)))
    result.add_trace(go.Scatter(x=[x[0], x[-1]], y=[0, 0], mode="lines",
                                name="axis" if shape["axisymmetric"] else "symmetry plane",
                                line=dict(color=DOMAIN, width=1, dash="dashdot")))
    return result


def meridional_figure(shape, depth):
    x = shape["x"] * MM
    if shape["axisymmetric"]:
        result = plots.figure("Meridional geometry", "z [mm]", "r [mm]")
        hub, shroud = np.zeros_like(x), shape["y"] * MM
        names = ("Axis", "Wall")
    else:
        result = plots.figure("Streamwise/span geometry", "Streamwise distance [mm]", "Span height [mm]")
        hub, shroud = np.zeros_like(x), np.full_like(x, depth * MM)
        names = ("Span start", "Span end")
    result.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([hub, shroud[::-1]]),
                                mode="none", fill="toself", fillcolor=plots.BLADE_FILL, hoverinfo="skip"))
    for values, name in zip((hub, shroud), names):
        result.add_trace(go.Scatter(x=x, y=values, mode="lines", name=name, line=dict(color=plots.EDGE, width=1)))
    for index, name in ((0, "inlet"), (len(x) - 1, "outlet")):
        result.add_trace(go.Scatter(x=[x[index], x[index]], y=[hub[index], shroud[index]], mode="lines",
                                    name=f"{name} flow domain", line=dict(color=DOMAIN, width=1.2, dash="dash")))
    result.update_yaxes(range=[0, 1.05 * float(shroud.max())], autorange=False, minallowed=0, constrain="domain")
    return result


def unique_edges(cells):
    edges = np.sort(np.stack([cells, np.roll(cells, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0)


def mesh_2d_figure(folder, axisymmetric, edge_width):
    with np.load(folder / "nozzle_section.npz", allow_pickle=False) as data:
        xy, quads = data["xy"] * MM, data["quads"]
        boundaries = {name[5:]: data[name] for name in data.files if name.startswith("edge_")}
    edges = unique_edges(quads)
    segments = np.full((len(edges), 3, 2), np.nan)
    segments[:, :2] = xy[edges]
    result = plots.figure("Nozzle 2D mesh", "Axial distance [mm]",
                          "Radius [mm]" if axisymmetric else "Height [mm]")
    result.add_trace(go.Scattergl(x=segments[:, :, 0].ravel(), y=segments[:, :, 1].ravel(), mode="lines",
                                  line=dict(color=plots.MESH_EDGE, width=edge_width), hoverinfo="skip"))
    colors = {"wall": plots.EDGE, "axis": "#526d79", "inlet": "#3a7bbf", "outlet": "#d0782f"}
    for name, pairs in boundaries.items():
        line = np.full((len(pairs), 3, 2), np.nan)
        line[:, :2] = xy[pairs]
        result.add_trace(go.Scattergl(x=line[:, :, 0].ravel(), y=line[:, :, 1].ravel(), mode="lines", name=name,
                                      line=dict(color=colors.get(name, plots.EDGE), width=1.6)))
    result.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
    return result


def rgb(color):
    return [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def display_panel():
    return html.Aside([
        html.Div(html.Strong("Configure display"), className="display-panel-header"),
        html.Section([
            html.Div([html.Button(text, id={"type": "nozzle-display-preset", "action": action}, n_clicks=0)
                      for text, action in (("All surfaces", "all"), ("Reset display", "reset"))],
                     className="display-presets"),
            dcc.Checklist(id="nozzle-display-surfaces", value=list(DEFAULT_SURFACES),
                          options=[{"label": name.title(), "value": name} for name in SURFACE_COLORS],
                          className="display-surfaces"),
            dcc.Checklist(id="nozzle-display-edges", options=[{"label": "Mesh edges", "value": "edges"}],
                          value=["edges"]),
        ]),
    ], className="display-panel")


def mesh_3d_scene(folder, selected, edges):
    surfaces = json.loads((folder / "browser_surfaces.json").read_text(encoding="utf-8"))
    representations = [dash_vtk.GeometryRepresentation(
        children=[dash_vtk.PolyData(points=(np.asarray(data["points"]) * MM).tolist(), polys=data["polys"])],
        property={"edgeVisibility": edges, "color": rgb(SURFACE_COLORS.get(name, "#a6b3b8")),
                  "edgeColor": [0.1, 0.1, 0.1]})
        for name, data in surfaces.items() if name in selected]
    view = dash_vtk.View(representations, background=[1, 1, 1], cameraPosition=[-1, -1, 1],
                         cameraViewUp=[0, 0, 1], style={"width": "100%", "height": "100%"})
    return view, list(surfaces)


def cross_figure(summary, settings, edge_width):
    radius = summary["exit_radius"] * MM
    if summary["axisymmetric"]:
        from turbo_moc.meshing.nozzle_mesh import o_grid_disk
        points, quads, _ = o_grid_disk(settings["circumferential_cells"] // 4, settings["radial_cells"],
                                       settings["wall_growth_ratio"])
        points = points * radius
        result = plots.figure("Cross-plane grid", "y [mm]", "z [mm]")
    else:
        from turbo_moc.meshing.nozzle_mesh import wall_distribution
        eta = wall_distribution(settings["radial_cells"], settings["wall_growth_ratio"])
        height = np.concatenate([-eta[::-1], eta[1:]]) * radius
        depth = np.linspace(-0.5, 0.5, settings["depth_cells"] + 1) * settings["planar_depth"] * MM
        z, y = np.meshgrid(depth, height, indexing="ij")
        points = np.column_stack([z.ravel(), y.ravel()])
        index = np.arange(len(points)).reshape(z.shape)
        quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], -1).reshape(-1, 4)
        result = plots.figure("Cross-plane grid", "Span [mm]", "Height [mm]")
    edges = unique_edges(quads)
    segments = np.full((len(edges), 3, 2), np.nan)
    segments[:, :2] = points[edges]
    result.add_trace(go.Scattergl(x=segments[:, :, 0].ravel(), y=segments[:, :, 1].ravel(), mode="lines",
                                  line=dict(color=plots.MESH_EDGE, width=edge_width), hoverinfo="skip"))
    return result


def table(rows):
    return html.Table([html.Tbody([html.Tr([html.Td(k), html.Td(str(v))]) for k, v in rows])],
                      className="report-table")


def report_2d(summary, report):
    return [table([("Cells", f"{summary['cells']:,}"), ("Nodes", f"{summary['nodes']:,}"),
                   ("Axial × radial cells", f"{summary['axial_cells']} × {summary['radial_cells']}"),
                   ("First wall cell height at the throat [mm]", f"{summary['first_wall_height'] * MM:.4g}"),
                   ("2D time [s]", f"{summary['elapsed_s']:.2f}")]),
            mesh_reporting.metric_table(report["metrics"], report["definitions"]),
            mesh_reporting.histogram_row(report["metrics"], report["definitions"])]


def report_3d(summary, report):
    sweep = ("Circumferential cells", summary["circumferential_cells"]) if summary["axisymmetric"] else (
        "Depth cells", summary["depth_cells"])
    return [table([("Volume cells", f"{summary['cells']:,}"), ("Volume nodes", f"{summary['nodes']:,}"),
                   ("Axial × radial cells", f"{summary['axial_cells']} × {summary['radial_cells']}"), sweep,
                   ("3D time [s]", f"{summary['elapsed_s']:.2f}")]),
            mesh_reporting.metric_table(report["metrics"], report["definitions"]),
            mesh_reporting.histogram_row(report["metrics"], report["definitions"])]


def load(reference, name):
    return json.loads((flow.directory(reference) / name).read_text(encoding="utf-8"))
