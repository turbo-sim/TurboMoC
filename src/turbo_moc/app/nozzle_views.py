"""Nozzle-mode figures, mesh scene and quality report."""
import json

import dash_vtk
import numpy as np
import plotly.graph_objects as go
from dash import html

from turbo_moc.app import mesh_reporting, plots
from turbo_moc.app import workflow as flow
from turbo_moc.geometry import nozzle as geometry

MM = 1e3
SURFACE_COLORS = {"wall": [0.55, 0.6, 0.63], "inlet": [0.35, 0.6, 0.85], "outlet": [0.9, 0.55, 0.3],
                  "side": [0.75, 0.78, 0.8]}


DOMAIN = "#72858e"


def section_figure(shape):
    x, y = shape["x"] * MM, shape["y"] * MM
    result = plots.figure("Nozzle cross section", "Axial distance [mm]",
                          "Radius [mm]" if shape["axisymmetric"] else "Height [mm]")
    for sign in (1, -1):
        result.add_trace(go.Scatter(x=x, y=sign * y, mode="lines", name="wall",
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


def mesh_3d_scene(folder):
    surfaces = json.loads((folder / "browser_surfaces.json").read_text(encoding="utf-8"))
    representations = []
    for name, data in surfaces.items():
        points = (np.asarray(data["points"]) * MM).tolist()
        representations.append(dash_vtk.GeometryRepresentation(
            children=[dash_vtk.PolyData(points=points, polys=data["polys"])],
            property={"edgeVisibility": True, "color": SURFACE_COLORS.get(name, [0.7, 0.7, 0.7]),
                      "edgeColor": [0.15, 0.15, 0.15], "opacity": 0.35 if name in ("wall", "side") else 1.0}))
    return dash_vtk.View(representations, background=[1, 1, 1], cameraPosition=[-1, -1, 1],
                         cameraViewUp=[0, 0, 1], style={"width": "100%", "height": "100%"})


def table(rows):
    return html.Table([html.Tbody([html.Tr([html.Td(k), html.Td(str(v))]) for k, v in rows])],
                      className="report-table")


def download(reference, filename, label):
    return html.A(label, href=f"/artifacts/{reference['session']}/{reference['job']}/{filename}",
                  className="download-link")


def summary_2d(summary, reference):
    return [table([("Cells", f"{summary['cells']:,}"), ("Nodes", f"{summary['nodes']:,}"),
                   ("Axial × radial cells", f"{summary['axial_cells']} × {summary['radial_cells']}"),
                   ("First wall cell height (throat) [mm]", f"{summary['first_wall_height'] * MM:.4g}"),
                   ("Nozzle length incl. convergent [mm]", f"{summary['length'] * MM:.4g}"),
                   ("Exit / throat radius", f"{summary['exit_radius'] / summary['throat_radius']:.4f}"),
                   ("2D time [s]", f"{summary['elapsed_s']:.2f}")]),
            download(reference, "nozzle_mesh_2d.cgns", "Download 2D CGNS")]


def summary_3d(summary, reference):
    sweep = ("Circumferential cells", summary["circumferential_cells"]) if summary["axisymmetric"] else (
        "Depth cells", summary["depth_cells"])
    return [table([("Volume cells (hexahedra)", f"{summary['cells']:,}"), ("Volume nodes", f"{summary['nodes']:,}"),
                   ("Axial × radial cells", f"{summary['axial_cells']} × {summary['radial_cells']}"), sweep,
                   ("Topology", "O-grid revolution" if summary["axisymmetric"] else "Extruded slab"),
                   ("3D time [s]", f"{summary['elapsed_s']:.2f}")]),
            download(reference, "nozzle_mesh_3d.cgns", "Download 3D CGNS")]


def quality(title, report):
    return [html.H3(title), mesh_reporting.metric_table(report["metrics"], report["definitions"]),
            mesh_reporting.histogram_row(report["metrics"], report["definitions"])]


def load(reference, name):
    return json.loads((flow.directory(reference) / name).read_text(encoding="utf-8"))
