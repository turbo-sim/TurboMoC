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


def section_figure(shape):
    x, y = shape["x"] * MM, shape["y"] * MM
    lower = "Axis" if shape["axisymmetric"] else "Symmetry plane"
    result = plots.figure("Nozzle cross section", "Axial distance [mm]",
                          "Radius [mm]" if shape["axisymmetric"] else "Height [mm]")
    result.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([y, -y[::-1]]),
                                fill="toself", fillcolor="rgba(140, 180, 200, 0.25)", mode="lines",
                                line=dict(color=plots.EDGE, width=1.6), hoverinfo="skip"))
    result.add_trace(go.Scatter(x=[x[0], x[-1]], y=[0, 0], mode="lines", name=lower,
                                line=dict(color="#526d79", width=1, dash="dashdot")))
    throat = shape["throat_index"]
    for index, label in ((throat, "Throat"), (len(x) - 1, "Exit")):
        result.add_trace(go.Scatter(x=[x[index], x[index]], y=[-y[index], y[index]], mode="lines",
                                    line=dict(color="#8a9aa1", width=1, dash="dot"), hoverinfo="skip"))
        result.add_annotation(x=x[index], y=y[index], text=f"{label} · {y[index]:.3f} mm", showarrow=False,
                              yshift=12, font=dict(size=11))
    result.update_layout(title=dict(text="Axisymmetric nozzle · meridional section" if shape["axisymmetric"]
                                    else "Planar nozzle · mid-depth section", font=dict(size=13)))
    return result


def solid_figure(shape, depth):
    result = go.Figure()
    for surface in geometry.surfaces(shape, depth):
        result.add_trace(go.Surface(x=surface["x"] * MM, y=surface["y"] * MM, z=surface["z"] * MM,
                                    colorscale=[[0, "#9fb7c2"], [1, "#9fb7c2"]], showscale=False, opacity=0.85,
                                    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2),
                                    name=surface["name"], hoverinfo="skip"))
    result.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=36, b=0), font=plots.APP_FONT,
                         uirevision="nozzle-solid", showlegend=False,
                         title=dict(text="Full revolution" if shape["axisymmetric"]
                                    else f"Planar slab · depth {depth * MM:.3g} mm", font=dict(size=13)),
                         scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]",
                                    zaxis_title="z [mm]", camera=dict(eye=dict(x=-1.2, y=-1.6, z=1.0))))
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
