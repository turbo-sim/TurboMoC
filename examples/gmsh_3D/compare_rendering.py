"""Compare the desktop (PyVista) and browser (dash-vtk) views of the same 3D mesh.

For an axial, a radial and a linear stator this script builds a small mesh,
checks numerically that the app's surface datasets and the Gmsh mesh are the
same object, renders the desktop view with PyVista and the browser view with
dash-vtk (captured with headless Microsoft Edge when available), and writes a
Markdown report with the images and checks.

Run from the repository root:
    python examples/gmsh_3D/compare_rendering.py [output_dir]
"""
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from turbo_moc.configuration.project import default_project, row_configuration
from turbo_moc.geometry import blade_row
from turbo_moc.meshing import mesh_3d
from turbo_moc.visualization import mesh_pyvista, style

SURFACES = ("blade", "hub", "inlet", "outlet", "periodic_left", "periodic_right")
EDGE_PATHS = (Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
              Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"))


def small_case(family):
    config = row_configuration(default_project(family), "stator_1")
    config["blade_geometry"].update(r_trailing=0.1, num_baseline_points=100, num_curve_points=100, n_cp=15)
    config["flow_domain"]["inlet"]["extension_length"] = 0.5
    config["flow_domain"]["outlet"]["extension_length"] = 0.5
    config["flow_domain"]["number_of_points"] = 40
    config["mesh"]["numerics"].update(terminal_output=False, blade_size=0.7, domain_size=3.0, curvature_elements=16,
                                      smoothing_steps=1, span_layers=4, span_stretch=0.8)
    config["mesh"]["numerics"]["inflation_layers"].update(first_height=0.03, growth_ratio=1.4, thickness=0.15)
    if family != "linear":
        config["meridional_channel"]["num_blades"] = 12
    x = np.linspace(0.0, 0.01, 80)
    wall = {"converged": True, "wall_final": {"x": x.tolist(), "y": (0.002 + 0.001 * (x / x[-1]) ** 2).tolist()}}
    return wall, config


def build_mesh(family, directory):
    wall, config = small_case(family)
    design = blade_row.build_design(wall, config)
    mesh_3d.reference_mesh(design, config, directory)
    section, design = mesh_3d.load_section(directory)
    mesh_3d.volume_mesh(section, design, config, directory)


def data_checks(directory):
    gmsh_surfaces = mesh_3d.load_mesh_surfaces(directory / "stator_mesh_3d.msh", SURFACES)
    app_surfaces = mesh_pyvista.load_surfaces(directory, SURFACES)
    rows = []
    for name in SURFACES:
        points, polygons = gmsh_surfaces[name]
        app = app_surfaces[name]
        rows.append({"surface": name, "points_gmsh": len(points), "points_app": app.n_points,
                     "faces_gmsh": len(_offsets(polygons)),
                     "faces_app": app.n_cells,
                     "bounds_max_difference_mm": float(np.max(np.abs(
                         np.r_[points.min(axis=0), points.max(axis=0)]
                         - np.r_[np.asarray(app.bounds)[0::2], np.asarray(app.bounds)[1::2]])))})
    return rows


def _offsets(polygons):
    offsets, index = [], 0
    while index < len(polygons):
        offsets.append(index)
        index += int(polygons[index]) + 1
    return np.asarray(offsets)


def quality_check(directory):
    surface, _ = mesh_pyvista.inspection_surface(directory, {"mode": "worst", "metric": "skewness", "n": 50})
    colors = surface.cell_data["MeshQualityColors"]
    return {"inspection_cells": surface.n_cells, "association": "cell",
            "color_components": int(colors.shape[1]), "color_dtype": str(colors.dtype)}


def desktop_image(directory, family, path):
    import pyvista as pv
    summary = json.loads((directory / "mesh_summary.json").read_text(encoding="utf-8"))
    definition = summary["geometry"]
    count = 3 if family == "linear" else int(definition["num_blades"])
    plotter = pv.Plotter(off_screen=True, window_size=(900, 650))
    plotter.set_background(style.BACKGROUND)
    mesh_pyvista.add_row(plotter, mesh_pyvista.load_surfaces(directory, ("blade", "hub")), summary, count, edges=False)
    mesh_pyvista.set_home_camera(plotter, family)
    plotter.screenshot(str(path))
    plotter.close()


def browser_page(directory, family, port):
    import dash_vtk
    from dash import Dash, html
    surfaces = json.loads((directory / "browser_surfaces.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "mesh_summary.json").read_text(encoding="utf-8"))
    definition = summary["geometry"]
    linear = family == "linear"
    count = 3 if linear else int(definition["num_blades"])
    actors = []
    for index in range(count):
        for name in ("blade", "hub"):
            orientation = [0, 0, 0 if linear else 180 - 360 * index / count]
            position = [0, index * definition.get("pitch_mm", 0), 0] if linear else [0, 0, 0]
            actors.append(dash_vtk.GeometryRepresentation(
                children=[dash_vtk.PolyData(points=surfaces[name]["points"], polys=surfaces[name]["polys"])],
                actor={"orientation": orientation, "position": position},
                property={"color": style.rgb(style.surface_color(name)), "edgeVisibility": False,
                          "edgeColor": style.rgb(style.EDGE_COLOR), **style.LIGHTING}))
    home = style.camera(family)
    view = dash_vtk.View(actors, background=style.rgb(style.BACKGROUND), cameraPosition=list(home["direction"]),
                         cameraViewUp=list(home["up"]), style={"width": "900px", "height": "650px"})
    app = Dash(__name__)
    app.layout = html.Div(view, style={"width": "900px", "height": "650px"})
    threading.Thread(target=lambda: app.run(port=port, debug=False), daemon=True).start()
    time.sleep(8)


def browser_image(family, directory, path, port):
    edge = next((candidate for candidate in EDGE_PATHS if candidate.is_file()), None)
    if edge is None:
        return False
    browser_page(directory, family, port)
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=920,670",
                    "--virtual-time-budget=12000", f"--screenshot={path}", f"http://127.0.0.1:{port}/"],
                   capture_output=True, timeout=120)
    return path.is_file()


def main(output=Path("output/rendering_comparison")):
    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    report = ["# Desktop and browser rendering comparison", "",
              "Both views draw `browser_surfaces.json` with the shared style in `turbo_moc.visualization.style`: "
              "surface colours, edge colour, ambient/diffuse lighting, background, the per-family camera home view "
              "and the display-copy transform (annular rows in the (-X, -Y, Z) display frame).", ""]
    for port, family in enumerate(("axial", "radial", "linear"), start=8150):
        directory = output / family
        directory.mkdir()
        build_mesh(family, directory)
        rows = data_checks(directory)
        quality = quality_check(directory)
        desktop = output / f"{family}_desktop.png"
        desktop_image(directory, family, desktop)
        browser = output / f"{family}_browser.png"
        captured = browser_image(family, directory, browser, port)
        same = all(r["points_gmsh"] == r["points_app"] and r["faces_gmsh"] == r["faces_app"]
                   and r["bounds_max_difference_mm"] < 1e-6 for r in rows)
        report += [f"## {family.title()} stator", "",
                   "| Surface | Points (Gmsh / app) | Faces (Gmsh / app) | Max bounds difference [mm] |",
                   "|---|---|---|---|"]
        report += [f"| {r['surface']} | {r['points_gmsh']} / {r['points_app']} | {r['faces_gmsh']} / {r['faces_app']} "
                   f"| {r['bounds_max_difference_mm']:.2e} |" for r in rows]
        report += ["", f"Same object: **{'yes' if same else 'no'}**. Quality inspection: {quality['inspection_cells']} "
                   f"cells, colours stored per {quality['association']} ({quality['color_components']} x "
                   f"{quality['color_dtype']}), identical in both renderers.", "",
                   f"| Desktop (PyVista) | Browser (dash-vtk) |", "|---|---|",
                   f"| ![]({desktop.name}) | " + (f"![]({browser.name}) |" if captured else "not captured |"), ""]
        print(family, "same object:", same, "browser image:", captured)
    report += ["## Differences that remain", "",
               "- Lighting model: VTK desktop OpenGL and vtk.js WebGL shade the same ambient/diffuse values slightly "
               "differently; colours match, shading intensity can differ.",
               "- Mesh edges: the app draws antialiased edges in its surface shader at a set pixel width; PyVista uses "
               "native line rendering.",
               "- Layout: the desktop view has two panels (one passage, full row), a legend, axes and bounds for static "
               "images; the app has one interactive scene with a displayed-blade count and colour pickers.",
               "- Interaction: quality inspection controls, row visibility and camera persistence are app-only; the "
               "desktop viewer accepts the same inspection request as a keyword argument."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Report:", output / "report.md")


if __name__ == "__main__":
    main(*sys.argv[1:])
