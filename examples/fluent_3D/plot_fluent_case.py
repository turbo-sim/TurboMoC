"""Save 400-DPI PNGs and open interactive windows; use --no-show for export only.

Run from any directory. The .cas.h5 supplies the mesh and its adjacent .dat.h5
supplies the solution. Install with `poetry install --extras fluent`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.path import Path as PolygonPath
from PIL import Image
from vtkmodules.vtkCommonDataModel import vtkCylinder
from ansys.fluent.core import (
    ScalarFieldDataRequest,
    SurfaceDataType,
    SurfaceFieldDataRequest,
)
from ansys.fluent.core.file_session import FileSession

HERE = Path(__file__).resolve().parent
CASE_FILE = HERE / "output/fluent/stator_fluent_3d.cas.h5"
DATA_FILE = CASE_FILE.with_name(CASE_FILE.name.replace(".cas.h5", ".dat.h5"))
OUTPUT_DIR = CASE_FILE.parent.parent / "pyvista"
PRESSURE = "Static pressure [kPa]"
EXPORT_DPI = 400
# Render at twice the window resolution: 8 x 5 inches for individual plots
# and 12 x 8 inches for the four-view figure, both at 400 DPI.
EXPORT_SCALE = 2


def three_passages(mesh, transform):
    """Display the solved passage and its two neighbours using the mesh pitch."""
    copies = [
        mesh.transform(np.linalg.matrix_power(transform, index), inplace=False)
        for index in (-1, 0, 1)
    ]
    return copies[0].merge(copies[1:], merge_points=False)


def camera_frame(blade, geometry):
    """Align the picture horizontally with the blade chord, above the hub."""
    axis = np.asarray(geometry["axis_direction"], dtype=float)
    origin = np.asarray(geometry["axis_origin"], dtype=float)
    radial = np.asarray(blade.center) - origin
    radial -= np.dot(radial, axis) * axis
    radial /= np.linalg.norm(radial)
    if geometry["kind"] == "axial":
        span, stream = radial, axis
    elif geometry["kind"] == "radial":
        span, stream = axis, radial
    else:
        span, stream = axis, np.array([1.0, 0.0, 0.0])
    stations = blade.points @ stream
    tolerance = 0.015 * np.ptp(stations)
    leading = blade.points[stations < stations.min() + tolerance].mean(axis=0)
    trailing = blade.points[stations > stations.max() - tolerance].mean(axis=0)
    chord = trailing - leading
    chord -= np.dot(chord, span) * span
    chord /= np.linalg.norm(chord)
    return span, chord, np.cross(span, chord)


def orient_camera(plotter, frame, meshes, view=(-0.35, 0.65, 1.0)):
    span, chord, up = frame
    along, across, above = view
    center = np.mean(np.asarray(plotter.bounds).reshape(3, 2), axis=1)
    direction = above * span + along * chord + across * up
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction, up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, -direction)
    points = np.vstack([mesh.points for mesh in meshes]) - center
    x, y = points @ right, points @ screen_up
    center += right * (x.min() + x.max()) / 2 + screen_up * (y.min() + y.max()) / 2
    plotter.camera_position = [center + direction, center, up]
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.focal_point = center
    plotter.camera.position = center + direction
    viewport = plotter.renderer.GetViewport()
    width, height = plotter.window_size
    aspect = (
        width * (viewport[2] - viewport[0]) / (height * (viewport[3] - viewport[1]))
    )
    plotter.camera.parallel_scale = 0.60 * max(np.ptp(y), np.ptp(x) / aspect)
    plotter.reset_camera_clipping_range()


def surface_edges(mesh):
    """Surface outlines and sharp joins only; exclude ordinary mesh edges."""
    return mesh.extract_feature_edges(
        feature_angle=35,
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
        clear_data=True,
    )


def closed_blade(blade, geometry):
    """Reconstruct hub/shroud caps from the two actual wall boundary rings."""
    rings = blade.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).strip(join=True)
    if rings.n_lines != 2:
        raise RuntimeError("Expected two blade end rings for the complete blade view.")
    caps, offset = [], 0
    for _ in range(rings.n_lines):
        count = rings.lines[offset]
        ids = rings.lines[offset + 1 : offset + count + 1]
        offset += count + 1
        if ids[0] == ids[-1]:
            ids = ids[:-1]
        points = rings.points[ids]
        radius = np.linalg.norm(points[:, :2], axis=1).mean()
        chart = np.zeros_like(points)
        if geometry["kind"] == "axial":
            chart[:, 0] = radius * np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
            chart[:, 1] = points[:, 2]
        else:
            chart[:, :2] = points[:, :2]
        # Interior points resolve the curved axial caps without splitting their
        # boundary edges, so cap and wall retain the same watertight ring.
        x, y = np.meshgrid(
            np.linspace(chart[:, 0].min(), chart[:, 0].max(), 80),
            np.linspace(chart[:, 1].min(), chart[:, 1].max(), 40),
        )
        grid = np.column_stack((x.ravel(), y.ravel()))
        grid = grid[PolygonPath(chart[:, :2]).contains_points(grid)]
        boundary_ids = np.arange(len(chart))
        signed_area = np.sum(
            chart[:, 0] * np.roll(chart[:, 1], -1)
            - np.roll(chart[:, 0], -1) * chart[:, 1]
        )
        if signed_area < 0:
            boundary_ids = boundary_ids[::-1]
        boundary = pv.PolyData(chart, np.r_[len(chart), boundary_ids])
        cloud = pv.PolyData(
            np.vstack((chart, np.column_stack((grid, np.zeros(len(grid))))))
        )
        cap = cloud.delaunay_2d(edge_source=boundary)
        if geometry["kind"] == "axial":
            angle = cap.points[:, 0] / radius
            cap.points = np.column_stack(
                (radius * np.cos(angle), radius * np.sin(angle), cap.points[:, 1])
            )
        else:
            cap.points[:, 2] = points[:, 2].mean()
        cap.points[: len(points)] = points
        caps.append(cap)
    solid = blade.copy()
    solid.clear_data()
    solid = solid.merge(caps, merge_points=True, tolerance=1e-10)
    if solid.n_open_edges:
        raise RuntimeError("The reconstructed blade caps did not close the surface.")
    return solid.compute_normals(consistent_normals=True, auto_orient_normals=True)


def add_surfaces(plotter, meshes, *, mesh_lines=False):
    fields = {scalar for _, _, scalar in meshes if scalar}
    limits = {
        scalar: (
            min(
                float(mesh[scalar].min())
                for mesh, _, field in meshes
                if field == scalar
            ),
            max(
                float(mesh[scalar].max())
                for mesh, _, field in meshes
                if field == scalar
            ),
        )
        for scalar in fields
    }
    for mesh, color, scalar in meshes:
        plotter.add_mesh(
            mesh,
            scalars=scalar,
            color=color if scalar is None else None,
            cmap="viridis",
            clim=limits.get(scalar),
            show_edges=mesh_lines,
            edge_color="black",
            line_width=0.45,
            lighting=scalar is None,
            smooth_shading=scalar is None and not mesh_lines,
            split_sharp_edges=scalar is None and not mesh_lines,
            scalar_bar_args=(
                {"title": scalar, "color": "black", "fmt": "%.1f"} if scalar else None
            ),
        )
        edges = surface_edges(mesh)
        if edges.n_cells:
            plotter.add_mesh(edges, color="black", line_width=1.6, lighting=False)


def export_and_show(plotter, path, title, *, show):
    """Save the initial camera view before inspection, even if the window is closed."""
    try:
        plotter.show(title=title, interactive=False, auto_close=False)
        pixels = plotter.screenshot(return_img=True, scale=EXPORT_SCALE)
        Image.fromarray(pixels).save(path, dpi=(EXPORT_DPI, EXPORT_DPI))
        print(f"Saved {path} ({EXPORT_DPI} DPI)", flush=True)
        if show:
            print("Inspect the plot; close its window or press q to advance.", flush=True)
            plotter.show(title=title, interactive=True, auto_close=False)
    finally:
        plotter.close()


def save_plot(meshes, filename, title, frame, *, show=True, mesh_lines=False):
    plotter = pv.Plotter(off_screen=not show, window_size=(1600, 1000))
    plotter.set_background("white")
    add_surfaces(plotter, meshes, mesh_lines=mesh_lines)
    plotter.add_text(title, font_size=16, color="black")
    plotter.add_axes()
    orient_camera(plotter, frame, [mesh for mesh, _, _ in meshes])
    path = OUTPUT_DIR / filename
    export_and_show(plotter, path, title, show=show)


def save_blade_views(blade, frame, show=True):
    plotter = pv.Plotter(
        shape=(2, 2), off_screen=not show, window_size=(2400, 1600), border=False
    )
    views = [
        ("Shroud-side oblique", (-0.55, 0.65, 1)),
        ("Opposite oblique", (0.45, -0.65, 1)),
        ("Shroud end", (0, 0, 1)),
        ("Hub end", (0, 0, -1)),
    ]
    for index, (title, view) in enumerate(views):
        plotter.subplot(index // 2, index % 2)
        plotter.set_background("white")
        add_surfaces(plotter, [(blade, "steelblue", None)])
        plotter.add_text(title, font_size=16, color="black")
        orient_camera(plotter, frame, [blade], view=view)
    path = OUTPUT_DIR / "single_blade_views.png"
    export_and_show(plotter, path, "Single blade: four views", show=show)


def main(show=True):
    for path in (CASE_FILE, DATA_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fluent case/data file: {path}")
    setup = json.loads((CASE_FILE.parent / "setup_summary.json").read_text())
    geometry = setup["geometry"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PyFluent's file reader gives the actual wall faces and stored face values.
    # FileSession reads HDF5 directly and never starts a licensed Fluent solver.
    session = FileSession(case_file_name=str(CASE_FILE), data_file_name=str(DATA_FILE))
    fields = session.fields.field_data
    names = [setup["zones"][role] for role in ("blade", "hub")]
    surfaces = fields.get_field_data(
        SurfaceFieldDataRequest(
            surfaces=names,
            data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity],
            flatten_connectivity=True,
        )
    )
    pressures = fields.get_field_data(
        ScalarFieldDataRequest(
            surfaces=names, field_name="SV_P", node_value=False, boundary_value=False
        )
    )
    walls = []
    for name in names:
        data = surfaces[name]
        wall = pv.PolyData(data.vertices, data.connectivity)
        values = np.asarray(pressures[name])
        if len(values) != wall.n_cells or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid saved pressure data on {name}.")
        wall.cell_data[PRESSURE] = values / 1000
        walls.append(wall)
    blade, hub = walls
    frame = camera_frame(blade, geometry)
    solid = closed_blade(blade, geometry)
    save_plot(
        [(blade, None, PRESSURE), (hub, None, PRESSURE)],
        "blade_hub_pressure.png",
        "Blade and hub pressure",
        frame,
        show=show,
    )

    # VTK's Fluent CFF reader loads volume cells and the adjacent solution file.
    volume = pv.get_reader(CASE_FILE).read().combine()
    if volume.n_cells != geometry["mesh"]["cell_count"]:
        raise RuntimeError("The Fluent volume reader returned an incomplete mesh.")
    mach = next(
        (
            name
            for name in ("SV_MACH", "Mach", "mach-number")
            if name in volume.cell_data
        ),
        None,
    )
    scalar = "Mach number" if mach else "Velocity magnitude [m/s]"
    volume.cell_data[scalar] = (
        volume.cell_data[mach]
        if mach
        else np.linalg.norm(
            np.column_stack(
                [volume.cell_data[f"SV_{axis}"] for axis in ("U", "V", "W")]
            ),
            axis=1,
        )
    )
    if not np.isfinite(volume.cell_data[scalar]).all():
        raise RuntimeError(f"Invalid saved {scalar} values.")
    # Average neighbouring cell values to mesh nodes for continuous contours.
    volume = volume.cell_data_to_point_data()
    section = geometry["section"]
    if section["type"] == "cylinder":
        cylinder = vtkCylinder()
        cylinder.SetCenter(*geometry["axis_origin"])
        cylinder.SetAxis(*geometry["axis_direction"])
        cylinder.SetRadius(section["radius_m"])
        midspan = volume.slice_implicit(cylinder)
    else:
        midspan = volume.slice(normal=section["normal"], origin=section["point_m"])
    if not midspan.n_cells:
        raise RuntimeError("The midspan cut has no faces inside the fluid mesh.")
    # Merge coincident cutter vertices and orient neighbouring polygons before
    # extracting outlines, otherwise slice seams can look like mesh edges.
    midspan = midspan.clean(tolerance=1e-8, absolute=True).compute_normals(
        consistent_normals=True
    )
    transform = np.asarray(geometry["transform"])
    blades = three_passages(solid, transform)
    hubs = three_passages(hub, transform)
    save_plot(
        [
            (hubs, "lightgrey", None),
            (blades, "steelblue", None),
            (three_passages(midspan, transform), None, scalar),
        ],
        "midspan_flow.png",
        f"Midspan {scalar} (three passages)",
        frame,
        show=show,
    )

    meshes = [(blades, "steelblue", None), (hubs, "lightgrey", None)]
    save_plot(
        meshes, "three_blades_hub_surface.png", "Three blades and hub", frame, show=show
    )
    save_plot(
        [
            (three_passages(blade, transform), "steelblue", None),
            (hubs, "lightgrey", None),
        ],
        "three_blades_hub_mesh.png",
        "Three blades and hub: boundary mesh",
        frame,
        show=show,
        mesh_lines=True,
    )
    save_blade_views(solid, frame, show=show)
    print(f"Read {volume.n_cells:,} cells; midspan uses {scalar}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open interactive PyVista windows (default); --no-show exports only",
    )
    main(show=parser.parse_args().show)
