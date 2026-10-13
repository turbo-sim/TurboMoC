"""PyVista mesh surface previews and display options."""

import json
from pathlib import Path

from turbo_moc.visualization import style


def pyvista_options(options):
    """Validate YAML surface selection and the mesh-line toggle."""
    from turbo_moc.meshing.mesh_3d import BOUNDARIES

    if not isinstance(options, dict):
        raise ValueError("mesh.pyvista must be a mapping.")
    selected = options.get("surfaces", {"blade": True})
    if not isinstance(selected, dict) or set(selected) - set(BOUNDARIES):
        raise ValueError(
            "mesh.pyvista.surfaces must map physical boundary names to booleans."
        )
    if any(type(value) is not bool for value in selected.values()):
        raise ValueError("Every mesh.pyvista.surfaces value must be true or false.")
    surfaces = tuple(name for name, visible in selected.items() if visible)
    if not surfaces:
        raise ValueError("Select at least one mesh.pyvista surface to display.")
    show_edges = options.get("show_mesh_edges", False)
    if type(show_edges) is not bool:
        raise ValueError("mesh.pyvista.show_mesh_edges must be true or false.")
    window_size = options.get("window_size", [1500, 800])
    if (not isinstance(window_size, (list, tuple)) or len(window_size) != 2 or
            any(type(value) is not int or value <= 0 for value in window_size)):
        raise ValueError("mesh.pyvista.window_size must contain two positive integers.")
    filename = options.get("figure_filename", "pyvista_mesh.png")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".png"):
        raise ValueError("mesh.pyvista.figure_filename must be a PNG filename.")
    return {"surfaces": surfaces, "show_mesh_edges": show_edges,
            "window_size": window_size, "figure_filename": filename}


def _family(definition):
    return definition.get("type") if definition.get("type") in ("axial", "radial", "linear") else "axial"


def load_surfaces(output_dir, surfaces):
    """Read the same boundary datasets the app displays (browser_surfaces.json)."""
    import numpy as np
    import pyvista as pv

    data = json.loads((Path(output_dir) / "browser_surfaces.json").read_text(encoding="utf-8"))
    missing = [name for name in surfaces if name not in data]
    if missing:
        raise ValueError(f"The mesh has no {', '.join(missing)} surface data.")
    return {name: pv.PolyData(np.asarray(data[name]["points"], dtype=float).reshape(-1, 3),
                              np.asarray(data[name]["polys"], dtype=np.int64)) for name in surfaces}


def inspection_surface(output_dir, inspection):
    """The app's quality-inspection cells, with their per-cell RGB badness colours."""
    import numpy as np
    import pyvista as pv
    from turbo_moc.meshing.inspection import inspect_mesh

    result = inspect_mesh(output_dir, **inspection)
    payload = result["surfaces"]["inspection"]
    mesh = pv.PolyData(np.asarray(payload["points"], dtype=float).reshape(-1, 3),
                       np.asarray(payload["polys"], dtype=np.int64))
    mesh.cell_data["MeshQualityColors"] = style.quality_rgb(payload["color_values"])
    return mesh, result.get("note", "")


def add_row(plotter, meshes, summary, count, *, role="stator", edges=False, placement=None, quality=None):
    """Add display copies exactly as the app's actors: same transform, colours and lighting."""
    definition = summary.get("geometry", summary.get("machine", {}))
    linear = definition.get("type") == "linear"
    placement = placement or (summary.get("assembly") or {}).get("placement")
    for index in range(count):
        matrix = style.copy_transform(index, is_linear=linear, num_blades=definition.get("num_blades"),
                                      pitch_mm=definition.get("pitch_mm"), placement=placement)
        for name, surface in meshes.items():
            context = quality is not None
            plotter.add_mesh(surface.transform(matrix, inplace=False),
                             color=style.CONTEXT_COLOR if context else style.surface_color(name, role),
                             opacity=0.18 if context else 1.0, show_edges=edges and not context,
                             edge_color=style.EDGE_COLOR, label=name if index == 0 and not context else None,
                             **style.LIGHTING)
        if quality is not None:
            plotter.add_mesh(quality.transform(matrix, inplace=False), scalars="MeshQualityColors", rgb=True,
                             preference="cell", show_edges=edges, edge_color=style.EDGE_COLOR, ambient=1.0,
                             diffuse=0.0)


def set_home_camera(plotter, family):
    home = style.camera(family)
    plotter.view_vector(home["direction"], viewup=home["up"])
    plotter.reset_camera()


def view_mesh_pyvista(output_dir, options, *, show=True, save_figure=False, inspection=None, role="stator"):
    """View one passage and the full row with the app's data and style; save a preview."""
    options = pyvista_options(options)
    if not show and not save_figure:
        return
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Install the project dependencies from the repository root: "
            '.venv/Scripts/python.exe -m pip install .'
        ) from exc

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "mesh_summary.json").read_text(encoding="utf-8"))
    definition = summary.get("geometry", summary.get("machine", {}))
    is_linear = definition.get("type") == "linear"
    num_blades = 3 if is_linear else int(definition["num_blades"])
    meshes = load_surfaces(output_dir, options["surfaces"])
    quality, note = inspection_surface(output_dir, inspection) if inspection else (None, "")

    def populate(plotter):
        for panel, count in enumerate((1, num_blades)):
            plotter.subplot(0, panel)
            plotter.set_background(style.BACKGROUND)
            title = (("Single blade" if tuple(meshes) == ("blade",) else "Single passage") if panel == 0
                     else (f"Linear cascade ({count} displayed blades)" if is_linear
                           else f"Full turbine row ({count} blades)"))
            plotter.add_text(title + (f" | {note}" if note else ""), position="upper_left", color="black", font_size=12)
            add_row(plotter, meshes, summary, count, role=role, edges=options["show_mesh_edges"], quality=quality)
            plotter.add_legend()
            plotter.show_axes()
            plotter.show_bounds(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="black")
            set_home_camera(plotter, _family(definition))
            plotter.add_text("Drag: rotate | Wheel: zoom | Shift + drag: pan", position="lower_left",
                             color="black", font_size=10)

    # Save the initial view using a separate offscreen window. Closing the
    # interactive window with its X button cannot destroy this screenshot.
    modes = ([True] if save_figure else []) + ([False] if show else [])
    for off_screen in modes:
        plotter = pv.Plotter(shape=(1, 2), window_size=options["window_size"], off_screen=off_screen)
        try:
            populate(plotter)
            if off_screen:
                plotter.screenshot(str(output_dir / options["figure_filename"]))
            else:
                plotter.show(auto_close=True)
        finally:
            plotter.close()
