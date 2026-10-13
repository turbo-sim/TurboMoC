"""PyVista mesh surface previews and display options."""

import json
from pathlib import Path


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


def view_mesh_pyvista(output_dir, options, *, show=True, save_figure=False):
    """View selected surfaces of one passage and its full row; save a safe preview."""
    from turbo_moc.meshing.mesh_3d import load_mesh_surfaces

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
    # New meshes record the general definition; retain support for older cases.
    definition = summary.get("geometry", summary.get("machine", {}))
    is_linear = definition.get("type") == "linear"
    num_blades = 3 if is_linear else int(definition["num_blades"])
    if num_blades < 1:
        raise ValueError("The saved case must have a positive blade count.")
    selected = load_mesh_surfaces(
        output_dir / "stator_mesh_3d.msh", options["surfaces"]
    )
    meshes = {
        name: pv.PolyData(points, polygons)
        for name, (points, polygons) in selected.items()
    }
    colors = {
        "blade": "orange",
        "hub": "silver",
        "shroud": "lightgrey",
        "inlet": "cornflowerblue",
        "outlet": "mediumseagreen",
        "periodic_left": "plum",
        "periodic_right": "khaki",
    }

    def populate(plotter):
        """Build the same two panels for image export and interactive inspection."""
        for panel, count in enumerate((1, num_blades)):
            plotter.subplot(0, panel)
            plotter.set_background("#f5f3ed")
            title = (
                ("Single blade" if tuple(meshes) == ("blade",) else "Single passage")
                if panel == 0
                else (f"Linear cascade ({count} displayed blades)" if is_linear
                      else f"Full turbine row ({count} blades)")
            )
            plotter.add_text(title, position="upper_left", color="black", font_size=16)
            for index in range(count):
                # Display copies follow the same one-pitch rigid transform as
                # the mesh, including a translation for a linear cascade.
                for name, surface in meshes.items():
                    if is_linear:
                        copy = surface.translate([0, index * definition["pitch_mm"], 0], inplace=False)
                    else:
                        copy = surface.rotate_z(-360.0 * index / num_blades, inplace=False)
                    plotter.add_mesh(
                        copy,
                        color=colors[name],
                        smooth_shading=True,
                        show_edges=options["show_mesh_edges"],
                        edge_color="black",
                        label=name if index == 0 else None,
                    )
            plotter.add_legend()
            plotter.show_axes()
            plotter.show_bounds(
                xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="black"
            )
            plotter.view_isometric()
            plotter.reset_camera()
            plotter.add_text(
                "Drag: rotate | Wheel: zoom | Shift + drag: pan",
                position="lower_left",
                color="black",
                font_size=10,
            )

    # Save the initial view using a separate offscreen window. Closing the
    # interactive window with its X button cannot destroy this screenshot.
    # Keep a distinct filename so Matplotlib's mesh_3d.png remains intact.
    modes = ([True] if save_figure else []) + ([False] if show else [])
    for off_screen in modes:
        plotter = pv.Plotter(
            shape=(1, 2), window_size=options["window_size"], off_screen=off_screen
        )
        try:
            populate(plotter)
            if off_screen:
                plotter.screenshot(str(output_dir / options["figure_filename"]))
            else:
                # Independent cameras; mouse interaction uses the hovered panel.
                plotter.show(auto_close=True)
        finally:
            plotter.close()
