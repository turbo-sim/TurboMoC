"""Conforming section meshing, spanwise deformation, sweep, and 3D export.

Volume assembly, checks and CGNS export live in turbo_moc.meshing.passage_volume
(shared with the app) and are re-exported here.

Reuses the 2D example for blade splines, smooth periodics, inflation layers,
and automatic core grading. Volume assembly and CGNS auditing follow the
ParaBlade mesh_examples/3D and 3D_v2 examples, without a ParaBlade dependency.
"""

import json
from pathlib import Path

import gmsh
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

import examples.gmsh.meshing_source as msh2
import examples.gmsh_3D.geometry_source as geo
import examples.workflow_logging as log
from turbo_moc.meshing.passage_volume import (  # noqa: F401 (re-exported)
    BC_TYPES,
    BOUNDARIES,
    ELEMENTS,
    SURFACES,
    add_cgns_periodicity,
    add_cgns_units,
    assemble_volume,
    boundary_types,
    collect_surface_mesh,
    export_cgns,
    validate_volume,
)

logger = log.get_logger("gmsh_3D.meshing")


# =============================================================================
# 1. Generate and extract a conforming 2D reference section
# =============================================================================


def generate_section(blade, curves, numerics):
    """Use the existing Gmsh spline, inflation, core grading and recombination."""
    gmsh.model.add("conformal_reference_section")
    blade_curves = msh2.add_blade_boundary(blade, numerics["blade_size"])
    surface, boundaries = msh2.add_periodic_passage(
        curves, blade_curves, blade["pitch"], numerics["domain_size"]
    )
    profile = msh2.calculate_inflation_profile(numerics["inflation_layers"])
    msh2.set_mesh_sizes(blade_curves, numerics, profile, blade["pitch"])
    msh2.add_inflation_layers(blade_curves, numerics)
    msh2.configure_recombination(surface, numerics)
    gmsh.model.mesh.generate(1)
    msh2.check_periodic_conformity(boundaries)
    transition = msh2.configure_core_transition(
        blade_curves, numerics, profile, blade["pitch"], use_mesh=True
    )
    gmsh.model.mesh.generate(2)
    msh2.check_periodic_conformity(boundaries)
    xy, cells, edges, local = msh2.collect_fluid_mesh(surface, boundaries)
    slave, master, _ = msh2.collect_periodic_nodes(boundaries, local, xy)
    blocks = {}
    for width, kind in ((3, 2), (4, 3)):
        selected = [cell for cell in cells if len(cell) == width]
        if selected:
            blocks[kind] = np.asarray(selected, dtype=np.int32) - 1
    if not len(xy) or (numerics.get("recombine", True) and 3 not in blocks):
        raise ValueError(
            "The reference section must have fluid cells and requested quadrilaterals."
        )
    gmsh.model.mesh.removeSizeCallback()
    return {
        "xy": xy,
        "cells": blocks,
        "boundaries": {name: values[:, :2] - 1 for name, values in edges.items()},
        "slave": slave - 1,
        "master": master - 1,
        "profile": profile,
        "transition": transition,
    }


# =============================================================================
# 2. Deform the core while prescribing the blade, collar and outer boundaries
# =============================================================================


class CoreDeformation:
    """Positive-weight harmonic core displacement with protected near-wall nodes.

    Every span station retains the reference connectivity. Prescribed collar
    coordinates are scaled with the blade, so smoothing cannot erase layers.
    This is a section-normal collar, not a 3D normal extrusion at a tapered wall.
    """

    def __init__(self, section, numerics):
        self.section = section
        xy = section["xy"]
        wall = np.unique(section["boundaries"]["blade"])
        # Distance to sampled wall nodes conservatively protects extra core rows.
        distance = cKDTree(xy[wall]).query(xy)[0]
        thickness = (
            section["profile"]["stack_thickness"]
            if section["profile"]["enabled"]
            else 0
        )
        protect = 1.5 * thickness + numerics["blade_size"]
        self.collar = np.flatnonzero(distance <= protect)
        outer = np.unique(
            np.concatenate(
                [
                    edges.ravel()
                    for name, edges in section["boundaries"].items()
                    if name != "blade"
                ]
            )
        )
        if np.intersect1d(self.collar, outer).size:
            raise ValueError(
                "Inflation/collar reaches a passage boundary; reduce layers/chord or blade count."
            )
        self.fixed = np.union1d(self.collar, outer)
        self.free = np.setdiff1d(np.arange(len(xy)), self.fixed)
        edges = np.unique(
            np.sort(
                np.concatenate(
                    [
                        np.column_stack(
                            (cells[:, j], cells[:, (j + 1) % cells.shape[1]])
                        )
                        for cells in section["cells"].values()
                        for j in range(cells.shape[1])
                    ]
                ),
                axis=1,
            ),
            axis=0,
        )
        weights = 1 / np.sum((xy[edges[:, 1]] - xy[edges[:, 0]]) ** 2, axis=1)
        adjacency = coo_matrix(
            (
                np.r_[weights, weights],
                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
            ),
            shape=(len(xy), len(xy)),
        ).tocsc()
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        self.factor = (
            splu(laplacian[self.free][:, self.free].tocsc()) if len(self.free) else None
        )
        self.coupling = laplacian[self.free][:, self.fixed]

    def deform(self, scale, reference, target, pitch):
        """Match a new section while keeping opposite periodic nodes paired."""
        warped = self.section["xy"] * scale
        result = warped.copy()
        for name, edges in self.section["boundaries"].items():
            if name != "blade":
                ids = np.unique(edges)
                result[ids] = geo.corresponding_boundary(
                    self.section["xy"][ids], name, reference, target
                )
        prescribed = result[self.fixed] - warped[self.fixed]
        if self.factor is not None:
            result[self.free] += self.factor.solve(-self.coupling @ prescribed)
        if not np.allclose(
            result[self.section["slave"]] - result[self.section["master"]],
            [pitch, 0],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Span deformation lost periodic correspondence.")
        for cells in self.section["cells"].values():
            p = result[cells]
            # Each corner must remain convex, not merely positive in total area.
            a, b = np.roll(p, -1, axis=1) - p, np.roll(p, -2, axis=1) - np.roll(
                p, -1, axis=1
            )
            if np.any(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0] <= 0):
                raise ValueError(
                    "Section deformation inverted a cell; reduce taper or refine the core."
                )
        return result


# =============================================================================
# 3. Python visualization of actual mesh faces; copies are display-only
# =============================================================================




def load_mesh_surfaces(mesh_file, surfaces=BOUNDARIES):
    """Read selected physical boundaries as compact points and polygon arrays."""
    surfaces = tuple(dict.fromkeys(surfaces))
    if not surfaces:
        raise ValueError("Select at least one surface to display.")
    unknown = set(surfaces) - set(BOUNDARIES)
    if unknown:
        raise ValueError(
            f"Unknown surfaces: {', '.join(sorted(unknown))}. "
            f"Available surfaces: {', '.join(BOUNDARIES)}."
        )
    mesh_file = Path(mesh_file)
    if not mesh_file.is_file():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_file}. Run run_mesh_generation.py first."
        )
    if gmsh.isInitialized():
        raise RuntimeError("Finish the current Gmsh session before loading a mesh.")

    gmsh.initialize(["blade_viewer", "-v", "0"], readConfigFiles=False)
    try:
        gmsh.open(str(mesh_file))
        xyz, faces = collect_surface_mesh()
        selected = {}
        for name in surfaces:
            blocks = [block for block in faces[name] if len(block)]
            if not blocks:
                raise ValueError(f"The mesh contains no '{name}' surface elements.")

            # Each surface retains only its own nodes, not the fluid-volume nodes.
            used = np.unique(np.concatenate([block.ravel() for block in blocks]))
            polygons = np.concatenate(
                [
                    np.column_stack(
                        (
                            np.full(len(block), block.shape[1]),
                            np.searchsorted(used, block),
                        )
                    ).ravel()
                    for block in blocks
                ]
            )
            selected[name] = (xyz[used], polygons)
        return selected
    finally:
        gmsh.finalize()


def pyvista_options(options):
    """Validate YAML surface selection and the mesh-line toggle."""
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
    return {"surfaces": surfaces, "show_mesh_edges": show_edges}


def view_mesh_pyvista(output_dir, options, *, show=True, save_figure=False):
    """View selected surfaces of one passage and its full row; save a safe preview."""
    options = pyvista_options(options)
    if not show and not save_figure:
        return
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Install the optional viewer from the repository root: "
            '.venv/Scripts/python.exe -m pip install ".[mesh-viz]"'
        ) from exc

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "mesh_summary.json").read_text(encoding="utf-8"))
    # New meshes record the general definition; retain support for older cases.
    num_blades = int(summary.get("geometry", summary.get("machine", {}))["num_blades"])
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
                else f"Full turbine row ({count} blades)"
            )
            plotter.add_text(title, position="upper_left", color="black", font_size=16)
            for index in range(count):
                # Copies follow the mesh's clockwise periodic rotation about Z.
                for name, surface in meshes.items():
                    rotated = surface.rotate_z(
                        -360.0 * index / num_blades, inplace=False
                    )
                    plotter.add_mesh(
                        rotated,
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
            shape=(1, 2), window_size=(1500, 800), off_screen=off_screen
        )
        try:
            populate(plotter)
            if off_screen:
                plotter.screenshot(str(output_dir / "pyvista_mesh.png"))
            else:
                # Independent cameras; mouse interaction uses the hovered panel.
                plotter.show(auto_close=True)
        finally:
            plotter.close()


def save_mesh_preview(channel, options, output_dir):
    """Matplotlib fallback: actual surface mesh, one or more rotational copies."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xyz, faces = collect_surface_mesh()
    fig = plt.figure(figsize=(10, 7), dpi=options["dpi"])
    ax = fig.add_subplot(111, projection="3d")
    displayed = []
    for copy in range(options["num_blades"]):
        angle = -copy * channel.pitch_angle
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = xyz @ rotation.T
        displayed.append(points)
        for name in ("hub", "blade", "periodic_left", "periodic_right"):
            polygons = [points[cell] for block in faces[name] for cell in block]
            ax.add_collection3d(
                Poly3DCollection(
                    polygons,
                    facecolor="white" if name == "blade" else options["domain_color"],
                    edgecolor=options["mesh_color"],
                    linewidth=0.12,
                    alpha=1,
                    rasterized=options["rasterize_mesh"],
                )
            )
            edges = np.concatenate(
                [
                    np.column_stack((block[:, j], block[:, (j + 1) % block.shape[1]]))
                    for block in faces[name]
                    for j in range(block.shape[1])
                ]
            )
            unique, counts = np.unique(
                np.sort(edges, axis=1), axis=0, return_counts=True
            )
            ax.add_collection3d(
                Line3DCollection(
                    points[unique[counts == 1]],
                    colors=options["edge_color"],
                    linewidths=0.7,
                )
            )
    points = np.vstack(displayed)
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = max(hi - lo)
    for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        middle = (a + b) / 2
        setter(middle - extent / 2, middle + extent / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x coordinate (mm)")
    ax.set_ylabel("y coordinate (mm)")
    ax.set_zlabel("z coordinate (mm)")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.pane.fill = False
    ax.view_init(elev=25, azim=-55)
    fig.tight_layout()
    try:
        for extension in ("png", "svg"):
            fig.savefig(
                Path(output_dir) / f"mesh_3d.{extension}",
                dpi=options["dpi"],
                bbox_inches="tight",
            )
    finally:
        plt.close(fig)


# =============================================================================
# 4. Complete workflow: section -> conformal stack -> XYZ -> export
# =============================================================================


def save_conformal_preview(channel, options, output_dir):
    """Label the reference mesh explicitly in dimensionless conformal coordinates."""
    import matplotlib.pyplot as plt

    data = msh2.collect_mesh_plot_data(1)
    data["xy"] = data["xy"] / channel.reference_radius
    fig, ax = msh2.plot_mesh_figure(
        data, pitch=channel.pitch_angle, options=options | {"num_blades": 1}
    )
    ax.set_xlabel(r"$\theta$ (rad, clockwise)", color="black")
    ax.set_ylabel(r"$-m'$", color="black")
    try:
        for extension in ("png", "svg"):
            fig.savefig(
                Path(output_dir) / f"conformal_mesh.{extension}",
                dpi=options["dpi"],
                bbox_inches="tight",
            )
    finally:
        plt.close(fig)


def create_mesh(
    nozzle,
    config,
    output_dir,
    *,
    geometry,
    save_figures=True,
    show_gmsh=False,
):
    """Mesh a common geometry definition, validate it, and export one passage."""
    if gmsh.isInitialized():
        raise RuntimeError(
            "Finish the existing Gmsh session before generating this mesh."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numerics = config["mesh"]["numerics"]
    options = msh2.mesh_figure_options(config["mesh"].get("matplotlib", {}))
    channel = geo.Channel(geometry)
    if options["num_blades"] > channel.num_blades:
        raise ValueError("Display blade count cannot exceed the physical num_blades.")
    eta = geo.span_stations(numerics)
    logger.info("  Fit the MoC blade and center a one-pitch conformal passage.")
    blade, curves = geo.build_section(nozzle, config, channel)
    gmsh.initialize(["moc_3d", "-v", str(numerics.get("verbosity", 2))])
    try:
        gmsh.option.setNumber("General.Terminal", int(numerics["terminal_output"]))
        logger.info(
            "  Mesh the reference plane with inflation layers and matching periodics."
        )
        section = generate_section(blade, curves, numerics)
        if save_figures:
            save_conformal_preview(channel, options, output_dir)
        logger.info("  Sweep corresponding sections into the physical 3D channel.")
        # A changing conformal chord requires core deformation, regardless of
        # the machine type. Reuse the reference plane whenever its scale fits.
        deformation = None
        mapped = []
        for value in eta:
            scale = channel.section_scale(value)
            if np.isclose(scale, 1, rtol=1e-12, atol=1e-12):
                q = section["xy"]
            else:
                if deformation is None:
                    deformation = CoreDeformation(section, numerics)
                _, target = geo.build_section(nozzle, config, channel, value)
                q = deformation.deform(scale, curves, target, channel.pitch)
            mapped.append(channel.map_points(q, value))
        xyz_sections = np.asarray(mapped)
        counts = assemble_volume(section, xyz_sections, channel)
        logger.info(
            "  Check cell Jacobians, exterior faces and rotational periodicity."
        )
        validation = validate_volume(section, xyz_sections, channel)
        result = {
            "machine": config.get("machine", {}),
            "geometry": geometry.to_dict(),
            "reference_radius_mm": channel.reference_radius,
            "reference_conformal_chord_mm": channel.reference_chord,
            "span_stations": eta.tolist(),
            "section_nodes": len(section["xy"]),
            "volume_nodes": int(xyz_sections.shape[0] * xyz_sections.shape[1]),
            "cells": counts,
            "quality": validation,
            "inflation_profile": section["profile"],
            "core_transition": section["transition"],
            "periodic_rotation_left_to_right": channel.rotation().tolist(),
        }
        gmsh.write(str(output_dir / "stator_mesh_3d.msh"))
        cgns = config["mesh"]["cgns"]
        if cgns["enabled"]:
            result["cgns"] = export_cgns(output_dir / cgns["filename"], cgns, channel)
        (output_dir / "mesh_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if save_figures:
            logger.info(
                "  Plot the actual 3D surface mesh; additional blades are rotational copies."
            )
            save_mesh_preview(channel, options, output_dir)
        if show_gmsh and config["mesh"]["show_gui"]:
            gmsh.fltk.run()
        log.summary(
            logger,
            "3D mesh",
            {
                "Volume cells": sum(counts.values()),
                "Minimum SICN": f"{validation['min_sicn']:.4f}",
                "Periodic node pairs": validation["periodic_node_pairs"],
                "Output": log.display_path(output_dir),
            },
        )
        return result
    finally:
        gmsh.model.mesh.removeSizeCallback()
        gmsh.finalize()
