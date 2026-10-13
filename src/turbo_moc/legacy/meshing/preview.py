"""Preserved Matplotlib PNG/SVG exporters from the first conformal 3D example.

These optional helpers are not used by the canonical app or meshing pipeline.
They require a live Gmsh mesh with the preserved boundary conventions. The
meridional-channel argument supplies pitch_angle and reference_radius, as in
turbo_moc.geometry.meridional_channel.MeridionalChannel.
"""
from pathlib import Path

import numpy as np

from turbo_moc.legacy.meshing import stator_mesh_2d as msh2
from turbo_moc.legacy.meshing.passage_volume import collect_surface_mesh

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

