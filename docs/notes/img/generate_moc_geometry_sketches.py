"""Regenerate the SVG sketches embedded in the Cartesian/stream-surface notes.

Run with the project's Python environment. Only Matplotlib and NumPy are needed.
The figures are schematic: their geometry is illustrative, not a blade design.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon
import numpy as np


OUTPUT = Path(__file__).resolve().parent
INK = "#172b3a"
MUTED = "#526774"
BLUE = "#0072b2"
ORANGE = "#c66b00"
TEAL = "#007f73"
GREY = "#a5b4bd"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def canvas(title, subtitle, height=5.4):
    fig = plt.figure(figsize=(11, height))
    fig.text(0.045, 0.945, title, fontsize=18, weight="bold")
    fig.text(0.045, 0.895, subtitle, fontsize=11, color=MUTED)
    return fig


def plane(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def arrow(ax, start, end, color=INK, width=1.7, style="-|>", **kwargs):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=width,
        color=color,
        shrinkA=0,
        shrinkB=0,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def ray(ax, origin, angle, length, color, **kwargs):
    end = np.asarray(origin) + length * np.array(
        [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
    )
    arrow(ax, origin, end, color=color, **kwargs)
    return end


def angle_arc(ax, origin, radius, start, end, color, label, label_radius=None):
    ax.add_patch(
        Arc(origin, 2 * radius, 2 * radius, theta1=start, theta2=end,
            linewidth=1.7, color=color)
    )
    middle = np.deg2rad((start + end) / 2)
    distance = label_radius if label_radius is not None else radius + 0.14
    ax.text(
        origin[0] + distance * np.cos(middle),
        origin[1] + distance * np.sin(middle),
        label, ha="center", va="center", color=color, fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )


def save(fig, name):
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def local_angles():
    fig = canvas(
        "Local flow angle and characteristic families",
        "Angles are measured in the tangent plane. The conformal plane preserves these angles.",
    )
    ax = plane(fig, [0.045, 0.12, 0.56, 0.68], (-1.4, 3.7), (-1.4, 2.8))
    origin = (0, 0)
    arrow(ax, (-0.55, 0), (3.5, 0), color=GREY, width=1)
    arrow(ax, (0, -1.15), (0, 2.55), color=GREY, width=1)
    ax.text(3.5, -0.25, r"$+\xi$ / meridional", ha="right", color=MUTED)
    ax.text(-0.1, 2.67, r"$+\eta$ / circumferential", ha="left", color=MUTED)
    flow = ray(ax, origin, 20, 3, TEAL, width=2.6)
    plus = ray(ax, origin, 55, 2.7, BLUE, width=2.2)
    minus = ray(ax, origin, -15, 3, ORANGE, width=2.2)
    normal = ray(ax, origin, 110, 1.8, MUTED, width=1.4, linestyle="--")
    ax.text(flow[0] + 0.05, flow[1] + 0.12, r"$\mathbf{v}=v\,\mathbf{e}_s$", color=TEAL)
    ax.text(plus[0] + 0.04, plus[1] + 0.07, r"$C_+$", color=BLUE, fontsize=15)
    ax.text(minus[0] + 0.08, minus[1] - 0.08, r"$C_-$", color=ORANGE, fontsize=15)
    ax.text(normal[0] - 0.12, normal[1] + 0.06, r"$\mathbf{e}_n$", color=MUTED)
    ax.plot(*origin, "o", color=INK, markersize=5)
    ax.text(-0.15, -0.25, r"$P$", fontsize=12)
    angle_arc(ax, origin, 0.78, 0, 20, TEAL, r"$\alpha$", 1.04)
    angle_arc(ax, origin, 1.36, 20, 55, BLUE, r"$\mu$", 1.59)
    angle_arc(ax, origin, 1.93, -15, 20, ORANGE, r"$\mu$", 2.13)
    fig.text(0.66, 0.74, "Convention used in the notes", weight="bold", fontsize=12)
    fig.text(0.66, 0.65, r"$C_+:\quad \alpha+\mu$", fontsize=16, color=BLUE)
    fig.text(0.66, 0.57, r"$C_-:\quad \alpha-\mu$", fontsize=16, color=ORANGE)
    fig.text(0.66, 0.44, "Planar compatibility", weight="bold", fontsize=12)
    fig.text(0.66, 0.36, r"$J_+=\alpha-\nu=\mathrm{constant}$", color=BLUE, fontsize=13)
    fig.text(0.66, 0.28, r"$J_-=\alpha+\nu=\mathrm{constant}$", color=ORANGE, fontsize=13)
    fig.text(0.66, 0.15, "On the stream sheet, these quantities\nchange through the geometric sources.", color=MUTED)
    fig.text(0.045, 0.045, "Only local tangents are shown; complete characteristics generally curve.", color=MUTED, fontsize=10)
    save(fig, "moc_local_angles.svg")


def meridional_geometry():
    fig = canvas(
        "Axisymmetric passage and a thin stream sheet",
        "Meridional section at fixed circumferential angle: rotate about the z-axis to form the annulus.",
        height=5.8,
    )
    ax = plane(fig, [0.04, 0.16, 0.67, 0.65], (-0.2, 7.4), (-0.2, 4.8))
    z = np.linspace(0.25, 6.6, 250)
    centre = 2.35 + 0.65 * np.tanh((z - 3.5) / 1.7)
    lower, upper = centre - 0.78, centre + 0.9
    ax.fill_between(z, lower - 0.28, lower, color="#e8edf0")
    ax.fill_between(z, upper, upper + 0.28, color="#e8edf0")
    ax.plot(z, lower, color=MUTED, linewidth=2)
    ax.plot(z, upper, color=MUTED, linewidth=2)
    slopes = 0.65 / 1.7 / np.cosh((z - 3.5) / 1.7) ** 2
    normals = np.stack((-slopes, np.ones_like(slopes))) / np.sqrt(1 + slopes**2)
    sheet = np.stack((z, centre))
    bounds_lower = sheet - 0.2 * normals
    bounds_upper = sheet + 0.2 * normals
    ax.add_patch(Polygon(np.concatenate((bounds_lower.T, bounds_upper.T[::-1])),
                         closed=True, facecolor="#d7eeea", edgecolor="none"))
    ax.plot(*bounds_lower, color=TEAL, linewidth=1, linestyle="--")
    ax.plot(*bounds_upper, color=TEAL, linewidth=1, linestyle="--")
    ax.plot(z, centre, color=BLUE, linewidth=2.5)
    ax.text(6.72, upper[-1], "Shroud", va="center", fontsize=10, color=MUTED)
    ax.text(6.72, lower[-1], "Hub", va="center", fontsize=10, color=MUTED)
    ax.text(5.75, centre[-1] + 0.23, "Sheet", color=BLUE)
    arrow(ax, (0, 0), (7.15, 0), color=MUTED, width=1.2)
    arrow(ax, (0, 0), (0, 4.5), color=MUTED, width=1.2)
    ax.text(7.2, -0.1, r"$z$", fontsize=14)
    ax.text(-0.15, 4.52, r"$r$", fontsize=14)
    ax.text(3.4, -0.22, "Machine axis", color=MUTED, ha="center", fontsize=10)
    point_z = 3.2
    point_r = 2.35 + 0.65 * np.tanh((point_z - 3.5) / 1.7)
    slope = 0.65 / 1.7 / np.cosh((point_z - 3.5) / 1.7) ** 2
    tangent = np.array([1, slope]) / np.hypot(1, slope)
    normal = np.array([-slope, 1]) / np.hypot(1, slope)
    point = np.array([point_z, point_r])
    ax.plot(*point, "o", color=BLUE, markersize=5)
    arrow(ax, point, point + 1.05 * tangent, color=BLUE, width=2)
    ax.text(4.15, 2.12, r"$+m,\ \mathbf{e}_m$", color=BLUE)
    arrow(ax, (point_z, 0.08), (point_z, point_r - 0.06), color=MUTED, style="<->", width=1.1)
    ax.text(point_z - 0.12, 1.03, r"$r(m)$", color=MUTED, ha="right")
    ax.plot([point_z, point_z], [0, point_r], color=GREY, linestyle=":", linewidth=1)
    arrow(ax, point - 0.2 * normal,
          point + 0.2 * normal, color=TEAL, style="<->", width=1.5)
    ax.annotate(r"$b(m)$: normal spacing", xy=point + 0.1 * normal,
                xytext=(1.9, 4.25), color=TEAL,
                arrowprops={"arrowstyle": "-", "color": TEAL, "linewidth": 1})
    arrow(ax, (0.25, 0.53), (point_z, 0.53), style="<->", color=MUTED, width=1)
    ax.text(1.72, 0.65, r"$z(m)-z(m_0)$", color=MUTED, ha="center", fontsize=10)
    ax.text(0.26, centre[0] - 0.43, r"$m_0$", color=BLUE)

    inset = plane(fig, [0.76, 0.43, 0.21, 0.32], (-1.1, 1.1), (-1.1, 1.1))
    origin = (0, 0)
    inset.add_patch(Arc(origin, 1.7, 1.7, theta1=0, theta2=360, color=GREY, linewidth=1))
    arrow(inset, (-1, 0), (1, 0), color=GREY, width=1)
    arrow(inset, (0, -1), (0, 1), color=GREY, width=1)
    p = ray(inset, origin, 40, 0.85, BLUE, width=1.8)
    tangent = np.array([-np.sin(np.deg2rad(40)), np.cos(np.deg2rad(40))])
    arrow(inset, p, p + 0.46 * tangent, color=TEAL, width=1.8)
    angle_arc(inset, origin, 0.42, 0, 40, ORANGE, r"$\theta$", 0.6)
    inset.text(0.92, -0.13, r"$x$", color=MUTED)
    inset.text(-0.15, 0.97, r"$y$", color=MUTED)
    inset.text(p[0] - 0.03, p[1] + 0.4, r"$+\mathbf{e}_\theta$", color=TEAL, fontsize=10)
    fig.text(0.755, 0.78, "View from the +z side", fontsize=10, weight="bold")
    fig.text(0.75, 0.29, "The dashed lines bound one\nthin stream sheet within\nthe full hub-to-shroud span.", fontsize=10, color=MUTED)
    fig.text(0.045, 0.055, "Thickness is exaggerated. Circumferential blade walls are resolved on the sheet, not in this section.", fontsize=10, color=MUTED)
    save(fig, "moc_axisymmetric_stream_sheet.svg")


def conformal_mapping():
    fig = canvas(
        "Surface of revolution and its conformal coordinates",
        "The mapping preserves local angles; the physical length scale is the local machine radius.",
        height=5.8,
    )
    ax = plane(fig, [0.035, 0.21, 0.46, 0.61], (-0.55, 5.35), (-2.2, 2.25))
    z = np.linspace(0, 4.25, 120)
    radius = 1 + 0.12 * z + 0.1 * np.sin(0.9 * z)

    def project(zp, rp, theta):
        return np.array([zp + 0.27 * rp * np.cos(theta),
                         rp * np.sin(theta) + 0.22 * rp * np.cos(theta)])

    # Oblique projection of the complete surface: circumferential rings and
    # meridional generators remain actual curves on the surface of revolution.
    theta = np.linspace(0, 2 * np.pi, 181)
    silhouette = np.concatenate([
        project(z, radius, np.full_like(z, np.pi / 2)).T,
        project(z[::-1], radius[::-1], np.full_like(z, -np.pi / 2)).T,
    ])
    ax.add_patch(Polygon(silhouette, color="#edf6fa", zorder=0))
    for index in (0, 23, 46, 69, 92, 119):
        curve = project(np.full_like(theta, z[index]), radius[index], theta)
        ax.plot(*curve, color=GREY, linewidth=0.8)
    for theta0 in np.linspace(0, 2 * np.pi, 9)[:-1]:
        curve = project(z, radius, np.full_like(z, theta0))
        ax.plot(*curve, color=GREY, linewidth=0.8)
    selected_theta = np.deg2rad(35)
    meridian = project(z, radius, np.full_like(z, selected_theta))
    ax.plot(*meridian, color=BLUE, linewidth=2.1)
    arrow(ax, meridian[:, 43], meridian[:, 68], color=BLUE, width=2)
    ax.text(2.62, 1.24, r"$+m$", color=BLUE)
    ring_theta = np.linspace(np.deg2rad(-45), np.deg2rad(28), 60)
    ring = project(np.full_like(ring_theta, z[85]), radius[85], ring_theta)
    ax.plot(*ring, color=TEAL, linewidth=2.1)
    arrow(ax, ring[:, -7], ring[:, -1], color=TEAL, width=2)
    ax.text(3.94, 0.1, r"$+\theta$", color=TEAL)
    arrow(ax, (-0.25, 0), (4.9, 0), color=MUTED, linestyle="--", width=1)
    ax.text(4.9, -0.25, r"$z$", color=MUTED)
    ax.text(-0.15, 1.92, "Physical stream surface", weight="bold", fontsize=11)
    ax.text(0.35, -1.98, r"$\mathbf{X}=(r\cos\theta,\ r\sin\theta,\ z)$", fontsize=11)

    mapped = plane(fig, [0.59, 0.21, 0.36, 0.6], (-0.45, 3.5), (-0.45, 2.8))
    for xi in np.linspace(0, 3, 6):
        mapped.plot([xi, xi], [0, 2.35], color="#d5dfe5", linewidth=0.9)
    for eta in np.linspace(0, 2.35, 6):
        mapped.plot([0, 3], [eta, eta], color="#d5dfe5", linewidth=0.9)
    arrow(mapped, (0, 0), (3.35, 0), color=MUTED, width=1.2)
    arrow(mapped, (0, 0), (0, 2.65), color=MUTED, width=1.2)
    mapped.text(3.27, -0.3, r"$\xi=m'$", fontsize=12, ha="right")
    mapped.text(-0.38, 2.66, r"$\eta=\theta$", fontsize=12)
    mapped.text(0.5, 2.62, "Conformal plane", weight="bold", fontsize=11)
    arrow(mapped, (0.15, 1.15), (2.82, 1.15), color=BLUE, width=2)
    mapped.text(2.86, 1.22, r"$+m$", color=BLUE, fontsize=11)
    arrow(mapped, (1.95, 0.15), (1.95, 2.17), color=TEAL, width=2)
    mapped.text(2.05, 2.1, r"$+\theta$", color=TEAL, fontsize=11)
    mapped.add_patch(Polygon([(0.45, 0.23), (1.05, 0.23), (1.05, 0.83), (0.45, 0.83)],
                            closed=True, facecolor="#fff3df", edgecolor=ORANGE, linewidth=1.5))
    mapped.text(0.75, 0.09, r"$d\xi$", color=ORANGE, ha="center", fontsize=10)
    mapped.text(0.32, 0.53, r"$d\eta$", color=ORANGE, ha="right", va="center", fontsize=10)
    fig.text(0.505, 0.61, r"$\longrightarrow$", fontsize=26, color=MUTED)
    fig.text(0.505, 0.51, "Map", ha="left", color=MUTED, fontsize=10)
    fig.text(0.045, 0.15, r"$\xi=\int_{m_0}^{m}d\bar m/r(\bar m),\qquad \eta=\theta$", fontsize=13)
    fig.text(0.59, 0.15, r"$d\ell_m=r\,d\xi,\qquad d\ell_\theta=r\,d\eta$", fontsize=13)
    fig.text(0.045, 0.06, "The square represents an infinitesimal patch: both physical directions have metric factor r.", color=MUTED, fontsize=10)
    save(fig, "moc_conformal_coordinates.svg")


def marching_limit():
    fig = canvas(
        "Supersonic flow and the meridional marching limit",
        "The characteristic cone, rather than total Mach number alone, controls a march in increasing ξ.",
        height=5.7,
    )
    for rect, alpha, subtitle in [
        ([0.05, 0.24, 0.4, 0.52], 20, "Both families advance in ξ"),
        ([0.55, 0.24, 0.4, 0.52], 70, "C₊ points toward decreasing ξ"),
    ]:
        ax = plane(fig, rect, (-1.45, 2.45), (-0.85, 2.5))
        ax.axvspan(-1.45, 0, facecolor="#f2f4f6")
        ax.axvline(0, color=GREY, linestyle="--", linewidth=1)
        arrow(ax, (-1.25, 0), (2.2, 0), color=GREY, width=1.1)
        ax.text(2.25, -0.17, r"$+\xi$", color=MUTED)
        ax.text(-1.32, 2.22, r"$\Delta\xi<0$", fontsize=10, color=MUTED)
        origin = (0, 0)
        p = ray(ax, origin, alpha + 30, 2.13, BLUE, width=2.1)
        n = ray(ax, origin, alpha - 30, 2.13, ORANGE, width=2.1)
        v = ray(ax, origin, alpha, 1.87, TEAL, width=2.5)
        ax.text(p[0] - 0.1, p[1] + 0.12, r"$C_+$", fontsize=13, color=BLUE)
        ax.text(n[0] + 0.07, n[1] - 0.05, r"$C_-$", fontsize=13, color=ORANGE)
        ax.text(v[0] + 0.03, v[1] + 0.04, r"$v$", fontsize=13, color=TEAL)
        angle_arc(ax, origin, 0.7, 0, alpha, TEAL, r"$\alpha$", 0.9)
        ax.plot(*origin, "o", color=INK, markersize=5)
        ax.set_title(subtitle, fontsize=12, color=INK, pad=12)
    fig.text(0.085, 0.19, r"$M=2,\quad\alpha=20^\circ,\quad\mu=30^\circ$", fontsize=13)
    fig.text(0.585, 0.19, r"$M=2,\quad\alpha=70^\circ,\quad\mu=30^\circ$", fontsize=13)
    fig.text(0.085, 0.135, r"$\alpha+\mu=50^\circ<90^\circ$", fontsize=12, color=BLUE)
    fig.text(0.585, 0.135, r"$\alpha+\mu=100^\circ>90^\circ$", fontsize=12, color=BLUE)
    fig.text(0.045, 0.055, r"Both families advance in ξ only when $|\alpha|+\mu<\pi/2$, equivalently $M\cos\alpha>1$ (forward flow).", fontsize=11, color=MUTED)
    save(fig, "moc_meridional_marching_limit.svg")


if __name__ == "__main__":
    local_angles()
    meridional_geometry()
    conformal_mapping()
    marching_limit()
