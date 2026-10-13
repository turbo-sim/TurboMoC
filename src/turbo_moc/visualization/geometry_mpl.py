"""Matplotlib views of blade and passage construction."""

import matplotlib.pyplot as plt

import turbo_moc


def plot_flow_domain(ax, domain):
    """Plot boundaries, reference sections, and extension control polygons."""
    for key in ("periodic_left", "periodic_right"):
        ax.plot(*domain[key].T, color="tab:purple", lw=1.5,
                label="Smooth periodic boundaries" if key.endswith("left") else None)
    for key, color in (("inlet", "tab:green"), ("outlet", "tab:orange")):
        ax.plot(*domain[key].T, color=color, lw=2, label=key.capitalize())
    for key, color, label in (
        ("convergent", "tab:red", "Integrated convergent reference"),
        ("divergent", "tab:blue", "MoC axis direction"),
        ("inlet_extension", "0.4", "Cubic B-spline extensions"),
        ("outlet_extension", "0.4", None),
    ):
        ax.plot(*domain[key].T, color=color, ls="--", lw=1.5, label=label)
    for key in ("inlet_control_points", "outlet_control_points"):
        ax.plot(*domain[key].T, ":o", color="0.6", ms=4,
                label="Extension control polygon" if key.startswith("inlet") else None)
    ax.plot(*domain["convergent"][0], "o", color="tab:red", ms=5, label="Reference start")


def plot_construction(blade, curves, *, figure_size=(14, 8)):
    """Overlay construction features on the package's Matplotlib blade plots."""
    fig, axes = plt.subplots(1, 2, figsize=figure_size, layout="constrained")
    detail, domain = axes
    for ax in axes:
        turbo_moc.mpl.plot_blade(
            blade, ax=ax, n_blades=1,
            show_control_points=ax is detail, show_cp_labels=False,
        )
        ax.set_xlabel("Pitchwise x (mm)")
        ax.set_ylabel("Axial y (mm); flow toward decreasing y")
        for spine in ax.spines.values():
            spine.set_visible(True)

    for name, color in (("suction", "tab:green"), ("pressure", "tab:orange")):
        detail.plot(*curves[name].T, color=color, lw=1, label=f"Raw {name} surface")
    detail.plot([], [], "--o", color=turbo_moc.mpl.COLOR_CP_LINE,
                mfc=turbo_moc.mpl.COLOR_CP_FACE, mec="black", ms=6,
                label="B-spline control polygon")
    camber = curves["construction_camberline"]
    detail.plot(*camber.T, color="tab:red", lw=3, ls="--",
                label="Construction camberline (convergent suction)")
    detail.plot(*camber[[0, -1]].T, "o", color="tab:red", ms=5)
    detail.set_title("Blade construction")

    draw_domain(blade, curves, domain)
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
        ax.margins(x=0.22, y=0.12)
    return fig, axes


def draw_domain(blade, curves, ax):
    """Draw the selected domain and report numerical side clearances."""
    from turbo_moc.geometry.passage_2d import assess_clearance

    plot_flow_domain(ax, curves)
    report = assess_clearance(blade, curves, clearance_samples=400)
    status = "" if report["contained"] else "\nBlade crosses a periodic boundary"
    ax.set_title(
        f"Smooth passage ({curves['domain_shift_mode']}); shift = {curves['shift_mm']:+.2f} mm\n"
        f"Clearance left/right: {report['left_clearance']:.2f} / {report['right_clearance']:.2f} mm{status}",
        color="black" if report["contained"] else "tab:red", fontsize=10,
    )
