"""Report figures, drawn at their printed size for an A4 page (182 mm text width)."""

import numpy as np

from .analysis import polydata, radius_at, span_line, unwrapped
from .extraction import FIELDS, ROWS, boundary

COLORS = {"stator_1": "#233c46", "rotor_1": "#c94f4f"}
SHADES = {"stator_1": "#6f8f9c", "rotor_1": "#e07f72"}
SPAN_STYLES = {0.25: ":", 0.5: "-", 0.75: "--"}
TEXT_WIDTH_IN = 7.2


def apply_style():
    import jaxprop as jxp
    import matplotlib.pyplot as plt
    jxp.set_plot_options(grid=False)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.0,
                         "axes.titlelocation": "left"})


class Saver:
    def __init__(self, directory, dpi):
        self.directory, self.dpi = directory, dpi
        directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig, name):
        import matplotlib.pyplot as plt
        fig.savefig(self.directory / f"{name}.png", dpi=self.dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return f"{self.directory.name}/{name}.png"


def cropped(image, pad=12):
    content = np.where(np.any(image[:, :, :3] < 245, axis=2))
    top, bottom = max(content[0].min() - pad, 0), min(content[0].max() + pad, image.shape[0])
    left, right = max(content[1].min() - pad, 0), min(content[1].max() + pad, image.shape[1])
    return image[top:bottom, left:right]


def render_blading(results, passages):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1300))
    plotter.set_background("white")
    z_blades = np.concatenate([results.blade[row].points[:, 2] for row in ROWS])
    margin = 0.25 * (z_blades.max() - z_blades.min())
    window = (-1, 1, -1, 1, z_blades.min() - margin, z_blades.max() + margin)
    half = passages // 2
    for row in ROWS:
        hub = results.hub[row].clip_box(window, invert=False)
        for k in range(-half, passages - half):
            angle = np.degrees(k * results.pitch[row])
            plotter.add_mesh(results.blade[row].rotate_z(angle, point=(0, 0, 0), inplace=False), color=SHADES[row],
                             smooth_shading=True, specular=0.3)
            plotter.add_mesh(hub.rotate_z(angle, point=(0, 0, 0), inplace=False), color="#d9dee0")
    theta = results.theta_ref
    e_r = np.array([np.cos(theta), np.sin(theta), 0.0])
    e_t = np.array([-np.sin(theta), np.cos(theta), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    focus = results.r_mid * e_r + e_z * z_blades.mean()
    plotter.camera.focal_point = focus
    plotter.camera.position = focus + 6 * (z_blades.max() - z_blades.min()) * (0.75 * e_r - 0.45 * e_z + 0.5 * e_t)
    plotter.camera.up = e_r
    plotter.camera.zoom(1.4)
    image = cropped(plotter.screenshot(return_img=True))
    plotter.close()
    return image


def geometry(results, options, save):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.85), gridspec_kw={"width_ratios": [1.1, 1]},
                             constrained_layout=True)
    axes[0].imshow(render_blading(results, options["passages_3d"]))
    axes[0].set_axis_off()
    axes[0].set_title(f"(a) Blading, {options['passages_3d']} passages per row")
    ax = axes[1]
    z_all = []
    for row in ROWS:
        points = results.blade[row].points
        z, r = points[:, 2] * 1e3, np.hypot(points[:, 0], points[:, 1]) * 1e3
        bins = np.digitize(r, np.linspace(r.min(), r.max(), 30))
        groups = np.unique(bins)
        lead = [z[bins == b].min() for b in groups]
        trail = [z[bins == b].max() for b in groups]
        ax.fill_betweenx([r[bins == b].mean() for b in groups], lead, trail, color=SHADES[row], lw=0)
        ax.text(0.5 * (min(lead) + max(trail)), results.r_mid * 1e3, row.split("_")[0].capitalize(), rotation=90,
                ha="center", va="center", color="w", fontsize=8)
        for name in ("inlet", "outlet"):
            z_plane = results.arrays[f"{boundary(results.setup, row, name)}|vertices"][:, 2].mean() * 1e3
            ax.axvline(z_plane, color="#8a9aa1", lw=0.8, ls="--")
        z_all += [results.hub[row].points[:, 2].min() * 1e3, results.hub[row].points[:, 2].max() * 1e3]
    extent = [min(z_all), max(z_all)]
    ax.plot(extent, [results.r_hub * 1e3] * 2, color="k", lw=1.6)
    ax.plot(extent, [results.r_shroud * 1e3] * 2, color="k", lw=1.6)
    for span, style in SPAN_STYLES.items():
        ax.plot(extent, [radius_at(results, span) * 1e3] * 2, style, color="#8a9aa1", lw=0.7)
        ax.text(extent[1] - 1, radius_at(results, span) * 1e3 + 0.15, f"{int(span * 100)} % span", ha="right",
                va="bottom", fontsize=6.5, color="#5b6b72")
    ax.text(results.interface * 1e3 + 1, results.r_shroud * 1e3 + 0.35, "mixing plane", ha="left", fontsize=6.5,
            color="#5b6b72")
    ax.set_xlabel("Axial coordinate z [mm]")
    ax.set_ylabel("Radius r [mm]")
    ax.set_title("(b) Meridional channel")
    ax.set_ylim(results.r_hub * 1e3 - 1.0, results.r_shroud * 1e3 + 1.4)
    return save(fig, "geometry")


class Midspan:
    """Unwrapped midspan cylinders of both rows, repeated over a common circumferential window."""

    def __init__(self, results, pitches):
        self.flats, self.surfaces = {}, {}
        for row in ROWS:
            mesh = results.midspan[row].copy()
            values = {field: mesh.point_data[field] for field in FIELDS}
            parts = results.kinematics.components(mesh.points, values)
            sound = results.model.speed_of_sound(values["absolute-pressure"])
            mesh.point_data["mach"] = (parts["c"] if row == "stator_1" else parts["w"]) / sound
            mesh.point_data["p_ratio"] = values["absolute-pressure"] / results.P0
            self.surfaces[row] = mesh
            flat = unwrapped(mesh, results.theta_ref, results.r_mid)
            flat = flat.compute_derivative(scalars="density", gradient="grad_rho")
            flat.point_data["grad_rho"] = np.linalg.norm(flat.point_data["grad_rho"], axis=1) * 1e3
            self.flats[row] = flat
        self.arc = {row: results.r_mid * results.pitch[row] * 1e3 for row in ROWS}
        centre = float(np.median(self.flats["stator_1"].points[:, 1]))
        self.band = (centre - 0.5 * pitches * self.arc["stator_1"], centre + 0.5 * pitches * self.arc["stator_1"])

    def extent(self, rows):
        return (min(self.flats[row].points[:, 0].min() for row in rows),
                max(self.flats[row].points[:, 0].max() for row in rows))

    def frame(self, ax, rows, xlabel=True):
        ax.set_xlim(*self.extent(rows))
        ax.set_ylim(*self.band)
        ax.set_aspect("equal")
        if xlabel:
            ax.set_xlabel("Axial coordinate z [mm]")
        ax.set_ylabel("Arc r·θ [mm]")

    def draw(self, ax, row, field, **style):
        import matplotlib.tri as mtri
        flat = self.flats[row]
        triangles = flat.faces.reshape(-1, 4)[:, 1:]
        lowest = int(np.floor((self.band[0] - flat.points[:, 1].max()) / self.arc[row]))
        highest = int(np.ceil((self.band[1] - flat.points[:, 1].min()) / self.arc[row]))
        artist = None
        for k in range(lowest, highest + 1):
            tri = mtri.Triangulation(flat.points[:, 0], flat.points[:, 1] + k * self.arc[row], triangles)
            artist = ax.tripcolor(tri, flat.point_data[field], shading="gouraud", rasterized=True, **style)
            if field == "mach":
                ax.tricontour(tri, flat.point_data[field], levels=[1.0], colors="w", linewidths=0.7)
        return artist


def midspan_fields(results, view, save):
    import matplotlib.pyplot as plt
    width = view.extent(ROWS)[1] - view.extent(ROWS)[0]
    panel = 6.1 * (view.band[1] - view.band[0]) / width
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 2 * panel + 0.75), sharex=True, constrained_layout=True)
    mach_max = max(float(np.max(view.flats[row].point_data["mach"])) for row in ROWS)
    for ax, field, label, cmap, limits, title in (
            (axes[0], "mach", "Mach number [-]", "turbo", (0, mach_max),
             "(a) Mach number: stator absolute, rotor relative"),
            (axes[1], "p_ratio", "p/P₀ [-]", "viridis", (0, 1), "(b) Static pressure")):
        for row in ROWS:
            artist = view.draw(ax, row, field, cmap=cmap, vmin=limits[0], vmax=limits[1])
        fig.colorbar(artist, ax=ax, label=label, pad=0.01, fraction=0.035)
        ax.axvline(results.interface * 1e3, color="w", lw=1.0, ls="--")
        view.frame(ax, ROWS, xlabel=ax is axes[1])
        ax.set_title(title, loc="left")
    return save(fig, "midspan_mach_pressure")


def density_gradient(view, percentile, save):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    paths = []
    for row, label in (("stator_1", "(a) Stator, absolute frame"), ("rotor_1", "(b) Rotor, relative frame")):
        span_x = view.extent([row])[1] - view.extent([row])[0]
        fig, ax = plt.subplots(figsize=(0.95 + 3.1 * span_x / (view.band[1] - view.band[0]), 3.45),
                               constrained_layout=True)
        gradient = view.flats[row].point_data["grad_rho"]
        view.flats[row].point_data["grad_rho_plot"] = np.clip(gradient, np.percentile(gradient, 5), None)
        artist = view.draw(ax, row, "grad_rho_plot", cmap="Greys", norm=LogNorm(
            vmin=np.percentile(gradient, 5), vmax=np.percentile(gradient, percentile)))
        fig.colorbar(artist, ax=ax, label=r"$|\nabla\rho|$ [kg m$^{-4}$]", pad=0.02, fraction=0.08)
        view.frame(ax, [row])
        ax.set_title(label, loc="left")
        paths.append(save(fig, f"density_gradient_{row.split('_')[0]}"))
    return paths


def blade_loading(results, save):
    """Blade p/P₀ and isentropic Mach at three spans; returns the figure path and the plotted data."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.9), sharex="col", constrained_layout=True)
    loading = {}
    model, P0 = results.model, results.P0
    for col, row in enumerate(ROWS):
        mesh = results.blade[row]
        z_le, z_te = mesh.points[:, 2].min(), mesh.points[:, 2].max()
        for span, style in SPAN_STYLES.items():
            line = span_line(mesh, radius_at(results, span))
            if line is None:
                continue
            pressure = line.point_data["absolute-pressure"]
            x = (line.points[:, 2] - z_le) / (z_te - z_le)
            if row == "stator_1":
                head = model.head(P0) - model.head(pressure)
            else:
                u = abs(results.omega) * radius_at(results, span)
                head = results.rotor_in["rothalpy"] + 0.5 * u ** 2 - model.head(pressure)
            mach = np.sqrt(2 * np.clip(head, 0, None)) / model.speed_of_sound(pressure)
            for cell in range(line.n_cells):
                order = line.get_cell(cell).point_ids
                axes[0, col].plot(x[order], pressure[order] / P0, style, color=COLORS[row], lw=1.0,
                                  label=f"{int(span * 100)} % span" if cell == 0 else None)
                axes[1, col].plot(x[order], mach[order], style, color=COLORS[row], lw=1.0)
            loading[f"{row}|{span}"] = {"x": x.tolist(), "p": pressure.tolist(), "M_is": mach.tolist()}
        axes[0, col].set_title(f"({'ab'[col]}) {row.split('_')[0].capitalize()} blade", loc="left")
        axes[0, col].legend(fontsize=7, frameon=False)
        axes[1, col].set_xlabel("Axial chord (z − z$_{LE}$)/c$_{ax}$ [-]")
    axes[0, 0].set_ylabel("p/P₀ [-]")
    axes[1, 0].set_ylabel("Isentropic Mach [-]")
    axes[1, 1].set_ylabel("Relative isentropic Mach [-]")
    return save(fig, "blade_loading"), loading


def stator_exit_pitchwise(results, save):
    import matplotlib.pyplot as plt
    mesh = polydata(results.arrays, boundary(results.setup, "stator_1", "outlet"))
    values = {field: mesh.point_data[field] for field in FIELDS}
    parts = results.kinematics.components(mesh.points, values)
    mesh.point_data["mach"] = parts["c"] / results.model.speed_of_sound(values["absolute-pressure"])
    mesh.point_data["alpha"] = np.degrees(np.arctan2(parts["c_t"], parts["c_m"]))
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.3), sharex=True, constrained_layout=True)
    for span, style in SPAN_STYLES.items():
        line = span_line(mesh, radius_at(results, span))
        theta = np.angle(np.exp(1j * (np.arctan2(line.points[:, 1], line.points[:, 0]) - results.theta_ref)))
        position = (theta - theta.min()) / results.pitch["stator_1"]
        order = np.argsort(position)
        for ax, field, scale in ((axes[0], "mach", 1.0), (axes[1], "absolute-pressure", 1 / results.P0),
                                 (axes[2], "alpha", 1.0)):
            ax.plot(position[order], line.point_data[field][order] * scale, style, color=COLORS["stator_1"],
                    lw=1.0, label=f"{int(span * 100)} % span")
    exit_, reference = results.exit_, results.design_stations["Stator exit (1)|absolute"]
    for ax, cfd, moc, label in ((axes[0], exit_["M"], reference["M"], "(a) Mach number [-]"),
                                (axes[1], exit_["p"] / results.P0, reference["p"] / results.P0, "(b) p/P₀ [-]"),
                                (axes[2], exit_["alpha"], reference["angle"], "(c) Flow angle α [deg]")):
        ax.axhline(cfd, color="#e08a3c", lw=1.2, label="CFD mass average")
        ax.axhline(moc, color="#5b6b72", lw=1.2, ls="-.", label="MoC design")
        ax.set_ylabel(label)
    fig.supxlabel("Pitchwise position from the blade wake [-]", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, fontsize=7.5, frameon=False)
    return save(fig, "stator_exit_pitchwise")


def residual_history(results, save):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.95, 2.75), constrained_layout=True)
    for column in results.residuals.columns:
        if column not in ("iter", "time/iter"):
            ax.semilogy(results.residuals["iter"], results.residuals[column], lw=0.9, label=column)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Scaled residual [-]")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    return save(fig, "residuals")


def state_diagram(results, save):
    """Temperature-entropy diagram: saturation dome and the design states of the stage."""
    import CoolProp.CoolProp as CP
    import jaxprop as jxp
    import matplotlib.pyplot as plt
    name = results.state["fluid_name"]
    T_crit = CP.PropsSI("Tcrit", name)
    T_dome = np.linspace(CP.PropsSI("Tmin", name) + 20, T_crit - 0.05, 160)
    s_liquid = [CP.PropsSI("S", "T", T, "Q", 0, name) / 1e3 for T in T_dome]
    s_vapour = [CP.PropsSI("S", "T", T, "Q", 1, name) / 1e3 for T in T_dome]
    fluid = jxp.Fluid(name)
    inlet = fluid.get_state(jxp.PQ_INPUTS, float(results.state["P0"]), float(results.state["Q0"]))
    stations = {item["station"] + "|" + item["frame"]: item for item in
                results.point["design"]["stator"]["stations"] + results.point["design"]["rotor"]["stations"]}
    points = [("0", "stator inlet, total", float(inlet.s), float(inlet.T)),
              ("1", "stator exit", stations["Stator exit (1)|absolute"]["s"], stations["Stator exit (1)|absolute"]["T"]),
              ("2", "rotor exit", stations["Rotor exit (2)|relative"]["s"], stations["Rotor exit (2)|relative"]["T"])]
    rotor = results.rotor_design
    if rotor is not None:
        relative = (fluid.get_state(jxp.PQ_INPUTS, rotor["P0_rel"], rotor["Q0_rel"]) if rotor["Q0_rel"] == rotor["Q0_rel"]
                    else fluid.get_state(jxp.PT_INPUTS, rotor["P0_rel"], rotor["T0_rel"]))
        points.insert(2, ("1r", "rotor relative total", float(relative.s), float(relative.T)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=True)
    for ax, zoom in zip(axes, (False, True)):
        ax.plot(s_liquid, T_dome, "k-", lw=1.0)
        ax.plot(s_vapour, T_dome, "k-", lw=1.0)
        isentrope = [float(inlet.s) / 1e3] * 2
        ax.plot(isentrope, [points[0][3], points[-1][3]], ":", color="#5b6b72", lw=0.9)
        for tag, label, s, T in points:
            ax.plot(s / 1e3, T, "o", color=COLORS["rotor_1"] if tag.startswith(("1r", "2")) else COLORS["stator_1"],
                    ms=4)
            ax.annotate(tag, (s / 1e3, T), textcoords="offset points", xytext=(5, 2), fontsize=7.5)
        ax.set_xlabel("Entropy s [kJ kg$^{-1}$ K$^{-1}$]")
        ax.set_ylabel("Temperature T [K]")
        if zoom:
            s_values = [p[2] / 1e3 for p in points]
            T_values = [p[3] for p in points]
            ax.set_xlim(min(s_values) - 0.08, min(s_values) + 0.45)
            ax.set_ylim(min(T_values) - 8, max(T_values) + 8)
            ax.set_title("(b) Stage states near the saturated-liquid line", loc="left")
        else:
            ax.set_title("(a) Saturation dome and the stage isentrope", loc="left")
    return save(fig, "state_diagram")


def starting_chart(results, save):
    """Isentropic and Kantrowitz contraction limits along the rotor relative isentrope, with the design point."""
    import matplotlib.pyplot as plt
    from turbo_moc.core import rotor_starting
    rotor, check = results.rotor_design, results.starting
    machs = np.linspace(1.05, max(2.0, check["M_inlet"] + 0.3), 14)
    curves = rotor_starting.starting_curves(rotor, machs)
    fig, ax = plt.subplots(figsize=(3.4, 2.75), constrained_layout=True)
    ax.plot(machs, curves[:, 0], "k:", lw=1.1, label="isentropic limit")
    ax.plot(machs, curves[:, 1], "k-", lw=1.1, label="Kantrowitz limit")
    ax.plot(check["M_inlet"], check["throat_ratio"], "o", color=COLORS["rotor_1"], ms=5, label="rotor design")
    ax.annotate(f"margin {check['margin']:.2f}", (check["M_inlet"], check["throat_ratio"]),
                textcoords="offset points", xytext=(6, -12), ha="left", fontsize=7.5)
    ax.set_xlabel("Relative inlet Mach number [-]")
    ax.set_ylabel("Area ratio A$_t$/A$_1$ [-]")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    return save(fig, "starting_chart")


def render_annulus(results, view, field, cmap, limits):
    """Midspan cylinder of both rows repeated over the full annulus, seen obliquely; blade sections are the gaps."""
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=(2000, 1700))
    plotter.set_background("white")
    for row in ROWS:
        sectors = int(round(2 * np.pi / results.pitch[row]))
        for k in range(sectors):
            angle = np.degrees(k * results.pitch[row])
            plotter.add_mesh(view.surfaces[row].rotate_z(angle, point=(0, 0, 0), inplace=False), scalars=field,
                             cmap=cmap, clim=limits, show_scalar_bar=False, lighting=False)
    z = np.concatenate([view.surfaces[row].points[:, 2] for row in ROWS])
    focus = np.array([0.0, 0.0, 0.5 * (z.min() + z.max())])
    plotter.camera.focal_point = focus
    plotter.camera.position = focus + 4.2 * results.r_mid * np.array([0.0, -0.5, -0.87])
    plotter.camera.up = (0.0, 0.0, 1.0)
    image = cropped(plotter.screenshot(return_img=True))
    plotter.close()
    return image


def annulus_fields(results, view, save):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    mach_max = max(float(np.max(view.surfaces[row].point_data["mach"])) for row in ROWS)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.6), constrained_layout=True)
    for ax, field, label, cmap, limits, title in (
            (axes[0], "mach", "Mach number [-]", "turbo", (0, mach_max), "(a) Mach: stator absolute, rotor relative"),
            (axes[1], "p_ratio", "p/P₀ [-]", "viridis", (0, 1), "(b) Static pressure")):
        ax.imshow(render_annulus(results, view, field, cmap, limits))
        ax.set_axis_off()
        ax.set_title(title, loc="left")
        fig.colorbar(ScalarMappable(Normalize(*limits), cmap), ax=ax, label=label, orientation="horizontal",
                     fraction=0.05, pad=0.02)
    return save(fig, "midspan_annulus")


def make_all(results, figures_dir, options):
    """Draw every report figure; returns their relative paths and the blade-loading data."""
    import matplotlib
    matplotlib.use("Agg")
    apply_style()
    save = Saver(figures_dir, options["dpi"])
    view = Midspan(results, options["passages_2d"])
    paths = {"geometry": geometry(results, options, save), "midspan": midspan_fields(results, view, save),
             "schlieren": density_gradient(view, options["schlieren_percentile"], save)}
    paths["annulus"] = annulus_fields(results, view, save)
    paths["loading"], loading = blade_loading(results, save)
    paths["pitchwise"] = stator_exit_pitchwise(results, save)
    if results.residuals is not None:
        paths["residuals"] = residual_history(results, save)
    if results.rotor_design is not None:
        paths["states"] = state_diagram(results, save)
        paths["starting"] = starting_chart(results, save)
    return paths, loading
