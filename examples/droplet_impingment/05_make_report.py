"""Paper-3 stage report: MoC stator + TurboDash impulse rotor, 3D barotropic RANS against the TurboDash meanline.

python 05_make_report.py
Needs 04_extract_results.py output in <run>/report/data. Writes report.md/.html/.pdf and figures to
settings.REPORT_DIR.
The focus is the flow the droplets meet: spanwise rotor-inlet kinematics, wetness and the impact angle of
droplets that lag the vapour.
"""
import json
import re

import numpy as np
import yaml

from turbo_moc.fluent.barotropic_model import BarotropicClosure
from turbo_moc.fluent.stage import document
from turbo_moc.fluent.stage.figures import COLORS, TEXT_WIDTH_IN, Saver, apply_style

from settings import REPORT_DIR as REPORT, ROTOR_DESIGN as TURBODASH, RUN_DIR as RUN

DATA = RUN / "report" / "data"
BLADES = {"stator_1": 44, "rotor_1": 120}
SPANS = (0.1, 0.5, 0.9)
SPAN_STYLES = {0.1: ":", 0.5: "-", 0.9: "--"}
METAL = {"stator_out": 72.0, "rotor_in": 52.00396384057661, "rotor_out": -52.003963840435446}


class Flow:
    """Cylindrical components about +x: c_t and w_t positive along the rotation (and the stator turning)."""

    def __init__(self, omega, model):
        self.omega, self.model = omega, model

    def __call__(self, xyz, values):
        theta = np.arctan2(xyz[:, 2], xyz[:, 1])
        radius = np.hypot(xyz[:, 1], xyz[:, 2])
        e_t = np.column_stack([-np.sin(theta), np.cos(theta)])
        c = np.column_stack([values["x-velocity"], values["y-velocity"], values["z-velocity"]])
        w = np.column_stack([values["relative-x-velocity"], values["relative-y-velocity"],
                             values["relative-z-velocity"]])
        p = values["absolute-pressure"]
        sound = self.model.speed_of_sound(p)
        return {"r": radius, "theta": theta, "p": p, "c_m": c[:, 0], "c_t": np.sum(c[:, 1:] * e_t, axis=1),
                "w_t": np.sum(w[:, 1:] * e_t, axis=1), "u": self.omega * radius,
                "c": np.linalg.norm(c, axis=1), "w": np.linalg.norm(w, axis=1), "a": sound,
                "M": np.linalg.norm(c, axis=1) / sound, "M_rel": np.linalg.norm(w, axis=1) / sound,
                "x": self.model.evaluate("barotropic_vapor_quality", p),
                "alpha_void": self.model.evaluate("barotropic_void_fraction", p),
                "H0": self.model.head(p) + 0.5 * np.sum(c ** 2, axis=1),
                "k": values["turb-kinetic-energy"]}


def face_values(arrays, surface, flow):
    values = {key.split("|")[-1]: arrays[key] for key in arrays if key.startswith(f"{surface}|face|")}
    parts = flow(arrays[f"{surface}|centroids"], values)
    velocity = np.column_stack([values["x-velocity"], values["y-velocity"], values["z-velocity"]])
    parts["flux"] = values["density"] * np.sum(velocity * arrays[f"{surface}|normals"], axis=1)
    return parts


def mass_average(parts, mask=None):
    weight = parts["flux"] if mask is None else parts["flux"] * mask
    weight = weight / weight.sum()
    result = {key: float(np.sum(weight * parts[key])) for key in
              ("p", "c_m", "c_t", "w_t", "u", "c", "w", "M", "M_rel", "x", "alpha_void", "H0")}
    result["r_c_t"] = float(np.sum(weight * parts["r"] * parts["c_t"]))
    result["kinetic"] = float(np.sum(weight * 0.5 * parts["c"] ** 2))
    result["alpha"] = float(np.degrees(np.arctan2(result["c_t"], result["c_m"])))
    result["beta"] = float(np.degrees(np.arctan2(result["w_t"], result["c_m"])))
    result["mass_flow"] = float(abs(np.sum(parts["flux"] if mask is None else parts["flux"] * mask)))
    return result


def span_profile(parts, r_hub, r_shroud, bins=24):
    """Mass averages in radial bands; bands with no face centroid (clustered TurboGrid layers) are skipped."""
    edges = np.linspace(r_hub, r_shroud, bins + 1)
    bands = [(parts["r"] >= low) & (parts["r"] < high) for low, high in zip(edges[:-1], edges[1:])]
    rows = [mass_average(parts, band) | {"span": float(np.sum(parts["flux"] * band * (parts["r"] - r_hub))
                                                       / np.sum(parts["flux"] * band) / (r_shroud - r_hub))}
            for band in bands if band.any()]
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def polydata(arrays, surface):
    import pyvista as pv
    mesh = pv.PolyData(arrays[f"{surface}|vertices"], arrays[f"{surface}|faces"])
    for key in arrays:
        if key.startswith(f"{surface}|node|"):
            mesh.point_data[key.split("|")[-1]] = arrays[key]
    return mesh.triangulate()


def residuals():
    """Iteration history of the last run segment (the transcript restarts at 1 after each initialisation)."""
    pattern = re.compile(r"^\s*(\d+)\s+((?:[-+0-9.]+e[-+]\d+\s+){6})")
    history = {}
    for path in [*sorted(RUN.glob("*.trn"), key=lambda item: item.stat().st_mtime), *RUN.glob("save_watch.log")]:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.match(line)
            if match:
                iteration = int(match.group(1))
                if iteration == 1:
                    history = {}
                history[iteration] = [float(value) for value in match.group(2).split()]
    iterations = np.array(sorted(history))
    return iterations, np.array([history[i] for i in iterations])


def compute():
    setup = json.loads((RUN / "setup_summary.json").read_text(encoding="utf-8"))
    reports = json.loads((DATA / "fluent_reports.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(DATA / "surfaces.npz"))
    state = setup["inlet_state"]
    model = BarotropicClosure.from_file(RUN / "barotropy" / "fluent_expressions_safe.json",
                                        float(state["p_min"]), float(state["P0"]))
    omega = float(setup["rotor_omega_rad_s"])
    flow = Flow(omega, model)
    r_hub, r_shroud = reports["radii_m"]
    faces = {name: face_values(arrays, name, flow) for name in
             ("stator_1_inlet", "stator_1_outlet", "rotor_1_inlet", "rotor_1_outlet")}
    stations = {name: mass_average(parts) for name, parts in faces.items()}
    values = reports["values"]
    flows = {name: abs(values[f"report_mass_flow({name})"]) * BLADES[name.rsplit("_", 1)[0]] for name in faces}
    torque = sum(values[f"report_moment_rotor_1(rotor_1_{wall})"] for wall in ("blade", "hub", "shroud"))
    torque *= BLADES["rotor_1"]
    stator_torque = BLADES["stator_1"] * sum(values[f"report_moment_stator_1(stator_1_{wall})"]
                                             for wall in ("blade", "hub", "shroud"))
    power = abs(torque) * omega
    mass_flow = flows["rotor_1_outlet"]
    work = power / mass_flow
    P0 = float(state["P0"])
    inlet, exit_, rotor_in, outlet = (stations[name] for name in faces)
    drop_ts = model.head(P0) - model.head(outlet["p"])
    euler = omega * (rotor_in["r_c_t"] - outlet["r_c_t"])
    design = yaml.safe_load(TURBODASH.read_text(encoding="utf-8"))
    return dict(setup=setup, reports=reports, arrays=arrays, state=state, model=model, omega=omega, flow=flow,
                r_hub=r_hub, r_shroud=r_shroud, faces=faces, stations=stations, flows=flows, torque=torque,
                stator_torque=stator_torque, power=power, mass_flow=mass_flow, work=work, euler=euler,
                drop_ts=drop_ts, drop_tt=drop_ts - outlet["kinetic"], P0=P0, design=design,
                profiles={name: span_profile(faces[name], r_hub, r_shroud) for name in
                          ("stator_1_outlet", "rotor_1_inlet", "rotor_1_outlet")},
                residuals=residuals())


# ----------------------------------------------------------------------------------------------- figures
def residual_figure(res, save):
    import matplotlib.pyplot as plt
    iterations, history = res["residuals"]
    fig, ax = plt.subplots(figsize=(0.5 * TEXT_WIDTH_IN, 2.3))
    for column, label in enumerate(("continuity", "$u_x$", "$u_y$", "$u_z$", "$k$", r"$\omega$")):
        ax.semilogy(iterations, history[:, column], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Scaled residual")
    ax.set_xlim(iterations[0], iterations[-1])
    ax.legend(ncol=3, fontsize=7, frameon=True, fancybox=False, edgecolor="k")
    return save(fig, "residuals")


def span_figure(res, save):
    """Pitch-averaged spanwise profiles at the stator exit, rotor inlet and rotor exit."""
    import matplotlib.pyplot as plt
    exit_, rin, rout = (res["profiles"][name] for name in ("stator_1_outlet", "rotor_1_inlet", "rotor_1_outlet"))
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN, 2.6), sharey=True, constrained_layout=True)
    stator, rotor = COLORS["stator_1"], COLORS["rotor_1"]
    ax = axes[0]
    ax.plot(exit_["alpha"], exit_["span"], color=stator, label=r"$\alpha_1$")
    ax.plot(rin["beta"], rin["span"], color=rotor, label=r"$\beta_1$")
    ax.plot(rout["beta"], rout["span"], color=rotor, ls="--", label=r"$\beta_2$")
    ax.plot(rout["alpha"], rout["span"], color=stator, ls="--", label=r"$\alpha_2$")
    for value in (METAL["stator_out"], METAL["rotor_in"], METAL["rotor_out"]):
        ax.axvline(value, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("Flow angle [deg]")
    ax.set_ylabel("Span fraction [-]")
    ax = axes[1]
    ax.plot(rin["beta"] - METAL["rotor_in"], rin["span"], color=rotor)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"Incidence $\beta_1-\beta_{1,\mathrm{metal}}$ [deg]")
    ax = axes[2]
    ax.plot(exit_["M"], exit_["span"], color=stator, label=r"$M_1$")
    ax.plot(rin["M_rel"], rin["span"], color=rotor, label=r"$M_\mathrm{rel,1}$")
    ax.plot(rout["M_rel"], rout["span"], color=rotor, ls="--", label=r"$M_\mathrm{rel,2}$")
    ax.set_xlabel("Mach number [-]")
    ax = axes[3]
    ax.plot(1 - exit_["x"], exit_["span"], color=stator, label="stator exit")
    ax.plot(1 - rout["x"], rout["span"], color=rotor, ls="--", label="rotor exit")
    ax.set_xlabel(r"Liquid mass fraction $1-x$ [-]")
    axes[0].set_ylim(0, 1)
    for ax in axes[[0, 2, 3]]:
        ax.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor="k", loc="best")
    return save(fig, "spanwise_profiles")


def unrolled(mesh, r_span, theta_ref, pitch, copies):
    """Constant-radius surface on (x, r theta) in mm, tiled over `copies` passages."""
    theta = np.angle(np.exp(1j * (np.arctan2(mesh.points[:, 2], mesh.points[:, 1]) - theta_ref)))
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    return [(mesh.points[:, 0] * 1e3, r_span * (theta + k * pitch) * 1e3, triangles) for k in copies]


def blade_section(arrays, row, r_span, theta_ref, pitch, copies):
    import pyvista as pv
    mesh = pv.PolyData(arrays[f"{row}_blade|vertices"], arrays[f"{row}_blade|faces"])
    line = mesh.contour([r_span], scalars=np.hypot(mesh.points[:, 1], mesh.points[:, 2])).strip()
    sections = []
    for cell in range(line.n_cells):
        points = line.get_cell(cell).points
        theta = np.angle(np.exp(1j * (np.arctan2(points[:, 2], points[:, 1]) - theta_ref)))
        sections += [(points[:, 0] * 1e3, r_span * (theta + k * pitch) * 1e3) for k in copies]
    return sections


def theta_reference(arrays, row):
    points = arrays[f"{row}_blade|vertices"]
    return float(np.angle(np.mean(np.exp(1j * np.arctan2(points[:, 2], points[:, 1])))))


def blade_to_blade(res, save):
    """Mach number on constant-radius surfaces (absolute in the stator, relative in the rotor)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    arrays, flow = res["arrays"], res["flow"]
    pitch = {row: 2 * np.pi / count for row, count in BLADES.items()}
    copies = {"stator_1": range(-2, 3), "rotor_1": range(-6, 7)}
    limits = (0.0, 1.6)
    fig, axes = plt.subplots(len(SPANS), 1, figsize=(TEXT_WIDTH_IN, 8.4), sharex=True, constrained_layout=True)
    for ax, span in zip(axes, reversed(SPANS)):
        r_span = res["r_hub"] + span * (res["r_shroud"] - res["r_hub"])
        for row, field in (("stator_1", "M"), ("rotor_1", "M_rel")):
            mesh = polydata(arrays, f"{row}_span{int(round(100 * span)):02d}")
            values = {key: mesh.point_data[key] for key in mesh.point_data}
            mach = flow(mesh.points, values)[field]
            ref = theta_reference(arrays, row)
            for x, y, triangles in unrolled(mesh, r_span, ref, pitch[row], copies[row]):
                ax.tripcolor(x, y, triangles, mach, cmap="magma", vmin=limits[0], vmax=limits[1],
                             shading="gouraud", rasterized=True)
            for x, y in blade_section(arrays, row, r_span, ref, pitch[row], copies[row]):
                ax.fill(x, y, color="w", ec="k", lw=0.6, zorder=3)
        ax.set_aspect("equal")
        ax.set_ylim(-70, 30)   # Same window at every span: one stator pitch at the hub, 2.4-2.9 rotor pitches
        ax.set_xlim(-24, 200)
        ax.set_title(f"{100 * span:.0f} % span", fontsize=8.5)
        ax.set_ylabel(r"$r\theta$ [mm]")
    axes[-1].set_xlabel("$x$ [mm]")
    fig.colorbar(ScalarMappable(Normalize(*limits), "magma"), ax=axes, location="right", shrink=0.6,
                 label=r"$M$ (stator), $M_\mathrm{rel}$ (rotor) [-]")
    return save(fig, "blade_to_blade_mach")


def blade_loading(res, save):
    """Static pressure on the blade at three spans against the axial chord fraction."""
    import matplotlib.pyplot as plt
    import pyvista as pv
    arrays = res["arrays"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.6), sharey=True, constrained_layout=True)
    loading = {}
    for ax, row in zip(axes, BLADES):
        mesh = pv.PolyData(arrays[f"{row}_blade|vertices"], arrays[f"{row}_blade|faces"])
        mesh.point_data["p"] = arrays[f"{row}_blade|node|absolute-pressure"]
        x_le, x_te = mesh.points[:, 0].min(), mesh.points[:, 0].max()
        for span in SPANS:
            r_span = res["r_hub"] + span * (res["r_shroud"] - res["r_hub"])
            line = mesh.contour([r_span], scalars=np.hypot(mesh.points[:, 1], mesh.points[:, 2])).strip()
            for cell in range(line.n_cells):   # Ordered polylines around the section
                ids = line.get_cell(cell).point_ids
                ax.plot((line.points[ids, 0] - x_le) / (x_te - x_le), line.point_data["p"][ids] / res["P0"],
                        color=COLORS[row], ls=SPAN_STYLES[span], label=f"{100 * span:.0f} % span" if cell == 0 else None)
            loading[f"{row}|{span}"] = float(np.ptp(line.point_data["p"]) / res["P0"])
        ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="k", ncol=3, loc="lower left")
        ax.set_xlabel(r"$(x-x_\mathrm{LE})/c_\mathrm{ax}$ [-]")
        ax.set_xlim(-0.02, 1.02)
        ax.set_title("Stator" if row == "stator_1" else "Rotor", fontsize=8.5)
    axes[0].set_ylabel(r"$p/P_{0}$ [-]")
    axes[0].set_ylim(0.0, 1.05)
    return save(fig, "blade_loading"), loading


def droplet_kinematics(res):
    """Rotor-relative impact of droplets that leave the stator along the vapour direction at k times its speed."""
    profile = res["profiles"]["stator_1_outlet"]
    slip = np.linspace(0.0, 1.0, 51)
    table = {}
    for span in SPANS:
        c_m, c_t = (np.interp(span, profile["span"], profile[key]) for key in ("c_m", "c_t"))
        u = res["omega"] * (
            res["r_hub"] + span * (res["r_shroud"] - res["r_hub"]))
        beta = np.degrees(np.arctan2(slip * c_t - u, slip * c_m))
        speed = np.hypot(slip * c_m, slip * c_t - u)
        table[span] = (beta, speed)
    return slip, table


def droplet_figure(res, save):
    import matplotlib.pyplot as plt
    slip, table = droplet_kinematics(res)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5), constrained_layout=True)
    for span, (beta, speed) in table.items():
        style = dict(color=COLORS["rotor_1"], ls=SPAN_STYLES[span], label=f"{100 * span:.0f} % span")
        axes[0].plot(slip, beta - METAL["rotor_in"], **style)
        axes[1].plot(slip, speed, **style)
    axes[0].axhline(0, color="0.6", lw=0.6)
    axes[0].set_ylabel(r"$\beta_d-\beta_{1,\mathrm{metal}}$ [deg]")
    axes[1].set_ylabel(r"Relative impact speed $w_d$ [m/s]")
    for ax in axes:
        ax.set_xlabel(r"Droplet-to-vapour speed ratio $c_d/c_1$ [-]")
        ax.set_xlim(0, 1)
    axes[1].legend(fontsize=7, frameon=True, fancybox=False, edgecolor="k")
    return save(fig, "droplet_impact_kinematics"), table


def rotor_cuts(res, save):
    """Relative Mach number and turbulent kinetic energy on axial cuts through the rotor."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    arrays, flow = res["arrays"], res["flow"]
    pitch = 2 * np.pi / BLADES["rotor_1"]
    ref = theta_reference(arrays, "rotor_1")
    names = ("rotor_1_x_le", "rotor_1_x_mid", "rotor_1_x_te")
    titles = ("3 mm upstream of the LE", "Mid axial chord", "3 mm downstream of the TE")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0), sharey=True, constrained_layout=True)
    limits = (0.0, 1.2)
    r_mid = 0.5 * (res["r_hub"] + res["r_shroud"])
    for ax, name, title in zip(axes, names, titles):
        mesh = polydata(arrays, name)
        values = {key: mesh.point_data[key] for key in mesh.point_data}
        mach = flow(mesh.points, values)["M_rel"]
        theta = np.angle(np.exp(1j * (np.arctan2(mesh.points[:, 2], mesh.points[:, 1]) - ref)))
        radius = np.hypot(mesh.points[:, 1], mesh.points[:, 2])
        span = (radius - res["r_hub"]) / (res["r_shroud"] - res["r_hub"])
        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        for k in range(-2, 3):
            ax.tripcolor(r_mid * (theta + k * pitch) * 1e3, span, triangles, mach, cmap="magma",
                         vmin=limits[0], vmax=limits[1], shading="gouraud", rasterized=True)
        ax.set_xlim(-1.5 * pitch * r_mid * 1e3, 1.5 * pitch * r_mid * 1e3)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(r"$r_\mathrm{m}\theta$ [mm]")
    axes[0].set_ylabel("Span fraction [-]")
    fig.colorbar(ScalarMappable(Normalize(*limits), "magma"), ax=axes, location="right", shrink=0.9,
                 label=r"$M_\mathrm{rel}$ [-]")
    return save(fig, "rotor_axial_cuts")


# ----------------------------------------------------------------------------------------------- document
def f(value, spec=".3f"):
    return "–" if value is None else format(value, spec)


def blocks(res, paths, loading, droplets):
    design = res["design"]["overall_performance"]
    st = res["stations"]
    inlet, exit_, rin, rout = (st[name] for name in ("stator_1_inlet", "stator_1_outlet", "rotor_1_inlet",
                                                     "rotor_1_outlet"))
    eta_ts, eta_tt = res["work"] / res["drop_ts"], res["work"] / res["drop_tt"]
    flows = res["flows"]
    imbalance = (flows["stator_1_inlet"] - flows["rotor_1_outlet"]) / flows["stator_1_inlet"]
    stator_ke = exit_["kinetic"] / (res["model"].head(res["P0"]) - res["model"].head(exit_["p"]))
    reaction = (res["model"].head(exit_["p"]) - res["model"].head(rout["p"])) / res["drop_ts"]
    iterations, history = res["residuals"]
    y_plus = {row: res["arrays"][f"{row}_blade|face|y-plus"] for row in BLADES}
    kpis = [("Mass flow rate", "kg/s", res["mass_flow"], design["mass_flow_rate"], ".2f"),
            ("Power", "kW", res["power"] / 1e3, design["power_actual"] / 1e3, ".1f"),
            ("Efficiency, total-to-static", "", eta_ts, design["efficiency_ts"], ".3f"),
            ("Rotor inlet rel. angle", "deg", rin["beta"], METAL["rotor_in"], ".1f")]
    perf = [
        ("Mass flow rate, full annulus", "kg/s", res["mass_flow"], design["mass_flow_rate"], ".2f"),
        ("Mass imbalance, stator inlet − rotor outlet", "%", 100 * imbalance, None, ".3f"),
        ("Torque, full rotor", "N m", abs(res["torque"]), design["shaft_torque"], ".0f"),
        ("Power", "kW", res["power"] / 1e3, design["power_actual"] / 1e3, ".1f"),
        ("Specific work (torque)", "kJ/kg", res["work"] / 1e3, design["power_actual"] / design["mass_flow_rate"] / 1e3,
         ".3f"),
        ("Specific work (Euler, Δ(r c_t))", "kJ/kg", res["euler"] / 1e3, None, ".3f"),
        ("Isentropic drop, total-to-static", "kJ/kg", res["drop_ts"] / 1e3, design["isentropic_enthalpy_drop"] / 1e3,
         ".3f"),
        ("Efficiency, total-to-static", "–", eta_ts, design["efficiency_ts"], ".3f"),
        ("Efficiency, total-to-total", "–", eta_tt, design["efficiency_tt"], ".3f"),
        ("Degree of reaction (isentropic head)", "–", reaction, 0.0, ".3f"),
        ("Stator kinetic-energy efficiency", "–", stator_ke, None, ".3f"),
        ("Exit kinetic-energy fraction", "–", rout["kinetic"] / res["drop_ts"], None, ".3f"),
    ]
    perf_rows = [[label, unit, f(cfd, spec), f(ref, spec),
                  "–" if ref in (None, 0.0) or "%" == unit else f"{100 * (cfd - ref) / ref:+.1f} %"]
                 for label, unit, cfd, ref, spec in perf]
    station_rows = []   # TurboDash design stations (meanline, cyclopentane_18_100_constantwidth) in brackets
    for label, s, frame, ref in (
            ("Stator exit (1)", exit_, "abs", dict(M=1.430, c=114.26, angle=72.0, x=0.2378)),
            ("Rotor inlet (1)", rin, "rel", dict(M=0.718, c=57.36, angle=52.00, x=0.2378)),
            ("Rotor exit (2)", rout, "abs", dict(M=0.498, c=39.76, angle=27.36, x=0.2378)),
            ("Rotor exit (2)", rout, "rel", dict(M=0.718, c=57.36, angle=-52.00, x=0.2378))):
        mach, speed, angle = ((s["M"], s["c"], s["alpha"]) if frame == "abs" else (s["M_rel"], s["w"], s["beta"]))
        station_rows.append([label, "absolute" if frame == "abs" else "relative",
                             f(s["p"] / 1e3, ".1f"), f"{mach:.3f} ({ref['M']:.3f})", f"{speed:.1f} ({ref['c']:.1f})",
                             f"{angle:.1f} ({ref['angle']:.1f})", f"{1 - s['x']:.3f} ({1 - ref['x']:.3f})"])
    slip = np.linspace(0.0, 1.0, 51)
    drop_rows = []
    for k in (1.0, 0.8, 0.6, 0.4, 0.2):
        i = int(np.argmin(np.abs(slip - k)))
        drop_rows.append([f"{k:.1f}"] + [f"{droplets[s][0][i] - METAL['rotor_in']:+.1f} / {droplets[s][1][i]:.1f}"
                                         for s in SPANS])
    rin_profile = res["profiles"]["rotor_1_inlet"]
    incidence = rin_profile["beta"] - METAL["rotor_in"]
    findings = [
        f"Mass flow {res['mass_flow']:.2f} kg/s against the TurboDash target {design['mass_flow_rate']:.2f} kg/s "
        f"({100 * (res['mass_flow'] / design['mass_flow_rate'] - 1):+.1f} %). The stator is choked (M₁ > 1), so the "
        "deficit is set at the stator throat, not by the rotor. It acts like a discharge coefficient of "
        f"{res['mass_flow'] / design['mass_flow_rate']:.3f} on the 44-blade MoC stator.",
        f"Stator exit pressure {exit_['p'] / 1e3:.1f} kPa against the 94.4 kPa design: the rotor takes a reaction of "
        f"{reaction:.3f} (design 0, pure impulse), which lowers M₁ and c₁. The "
        f"{100 * (1 - res['power'] / design['power_actual']):.1f} % power deficit splits into "
        f"{100 * (1 - res['mass_flow'] / design['mass_flow_rate']):.1f} % less flow and "
        f"{100 * (1 - res['work'] * design['mass_flow_rate'] / design['power_actual']):.1f} % less specific work.",
        f"Stator exit: mass-averaged M₁ = {exit_['M']:.3f} and α₁ = {exit_['alpha']:.1f}° (design 1.430 and 72.0°).",
        f"Rotor inlet: β₁ = {rin['beta']:.1f}° against the 52.0° metal angle; spanwise the incidence runs from "
        f"{incidence.max():+.1f}° near the hub to {incidence.min():+.1f}° near the tip. The stator holds α₁ ≈ 72° over "
        "the span while u = ωr grows by 25 % from hub to tip, and the rotor blade is untwisted, so the "
        f"incidence swings by about {np.ptp(incidence):.0f}°. Relative Mach M_rel,1 = {rin['M_rel']:.3f}.",
        f"Shaft power {res['power'] / 1e3:.1f} kW and η_ts = {eta_ts:.3f}, against TurboDash "
        f"{design['power_actual'] / 1e3:.1f} kW and {design['efficiency_ts']:.3f}. TurboDash includes a tip-clearance "
        "loss (ζ = 0.083 of the rotor's 0.217); the CFD rotor has no tip gap.",
        f"Torque work and Euler work ω Δ(r c_t) agree to {100 * (res['euler'] - res['work']) / res['work']:+.2f} %: "
        "the angular-momentum balance closes in this 3D case.",
        f"Wetness: the liquid mass fraction is {1 - exit_['x']:.3f} at the stator exit and {1 - rout['x']:.3f} at the "
        f"rotor exit. In equilibrium the void fraction at the rotor inlet is {exit_['alpha_void']:.3f}, so most of "
        "the volume is vapour but about three quarters of the mass is liquid.",
        f"Convergence: {iterations[-1]} iterations. Final scaled residuals: continuity {history[-1, 0]:.1e}, "
        f"k {history[-1, 4]:.1e}, ω {history[-1, 5]:.1e}. Mass imbalance {100 * imbalance:+.3f} %.",
    ]
    return [
        ("title", "Paper 3 stage: MoC stator + TurboDash impulse rotor",
         "3D steady RANS (k-ω SST), equilibrium barotropic two-phase model, mixing plane, compared with the "
         "TurboDash meanline design",
         [("Stator", "44 blades, MoC, chord 44.0 mm"), ("Rotor", "120 blades, TurboDash impulse"),
          ("Speed", f"{res['setup']['rpm']:.1f} rpm"), ("P₀ / p_out", "251.3 / 94.4 kPa"),
          ("Inlet quality", f"{res['state']['Q0']:.4f}"), ("Cells", "540 200")]),
        ("kpis", kpis),
        ("note", "KPI reference values are the TurboDash meanline design (cyclopentane_18_100_constantwidth). "
                 "The rotor-angle card compares the CFD mass-averaged β₁ with the 52.0° rotor metal angle."),
        ("h2", "Findings"),
        ("bullets", findings),
        ("h2", "Stage performance"),
        ("table", "CFD stage performance against the TurboDash meanline. Isentropic quantities come from the "
                  "barotropic closure used by the solver, H(p) = ∫dp/ρ; power is the rotor torque times ω.",
         ["Quantity", "Unit", "CFD", "TurboDash", "Δ"], None, perf_rows, "numeric-2"),
        ("table", "Mass-averaged station values, with the TurboDash design in brackets. Station 1 is the mixing "
                  "plane: stator side (absolute) and rotor side (relative).",
         ["Station", "Frame", "p [kPa]", "Mach", "Speed [m/s]", "Angle [deg]", "Liquid fraction 1−x"],
         None, station_rows, "numeric-2"),
        ("h2", "Spanwise flow at the mixing plane and rotor exit"),
        ("figure", [paths["span"]],
         "Pitch-averaged spanwise profiles. Left: flow angles from axial (dotted lines are the metal angles 72°, "
         "52° and −52°). Second: rotor incidence. Third: stator-exit absolute Mach and rotor relative Mach. "
         "Right: equilibrium liquid mass fraction."),
        ("h2", "Blade-to-blade flow"),
        ("figure", [paths["b2b"]],
         "Mach number on constant-radius surfaces at 90, 50 and 10 % span (top to bottom), unrolled to (x, rθ): "
         "absolute in the stator, relative in the rotor. The two rows meet at the mixing plane (x = 75.1 mm), so "
         "their circumferential phase is arbitrary."),
        ("figure", [paths["cuts"]],
         "Relative Mach number on axial cuts through three rotor passages: upstream of the leading edge, at mid "
         "axial chord, and downstream of the trailing edge."),
        ("figure", [paths["loading"]],
         "Blade static pressure against axial chord fraction at 10, 50 and 90 % span."),
        ("h2", "Droplet impact kinematics (input to the DPM study)"),
        ("bullets", [
            "The barotropic model carries the liquid in equilibrium with the vapour: no slip. Larger droplets lag "
            "the vapour, so they reach the rotor with a different relative angle and speed.",
            "Estimate: a droplet leaves the stator along the CFD vapour direction at speed c_d = k·c₁. Its rotor-"
            "relative angle is β_d = atan((k c_t − u)/(k c_m)) and its impact speed is w_d = |k c₁ − u|. Here c_m, "
            "c_t are the pitch-averaged CFD values at each span and u = ωr.",
            "As k falls, β_d swings from the inlet metal angle through axial towards −90° (k → 0). The large negative "
            "incidence means the blade sweeps into the droplets with its suction side, which is the classic "
            "braking impact of wet turbines. The impact speed is lowest near k ≈ 0.5 (15–23 m/s) and returns to "
            "the blade speed u as k → 0. The DPM runs will show where they actually land.",
        ]),
        ("figure", [paths["droplet"]],
         "Rotor-relative impact angle (as incidence on the 52° metal angle) and impact speed of droplets moving "
         "at a fraction k of the local vapour speed at the stator exit."),
        ("table", "Droplet incidence / relative impact speed [deg / m/s] for selected speed ratios k.",
         ["k = c_d/c₁", "10 % span", "50 % span", "90 % span"], None, drop_rows, "numeric-1"),
        ("h2", "Numerics"),
        ("columns", [("figure", [paths["residuals"]], "Scaled residual history of the final run segment.")],
         [("table", "Solver and mesh.", ["Item", "Value"], None, [
             ["Mesh", "TurboGrid, 540 200 cells (stator + rotor)"],
             ["Turbulence", "k-ω SST, 5 % intensity, μt/μ = 10 at the inlet"],
             ["Fluid", "Cyclopentane, equilibrium barotropic ρ(p), μ(p) (8th-order fit)"],
             ["Interface", "Mixing plane at x = 75.1 mm"],
             ["Iterations", f"{iterations[-1]}"],
             ["Blade y⁺ (stator) mean / max", f"{y_plus['stator_1'].mean():.1f} / {y_plus['stator_1'].max():.1f}"],
             ["Blade y⁺ (rotor) mean / max", f"{y_plus['rotor_1'].mean():.1f} / {y_plus['rotor_1'].max():.1f}"],
             ["Stator torque, full row (reaction, no work)",f"{res['stator_torque']:.0f} N m"],
         ], "keyvalue")], 0.5),
    ]


def main():
    apply_style()
    res = compute()
    save = Saver(REPORT / "figures", 300)
    paths = {"residuals": residual_figure(res, save), "span": span_figure(res, save),
             "b2b": blade_to_blade(res, save), "cuts": rotor_cuts(res, save)}
    paths["loading"], loading = blade_loading(res, save)
    paths["droplet"], droplets = droplet_figure(res, save)
    content = blocks(res, paths, loading, droplets)
    pdf = document.write_documents(content, REPORT)
    for name in ("report.md", "report.html"):   # The shared renderer labels the reference as the MoC design
        path = REPORT / name
        path.write_text(path.read_text(encoding="utf-8").replace("MoC design", "TurboDash")
                        .replace("| | CFD | MoC |", "| | CFD | TurboDash |"), encoding="utf-8")
    pdf = document.write_pdf(REPORT / "report.html")
    (REPORT / "report.json").write_text(json.dumps({
        "stations": res["stations"], "flows": res["flows"], "torque_Nm": res["torque"], "power_W": res["power"],
        "eta_ts": res["work"] / res["drop_ts"], "eta_tt": res["work"] / res["drop_tt"], "loading": loading},
        indent=2, default=float), encoding="utf-8")
    print("report:", REPORT / "report.md", pdf)


if __name__ == "__main__":
    main()
