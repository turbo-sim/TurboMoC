"""Report the solved Fluent stage: figures and tables, with the MoC design alongside.

Extraction reads the solved case and data from output/fluent in a headless
Fluent session. It writes surface data, Fluent's own integrals and the residual
history to output/report/data. The figures and tables are built from those files
only, so ``--plots-only`` reruns them without Fluent.

Run from the repository root:
    python examples/fluent_3D/stage/report_fluent_stage.py
    python examples/fluent_3D/stage/report_fluent_stage.py --plots-only
"""
import argparse
import json
import re
import time
from pathlib import Path
import jaxprop as jxp

import numpy as np
import yaml

jxp.set_plot_options(grid=False)

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_stage.yaml"
ROWS = ("stator_1", "rotor_1")
WALLS = ("blade", "hub", "shroud")
NODE_FIELDS = ("absolute-pressure", "density", "x-velocity", "y-velocity", "z-velocity",
               "relative-x-velocity", "relative-y-velocity", "relative-z-velocity")
FACE_FIELDS = NODE_FIELDS
COLORS = {"stator_1": "#233c46", "rotor_1": "#c94f4f"}
SHADES = {"stator_1": "#6f8f9c", "rotor_1": "#e07f72"}
SPAN_STYLES = {0.25: ":", 0.5: "-", 0.75: "--"}


def boundary(setup, row, name):
    return setup["boundaries"][f"{row}_fluid_{name}"]


def midspan_surface(row):
    return f"{row}_midspan"


class Barotropic:
    """The Fluent barotropic closure evaluated in Python, plus the isentropic head H(p) = ∫ dp/ρ."""

    def __init__(self, expressions_file, p_min, p_max):
        expressions = json.loads(Path(expressions_file).read_text(encoding="utf-8"))
        self.sources = {name: re.sub(r"\s*\[[^\]]*\]", "", text).replace("IF(", "where(")
                        for name, text in expressions.items()}
        grid = np.geomspace(0.8 * p_min, 1.05 * p_max, 4000)
        inverse = 1.0 / self.density(grid)
        self._p = grid
        self._head = np.concatenate([[0.0], np.cumsum(0.5 * (inverse[1:] + inverse[:-1]) * np.diff(grid))])

    def _evaluate(self, name, pressure):
        pressure = np.asarray(pressure, dtype=float)
        return np.asarray(eval(self.sources[name], {"where": np.where, "exp": np.exp,
                                                    "AbsolutePressure": pressure}), dtype=float)

    def density(self, pressure):
        return self._evaluate("barotropic_density", pressure)

    def speed_of_sound(self, pressure):
        return self._evaluate("barotropic_speed_of_sound", pressure)

    def head(self, pressure):
        return np.interp(pressure, self._p, self._head)

    def pressure_from_head(self, head):
        return np.interp(head, self._head, self._p)


def extract(case_file, setup, data_dir, options, spans):
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest

    data_dir.mkdir(parents=True, exist_ok=True)
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=3,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode="no_gui", start_timeout=options["start_timeout"], cwd=str(data_dir))
    try:
        settings = solver.settings
        settings.file.read_case_data(file_name=str(case_file))
        r_mid = 0.5 * (spans["hub"] + spans["shroud"])
        iso = settings.results.surfaces.iso_surface
        for row in ROWS:
            iso.create(name=midspan_surface(row))
            surface = iso[midspan_surface(row)]
            surface.field = "radial-coordinate"
            surface.zones = [setup["cell_zones"][f"{row}_fluid"]]
            surface.iso_values = [r_mid]

        surfaces = [boundary(setup, row, name) for row in ROWS for name in ("inlet", "outlet", *WALLS)]
        surfaces += [midspan_surface(row) for row in ROWS]
        field_data = solver.fields.field_data
        arrays = {}
        for surface in surfaces:
            mesh = field_data.get_field_data(SurfaceFieldDataRequest(
                surfaces=[surface], data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity,
                                                SurfaceDataType.FacesNormal, SurfaceDataType.FacesCentroid]))[surface]
            faces = list(mesh.connectivity)
            arrays[f"{surface}|vertices"] = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
            arrays[f"{surface}|faces"] = np.concatenate([[len(face), *face] for face in faces]).astype(np.int64)
            arrays[f"{surface}|normals"] = np.asarray(mesh.face_normals, dtype=float).reshape(-1, 3)
            arrays[f"{surface}|centroids"] = np.asarray(mesh.face_centroids, dtype=float).reshape(-1, 3)
            for field in NODE_FIELDS:
                arrays[f"{surface}|node|{field}"] = np.asarray(field_data.get_field_data(ScalarFieldDataRequest(
                    field_name=field, surfaces=[surface], node_value=True))[surface], dtype=float)
            for field in FACE_FIELDS + (("y-plus",) if any(surface.endswith(f"_{w}") for w in WALLS) else ()):
                arrays[f"{surface}|face|{field}"] = np.asarray(field_data.get_field_data(ScalarFieldDataRequest(
                    field_name=field, surfaces=[surface], node_value=False))[surface], dtype=float)
        np.savez_compressed(data_dir / "surfaces.npz", **arrays)

        definitions = settings.solution.report_definitions
        flow_surfaces = [boundary(setup, row, name) for row in ROWS for name in ("inlet", "outlet")]
        definitions.surface.create(name="report_mass_flow")
        mass_flow = definitions.surface["report_mass_flow"]
        mass_flow.report_type = "surface-massflowrate"
        mass_flow.surface_names = flow_surfaces
        mass_flow.per_surface = True
        moments = {}
        for row in ROWS:
            name = f"report_moment_{row}"
            definitions.moment.create(name=name)
            moment = definitions.moment[name]
            moment.report_output_type = "Moment"
            moment.mom_center = [0.0, 0.0, 0.0]
            moment.mom_axis = [0.0, 0.0, 1.0]
            moment.zones = [boundary(setup, row, wall) for wall in WALLS]
            moment.per_zone = True
            moments[row] = name
        values = definitions.compute(report_defs=["report_mass_flow", *moments.values()])
        values = {key: value[0] for item in values for key, value in item.items()}
        reports = {"mass_flow": {surface: values[f"report_mass_flow({surface})"] for surface in flow_surfaces},
                   "moment": {row: {wall: values[f"{name}({boundary(setup, row, wall)})"] for wall in WALLS}
                              for row, name in moments.items()}}

        lines = []
        solver.transcript.register_callback(lines.append)
        solver.transcript.start(write_to_stdout=False)
        settings.mesh.quality()
        time.sleep(2)
        text = "".join(lines)
        quality = {key: float(value) for key, value in (
            ("min_orthogonal_quality", re.search(r"Minimum Orthogonal Quality =\s*([0-9.eE+-]+)", text).group(1)),
            ("max_aspect_ratio", re.search(r"Maximum Aspect Ratio =\s*([0-9.eE+-]+)", text).group(1)))}
        reports["mesh_quality"] = quality
        reports["methods"] = settings.solution.methods.get_state()
        reports["viscous"] = settings.setup.models.viscous.model()
        (data_dir / "fluent_reports.json").write_text(json.dumps(reports, indent=2, default=float),
                                                      encoding="utf-8")
    finally:
        solver.exit()


def residual_table(transcript_dir, data_dir):
    import barotropy
    transcripts = sorted(Path(transcript_dir).glob("*.trn"), key=lambda path: path.stat().st_size)
    parsed = barotropy.read_residual_file(str(transcripts[-1]))
    history = parsed[0] if isinstance(parsed, tuple) else parsed
    history = history.drop_duplicates("iter", keep="last").sort_values("iter")
    history.to_csv(data_dir / "residuals.csv", index=False)


def polydata(arrays, surface, values="node"):
    import pyvista as pv
    mesh = pv.PolyData(arrays[f"{surface}|vertices"], arrays[f"{surface}|faces"])
    for key in arrays:
        prefix = f"{surface}|{values}|"
        if key.startswith(prefix):
            target = mesh.point_data if values == "node" else mesh.cell_data
            target[key[len(prefix):]] = arrays[key]
    return mesh


class Kinematics:
    """Absolute/relative velocity components in the design convention.

    Tangential components are positive along the stator turning and angles are
    measured from the axial direction, as in the app's stage statistics.
    """

    def __init__(self, omega, swirl_sign, flow_sign):
        self.omega, self.swirl_sign, self.flow_sign = omega, swirl_sign, flow_sign

    def components(self, xyz, values):
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        e_theta = np.column_stack([-np.sin(theta), np.cos(theta)])
        velocity = np.column_stack([values["x-velocity"], values["y-velocity"]])
        relative = np.column_stack([values["relative-x-velocity"], values["relative-y-velocity"]])
        result = {
            "radius": radius, "theta": theta,
            "c_m": self.flow_sign * values["z-velocity"],
            "c_t": self.swirl_sign * np.sum(velocity * e_theta, axis=1),
            "w_t": self.swirl_sign * np.sum(relative * e_theta, axis=1),
            "u": self.swirl_sign * self.omega * radius,
        }
        result["c"] = np.sqrt(values["x-velocity"] ** 2 + values["y-velocity"] ** 2 + values["z-velocity"] ** 2)
        result["w"] = np.sqrt(values["relative-x-velocity"] ** 2 + values["relative-y-velocity"] ** 2
                              + values["relative-z-velocity"] ** 2)
        return result


def mass_averages(arrays, surface, model, kinematics, rotating):
    values = {field: arrays[f"{surface}|face|{field}"] for field in FACE_FIELDS}
    normals = arrays[f"{surface}|normals"]
    velocity = np.column_stack([values["x-velocity"], values["y-velocity"], values["z-velocity"]])
    flux = values["density"] * np.sum(velocity * normals, axis=1)
    weight = flux / flux.sum()
    parts = kinematics.components(arrays[f"{surface}|centroids"], values)
    pressure = values["absolute-pressure"]
    sound = model.speed_of_sound(pressure)
    average = lambda field: float(np.sum(weight * field))
    result = {"mass_flow": float(abs(flux.sum())), "p": average(pressure), "rho": average(values["density"]),
              "c": average(parts["c"]), "c_m": average(parts["c_m"]), "c_t": average(parts["c_t"]),
              "M": average(parts["c"] / sound), "H0": average(model.head(pressure) + 0.5 * parts["c"] ** 2),
              "r_c_t": average(parts["radius"] * parts["c_t"]), "kinetic": average(0.5 * parts["c"] ** 2),
              "backflow_fraction": float(np.sum(np.abs(flux[np.sign(flux) != np.sign(flux.sum())]))
                                         / np.sum(np.abs(flux)))}
    result["alpha"] = float(np.degrees(np.arctan2(result["c_t"], result["c_m"])))
    if rotating:
        result.update({"w": average(parts["w"]), "w_t": average(parts["w_t"]), "u": average(parts["u"]),
                       "M_rel": average(parts["w"] / sound),
                       "rothalpy": average(model.head(pressure) + 0.5 * parts["w"] ** 2 - 0.5 * parts["u"] ** 2)})
        result["beta"] = float(np.degrees(np.arctan2(result["w_t"], result["c_m"])))
    return result


def span_line(mesh, radius):
    line = mesh.contour([radius], scalars=np.hypot(mesh.points[:, 0], mesh.points[:, 1])).strip()
    if line.n_points == 0:
        return None
    return line


def unwrapped(mesh, theta_ref, r_mid):
    """Midspan cylinder unrolled onto (axial, circumferential arc), both in mm."""
    import pyvista as pv
    theta = np.angle(np.exp(1j * (np.arctan2(mesh.points[:, 1], mesh.points[:, 0]) - theta_ref)))
    flat = pv.PolyData(np.column_stack([mesh.points[:, 2] * 1e3, r_mid * theta * 1e3,
                                        np.zeros(mesh.n_points)]), mesh.faces)
    for key in mesh.point_data:
        flat.point_data[key] = mesh.point_data[key]
    return flat.triangulate()


def report(output_dir, workflow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv

    data_dir, figures = output_dir / "report" / "data", output_dir / "report" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.0,
                         "axes.titlelocation": "left"})
    options = workflow["report"]
    dpi = options["dpi"]
    setup = json.loads((output_dir / "fluent" / "setup_summary.json").read_text(encoding="utf-8"))
    point = json.loads((output_dir / "operating_point.json").read_text(encoding="utf-8"))
    reports = json.loads((data_dir / "fluent_reports.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(data_dir / "surfaces.npz"))
    design = point["design"]
    blades = design["num_blades"]
    state = setup["inlet_state"]
    P0 = float(state["P0"])
    model = Barotropic(output_dir / "barotropy" / "fluent_expressions_safe.json", float(state["p_min"]), P0)
    omega = float(setup["rotor_omega_rad_s"])
    hub = {row: polydata(arrays, boundary(setup, row, "hub")) for row in ROWS}
    shroud = {row: polydata(arrays, boundary(setup, row, "shroud")) for row in ROWS}
    r_hub = float(np.hypot(*hub["stator_1"].points[:, :2].T).mean())
    r_shroud = float(np.hypot(*shroud["stator_1"].points[:, :2].T).mean())
    r_mid = 0.5 * (r_hub + r_shroud)
    radius_at = lambda span: r_hub + span * (r_shroud - r_hub)
    inlet = arrays[f"{boundary(setup, 'stator_1', 'inlet')}|face|z-velocity"]
    kinematics = Kinematics(omega, float(point["swirl_sign"]), float(np.sign(np.mean(inlet))))
    pitch = {row: 2 * np.pi / blades[row] for row in ROWS}
    blade = {row: polydata(arrays, boundary(setup, row, "blade")) for row in ROWS}
    theta_ref = float(np.angle(np.mean(np.exp(1j * np.arctan2(blade["stator_1"].points[:, 1],
                                                                blade["stator_1"].points[:, 0])))))

    stations = {
        "Stator inlet (0)": mass_averages(arrays, boundary(setup, "stator_1", "inlet"), model, kinematics, False),
        "Stator exit (1)": mass_averages(arrays, boundary(setup, "stator_1", "outlet"), model, kinematics, False),
        "Rotor inlet (1)": mass_averages(arrays, boundary(setup, "rotor_1", "inlet"), model, kinematics, True),
        "Rotor exit (2)": mass_averages(arrays, boundary(setup, "rotor_1", "outlet"), model, kinematics, True),
    }
    exit_, rotor_in, outlet = stations["Stator exit (1)"], stations["Rotor inlet (1)"], stations["Rotor exit (2)"]
    flows = reports["mass_flow"]
    m_in = abs(flows[boundary(setup, "stator_1", "inlet")])
    m_out = abs(flows[boundary(setup, "rotor_1", "outlet")])
    m_exit = abs(flows[boundary(setup, "stator_1", "outlet")])
    m_rotor = abs(flows[boundary(setup, "rotor_1", "inlet")])
    torque_passage = sum(reports["moment"]["rotor_1"][wall] for wall in ("blade", "hub"))
    torque = torque_passage * blades["rotor_1"]
    power = torque * omega
    mass_flow = m_out * blades["rotor_1"]
    work = power / mass_flow
    euler = abs(omega) * (rotor_in["r_c_t"] - outlet["r_c_t"])
    drop_ts = model.head(P0) - model.head(outlet["p"])
    drop_tt = drop_ts - outlet["kinetic"]
    stator_drop = model.head(P0) - model.head(exit_["p"])
    design_stage = dict(point["stage"])
    design_stations = {f'{item["station"]}|{item["frame"]}': item
                       for item in design["stator"]["stations"] + design["rotor"]["stations"]}
    performance = [
        ("Mass flow rate, full annulus", "kg/s", mass_flow, design["stator"]["mass_flow"], "relative", ".3f"),
        ("Mass flow rate, one passage", "kg/s", m_out, None, None, ".4f"),
        ("Torque, full rotor", "N m", abs(torque), None, None, ".2f"),
        ("Power", "kW", power / 1e3, design_stage.get("Power, annulus estimate [kW]"), "relative", ".2f"),
        ("Specific work (torque)", "kJ/kg", work / 1e3, design_stage.get("Specific work [kJ/kg]"), "relative", ".3f"),
        ("Specific work (Euler)", "kJ/kg", euler / 1e3, design_stage.get("Specific work [kJ/kg]"), "relative", ".3f"),
        ("Isentropic drop, total-to-static", "kJ/kg", drop_ts / 1e3,
         design_stage.get("Isentropic drop, total-to-static [kJ/kg]"), "relative", ".3f"),
        ("Efficiency, total-to-static", "–", work / drop_ts, design_stage.get("Efficiency, total-to-static [-]"),
         "absolute", ".3f"),
        ("Efficiency, total-to-total", "–", work / drop_tt, None, None, ".3f"),
        ("Degree of reaction", "–", (model.head(exit_["p"]) - model.head(outlet["p"])) / drop_ts,
         design_stage.get("Degree of reaction [-]"), "absolute", ".3f"),
        ("Exit kinetic-energy fraction", "–", outlet["kinetic"] / drop_ts,
         design_stage.get("Exit kinetic energy fraction [-]"), "absolute", ".3f"),
        ("Stator kinetic-energy efficiency", "–", exit_["kinetic"] / stator_drop, 1.0, "absolute", ".3f"),
    ]
    station_rows = [
        ("Stator exit (1)", "absolute", exit_, "Stator exit (1)|absolute", "M", "c", "alpha"),
        ("Rotor inlet (1)", "absolute", rotor_in, "Rotor inlet (1)|absolute", "M", "c", "alpha"),
        ("Rotor inlet (1)", "relative", rotor_in, "Rotor inlet (1)|relative", "M_rel", "w", "beta"),
        ("Rotor exit (2)", "absolute", outlet, "Rotor exit (2)|absolute", "M", "c", "alpha"),
        ("Rotor exit (2)", "relative", outlet, "Rotor exit (2)|relative", "M_rel", "w", "beta"),
    ]
    residuals = None
    if (data_dir / "residuals.csv").exists():
        import pandas as pd
        residuals = pd.read_csv(data_dir / "residuals.csv")
    y_plus = {row: arrays[f"{boundary(setup, row, 'blade')}|face|y-plus"] for row in ROWS}
    mesh_report = json.loads((output_dir / "mesh" / "mesh_report.json").read_text(encoding="utf-8"))
    project = yaml.safe_load((output_dir / "project.yaml").read_text(encoding="utf-8"))
    gap = next(row for row in project["rows"] if row["role"] == "rotor")["geometry"].get("axial_interspace_mm", 2.0)

    figure_list = {}

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return f"figures/{name}.png"

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1300))
    plotter.set_background("white")
    z_blades = np.concatenate([blade[row].points[:, 2] for row in ROWS])
    margin = 0.25 * (z_blades.max() - z_blades.min())
    window = (-1, 1, -1, 1, z_blades.min() - margin, z_blades.max() + margin)
    half = options["passages_3d"] // 2
    for row in ROWS:
        surface = hub[row].clip_box(window, invert=False)
        for k in range(-half, options["passages_3d"] - half):
            angle = np.degrees(k * pitch[row])
            plotter.add_mesh(blade[row].rotate_z(angle, point=(0, 0, 0), inplace=False), color=SHADES[row],
                             smooth_shading=True, specular=0.3)
            plotter.add_mesh(surface.rotate_z(angle, point=(0, 0, 0), inplace=False), color="#d9dee0")
    e_r = np.array([np.cos(theta_ref), np.sin(theta_ref), 0.0])
    e_t = np.array([-np.sin(theta_ref), np.cos(theta_ref), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    focus = r_mid * e_r + e_z * z_blades.mean()
    distance = 6 * (z_blades.max() - z_blades.min())
    plotter.camera.focal_point = focus
    plotter.camera.position = focus + distance * (0.75 * e_r - 0.45 * e_z + 0.5 * e_t)
    plotter.camera.up = e_r
    plotter.camera.zoom(1.4)
    image = cropped(plotter.screenshot(return_img=True))
    plotter.close()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.1, 1]},
                             constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[0].set_title(f"(a) Blading, {options['passages_3d']} passages per row")
    ax = axes[1]
    z_all = []
    for row in ROWS:
        z = blade[row].points[:, 2] * 1e3
        r = np.hypot(*blade[row].points[:, :2].T) * 1e3
        edges = np.linspace(r.min(), r.max(), 30)
        bins = np.digitize(r, edges)
        lead = [z[bins == b].min() for b in np.unique(bins)]
        trail = [z[bins == b].max() for b in np.unique(bins)]
        radii = [r[bins == b].mean() for b in np.unique(bins)]
        ax.fill_betweenx(radii, lead, trail, color=SHADES[row], lw=0)
        ax.text(0.5 * (min(lead) + max(trail)), r_mid * 1e3, row.split("_")[0].capitalize(), rotation=90,
                ha="center", va="center", color="w", fontsize=8)
        for name in ("inlet", "outlet"):
            zz = arrays[f"{boundary(setup, row, name)}|vertices"][:, 2].mean() * 1e3
            ax.axvline(zz, color="#8a9aa1", lw=0.8, ls="--")
        z_all += [hub[row].points[:, 2].min() * 1e3, hub[row].points[:, 2].max() * 1e3]
    ax.plot([min(z_all), max(z_all)], [r_hub * 1e3] * 2, color="k", lw=1.6)
    ax.plot([min(z_all), max(z_all)], [r_shroud * 1e3] * 2, color="k", lw=1.6)
    for span, style in SPAN_STYLES.items():
        ax.plot([min(z_all), max(z_all)], [radius_at(span) * 1e3] * 2, style, color="#8a9aa1", lw=0.7)
        ax.text(max(z_all) - 1, radius_at(span) * 1e3 + 0.15, f"{int(span * 100)} % span", ha="right", va="bottom",
                fontsize=6.5, color="#5b6b72")
    interface = arrays[f"{boundary(setup, 'stator_1', 'outlet')}|vertices"][:, 2].mean() * 1e3
    ax.text(interface + 1, r_shroud * 1e3 + 0.35, "mixing plane", ha="left", fontsize=6.5, color="#5b6b72")
    ax.set_xlabel("Axial coordinate z [mm]")
    ax.set_ylabel("Radius r [mm]")
    ax.set_title("(b) Meridional channel")
    ax.set_ylim(r_hub * 1e3 - 1.0, r_shroud * 1e3 + 1.4)
    figure_list["geometry"] = save(fig, "geometry")

    flats = {}
    for row in ROWS:
        mesh = polydata(arrays, midspan_surface(row))
        values = {field: mesh.point_data[field] for field in NODE_FIELDS}
        parts = kinematics.components(mesh.points, values)
        sound = model.speed_of_sound(values["absolute-pressure"])
        mesh.point_data["mach"] = (parts["c"] if row == "stator_1" else parts["w"]) / sound
        mesh.point_data["p_ratio"] = values["absolute-pressure"] / P0
        flat = unwrapped(mesh, theta_ref, r_mid)
        flat = flat.compute_derivative(scalars="density", gradient="grad_rho")
        flat.point_data["grad_rho"] = np.linalg.norm(flat.point_data["grad_rho"], axis=1) * 1e3
        flats[row] = flat
    arc = {row: r_mid * pitch[row] * 1e3 for row in ROWS}
    copies = options["passages_2d"]
    centre = float(np.median(flats["stator_1"].points[:, 1]))
    band = (centre - 0.5 * copies * arc["stator_1"], centre + 0.5 * copies * arc["stator_1"])
    extent = lambda rows: (min(flats[row].points[:, 0].min() for row in rows),
                           max(flats[row].points[:, 0].max() for row in rows))

    def frame(ax, rows, xlabel=True):
        ax.set_xlim(*extent(rows))
        ax.set_ylim(*band)
        ax.set_aspect("equal")
        if xlabel:
            ax.set_xlabel("Axial coordinate z [mm]")
        ax.set_ylabel("Arc r·θ [mm]")

    def draw(ax, row, field, **style):
        flat = flats[row]
        triangles = flat.faces.reshape(-1, 4)[:, 1:]
        lowest = int(np.floor((band[0] - flat.points[:, 1].max()) / arc[row]))
        highest = int(np.ceil((band[1] - flat.points[:, 1].min()) / arc[row]))
        result = None
        for k in range(lowest, highest + 1):
            tri = mtri.Triangulation(flat.points[:, 0], flat.points[:, 1] + k * arc[row], triangles)
            result = ax.tripcolor(tri, flat.point_data[field], shading="gouraud", rasterized=True, **style)
            if field == "mach":
                ax.tricontour(tri, flat.point_data[field], levels=[1.0], colors="w", linewidths=0.7)
        return result

    width = extent(ROWS)[1] - extent(ROWS)[0]
    panel = 6.1 * (band[1] - band[0]) / width
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 2 * panel + 0.75), sharex=True, constrained_layout=True)
    mach_max = max(float(np.max(flats[row].point_data["mach"])) for row in ROWS)
    for ax, field, label, cmap, limits, title in (
            (axes[0], "mach", "Mach number [-]", "turbo", (0, mach_max),
             "(a) Mach number: stator absolute, rotor relative"),
            (axes[1], "p_ratio", "p/P₀ [-]", "viridis", (0, 1), "(b) Static pressure")):
        for row in ROWS:
            artist = draw(ax, row, field, cmap=cmap, vmin=limits[0], vmax=limits[1])
        fig.colorbar(artist, ax=ax, label=label, pad=0.01, fraction=0.035)
        ax.axvline(interface, color="w", lw=1.0, ls="--")
        frame(ax, ROWS, xlabel=ax is axes[1])
        ax.set_title(title, loc="left")
    figure_list["midspan"] = save(fig, "midspan_mach_pressure")

    from matplotlib.colors import LogNorm
    schlieren = []
    for row, label in (("stator_1", "(a) Stator, absolute frame"), ("rotor_1", "(b) Rotor, relative frame")):
        span_x = extent([row])[1] - extent([row])[0]
        fig, ax = plt.subplots(figsize=(0.95 + 3.1 * span_x / (band[1] - band[0]), 3.45), constrained_layout=True)
        gradient = flats[row].point_data["grad_rho"]
        flats[row].point_data["grad_rho_plot"] = np.clip(gradient, np.percentile(gradient, 5), None)
        artist = draw(ax, row, "grad_rho_plot", cmap="Greys", norm=LogNorm(
            vmin=np.percentile(gradient, 5), vmax=np.percentile(gradient, options["schlieren_percentile"])))
        fig.colorbar(artist, ax=ax, label=r"$|\nabla\rho|$ [kg m$^{-4}$]", pad=0.02, fraction=0.08)
        frame(ax, [row])
        ax.set_title(label, loc="left")
        schlieren.append(save(fig, f"density_gradient_{row.split('_')[0]}"))
    figure_list["schlieren"] = schlieren

    rothalpy = rotor_in["rothalpy"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex="col", constrained_layout=True)
    loading = {}
    for col, row in enumerate(ROWS):
        mesh = blade[row]
        z = mesh.points[:, 2]
        z_le, z_te = z.min(), z.max()
        for span, style in SPAN_STYLES.items():
            line = span_line(mesh, radius_at(span))
            if line is None:
                continue
            pressure = line.point_data["absolute-pressure"]
            x = (line.points[:, 2] - z_le) / (z_te - z_le)
            if row == "stator_1":
                head = model.head(P0) - model.head(pressure)
            else:
                u = abs(omega) * radius_at(span)
                head = rothalpy + 0.5 * u ** 2 - model.head(pressure)
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
    figure_list["loading"] = save(fig, "blade_loading")

    outlet_mesh = polydata(arrays, boundary(setup, "stator_1", "outlet"))
    values = {field: outlet_mesh.point_data[field] for field in NODE_FIELDS}
    parts = kinematics.components(outlet_mesh.points, values)
    outlet_mesh.point_data["mach"] = parts["c"] / model.speed_of_sound(values["absolute-pressure"])
    outlet_mesh.point_data["alpha"] = np.degrees(np.arctan2(parts["c_t"], parts["c_m"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True, constrained_layout=True)
    reference = design_stations["Stator exit (1)|absolute"]
    for span, style in SPAN_STYLES.items():
        line = span_line(outlet_mesh, radius_at(span))
        theta = np.angle(np.exp(1j * (np.arctan2(line.points[:, 1], line.points[:, 0]) - theta_ref)))
        position = (theta - theta.min()) / pitch["stator_1"]
        order = np.argsort(position)
        for ax, field, scale in ((axes[0], "mach", 1.0), (axes[1], "absolute-pressure", 1 / P0),
                                 (axes[2], "alpha", 1.0)):
            ax.plot(position[order], line.point_data[field][order] * scale, style, color=COLORS["stator_1"], lw=1.0,
                    label=f"{int(span * 100)} % span")
    for ax, cfd, moc, tag, label in ((axes[0], exit_["M"], reference["M"], "(a)", "Mach number [-]"),
                                     (axes[1], exit_["p"] / P0, reference["p"] / P0, "(b)", "p/P₀ [-]"),
                                     (axes[2], exit_["alpha"], reference["angle"], "(c)", "Flow angle α [deg]")):
        ax.axhline(cfd, color="#e08a3c", lw=1.2, label="CFD mass average")
        ax.axhline(moc, color="#5b6b72", lw=1.2, ls="-.", label="MoC design")
        ax.set_ylabel(f"{tag} {label}")
    fig.supxlabel("Pitchwise position from the blade wake [-]", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, fontsize=7.5, frameon=False)
    figure_list["pitchwise"] = save(fig, "stator_exit_pitchwise")

    if residuals is not None:
        fig, ax = plt.subplots(figsize=(2.95, 2.75), constrained_layout=True)
        for column in residuals.columns:
            if column not in ("iter", "time/iter"):
                ax.semilogy(residuals["iter"], residuals[column], lw=0.9, label=column)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Scaled residual [-]")
        ax.legend(fontsize=6.5, ncol=2, frameon=False)
        figure_list["residuals"] = save(fig, "residuals")

    blocks = report_blocks(locals())
    report_dir = output_dir / "report"
    (report_dir / "report.md").write_text(render_markdown(blocks), encoding="utf-8")
    (report_dir / "report.html").write_text(render_html(blocks, report_dir), encoding="utf-8")
    (report_dir / "report.json").write_text(json.dumps({
        "stations": stations, "performance": {row[0]: {"cfd": row[2], "moc": row[3]} for row in performance},
        "torque_passage": torque_passage, "power_W": power, "fluent_reports": reports, "blade_loading": loading},
        indent=2, default=float), encoding="utf-8")
    return report_dir


def cropped(image, pad=12):
    content = np.where(np.any(image[:, :, :3] < 245, axis=2))
    top, bottom = max(content[0].min() - pad, 0), min(content[0].max() + pad, image.shape[0])
    left, right = max(content[1].min() - pad, 0), min(content[1].max() + pad, image.shape[1])
    return image[top:bottom, left:right]


def number(value, spec):
    return "–" if value is None else format(value, spec)


def report_blocks(scope):
    """Turn the computed results into an ordered list of document blocks."""
    s = type("Scope", (), scope)
    performance, stations_table = s.performance, []
    for name, frame, values, key, mach, speed, angle in s.station_rows:
        reference = s.design_stations.get(key, {})
        stations_table.append([name, frame, number(values["p"] / 1e3, ".2f"),
                               number(reference.get("p") and reference["p"] / 1e3, ".2f"),
                               number(values[mach], ".3f"), number(reference.get("M"), ".3f"),
                               number(values[angle], ".1f"), number(reference.get("angle"), ".1f"),
                               number(values[speed], ".1f")])
    perf_rows = []
    for label, unit, cfd, moc, kind, spec in performance:
        if cfd is None or moc is None or kind is None:
            delta = "–"
        elif kind == "relative":
            delta = f"{100 * (cfd - moc) / moc:+.1f} %"
        else:
            delta = format(cfd - moc, "+" + spec)
        perf_rows.append([label, unit, number(cfd, spec), number(moc, spec), delta])

    lookup = {row[0]: row for row in performance}
    design_exit = s.design_stations["Stator exit (1)|absolute"]
    design_rotor = s.design_stations["Rotor exit (2)|relative"]
    work, euler = lookup["Specific work (torque)"][2], lookup["Specific work (Euler)"][2]
    flow, flow_moc = lookup["Mass flow rate, full annulus"][2], lookup["Mass flow rate, full annulus"][3]
    eta, eta_moc = lookup["Efficiency, total-to-static"][2], lookup["Efficiency, total-to-static"][3]
    highlights = [
        f"Mass flow {flow:.3f} kg/s, {100 * (flow - flow_moc) / flow_moc:+.1f} % against the MoC annulus estimate; "
        f"the stator is choked and sets it.",
        f"Stator exit: M = {s.exit_['M']:.3f} (design {design_exit['M']:.3f}) and α = {s.exit_['alpha']:.1f}° "
        f"(design {design_exit['angle']:.1f}°), at {s.exit_['p'] / 1e3:.1f} kPa against {design_exit['p'] / 1e3:.1f} kPa "
        f"designed ({100 * (s.exit_['p'] - design_exit['p']) / design_exit['p']:+.1f} %).",
        f"Rotor exit relative angle {s.outlet['beta']:.1f}° against {design_rotor['angle']:.1f}° designed: "
        f"{abs(s.outlet['beta'] - design_rotor['angle']):.1f}° less turning, hence the lower work.",
        f"Specific work {work:.3f} kJ/kg from the rotor torque and {euler:.3f} kJ/kg from the Euler equation "
        f"({100 * abs(work - euler) / work:.1f} % apart); total-to-static efficiency {eta:.3f} against {eta_moc:.3f}.",
    ]
    if s.residuals is not None:
        equations = [c for c in s.residuals.columns if c not in ("iter", "time/iter")]
        final = s.residuals.iloc[-1]
        worst = max(equations, key=lambda eq: final[eq])
        highlights.append(f"{int(final['iter'])} iterations; largest final residual {final[worst]:.1e} ({worst}); "
                          f"mass imbalance {100 * (s.m_in - s.m_out) / s.m_in:+.2f} % inlet to outlet, "
                          f"{100 * (s.m_exit - s.m_rotor) / s.m_exit:+.3f} % across the mixing plane.")

    mesh_rows = []
    for row in ROWS:
        summary = s.mesh_report["rows"][row]["meshes"]["3d"]["summary"]
        cells, quality = summary["cells"], summary["quality"]
        mesh_rows.append([row.split("_")[0].capitalize(), f"{cells.get('hexahedra', 0):,}",
                          f"{cells.get('prisms', 0):,}", number(quality["min_determinant_jacobian"], ".2e"),
                          number(quality["min_sicn"], ".3f"),
                          f"{np.mean(s.y_plus[row]):.0f} / {np.max(s.y_plus[row]):.0f}"])
    methods, schemes = s.reports["methods"], s.reports["methods"]["discretization_scheme"]
    setup_rows = [
        ["Solver", f"Pressure-based {methods['p_v_coupling']['flow_scheme'].lower()}, steady (pseudo-transient)"],
        ["Turbulence", "SST k-ω" if s.reports["viscous"] == "k-omega" else s.reports["viscous"]],
        ["Discretization", f"{schemes['mom'].replace('-', ' ')} (momentum, density, k, ω); pressure "
                           f"{schemes['pressure']}; {methods['gradient_scheme'].replace('-', ' ')} gradients"],
        ["Rotor speed", f"{abs(s.point['rpm']):.0f} rpm ({abs(s.omega):.1f} rad/s)"],
        ["Boundary conditions", f"P₀ = {s.P0 / 1e3:.1f} kPa at the inlet; p = {s.point['rotor_exit_pressure'] / 1e3:.2f} "
                                f"kPa at the outlet"],
        ["Mass imbalance", f"{100 * (s.m_in - s.m_out) / s.m_in:+.3f} % (inlet → outlet), "
                           f"{100 * (s.m_exit - s.m_rotor) / s.m_exit:+.3f} % (mixing plane)"],
    ]
    if s.residuals is not None:
        setup_rows.append(["Iterations", f"{int(final['iter'])}"])
        setup_rows.append(["Final residuals", ", ".join(f"{eq} {final[eq]:.0e}" for eq in equations)])
    figures = s.figure_list
    operating = [("Fluid", s.state["fluid_name"]), ("P₀", f"{s.P0 / 1e3:.1f} kPa"),
                 ("Inlet quality", f"{s.state['Q0']:.3f}"), ("Exit pressure", f"{s.point['rotor_exit_pressure'] / 1e3:.2f} kPa"),
                 ("Speed", f"{abs(s.point['rpm']):.0f} rpm"),
                 ("Blades", f"{s.blades['stator_1']} stator / {s.blades['rotor_1']} rotor"),
                 ("Row gap", f"{s.gap:g} mm")]
    kpis = [(label, unit, lookup[key][2], lookup[key][3], spec) for label, unit, key, spec in (
        ("Mass flow", "kg/s", "Mass flow rate, full annulus", ".3f"), ("Power", "kW", "Power", ".2f"),
        ("Specific work", "kJ/kg", "Specific work (torque)", ".3f"),
        ("Total-to-static efficiency", "", "Efficiency, total-to-static", ".3f"))]
    return [
        ("title", "Supersonic axial stage: CFD assessment of the MoC design",
         "Three-dimensional RANS of the stator–rotor stage (Fluent, mixing plane) against the "
         "method-of-characteristics design", operating),
        ("kpis", kpis),
        ("h2", "Highlights"), ("bullets", highlights),
        ("h2", "Stage performance"),
        ("table", "Stage performance from CFD and from the MoC design; Δ is relative for dimensional quantities and "
                  "absolute otherwise. Isentropic drops use H(p) = ∫dp/ρ of the barotropic closure.",
         ["Quantity", "Unit", "CFD", "MoC", "Δ"], None, perf_rows, "numeric-3"),
        ("table", "Mass-weighted averages at the row interfaces. Angles from the axial direction, positive along the "
                  "stator turning; the stator exit is evaluated on the stator side of the mixing plane.",
         ["Station", "Frame", "CFD", "MoC", "CFD", "MoC", "CFD", "MoC", "CFD"],
         [("", 2), ("p [kPa]", 2), ("Mach [-]", 2), ("Angle [deg]", 2), ("V [m/s]", 1)], stations_table,
         "numeric-2"),
        ("h2", "Geometry and mesh"),
        ("figure", [figures["geometry"]], "Stage geometry. (a) Blading over five passages per row, rendered from the "
                                          "Fluent surfaces; (b) meridional channel with the row inlets, the mixing "
                                          "plane, the outlet (dashed) and the 25/50/75 % spans."),
        ("table", "Volume meshes (app mesher) and blade y⁺ from the solution. Fluent: minimum orthogonal quality "
                  f"{s.reports['mesh_quality']['min_orthogonal_quality']:.3f}, maximum aspect ratio "
                  f"{s.reports['mesh_quality']['max_aspect_ratio']:.0f}.",
         ["Row", "Hexahedra", "Prisms", "Min. det. Jacobian", "Min. SICN", "Blade y⁺ mean / max"], None, mesh_rows,
         "numeric-1"),
        ("h2", "Simulation setup and convergence"),
        ("columns", [("table", "Solver settings and convergence.", ["Item", "Setting"], None, setup_rows, "keyvalue"),
                     ("note", "Assumptions: steady RANS with the rotor in a moving reference frame; circumferentially averaged mixing "
                 "plane between the rows; one passage per row with rotational periodicity; barotropic equilibrium "
                 "closure along the inlet isentrope (energy equation off, adiabatic walls); no tip clearance; rotor "
                 "shroud stationary.")],
         [("figure", [figures["residuals"]], "Residual history.")] if "residuals" in figures else [], 0.58),
        ("h2", "Flow field"),
        ("figure", [figures["midspan"]], "Midspan (r = %.1f mm) on the unwrapped cylinder. White line: M = 1; dashed "
                                         "line: mixing plane." % (s.r_mid * 1e3)),
        ("figure", figures["schlieren"], "Numerical schlieren |∇ρ| at midspan, logarithmic grey scale between the 5th "
                                         f"and {s.options['schlieren_percentile']}th percentiles."),
        ("h2", "Blade loading and stator exit"),
        ("figure", [figures["loading"]], "Blade static pressure and isentropic Mach number at 25, 50 and 75 % span. "
                                         "Stator from P₀; rotor relative, from the rotor-inlet rothalpy at the local "
                                         "radius."),
        ("figure", [figures["pitchwise"]], "Stator exit, stator side of the mixing plane: pitchwise distributions at "
                                           "three spans against the mass average (Table 2) and the MoC design."),
    ]


def render_markdown(blocks):
    lines, tables, figures = [], 0, 0
    for block in blocks:
        kind = block[0]
        if kind == "title":
            lines += [f"# {block[1]}", "", f"*{block[2]}*", "",
                      " · ".join(f"**{key}** {value}" for key, value in block[3]), ""]
        elif kind == "kpis":
            lines += ["| | CFD | MoC |", "| --- | --- | --- |"]
            lines += [f"| {label} {f'[{unit}]' if unit else ''} | {number(cfd, spec)} | {number(moc, spec)} |"
                      for label, unit, cfd, moc, spec in block[1]] + [""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "bullets":
            lines += [f"- {item}" for item in block[1]] + [""]
        elif kind == "note":
            lines += [f"> {block[1]}", ""]
        elif kind == "table":
            tables += 1
            _, caption, headers, groups, rows, _style = block
            if groups:
                expanded = [label for label, span in groups for _ in range(span)]
                headers = [f"{group} {header}".strip() if group else header for group, header in zip(expanded, headers)]
            lines += [f"**Table {tables}.** {caption}", "", "| " + " | ".join(headers) + " |",
                      "|" + " --- |" * len(headers)] + ["| " + " | ".join(row) + " |" for row in rows] + [""]
        elif kind == "figure":
            figures += 1
            lines += [f"![Figure {figures}]({path})" for path in block[1]]
            lines += ["", f"**Figure {figures}.** {block[2]}", ""]
        elif kind == "columns":
            nested = render_markdown(block[1] + block[2]).splitlines()
            lines += nested + [""]
    return "\n".join(lines)


REPORT_STYLE = """
@page { size: A4; margin: 15mm 14mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt "Times New Roman", serif;
                         color: #6b7b82; } }
* { box-sizing: border-box; }
body { margin: 0; font: 10pt/1.38 "Times New Roman", Times, serif; color: #1c2a30; }
header { border-bottom: 1.5pt solid #233c46; padding-bottom: 7pt; margin-bottom: 9pt; }
header h1 { font-size: 18pt; margin: 0 0 3pt; color: #1f3741; letter-spacing: -0.2pt; }
header .subtitle { font-style: italic; color: #4a5b62; margin: 0 0 6pt; }
.chips { display: flex; flex-wrap: wrap; gap: 3pt 12pt; font-size: 8.6pt; color: #33454c; }
.chips b { color: #1f3741; font-weight: 600; margin-right: 2pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7pt; margin: 2pt 0 4pt; }
.kpi { border: 0.6pt solid #c9d2d6; border-top: 2.2pt solid #233c46; padding: 5pt 8pt 6pt; background: #f7f9f9; }
.kpi .label { font-size: 8.6pt; color: #4a5b62; font-variant: small-caps; letter-spacing: 0.3pt; }
.kpi .value { font-size: 19pt; color: #1f3741; line-height: 1.15; }
.kpi .value small { font-size: 9pt; color: #4a5b62; margin-left: 3pt; }
.kpi .reference { font-size: 8.4pt; color: #4a5b62; }
.kpi .delta { color: #1f3741; font-weight: 600; margin-left: 5pt; }
h2 { font-size: 12.5pt; color: #1f3741; margin: 12pt 0 5pt; padding-bottom: 2pt;
     border-bottom: 0.6pt solid #c9d2d6; break-after: avoid; }
ul { margin: 2pt 0 4pt; padding-left: 14pt; }
li { margin: 1.5pt 0; padding-left: 2pt; }
li::marker { content: "•  "; color: #233c46; }
figure, .table { margin: 6pt 0 9pt; break-inside: avoid; }
figure { text-align: center; }
figure .row { display: flex; justify-content: center; gap: 6pt; align-items: flex-end; }
figure img { display: block; margin: 0 auto; }
figcaption, .table .caption { font-size: 8.8pt; color: #33454c; text-align: left; line-height: 1.3; }
figcaption { margin-top: 4pt; }
.table .caption { margin-bottom: 3pt; }
figcaption b, .table .caption b { color: #1f3741; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 8.6pt; font-variant-numeric: tabular-nums; }
thead th { border-top: 1pt solid #233c46; border-bottom: 0.6pt solid #233c46; font-weight: 600; color: #1f3741;
           padding: 2.5pt 5pt; text-align: left; }
thead tr.group th { border-bottom: 0.4pt solid #9fb0b6; border-top: 1pt solid #233c46; text-align: center; }
thead tr.group + tr th { border-top: none; }
tbody td { padding: 2.2pt 5pt; border-bottom: 0.3pt solid #e1e7e9; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1pt solid #233c46; }
table.numeric-1 td:nth-child(n+2), table.numeric-1 th:nth-child(n+2),
table.numeric-2 td:nth-child(n+3), table.numeric-2 tr:not(.group) th:nth-child(n+3),
table.numeric-3 td:nth-child(n+3), table.numeric-3 th:nth-child(n+3) { text-align: right; }
table.keyvalue td:first-child { color: #4a5b62; white-space: nowrap; width: 28%; }
.columns { display: flex; gap: 12pt; align-items: flex-start; break-inside: avoid; }
.columns > div > :first-child { margin-top: 0; }
.columns figure { margin-top: 0; }
.note { font-size: 8.8pt; color: #33454c; background: #f3f6f7; border-left: 2pt solid #9fb0b6;
        padding: 4pt 8pt; margin: 4pt 0 6pt; break-inside: avoid; }
"""


def render_html(blocks, report_dir, text_width_mm=182.0):
    import html
    from PIL import Image
    escape = html.escape
    counters = {"table": 0, "figure": 0}

    def image_size(path):
        with Image.open(report_dir / path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (200, 200))[0] or 200
        return width, height, width / dpi * 25.4

    def minus(value):
        return re.sub(r"(?<![\w.])-(?=\d)", "\u2212", value)

    def render(block, container_mm=text_width_mm):
        kind = block[0]
        if kind == "title":
            chips = "".join(f"<span><b>{escape(key)}</b>{escape(value)}</span>" for key, value in block[3])
            return (f"<header><h1>{escape(block[1])}</h1><p class='subtitle'>{escape(block[2])}</p>"
                    f"<div class='chips'>{chips}</div></header>")
        if kind == "kpis":
            cards = []
            for label, unit, cfd, moc, spec in block[1]:
                delta = ""
                if moc:
                    change = 100 * (cfd - moc) / moc
                    delta = f"<span class='delta'>{minus(f'{change:+.1f}')} %</span>"
                cards.append(f"<div class='kpi'><div class='label'>{escape(label)}</div>"
                             f"<div class='value'>{minus(format(cfd, spec))}<small>{escape(unit)}</small></div>"
                             f"<div class='reference'>MoC design {number(moc, spec)}{delta}</div></div>")
            return f"<div class='kpis'>{''.join(cards)}</div>"
        if kind == "h2":
            return f"<h2>{escape(block[1])}</h2>"
        if kind == "bullets":
            return "<ul>" + "".join(f"<li>{minus(escape(item))}</li>" for item in block[1]) + "</ul>"
        if kind == "note":
            return f"<div class='note'>{escape(block[1])}</div>"
        if kind == "table":
            counters["table"] += 1
            _, caption, headers, groups, rows, style = block
            head = ""
            if groups:
                head += "<tr class='group'>" + "".join(f"<th colspan='{span}'>{escape(label)}</th>"
                                                       for label, span in groups) + "</tr>"
            head += "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
            body = "".join("<tr>" + "".join(f"<td>{minus(escape(cell))}</td>" for cell in row) + "</tr>"
                           for row in rows)
            return (f"<div class='table'><div class='caption'><b>Table {counters['table']}.</b> {escape(caption)}"
                    f"</div><table class='{style}'><thead>{head}</thead><tbody>{body}</tbody></table></div>")
        if kind == "figure":
            counters["figure"] += 1
            sizes = [image_size(path) for path in block[1]]
            if len(sizes) == 1:
                width = min(100.0, 100 * sizes[0][2] / container_mm)
                images = f"<img src='{block[1][0]}' style='width:{width:.1f}%'>"
            else:
                ratios = [w / h for w, h, _ in sizes]
                share = [100 * ratio / sum(ratios) for ratio in ratios]
                images = "<div class='row'>" + "".join(
                    f"<img src='{path}' style='width:calc({part:.2f}% - 4pt)'>"
                    for path, part in zip(block[1], share)) + "</div>"
            return (f"<figure>{images}<figcaption><b>Figure {counters['figure']}.</b> {escape(block[2])}"
                    f"</figcaption></figure>")
        if kind == "columns":
            left = "".join(render(item, container_mm * block[3]) for item in block[1])
            right = "".join(render(item, container_mm * (1 - block[3]) - 12 * 0.353) for item in block[2])
            return (f"<div class='columns'><div style='flex:{block[3]}'>{left}</div>"
                    f"<div style='flex:{1 - block[3]}'>{right}</div></div>")
        raise ValueError(f"Unknown report block {kind!r}")

    body = "".join(render(block) for block in blocks)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{escape(blocks[0][1])}</title>"
            f"<style>{REPORT_STYLE}</style></head><body>{body}</body></html>")


def browser():
    import shutil
    candidates = [shutil.which(name) for name in ("msedge", "chrome", "google-chrome", "chromium")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next((path for path in candidates if path and Path(path).exists()), None)


def write_pdf(html_file):
    """Print the HTML report to PDF with a headless Chromium browser."""
    import subprocess
    executable = browser()
    if executable is None:
        print("    No Chrome or Edge found: PDF skipped")
        return None
    pdf_file = html_file.with_suffix(".pdf")
    subprocess.run([executable, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_file}", html_file.as_uri()], check=True, capture_output=True,
                   timeout=120)
    return pdf_file


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=CONFIG_FILE)
    parser.add_argument("--plots-only", action="store_true", help="reuse the extracted data, do not start Fluent")
    args = parser.parse_args()
    config_file = Path(args.config).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    data_dir = output_dir / "report" / "data"
    setup = json.loads((output_dir / "fluent" / "setup_summary.json").read_text(encoding="utf-8"))
    if not args.plots_only:
        case_file = output_dir / "fluent" / workflow["fluent"]["case_filename"]
        if not case_file.with_name(case_file.name.replace(".cas.h5", ".dat.h5")).exists():
            raise FileNotFoundError(f"Solve and save the case first: no data file next to {case_file}")
        mesh = json.loads((output_dir / "mesh" / "mesh_report.json").read_text(encoding="utf-8"))
        channel = mesh["rows"]["stator_1"]["meshes"]["3d"]["summary"]["geometry"]
        spans = {"hub": channel["hub"]["control_points_rz_mm"][0][0] / 1e3,
                 "shroud": channel["shroud"]["control_points_rz_mm"][0][0] / 1e3}
        print("1/3 Extract surfaces and Fluent reports (headless Fluent)")
        extract(case_file, setup, data_dir, workflow["fluent"], spans)
        residual_table(output_dir / "fluent", data_dir)
    print("2/3 Figures and tables")
    report_dir = report(output_dir, workflow)
    pdf_file = write_pdf(report_dir / "report.html")
    print(f"3/3 Report written: {report_dir / 'report.md'}" + (f" and {pdf_file.name}" if pdf_file else ""))


if __name__ == "__main__":
    main()
