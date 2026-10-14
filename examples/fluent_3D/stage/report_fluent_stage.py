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
    performance = {
        "Mass flow rate, full annulus [kg/s]": (mass_flow, design["stator"]["mass_flow"]),
        "Mass flow rate, one passage [kg/s]": (m_out, None),
        "Torque, full rotor [N m]": (abs(torque), None),
        "Power [kW]": (power / 1e3, design_stage.get("Power, annulus estimate [kW]")),
        "Specific work, torque [kJ/kg]": (work / 1e3, design_stage.get("Specific work [kJ/kg]")),
        "Specific work, Euler [kJ/kg]": (euler / 1e3, design_stage.get("Specific work [kJ/kg]")),
        "Isentropic drop, total-to-static [kJ/kg]": (drop_ts / 1e3,
                                                     design_stage.get("Isentropic drop, total-to-static [kJ/kg]")),
        "Efficiency, total-to-static [-]": (work / drop_ts, design_stage.get("Efficiency, total-to-static [-]")),
        "Efficiency, total-to-total [-]": (work / drop_tt, None),
        "Degree of reaction [-]": ((model.head(exit_["p"]) - model.head(outlet["p"])) / drop_ts,
                                   design_stage.get("Degree of reaction [-]")),
        "Exit kinetic energy fraction [-]": (outlet["kinetic"] / drop_ts,
                                             design_stage.get("Exit kinetic energy fraction [-]")),
        "Stator kinetic-energy efficiency [-]": (exit_["kinetic"] / stator_drop, 1.0),
    }
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

    def fmt(value, digits=4):
        return "–" if value is None else f"{value:.{digits}g}"

    lines = ["# Stator–rotor stage: Fluent results against the MoC design", "",
             f"Cyclopentane, P₀ = {P0 / 1e3:.1f} kPa, inlet quality {state['Q0']:.3f}, "
             f"rotor exit pressure {point['rotor_exit_pressure'] / 1e3:.2f} kPa, "
             f"{abs(point['rpm']):.0f} rpm, {blades['stator_1']} stator and {blades['rotor_1']} rotor blades.", "",
             "## Mesh", "", "| Row | Hexahedra | Prisms | Min. det. Jacobian | Min. SICN | Blade y⁺ mean / max |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in ROWS:
        summary = mesh_report["rows"][row]["meshes"]["3d"]["summary"]
        cells, quality = summary["cells"], summary["quality"]
        lines.append(f"| {row.split('_')[0].capitalize()} | {cells.get('hexahedra', 0):,} | {cells.get('prisms', 0):,} "
                     f"| {fmt(quality['min_determinant_jacobian'], 3)} | {fmt(quality['min_sicn'], 3)} "
                     f"| {np.mean(y_plus[row]):.1f} / {np.max(y_plus[row]):.1f} |")
    lines += ["", f"Fluent: minimum orthogonal quality {reports['mesh_quality']['min_orthogonal_quality']:.3f}, "
                  f"maximum aspect ratio {reports['mesh_quality']['max_aspect_ratio']:.1f}.", "",
              "## Simulation setup and convergence", "", "| Item | Value |", "| --- | --- |"]
    methods = reports["methods"]
    schemes = methods["discretization_scheme"]
    lines += [f"| Solver | Pressure-based, {methods['p_v_coupling']['flow_scheme']}, steady pseudo-transient |",
              f"| Turbulence | {'SST k-ω' if reports['viscous'] == 'k-omega' else reports['viscous']} |",
              f"| Spatial discretization | pressure {schemes['pressure']}, momentum {schemes['mom']}, "
              f"density {schemes['density']}, k/ω {schemes['k']}; gradients {methods['gradient_scheme']} |",
              f"| Mass imbalance, stator inlet → rotor outlet | {100 * (m_in - m_out) / m_in:+.3f} % |",
              f"| Mixing-plane mass imbalance | {100 * (m_exit - m_rotor) / m_exit:+.3f} % |",
              f"| Reversed mass flux at the rotor exit | {100 * outlet['backflow_fraction']:.2f} % |"]
    if residuals is not None:
        equations = [column for column in residuals.columns if column not in ("iter", "time/iter")]
        final = residuals.iloc[-1]
        lines.append(f"| Iterations | {int(final['iter'])} |")
        lines.append("| Final residuals (min / max) | " + ", ".join(
            f"{eq} {final[eq]:.1e}" for eq in equations) + f" ({min(final[equations]):.1e} / "
            f"{max(final[equations]):.1e}) |")
    lines += ["", "Assumptions: steady RANS; rotor in a moving reference frame; stator–rotor coupling through a "
                  "circumferentially averaged mixing plane; one passage per row with rotational periodicity; "
                  "barotropic equilibrium closure along the inlet isentrope (energy equation off, adiabatic); "
                  "no tip clearance; hub and blades no-slip in their own frame, rotor shroud stationary.", "",
              "## Stage performance", "", "| Quantity | CFD | MoC design |", "| --- | --- | --- |"]
    lines += [f"| {name} | {fmt(cfd)} | {fmt(moc)} |" for name, (cfd, moc) in performance.items()]
    lines += ["", "Isentropic drops use H(p) = ∫dp/ρ of the barotropic closure. Torque and power come from the "
                  "Fluent moment on the rotor blade and hub; Euler work from the mass-averaged r·c_t.", "",
              "## Mass-weighted averages", "",
              "| Station | Frame | p [kPa] | Mach [-] | Velocity [m/s] | Flow angle [deg] | MoC p | MoC Mach | MoC angle |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, frame, values, key, mach, speed, angle in station_rows:
        reference = design_stations.get(key, {})
        lines.append(f"| {name} | {frame} | {values['p'] / 1e3:.2f} | {values[mach]:.3f} | {values[speed]:.1f} "
                     f"| {values[angle]:.2f} | {fmt(reference.get('p', None) and reference['p'] / 1e3)} "
                     f"| {fmt(reference.get('M'))} | {fmt(reference.get('angle'))} |")
    lines += ["", "Angles are measured from the axial direction, positive along the stator turning. Averages "
                  "on the stator outlet are taken on the stator side of the mixing plane.", ""]

    figure_list = []

    def save(fig, name, caption):
        fig.savefig(figures / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_list.append((name, caption))

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
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
    image = plotter.screenshot(return_img=True)
    plotter.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[0].set_title(f"{options['passages_3d']} passages per row (stator blue-grey, rotor red)")
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
        ax.fill_betweenx(radii, lead, trail, color=COLORS[row], alpha=0.75, lw=0,
                         label=row.split("_")[0].capitalize())
        for name in ("inlet", "outlet"):
            zz = arrays[f"{boundary(setup, row, name)}|vertices"][:, 2].mean() * 1e3
            ax.axvline(zz, color="#8a9aa1", lw=1, ls="--")
        z_all += [hub[row].points[:, 2].min() * 1e3, hub[row].points[:, 2].max() * 1e3]
    ax.plot([min(z_all), max(z_all)], [r_hub * 1e3] * 2, color="k", lw=2)
    ax.plot([min(z_all), max(z_all)], [r_shroud * 1e3] * 2, color="k", lw=2)
    for span, style in SPAN_STYLES.items():
        ax.plot([min(z_all), max(z_all)], [radius_at(span) * 1e3] * 2, style, color="#8a9aa1", lw=0.8)
        ax.text(max(z_all), radius_at(span) * 1e3, f" {int(span * 100)} %", va="center", fontsize=8,
                color="#5b6b72")
    interface = arrays[f"{boundary(setup, 'stator_1', 'outlet')}|vertices"][:, 2].mean() * 1e3
    ax.text(interface, r_shroud * 1e3 + 0.4, "mixing plane", ha="center", fontsize=8, color="#5b6b72")
    ax.set_xlabel("Axial coordinate z [mm]")
    ax.set_ylabel("Radius r [mm]")
    ax.set_title("Meridional view: constant hub and shroud radii (no flaring)")
    ax.set_ylim(r_hub * 1e3 - 1.5, r_shroud * 1e3 + 1.5)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "geometry", "Three-dimensional blading and meridional channel; dashed lines mark the row inlets, "
                          "the mixing plane and the outlet, thin lines the 25/50/75 % spans.")

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

    def frame(ax, rows):
        ax.set_xlim(min(flats[row].points[:, 0].min() for row in rows),
                    max(flats[row].points[:, 0].max() for row in rows))
        ax.set_ylim(*band)
        ax.set_aspect("equal")
        ax.set_xlabel("Axial coordinate z [mm]")
        ax.set_ylabel("Circumferential arc r·θ [mm]")

    def draw(ax, row, field, **style):
        flat = flats[row]
        triangles = flat.faces.reshape(-1, 4)[:, 1:]
        lowest = int(np.floor((band[0] - flat.points[:, 1].max()) / arc[row]))
        highest = int(np.ceil((band[1] - flat.points[:, 1].min()) / arc[row]))
        result = None
        for k in range(lowest, highest + 1):
            tri = mtri.Triangulation(flat.points[:, 0], flat.points[:, 1] + k * arc[row], triangles)
            result = ax.tripcolor(tri, flat.point_data[field], shading="gouraud", **style)
            if field == "mach":
                ax.tricontour(tri, flat.point_data[field], levels=[1.0], colors="w", linewidths=0.8)
        return result

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    mach_max = max(float(np.max(flats[row].point_data["mach"])) for row in ROWS)
    for ax, field, label, cmap, limits in (
            (axes[0], "mach", "Mach number [-] (stator absolute, rotor relative)", "turbo", (0, mach_max)),
            (axes[1], "p_ratio", "Static pressure p/P₀ [-]", "viridis", (0, 1))):
        for row in ROWS:
            artist = draw(ax, row, field, cmap=cmap, vmin=limits[0], vmax=limits[1])
        fig.colorbar(artist, ax=ax, label=label, pad=0.01)
        ax.axvline(interface, color="w", lw=1.2, ls="--")
        frame(ax, ROWS)
    axes[0].set_xlabel("")
    axes[0].set_title("Midspan (r = %.1f mm), unwrapped; white line M = 1, dashed line the mixing plane" % (r_mid * 1e3))
    save(fig, "midspan_mach_pressure", "Midspan Mach number and static pressure on the unwrapped cylinder.")

    from matplotlib.colors import LogNorm
    for row, label in (("stator_1", "absolute (stationary) frame"), ("rotor_1", "relative (rotating) frame")):
        width = flats[row].points[:, 0].max() - flats[row].points[:, 0].min()
        fig, ax = plt.subplots(figsize=(1.2 + 7.5 * width / (band[1] - band[0]), 7.5))
        gradient = flats[row].point_data["grad_rho"]
        flats[row].point_data["grad_rho_plot"] = np.clip(gradient, np.percentile(gradient, 5), None)
        artist = draw(ax, row, "grad_rho_plot", cmap="Greys", norm=LogNorm(
            vmin=np.percentile(gradient, 5), vmax=np.percentile(gradient, options["schlieren_percentile"])))
        fig.colorbar(artist, ax=ax, label=r"$|\nabla\rho|$ [kg m$^{-4}$]", pad=0.02)
        frame(ax, [row])
        ax.set_title(f"{row.split('_')[0].capitalize()} midspan " + r"$|\nabla\rho|$" + f", {label}")
        save(fig, f"density_gradient_{row.split('_')[0]}",
             f"Numerical schlieren |∇ρ| at midspan, {row.split('_')[0]} passage ({label}), logarithmic grey scale "
             f"between the 5th and {options['schlieren_percentile']}th percentiles.")

    rothalpy = rotor_in["rothalpy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col")
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
                axes[0, col].plot(x[order], pressure[order] / P0, style, color=COLORS[row], lw=1.3,
                                  label=f"{int(span * 100)} % span" if cell == 0 else None)
                axes[1, col].plot(x[order], mach[order], style, color=COLORS[row], lw=1.3)
            loading[f"{row}|{span}"] = {"x": x.tolist(), "p": pressure.tolist(), "M_is": mach.tolist()}
        axes[0, col].set_title(f"{row.split('_')[0].capitalize()} blade")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_xlabel("Normalized axial chord (z − z_LE)/c_ax [-]")
    axes[0, 0].set_ylabel("Static pressure p/P₀ [-]")
    axes[1, 0].set_ylabel("Isentropic Mach number [-]")
    axes[1, 1].set_ylabel("Relative isentropic Mach number [-]")
    save(fig, "blade_loading", "Blade pressure and isentropic Mach number at 25, 50 and 75 % span. Stator: from "
                               "P₀; rotor: relative, from the rotor-inlet rothalpy at the local radius.")

    outlet_mesh = polydata(arrays, boundary(setup, "stator_1", "outlet"))
    values = {field: outlet_mesh.point_data[field] for field in NODE_FIELDS}
    parts = kinematics.components(outlet_mesh.points, values)
    outlet_mesh.point_data["mach"] = parts["c"] / model.speed_of_sound(values["absolute-pressure"])
    outlet_mesh.point_data["alpha"] = np.degrees(np.arctan2(parts["c_t"], parts["c_m"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    reference = design_stations["Stator exit (1)|absolute"]
    for span, style in SPAN_STYLES.items():
        line = span_line(outlet_mesh, radius_at(span))
        theta = np.angle(np.exp(1j * (np.arctan2(line.points[:, 1], line.points[:, 0]) - theta_ref)))
        position = (theta - theta.min()) / pitch["stator_1"]
        order = np.argsort(position)
        for ax, field, scale in ((axes[0], "mach", 1.0), (axes[1], "absolute-pressure", 1 / P0),
                                 (axes[2], "alpha", 1.0)):
            ax.plot(position[order], line.point_data[field][order] * scale, style, color=COLORS["stator_1"], lw=1.3,
                    label=f"{int(span * 100)} % span")
    for ax, cfd, moc, label in ((axes[0], exit_["M"], reference["M"], "Mach number [-]"),
                                (axes[1], exit_["p"] / P0, reference["p"] / P0, "Static pressure p/P₀ [-]"),
                                (axes[2], exit_["alpha"], reference["angle"], "Flow angle α [deg]")):
        ax.axhline(cfd, color="#e08a3c", lw=1.4, label="CFD mass average")
        ax.axhline(moc, color="#5b6b72", lw=1.4, ls="-.", label="MoC design")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Normalized pitchwise position [-]")
    axes[0].set_title("Stator exit (stator side of the mixing plane)")
    save(fig, "stator_exit_pitchwise", "Pitchwise Mach number, static pressure and flow angle at the stator exit, "
                                       "against the mass average and the MoC design.")

    if residuals is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for column in residuals.columns:
            if column not in ("iter", "time/iter"):
                ax.semilogy(residuals["iter"], residuals[column], lw=1.2, label=column)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Scaled residual [-]")
        ax.legend(fontsize=8, ncol=3)
        save(fig, "residuals", "Residual history.")

    lines += ["## Figures", ""]
    lines += [f"![{caption}](figures/{name}.png)\n*{caption}*\n" for name, caption in figure_list]
    report_dir = output_dir / "report"
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "report.json").write_text(json.dumps({
        "stations": stations, "performance": {key: {"cfd": cfd, "moc": moc} for key, (cfd, moc) in performance.items()},
        "torque_passage": torque_passage, "power_W": power, "fluent_reports": reports, "blade_loading": loading},
        indent=2, default=float), encoding="utf-8")
    return report_dir


PDF_STYLE = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; color: #1d2b31; line-height: 1.4; }
h1 { font-size: 17pt; color: #233c46; border-bottom: 2px solid #233c46; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #233c46; margin-top: 18px; break-after: avoid; }
table { border-collapse: collapse; width: 100%; box-sizing: border-box; margin: 6px 0 10px; font-size: 8.5pt;
        break-inside: avoid; }
th, td { border: 1px solid #c9d2d6; padding: 3px 6px; text-align: left; }
th { background: #eef2f3; }
img { max-width: 100%; max-height: 118mm; display: block; margin: 10px auto 4px; }
p:has(img) { break-inside: avoid; margin-bottom: 14px; }
em { color: #4a5b62; font-size: 9pt; }
"""


def browser():
    import shutil
    candidates = [shutil.which(name) for name in ("msedge", "chrome", "google-chrome", "chromium")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next((path for path in candidates if path and Path(path).exists()), None)


def write_pdf(markdown_file):
    """Render the Markdown report to HTML and print it to PDF with a headless Chromium browser."""
    import subprocess
    from markdown_it import MarkdownIt
    executable = browser()
    if executable is None:
        print("    No Chrome or Edge found: PDF skipped")
        return None
    body = MarkdownIt("commonmark").enable("table").render(markdown_file.read_text(encoding="utf-8"))
    html_file = markdown_file.with_suffix(".html")
    html_file.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{PDF_STYLE}</style>"
                         f"</head><body>{body}</body></html>", encoding="utf-8")
    pdf_file = markdown_file.with_suffix(".pdf")
    subprocess.run([executable, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_file}", html_file.as_uri()], check=True, capture_output=True,
                   timeout=120)
    html_file.unlink()
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
    pdf_file = write_pdf(report_dir / "report.md")
    print(f"3/3 Report written: {report_dir / 'report.md'}" + (f" and {pdf_file.name}" if pdf_file else ""))


if __name__ == "__main__":
    main()
