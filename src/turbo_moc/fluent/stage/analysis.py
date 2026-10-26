"""Stage performance from the extracted Fluent data, with the MoC design alongside.

Velocity components follow the app's design convention: tangential components
are positive along the stator turning and angles are measured from the axial
direction. Isentropic quantities use the barotropic closure the solver used,
with H(p) = ∫ dp/ρ as the isentropic head.
"""

import json
from types import SimpleNamespace

import numpy as np
import yaml

from ..barotropic_model import BarotropicClosure
from .extraction import FIELDS, ROWS, boundary, midspan_surface


class Kinematics:
    def __init__(self, omega, swirl_sign, flow_sign):
        self.omega, self.swirl_sign, self.flow_sign = omega, swirl_sign, flow_sign

    def components(self, xyz, values):
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        e_theta = np.column_stack([-np.sin(theta), np.cos(theta)])
        velocity = np.column_stack([values["x-velocity"], values["y-velocity"]])
        relative = np.column_stack([values["relative-x-velocity"], values["relative-y-velocity"]])
        return {
            "radius": radius, "theta": theta,
            "c_m": self.flow_sign * values["z-velocity"],
            "c_t": self.swirl_sign * np.sum(velocity * e_theta, axis=1),
            "w_t": self.swirl_sign * np.sum(relative * e_theta, axis=1),
            "u": self.swirl_sign * self.omega * radius,
            "c": np.sqrt(values["x-velocity"] ** 2 + values["y-velocity"] ** 2 + values["z-velocity"] ** 2),
            "w": np.sqrt(values["relative-x-velocity"] ** 2 + values["relative-y-velocity"] ** 2
                         + values["relative-z-velocity"] ** 2),
        }


def mass_averages(arrays, surface, model, kinematics, rotating):
    """Mass-weighted averages over a boundary surface from facet values and facet area vectors."""
    values = {field: arrays[f"{surface}|face|{field}"] for field in FIELDS}
    velocity = np.column_stack([values["x-velocity"], values["y-velocity"], values["z-velocity"]])
    flux = values["density"] * np.sum(velocity * arrays[f"{surface}|normals"], axis=1)
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


def polydata(arrays, surface, values="node"):
    import pyvista as pv
    mesh = pv.PolyData(arrays[f"{surface}|vertices"], arrays[f"{surface}|faces"])
    prefix = f"{surface}|{values}|"
    target = mesh.point_data if values == "node" else mesh.cell_data
    for key in arrays:
        if key.startswith(prefix):
            target[key[len(prefix):]] = arrays[key]
    return mesh


def span_line(mesh, radius):
    line = mesh.contour([radius], scalars=np.hypot(mesh.points[:, 0], mesh.points[:, 1])).strip()
    return line if line.n_points else None


def unwrapped(mesh, theta_ref, r_mid):
    """Midspan cylinder unrolled onto (axial, circumferential arc), both in mm."""
    import pyvista as pv
    theta = np.angle(np.exp(1j * (np.arctan2(mesh.points[:, 1], mesh.points[:, 0]) - theta_ref)))
    flat = pv.PolyData(np.column_stack([mesh.points[:, 2] * 1e3, r_mid * theta * 1e3, np.zeros(mesh.n_points)]),
                       mesh.faces)
    for key in mesh.point_data:
        flat.point_data[key] = mesh.point_data[key]
    return flat.triangulate()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rotor_design_result(output_dir):
    """Vortex rotor result of the latest app session, with its relative stagnation state."""
    results = sorted((output_dir / "jobs").glob("*/*/rotor_result.json"), key=lambda path: path.stat().st_mtime)
    return read_json(results[-1]) if results else None


def compute(output_dir):
    """Load the extracted data and compute stations, stage performance and the design comparison."""
    data_dir = output_dir / "report" / "data"
    setup = read_json(output_dir / "fluent" / "setup_summary.json")
    point = read_json(output_dir / "operating_point.json")
    reports = read_json(data_dir / "fluent_reports.json")
    arrays = dict(np.load(data_dir / "surfaces.npz"))
    design = point["design"]
    blades = design["num_blades"]
    sectors = {row: blades[row] // int(setup.get("passages", {}).get(row, 1)) for row in ROWS}
    state = setup["inlet_state"]
    P0 = float(state["P0"])
    model = BarotropicClosure.from_file(output_dir / "barotropy" / "fluent_expressions_safe.json",
                                        float(state["p_min"]), P0)
    omega = float(setup["rotor_omega_rad_s"])
    surfaces = {name: {row: polydata(arrays, boundary(setup, row, name)) for row in ROWS}
                for name in ("hub", "shroud", "blade")}
    r_hub = float(np.hypot(*surfaces["hub"]["stator_1"].points[:, :2].T).mean())
    r_shroud = float(np.hypot(*surfaces["shroud"]["stator_1"].points[:, :2].T).mean())
    inlet = arrays[f"{boundary(setup, 'stator_1', 'inlet')}|face|z-velocity"]
    kinematics = Kinematics(omega, float(point["swirl_sign"]), float(np.sign(np.mean(inlet))))
    stator_blade = surfaces["blade"]["stator_1"].points
    theta_ref = float(np.angle(np.mean(np.exp(1j * np.arctan2(stator_blade[:, 1], stator_blade[:, 0])))))

    stations = {
        "Stator inlet (0)": mass_averages(arrays, boundary(setup, "stator_1", "inlet"), model, kinematics, False),
        "Stator exit (1)": mass_averages(arrays, boundary(setup, "stator_1", "outlet"), model, kinematics, False),
        "Rotor inlet (1)": mass_averages(arrays, boundary(setup, "rotor_1", "inlet"), model, kinematics, True),
        "Rotor exit (2)": mass_averages(arrays, boundary(setup, "rotor_1", "outlet"), model, kinematics, True),
    }
    exit_, rotor_in, outlet = stations["Stator exit (1)"], stations["Rotor inlet (1)"], stations["Rotor exit (2)"]
    flows = {key: abs(reports["mass_flow"][boundary(setup, row, name)]) * sectors[row] for key, row, name in (
        ("inlet", "stator_1", "inlet"), ("stator_exit", "stator_1", "outlet"),
        ("rotor_inlet", "rotor_1", "inlet"), ("outlet", "rotor_1", "outlet"))}
    torque_passage = sum(reports["moment"]["rotor_1"][wall] for wall in ("blade", "hub"))
    torque = torque_passage * sectors["rotor_1"]
    power = torque * omega
    mass_flow = flows["outlet"]
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
        ("Mass flow rate, one rotor passage", "kg/s", flows["outlet"] / blades["rotor_1"], None, None, ".4f"),
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
        ("Stator exit (1)", "absolute", exit_, "M", "c", "alpha"),
        ("Rotor inlet (1)", "absolute", rotor_in, "M", "c", "alpha"),
        ("Rotor inlet (1)", "relative", rotor_in, "M_rel", "w", "beta"),
        ("Rotor exit (2)", "absolute", outlet, "M", "c", "alpha"),
        ("Rotor exit (2)", "relative", outlet, "M_rel", "w", "beta"),
    ]
    residuals = None
    if (data_dir / "residuals.csv").exists():
        import pandas as pd
        residuals = pd.read_csv(data_dir / "residuals.csv")
    project = yaml.safe_load((output_dir / "project.yaml").read_text(encoding="utf-8"))
    gap = next(row for row in project["rows"] if row["role"] == "rotor")["geometry"].get("axial_interspace_mm", 2.0)
    return SimpleNamespace(
        setup=setup, point=point, reports=reports, arrays=arrays, blades=blades, state=state, P0=P0, model=model,
        omega=omega, hub=surfaces["hub"], blade=surfaces["blade"], r_hub=r_hub, r_shroud=r_shroud,
        r_mid=0.5 * (r_hub + r_shroud), kinematics=kinematics, theta_ref=theta_ref,
        pitch={row: 2 * np.pi / sectors[row] for row in ROWS},
        interface=float(arrays[f"{boundary(setup, 'stator_1', 'outlet')}|vertices"][:, 2].mean()),
        stations=stations, exit_=exit_, rotor_in=rotor_in, outlet=outlet, flows=flows,
        torque_passage=torque_passage, power=power, performance=performance, station_rows=station_rows,
        design_stations=design_stations, residuals=residuals, gap=gap,
        y_plus={row: arrays[f"{boundary(setup, row, 'blade')}|face|y-plus"] for row in ROWS},
        mesh_report=read_json(output_dir / "mesh" / "mesh_report.json"),
        midspan={row: polydata(arrays, midspan_surface(row)) for row in ROWS},
        rotor_design=rotor_design_result(output_dir))


def radius_at(results, span):
    return results.r_hub + span * (results.r_shroud - results.r_hub)
