"""2D rotor alone on the medium midspan mesh, translating at U = omega r_m, fed with the midspan state of the 3D stage.

python 07_midspan_2D_rotor.py extract [processors]   midspan rotor-inlet state of the 3D medium stage -> inlet_midspan.json
python 07_midspan_2D_rotor.py setup [processors]     writes the 2D case
python 07_midspan_2D_rotor.py solve [processors]     solves it with the staged procedure of run_night_recipe.py
python 07_midspan_2D_rotor.py report [processors]    mass flow, blade force, work and the inlet/outlet averages

Mesh: meshes/midspan_2D/rotor_medium_midspan_2D.cgns, x axial and y = r_m (theta - theta0), flow along +x; +y is +theta, the
3D rotation direction, so the rotor zone translates at +U along y. On a cylinder the rotation forces are radial, so the
blade-to-blade equations are those of the translating cascade; radial effects are left out. Flows are per metre of depth.

Inlet (the rotor inlet plane, x = 49 mm), absolute frame, the 3D stage's pitch-averaged midspan state: mass-flow inlet
(rho V_ax pitch) with the absolute flow direction, static pressure, k and omega (default), or with ROTOR_2D_INLET=pressure
a pressure inlet whose total pressure follows the barotropic isentrope (Fluent's pressure-defined density otherwise takes
p0 - p = rho V^2/2; planar.total_pressure_expression). Outlet: 94.4 kPa as in the stage.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from turbo_moc.fluent.barotropic_model import BarotropicClosure, generate_barotropic_model
from turbo_moc.fluent.common import boundary_types, configure_numerics, load_dependencies
from turbo_moc.fluent.stage import planar, planar_mesh
from turbo_moc.fluent.stage.case import ROTOR, configure_material, zone
from turbo_moc.io.cgns_adf import read_adf

from settings import STATOR_CONFIG

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fluent_2D_stage"))
stage_3d = importlib.import_module("01_setup_stage_case")   # Same speed, outlet pressure, options and barotropic model
recipe = importlib.import_module("run_night_recipe")

MESH_FILE = HERE / "meshes" / "midspan_2D" / "rotor_medium_midspan_2D.cgns"
STAGE_3D = Path(os.environ.get("DROPLET_STAGE_3D", Path.home() / "OneDrive - Danmarks Tekniske Universitet" /
                               "Documents/projects/ASME TurboExpo2027/paper 3 - DPM_impingement/Barotropic_simulations/"
                               "Grid independence study/Simulations/medium_sim/stage_fluent_medium_2order_tsn_converged.cas.h5"))
RUN_DIR = HERE / "run_2D_rotor"
INLET_FILE = RUN_DIR / "inlet_midspan.json"
# Inlets: mass_flow / pressure / pressure_constant (absolute frame: these cycled or reversed on 2026-11-07, the absolute
# angle alpha lets the rotor incidence drift with the mass flow), pressure_relative / mass_flow_relative (rotor frame,
# beta fixed: pressure_relative converged steadily but passed 9.5% more mass than 3D, as planar 2D has AVDR = 1 while
# the 3D midspan stream tube contracts, AVDR 1.083)
INLET = os.environ.get("ROTOR_2D_INLET", "mass_flow")
# Outlet: "stage" = the stage's 94.4 kPa (the 3D hub value, radial equilibrium); "midspan" = the 3D static pressure at
# the rotor outlet at midspan (95.39 kPa, midspan_3D_inlet_outlet.json)
OUTLET = os.environ.get("ROTOR_2D_OUTLET", "stage")
TAG = INLET + ("_pout_midspan" if OUTLET == "midspan" else "")
CASE_NAME = f"rotor_2D_{TAG}.cas.h5"
SOLVED_NAME = f"rotor_2D_{TAG}_solved.cas.h5"
SUMMARY_NAME = f"setup_summary_{TAG}.json"
START_RAMP = ((0.02, 100), (0.05, 150), (0.1, 150))      # (time scale, iterations) before the recipe's phases
START_FINAL = 0.3                                        # First-order time scale of the phases (halved after a crash)
BLADES = 120
EDGE_NAMES = {"INFLOW": "inlet", "OUTFLOW": "outlet", "BLADE": "blade", "PER1": "periodic_left", "PER2": "periodic_right"}
FIELDS_3D = ("absolute-pressure", "density", "x-velocity", "y-velocity", "z-velocity", "turb-kinetic-energy",
             "specific-diss-rate")


def read_section():
    """The rotor midspan mesh as a planar_mesh section, with the radius of the unwrapping."""
    zone_2d = read_adf(str(MESH_FILE)).find_all("CGNSBase_t")[0].find_all("Zone_t")[0]
    grid = zone_2d.get(label="GridCoordinates_t")
    xy = np.column_stack([np.asarray(grid.get(f"Coordinate{axis}").data, dtype=float) for axis in "XY"])
    sections = {section.name: np.asarray(section.get("ElementConnectivity").data).ravel().astype(np.int64) - 1
                for section in zone_2d.find_all("Elements_t")}
    note = zone_2d.get("SurfaceExtraction").text
    radius = float(note.split("cylinder of radius ")[1].split()[0])
    return {"xy": xy, "cells": list(sections["MIDSPAN"].reshape(-1, 4)), "radius": radius,
            "boundaries": {name: sections[label].reshape(-1, 2) for label, name in EDGE_NAMES.items()}}


def pitch(section):
    left, right = (section["xy"][np.unique(section["boundaries"][f"periodic_{side}"])] for side in ("left", "right"))
    left, right = left[np.argsort(left[:, 0])], right[np.argsort(right[:, 0])]
    shift = right - left
    if len(left) != len(right) or np.ptp(shift, axis=0).max() > 1e-6:
        raise ValueError(f"The periodic sides are not one constant translation apart (spread {np.ptp(shift, axis=0)} m).")
    return float(shift[:, 1].mean())


def outlet_pressure():
    if OUTLET == "midspan":
        midspan = json.loads((RUN_DIR / "midspan_3D_inlet_outlet.json").read_text(encoding="utf-8"))
        return float(midspan["rotor_1_outlet"]["absolute-pressure"])
    return stage_3d.P_OUT


def extract(processors):
    """Pitch-averaged rotor-inlet state of the 3D stage at the midspan radius (machine axis x, theta = atan2(z, y)).

    The rotor inlet is the downstream side of the mixing plane, so its values are uniform in pitch within each
    spanwise band: the faces are binned in radius and the band values interpolated to the midspan radius.
    """
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest
    _, pyfluent = load_dependencies(False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    radius = read_section()["radius"]
    solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                    processor_count=processors, ui_mode="no_gui", start_timeout=600, cwd=str(RUN_DIR))
    try:
        solver.settings.file.read_case_data(file_name=str(STAGE_3D))
        data, surface = solver.fields.field_data, "rotor_1_inlet"
        centroid = np.asarray(data.get_field_data(SurfaceFieldDataRequest(
            surfaces=[surface], data_types=[SurfaceDataType.FacesCentroid]))[surface].face_centroids,
            dtype=float).reshape(-1, 3)
        values = {field: np.asarray(data.get_field_data(ScalarFieldDataRequest(
            field_name=field, surfaces=[surface], node_value=False))[surface], dtype=float) for field in FIELDS_3D}
    finally:
        solver.exit()
    r = np.hypot(centroid[:, 1], centroid[:, 2])
    theta = np.arctan2(centroid[:, 2], centroid[:, 1])
    values["v_axial"] = values["x-velocity"]
    values["v_theta"] = values["z-velocity"] * np.cos(theta) - values["y-velocity"] * np.sin(theta)
    values["v_radial"] = values["y-velocity"] * np.cos(theta) + values["z-velocity"] * np.sin(theta)
    levels = np.round(r, 6)                              # TurboGrid span layers: one radius per band of faces
    radii = np.unique(levels)
    keys = ("absolute-pressure", "density", "v_axial", "v_theta", "v_radial", "turb-kinetic-energy", "specific-diss-rate")
    state = {key: float(np.interp(radius, radii, [values[key][levels == level].mean() for level in radii]))
             for key in keys}
    nearest = levels == radii[np.argmin(np.abs(radii - radius))]
    spread = {key: float(np.ptp(values[key][nearest])) for key in keys}
    pitch_m = 2 * np.pi * radius / BLADES
    state.update(radius_m=radius, x_m=float(centroid[:, 0].mean()), source=str(STAGE_3D), span_levels=len(radii),
                 mass_flow_per_depth=state["density"] * state["v_axial"] * pitch_m,
                 alpha_deg=float(np.degrees(np.arctan2(state["v_theta"], state["v_axial"]))),
                 pitch_spread_at_midspan=spread)
    INLET_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


def inflow(state, closure, speed):
    """Absolute total pressure on the barotropic isentrope from the in-plane velocity, and the relative Mach number."""
    velocity_sq = state["v_axial"] ** 2 + state["v_theta"] ** 2
    p = state["absolute-pressure"]
    total = float(closure.pressure_from_head(closure.head(p) + 0.5 * velocity_sq))
    w = np.hypot(state["v_axial"], state["v_theta"] - speed)
    return {"P0": total, "P0_fluent": p + 0.5 * state["density"] * velocity_sq, "p": p, "V": float(np.sqrt(velocity_sq)),
            "W": float(w), "P0_rel_fluent": float(p + 0.5 * state["density"] * w ** 2),
            "beta_deg": float(np.degrees(np.arctan2(state["v_theta"] - speed, state["v_axial"]))),
            "M_abs": float(np.sqrt(velocity_sq) / closure.speed_of_sound(p)), "M_rel": float(w / closure.speed_of_sound(p))}


def setup(processors):
    barotropy, pyfluent = load_dependencies(False)
    state = json.loads(INLET_FILE.read_text(encoding="utf-8"))
    section = read_section()
    sector = pitch(section)
    speed = stage_3d.RPM * 2 * np.pi / 60 * section["radius"]
    mesh_file = planar_mesh.write_cgns(RUN_DIR / "mesh" / "rotor_2D.cgns", {"rotor_1": section}, length_scale=1.0)
    config = yaml.safe_load(STATOR_CONFIG.read_text(encoding="utf-8"))
    parameters = dict(config["rows"][0]["geometry"]["method_of_characteristics"], fluid_name="Cyclopentane")
    expressions, inlet_state = generate_barotropic_model(barotropy, {"nozzle": parameters}, {}, stage_3d.BAROTROPIC,
                                                         RUN_DIR / "barotropy")
    closure = BarotropicClosure.from_file(RUN_DIR / "barotropy" / "fluent_expressions_safe.json",
                                          float(inlet_state["p_min"]), float(inlet_state["P0"]))
    flow = inflow(state, closure, speed)
    print(f"pitch {sector * 1e3:.4f} mm, U = {speed:.4f} m/s; inflow {flow}", flush=True)
    options = dict(stage_3d.OPTIONS, case_filename=CASE_NAME)
    options["residual_equations"] = {key: value for key, value in options["residual_equations"].items()
                                     if key != "z-velocity"}
    solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=2, precision="double",
                                    processor_count=processors, ui_mode="no_gui", start_timeout=600, cwd=str(RUN_DIR))
    try:
        settings = solver.settings
        settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
        settings.mesh.check()
        types = boundary_types(settings.setup.boundary_conditions)
        zones = {f"{ROTOR}_{name}": zone(types, "rotor_1", name) for name in planar_mesh.BOUNDARIES}
        names = configure_material(settings, expressions, options["material_name"])
        modify = settings.mesh.modify_zones
        modify.zone_type(zone_names=[zones[f"{ROTOR}_inlet"]], new_type="pressure-inlet")
        modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
        modify.make_periodic(zone_name=zones[f"{ROTOR}_periodic_left"], shadow_zone_name=zones[f"{ROTOR}_periodic_right"],
                             rotate_periodic=False, create=True, auto_translation=False, direction=[0.0, sector])
        periodic = settings.setup.boundary_conditions.periodic
        for name in periodic.get_object_names():
            periodic[name].periodic.rotationally_periodic = "Translational"
        configure_numerics(settings, options)
        frame = settings.setup.cell_zone_conditions.fluid[names[ROTOR]].reference_frame
        frame.frame_motion = True
        frame.reference_frame_velocity = [0.0, speed]
        alpha = np.radians(state["alpha_deg"])
        if INLET.startswith("pressure"):
            inlet = settings.setup.boundary_conditions.pressure_inlet[zones[f"{ROTOR}_inlet"]]
            if INLET == "pressure":
                settings.setup.named_expressions["inlet_total_pressure"] = {"definition": planar.total_pressure_expression(
                    closure, flow["P0"], flow["p"])}
                inlet.momentum.gauge_total_pressure.value = "inlet_total_pressure"
            elif INLET == "pressure_relative":   # rotor frame: p0_rel = p + rho W^2 / 2 and the relative angle beta fixed,
                inlet.momentum.reference_frame = "Relative to Adjacent Cell Zone"   # so incidence cannot drift with mass
                inlet.momentum.gauge_total_pressure.value = flow["P0_rel_fluent"]  # flow; inlet subsonic (M_rel 0.67)
                alpha = np.arctan2(state["v_theta"] - speed, state["v_axial"])
            else:   # pressure_constant: Fluent's own p0 = p + rho V^2 / 2 at the 3D state, what the 3D mixing plane passed on
                inlet.momentum.gauge_total_pressure.value = flow["P0_fluent"]
            inlet.momentum.prevent_reverse_flow = True
            inlet.momentum.supersonic_or_initial_gauge_pressure.value = flow["p"]
            inlet.momentum.direction_specification_method = "Direction Vector"
        else:
            modify.zone_type(zone_names=[zones[f"{ROTOR}_inlet"]], new_type="mass-flow-inlet")
            inlet = settings.setup.boundary_conditions.mass_flow_inlet[zones[f"{ROTOR}_inlet"]]
            if INLET == "mass_flow_relative":   # 3D midspan mass flow with the relative angle beta fixed (rotor frame)
                inlet.momentum.reference_frame = "Relative to Adjacent Cell Zone"
                alpha = np.arctan2(state["v_theta"] - speed, state["v_axial"])
            inlet.momentum.mass_flow_specification = "Mass Flow Rate"
            inlet.momentum.mass_flow_rate.value = state["mass_flow_per_depth"]
            inlet.momentum.supersonic_gauge_pressure.value = flow["p"]
            inlet.momentum.direction_specification = "Direction Vector"
        for component, value in zip(inlet.momentum.flow_direction, (np.cos(alpha), np.sin(alpha))):
            component.value = value
        inlet.turbulence.turbulence_specification = "K and Omega"
        turbulence = inlet.turbulence.get_active_child_names()
        k_name = next(name for name in turbulence if "kinetic" in name)
        omega_name = next(name for name in turbulence if "dissipation" in name)
        getattr(inlet.turbulence, k_name).value = state["turb-kinetic-energy"]
        getattr(inlet.turbulence, omega_name).value = state["specific-diss-rate"]
        planar.outlet_boundary(settings, zones, outlet_pressure(), options)
        settings.setup.boundary_conditions.pressure_outlet[zones[f"{ROTOR}_outlet"]].momentum.prevent_reverse_flow = True
        settings.setup.named_expressions["relative_mach"] = {"definition": (
            f"sqrt(Velocity.x^2 + (Velocity.y - {speed:.6f} [m/s])^2) / barotropic_speed_of_sound")}
        settings.mesh.check()
        settings.file.batch_options.confirm_overwrite = False
        settings.file.write_case(file_name=str(RUN_DIR / CASE_NAME))
        print("inlet", inlet.get_state(), flush=True)
    finally:
        solver.exit()
    (RUN_DIR / SUMMARY_NAME).write_text(json.dumps({
        "inflow": flow, "inlet_midspan_3D": state, "blade_speed_m_s": speed, "pitch_m": sector, "radius_m": section["radius"],
        "inlet_type": INLET, "outlet_pressure": outlet_pressure(), "rpm": stage_3d.RPM, "cells": len(section["cells"]), "zones": zones,
        "cell_zones": names}, indent=2, default=float), encoding="utf-8")
    print("saved", RUN_DIR / CASE_NAME, flush=True)


def launch_2d(processors):
    _, pyfluent = load_dependencies(False)
    return pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=2, precision="double",
                                  processor_count=processors, ui_mode="no_gui", start_timeout=600, cwd=str(RUN_DIR))


def start_from_3d(settings, monitor, attempt, final):
    """Standard initialization with the 3D midspan inflow (absolute frame), a gentle first-order ramp, then the recipe's
    phases. A hybrid initialization of the barotropic flow started a reversal at the outlet that diverged by iteration
    ~110 with the pressure inlet. The ramp scales with ``final``, halved after each crash."""
    state = json.loads(INLET_FILE.read_text(encoding="utf-8"))
    initialization = settings.solution.initialization
    monitor.crashed, monitor.residuals, monitor.iteration = False, None, 0
    initialization.initialization_type = "standard"
    if "reference_frame" in initialization.get_active_child_names():
        initialization.reference_frame = "absolute"
    defaults = initialization.defaults
    for name, value in (("pressure", state["absolute-pressure"]), ("x-velocity", state["v_axial"]),
                        ("y-velocity", state["v_theta"]), ("k", state["turb-kinetic-energy"]),
                        ("omega", state["specific-diss-rate"])):
        if name in defaults.get_state():
            defaults[name] = value
    recipe.log(f"attempt {attempt}: standard initialization {defaults.get_state()}")
    initialization.standard_initialize()
    recipe.set_schemes(settings, "first")
    for fraction, iterations in START_RAMP:
        scale = fraction * final / START_FINAL
        recipe.log(f"attempt {attempt}: {iterations} iterations at time scale {scale:.3g}")
        if not recipe.run(settings, monitor, iterations, scale):
            return False
    return recipe.phases(settings, monitor, RUN_DIR, final=final)


def solve(processors):
    (RUN_DIR / "autosave").mkdir(exist_ok=True)
    solver = launch_2d(processors)
    monitor = recipe.Monitor()
    solver.transcript.register_callback(monitor)
    settings = solver.settings
    try:
        settings.file.read_case(file_name=str(RUN_DIR / CASE_NAME))
        settings.file.batch_options.confirm_overwrite = False
        relax = settings.solution.controls.pseudo_time_explicit_relaxation_factor.global_dt_pseudo_relax
        for equation, value in recipe.DEFAULT_RELAXATION.items():
            if equation in set(relax.get_state()):
                relax[equation] = value
        auto = settings.file.auto_save
        auto.root_name = str(RUN_DIR / "autosave" / f"rotor_{TAG}")
        auto.data_frequency = 250
        auto.retain_most_recent_files = True
        auto.max_files = 2
        final = START_FINAL
        for attempt in range(1, recipe.ATTEMPTS + 1):
            try:
                if attempt <= 2:
                    done = start_from_3d(settings, monitor, attempt, final)
                else:   # Fallback: the recipe's hybrid start; a first hybrid initialization often crashes, a second works
                    done = recipe.procedure(settings, monitor, RUN_DIR, attempt, final)
            except Exception as error:
                recipe.log(f"attempt {attempt} raised {str(error).splitlines()[0][:120]}")
                done = False
            if done:
                recipe.log(f"finished at iteration {monitor.iteration}, residuals {monitor.residuals}")
                break
            recipe.log(f"attempt {attempt} crashed at iteration {monitor.iteration}: halving the time scales")
            final /= 2
        else:
            recipe.log("all attempts crashed")
        settings.file.write_case_data(file_name=str(RUN_DIR / SOLVED_NAME))
        recipe.log("saved", RUN_DIR / SOLVED_NAME)
    finally:
        solver.exit()


def report(processors):
    from ansys.fluent.core import ScalarFieldDataRequest
    summary = json.loads((RUN_DIR / SUMMARY_NAME).read_text(encoding="utf-8"))
    zones, speed = summary["zones"], summary["blade_speed_m_s"]
    solver = launch_2d(processors)
    settings = solver.settings
    try:
        settings.file.read_case_data(file_name=str(RUN_DIR / SOLVED_NAME))
        definitions = settings.solution.report_definitions
        for key in ("inlet", "outlet"):
            definitions.surface.create(name=f"mass_{key}")
            definitions.surface[f"mass_{key}"].report_type = "surface-massflowrate"
            definitions.surface[f"mass_{key}"].surface_names = [zones[f"{ROTOR}_{key}"]]
        definitions.force.create(name="blade_force_y")
        definitions.force["blade_force_y"].zones = [zones[f"{ROTOR}_blade"]]
        definitions.force["blade_force_y"].force_vector = [0.0, 1.0]
        values = {key: value[0] for item in definitions.compute(report_defs=["mass_inlet", "mass_outlet", "blade_force_y"])
                  for key, value in item.items()}
        work = values["blade_force_y"] * speed / values["mass_inlet"]
        print(f"per metre of depth and passage: mass flow {values['mass_inlet']:.6g} / {values['mass_outlet']:.6g} kg/s, "
              f"blade force along U {values['blade_force_y']:.6g} N, specific work {work:.1f} J/kg", flush=True)
        data = solver.fields.field_data
        for key in ("inlet", "outlet"):
            surface = zones[f"{ROTOR}_{key}"]
            mean = {field: float(np.mean(data.get_field_data(ScalarFieldDataRequest(
                field_name=field, surfaces=[surface], node_value=False))[surface]))
                for field in ("absolute-pressure", "density", "x-velocity", "y-velocity")}
            print(f"{key}: p {mean['absolute-pressure']:.0f} Pa, rho {mean['density']:.4f}, V_ax {mean['x-velocity']:.2f}, "
                  f"V_theta {mean['y-velocity']:.2f}, W_theta {mean['y-velocity'] - speed:.2f} m/s, "
                  f"alpha {np.degrees(np.arctan2(mean['y-velocity'], mean['x-velocity'])):.2f} deg", flush=True)
        reference = summary["inlet_midspan_3D"]
        print(f"3D midspan reference: rho V_ax pitch {reference['mass_flow_per_depth']:.6g} kg/s per metre (2D/3D "
              f"{values['mass_inlet'] / reference['mass_flow_per_depth']:.4f}), static pressure "
              f"{reference['absolute-pressure']:.0f} Pa, alpha {reference['alpha_deg']:.2f} deg", flush=True)
    finally:
        solver.exit()


if __name__ == "__main__":
    mode = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    {"extract": extract, "setup": setup, "solve": solve, "report": report}[mode](count)
    sys.stdout.flush()
    os._exit(0)  # pyfluent's transcript-streaming thread would keep the interpreter alive
