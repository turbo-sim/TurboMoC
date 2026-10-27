"""Paper 3 stage: MoC stator (44 blades) + TurboDash impulse rotor (120 blades), TurboGrid meshes, barotropic Fluent.

python 01_setup_stage_case.py <no_gui|gui> [processors]

Meshes: put TurboGrid's stator.gtm and rotor.gtm in settings.MESH_DIR. Fluent imports CFX definitions, not .gtm, so
each is converted with cfx5gtmconv to a .def next to it (again whenever the .gtm is newer). Axis x. After the rotor
is appended it is translated axially so its inlet sits AXIAL_GAP downstream of the stator outlet.
Writes the case, setup_summary.json and the barotropic model to settings.RUN_DIR.
"""
import json
import subprocess
import sys

import numpy as np
import yaml

from turbo_moc.fluent.common import MACH_EXPRESSION, boundary_types, configure_numerics, load_dependencies
from turbo_moc.fluent.barotropic_model import generate_barotropic_model
from turbo_moc.fluent.stage.case import STATOR, ROTOR, configure_boundaries, configure_material, create_turbo_interface

from settings import CASE_NAME, MESH_DIR, RUN_DIR, STATOR_CONFIG, cfx5gtmconv

RPM, P_OUT = 1235.1943016058071, 94400.0   # Rotor speed [rpm], outlet static pressure [Pa]
AXIAL_GAP = 0.0                             # Rotor inlet minus stator outlet axial position [m]; 0: the planes coincide
OPTIONS = {
    "product_version": "24.2", "material_name": "barotropic_fluid", "case_filename": CASE_NAME,
    "turbulence_intensity": 0.05, "turbulence_viscosity_ratio": 10.0,
    "density_relaxation": 0.5, "turbulence_relaxation": 0.3, "high_order_relaxation_factor": 0.15,
    "solution_strategy": [{"iterations": 100, "time_scale_factor": 0.05, "order": "first"}],
    "residual_equations": {name: 1.0e-4 for name in ("continuity", "x-velocity", "y-velocity", "z-velocity", "k", "omega")},
}
BAROTROPIC = {"calculation_type": "equilibrium", "efficiency": 1.0, "pressure_ratio": 251300.0 / (P_OUT / 100),
              "polynomial_degree": 8, "segment_pressure_ratio": 1.3, "ODE_solver": "LSODA", "ODE_tolerance": 1.0e-9,
              "save_figures": True}
PARTS = {"inlet": ["inflow-passage"], "outlet": ["outflow-outblock"], "blade": ["bld-"], "hub": ["hub-"],
         "shroud": ["shroud-"], "periodic_left": ["per1-"], "periodic_right": ["per2-"]}
JOIN = ["outflow-passage", "inflow-outblock"]   # TurboGrid passage block / outlet block internal face


def convert_meshes():
    """TurboGrid .gtm -> CFX .def for each row, when the .def is missing or older than its .gtm."""
    for row in ("stator", "rotor"):
        source, target = MESH_DIR / f"{row}.gtm", MESH_DIR / f"{row}.def"
        if source.exists() and (not target.exists() or source.stat().st_mtime > target.stat().st_mtime):
            print(f"converting {source.name} -> {target.name}", flush=True)
            subprocess.run([cfx5gtmconv(), "-overwrite", str(source), str(target)], cwd=MESH_DIR, check=True)
        if not target.exists():
            raise FileNotFoundError(f"Put the TurboGrid mesh {source.name} (or {target.name}) in {MESH_DIR}.")


def face_centroids(solver, zone):
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
    data = solver.fields.field_data.get_field_data(SurfaceFieldDataRequest(
        surfaces=[zone], data_types=[SurfaceDataType.FacesCentroid]))[zone].face_centroids
    return np.asarray(data, dtype=float).reshape(-1, 3)


def centroid(solver, zone):
    return face_centroids(solver, zone).mean(axis=0)


def axial_position(solver, zone):
    """Axial (x) position of a row interface, which must be a plane x = const for the mixing plane."""
    x = face_centroids(solver, zone)[:, 0]
    if np.ptp(x) > 1e-6:
        raise ValueError(f"{zone} is not an axial plane (x spread {np.ptp(x) * 1e3:.3f} mm).")
    return float(x.mean())


def tidy_row(solver, row):
    """Rename the imported CFX zones to <row>_<part> (merging the passage and outlet-block pieces)."""
    settings = solver.settings
    modify = settings.mesh.modify_zones
    cells = [name for name in settings.setup.cell_zone_conditions.fluid.get_object_names()
             if not name.startswith(("stator_1", "rotor_1"))]
    modify.zone_name(zone_name=cells[0], new_name=f"{row}_fluid")
    def matching(prefixes):
        return [name for name in boundary_types(settings.setup.boundary_conditions)
                if name.startswith(tuple(prefixes)) and name not in JOIN]
    modify.fuse_face_zones(zone_names=JOIN, zone_name=f"{row}_block_join")
    print(row, "after fusing the block join:", sorted(boundary_types(settings.setup.boundary_conditions)), flush=True)
    for part, prefixes in PARTS.items():
        names = matching(prefixes)
        if len(names) > 1:
            modify.merge_zones(zone_names=names)
            names = matching(prefixes)
        modify.zone_name(zone_name=names[0], new_name=f"{row}_{part}")


def build_rotor_case(pyfluent, processors):
    solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                    processor_count=processors, ui_mode="no_gui", start_timeout=600, cwd=str(MESH_DIR))
    try:
        solver.settings.file.import_.read(file_type="cfx-definition", file_name=str(MESH_DIR / "rotor.def"))
        tidy_row(solver, "rotor_1")
        solver.settings.file.batch_options.confirm_overwrite = False
        solver.settings.file.write_case(file_name=str(MESH_DIR / "rotor_renamed.cas.h5"))
    finally:
        solver.exit()


def main(ui_mode="no_gui", processors=12):
    processors = int(processors)
    barotropy, pyfluent = load_dependencies(False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    convert_meshes()
    config = yaml.safe_load(STATOR_CONFIG.read_text(encoding="utf-8"))
    parameters = dict(config["rows"][0]["geometry"]["method_of_characteristics"], fluid_name="Cyclopentane")
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, {}, BAROTROPIC, RUN_DIR / "barotropy")
    print("inlet state", state)

    build_rotor_case(pyfluent, processors)
    solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                    processor_count=processors, ui_mode=ui_mode, start_timeout=600, cwd=str(RUN_DIR),
                                    cleanup_on_exit=False)
    settings = solver.settings
    settings.file.import_.read(file_type="cfx-definition", file_name=str(MESH_DIR / "stator.def"))
    tidy_row(solver, "stator_1")
    settings.mesh.modify_zones.append_mesh(file_name=str(MESH_DIR / "rotor_renamed.cas.h5"))
    types = boundary_types(settings.setup.boundary_conditions)
    zones = {f"{row}_fluid_{part}": f"{row}_{part}" for row in ("stator_1", "rotor_1") for part in PARTS}
    missing = [zone for zone in zones.values() if zone not in types]
    if missing:
        raise RuntimeError(f"Missing zones after the append: {missing}; have {sorted(types)}")
    # Each TurboGrid mesh has its own axial origin: move the rotor so its inlet meets the stator outlet
    shift = axial_position(solver, "stator_1_outlet") + AXIAL_GAP - axial_position(solver, "rotor_1_inlet")
    settings.mesh.modify_zones.translate_zone(zone_names=["rotor_1_fluid"], offset=[shift, 0.0, 0.0])
    settings.mesh.check()
    gap = {zone: axial_position(solver, zone) for zone in ("stator_1_outlet", "rotor_1_inlet")}
    print(f"rotor translated by {shift * 1e3:.3f} mm; interface axial positions [m]", gap, flush=True)

    names = configure_material(settings, expressions, OPTIONS["material_name"])
    fluids = settings.setup.cell_zone_conditions.fluid
    for name in names.values():   # Machine axis X: rotational periodicity and frame motion about +x
        frame = fluids[name].reference_frame
        frame.reference_frame_axis_origin = [0.0, 0.0, 0.0]
        frame.reference_frame_axis_direction = [1.0, 0.0, 0.0]
    modify = settings.mesh.modify_zones
    modify.zone_type(zone_names=[zones[f"{STATOR}_inlet"]], new_type="pressure-inlet")
    modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
    modify.zone_type(zone_names=[zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"]], new_type="interface")
    for row in (STATOR, ROTOR):
        modify.make_periodic(zone_name=zones[f"{row}_periodic_left"], shadow_zone_name=zones[f"{row}_periodic_right"],
                             rotate_periodic=True, create=True, auto_translation=True)
    configure_numerics(settings, OPTIONS)
    # The MoC stator turns the flow to +theta = atan2(z, y), counterclockwise about +x: the rotor turns with it
    omega = RPM * 2 * np.pi / 60
    rotor = fluids[names[ROTOR]].reference_frame
    rotor.frame_motion = True
    rotor.mrf_omega.value = omega
    create_turbo_interface(solver, "stator_rotor_mixing_plane", zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"],
                           "mixing_plane")
    configure_boundaries(settings, zones, state, {"rotor_exit_pressure": P_OUT}, OPTIONS)
    w = f"({omega:.6f} [s^-1])"
    settings.setup.named_expressions["relative_mach"] = {"definition": (
        f"sqrt(Velocity.x^2 + (Velocity.y + {w}*Position.z)^2 + (Velocity.z - {w}*Position.y)^2)"
        " / barotropic_speed_of_sound")}
    settings.mesh.check()
    case_file = RUN_DIR / OPTIONS["case_filename"]
    settings.file.batch_options.confirm_overwrite = False
    settings.file.write_case(file_name=str(case_file))
    (RUN_DIR / "setup_summary.json").write_text(json.dumps({
        "inlet_state": state, "rotor_omega_rad_s": omega, "rpm": RPM, "outlet_pressure": P_OUT,
        "rotor_shift_m": shift, "interface_x_m": gap, "blades": {"stator_1": 44, "rotor_1": 120},
        "zones": zones, "mach_expression": MACH_EXPRESSION}, indent=2, default=float), encoding="utf-8")
    print(f"saved {case_file}")
    if ui_mode == "no_gui":
        solver.exit()


if __name__ == "__main__":
    main(*sys.argv[1:])

