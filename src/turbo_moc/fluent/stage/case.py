"""Fluent stage case: barotropic material, rotating rotor zone, mixing plane, periodicity and boundaries."""

import numpy as np

from ..common import MACH_EXPRESSION, boundary_types, configure_numerics

STATOR, ROTOR = "stator_1_fluid", "rotor_1_fluid"
BOUNDARIES = ("inlet", "outlet", "blade", "hub", "shroud", "periodic_left", "periodic_right")


def zone(types, row, name):
    return next(key for key in types if key.endswith(f"{row}_{name}"))


INTERFACES = {"mixing_plane": 2, "frozen_rotor": 1}


def create_turbo_interface(solver, name, zone1, zone2, kind):
    """General turbo interface: mixing plane (averaged) or frozen rotor (no pitch change).

    Fluent 24.2 creates a working interface only through the TUI command
    ``turbo-create`` (0 pitch-scale, 1 no pitch-scale, 2 mixing plane, 3 standard);
    the settings-API turbo interface is listed but leaves the interface zones
    unassigned at initialization. Both need the coupled solver and periodic
    zones beside the interface.
    """
    settings = solver.settings
    settings.setup.turbo_models.enabled = True
    solver.execute_tui(f"/define/turbo-model/turbo-create {name} {zone1} () {zone2} () {INTERFACES[kind]}")
    if name not in settings.setup.mesh_interfaces.interface.get_object_names():
        raise RuntimeError(f"Fluent did not create the {kind.replace('_', ' ')} interface {name!r}.")


def mean_angle(solver, zone):
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
    centroids = solver.fields.field_data.get_field_data(SurfaceFieldDataRequest(
        surfaces=[zone], data_types=[SurfaceDataType.FacesCentroid]))[zone].face_centroids
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 3)
    return float(np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0])).mean())


def replicate_passages(solver, row, zones, passages):
    """Copy a row's passage by rotation and fuse the copies into one cell zone of ``passages`` blades.

    The frozen rotor needs both rows to span the same sector. The copied
    periodic faces that coincide become one interior zone; the two outer ones
    become the row's periodic pair. ``copy_move_cell_zone`` takes radians.
    """
    settings = solver.settings
    modify = settings.mesh.modify_zones
    cell = next(name for name in settings.setup.cell_zone_conditions.fluid.get_object_names() if name.startswith(row))
    pitch = (mean_angle(solver, zones[f"{row}_periodic_left"]) - mean_angle(solver, zones[f"{row}_periodic_right"]))
    for copy in range(1, passages):
        modify.copy_move_cell_zone(cell_zone_name=cell, translate=False, rotation_angle=np.radians(pitch * copy),
                                   offset=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0])
    prefix = zones[f"{row}_periodic_left"].rsplit("_periodic_left", 1)[0]
    periodic = [name for name in boundary_types(settings.setup.boundary_conditions)
                if name.startswith(prefix) and "_periodic_" in name]
    angles = {name: mean_angle(solver, name) for name in periodic}
    tolerance = 1e-3 * abs(pitch)
    inner = [name for name in periodic if any(other != name and abs(angles[other] - angles[name]) < tolerance
                                              for other in periodic)]
    if len(inner) != 2 * (passages - 1):
        raise RuntimeError(f"{row}: {len(inner)} coincident periodic faces after copying, "
                           f"expected {2 * (passages - 1)}.")
    modify.fuse_face_zones(zone_names=inner, zone_name=f"{row}_passage_joins")
    modify.merge_zones(zone_names=[name for name in settings.setup.cell_zone_conditions.fluid.get_object_names()
                                   if name.startswith(row)])
    for name in BOUNDARIES:
        if name.startswith("periodic"):
            continue
        copies = [zone for zone in boundary_types(settings.setup.boundary_conditions)
                  if zone == zones[f"{row}_{name}"] or zone.startswith(zones[f"{row}_{name}"] + ":")]
        if len(copies) > 1:
            modify.merge_zones(zone_names=copies)
    outer = sorted((name for name in periodic if name not in inner), key=angles.get)
    zones = dict(zones)
    zones[f"{row}_periodic_right"], zones[f"{row}_periodic_left"] = outer
    for name in BOUNDARIES:
        if not name.startswith("periodic"):
            zones[f"{row}_{name}"] = next(zone for zone in boundary_types(settings.setup.boundary_conditions)
                                          if zone == zones[f"{row}_{name}"]
                                          or zone.startswith(zones[f"{row}_{name}"] + ":"))
    return zones


def configure_material(settings, expressions, material_name):
    general = settings.setup.general
    general.solver.type = "pressure-based"
    general.solver.time = "steady"
    general.operating_conditions.operating_pressure = 0.0
    general.operating_conditions.gravity.enable = False
    models = settings.setup.models
    models.energy.enabled = False
    models.viscous.model = "k-omega"
    models.viscous.k_omega_model = "sst"
    for name, definition in expressions.items():
        settings.setup.named_expressions[name] = {"definition": definition}
    settings.setup.named_expressions[MACH_EXPRESSION] = {"definition": "VelocityMagnitude / barotropic_speed_of_sound"}
    material = settings.setup.materials.fluid.create(name=material_name)
    material.density = {"option": "expression", "expression": "barotropic_density"}
    material.viscosity = {"option": "expression", "expression": "barotropic_viscosity"}
    fluids = settings.setup.cell_zone_conditions.fluid
    names = {row: name for row in (STATOR, ROTOR) for name in fluids.get_object_names() if name.startswith(row)}
    for name in names.values():
        fluids[name].general.material = material_name
    return names


def configure_boundaries(settings, zones, state, point, options):
    bc = settings.setup.boundary_conditions
    inlet = bc.pressure_inlet[zones[f"{STATOR}_inlet"]]
    inlet.momentum.gauge_total_pressure.value = state["P0"]
    inlet.momentum.supersonic_or_initial_gauge_pressure.value = state["P0"]
    inlet.momentum.direction_specification_method = "Normal to Boundary"
    inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    inlet.turbulence.turbulent_intensity = options["turbulence_intensity"]
    inlet.turbulence.turbulent_viscosity_ratio = options["turbulence_viscosity_ratio"]
    outlet = bc.pressure_outlet[zones[f"{ROTOR}_outlet"]]
    outlet.momentum.gauge_pressure.value = point["rotor_exit_pressure"]
    outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
    outlet.turbulence.backflow_turbulent_intensity = options["turbulence_intensity"]
    outlet.turbulence.backflow_turbulent_viscosity_ratio = options["turbulence_viscosity_ratio"]
    shroud = bc.wall[zones[f"{ROTOR}_shroud"]].momentum
    shroud.wall_motion = "Moving Wall"
    shroud.relative = False
    shroud.rotating = False
    shroud.speed.value = 0.0
    if options.get("slip_endwalls", False):
        settings.mesh.modify_zones.zone_type(
            zone_names=[zones[f"{row}_{side}"] for row in (STATOR, ROTOR) for side in ("hub", "shroud")],
            new_type="symmetry")
    if options.get("slip_blades", False):  # Zero shear but still walls, so the blade torque is reported
        for row in (STATOR, ROTOR):
            bc.wall[zones[f"{row}_blade"]].momentum.shear_condition = "Specified Shear"


def configure_stage(solver, mesh_file, state, expressions, point, options):
    """Import the stage CGNS and configure the complete, unsolved case.

    Order matters in Fluent 24.2: periodic zones before the mixing plane
    (otherwise ``make_periodic`` prompts interactively), the coupled scheme and
    the rotor frame motion before ``turbo-create``.
    """
    settings = solver.settings
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    types = boundary_types(settings.setup.boundary_conditions)
    zones = {f"{row}_{name}": zone(types, row.split("_fluid")[0], name) for row in (STATOR, ROTOR) for name in BOUNDARIES}
    kind = options.get("interface", "mixing_plane")
    passages = {STATOR: 1, ROTOR: 1}
    if kind == "frozen_rotor":
        counts = point["design"]["num_blades"]
        if counts["rotor_1"] % counts["stator_1"]:
            raise ValueError(f"Frozen rotor: {counts['rotor_1']} rotor blades are not a multiple of "
                             f"{counts['stator_1']} stator blades, so the rows cannot span the same sector.")
        passages[ROTOR] = counts["rotor_1"] // counts["stator_1"]
        zones = replicate_passages(solver, ROTOR, zones, passages[ROTOR])
        settings.mesh.check()
    names = configure_material(settings, expressions, options["material_name"])
    modify = settings.mesh.modify_zones
    modify.zone_type(zone_names=[zones[f"{STATOR}_inlet"]], new_type="pressure-inlet")
    modify.zone_type(zone_names=[zones[f"{ROTOR}_outlet"]], new_type="pressure-outlet")
    modify.zone_type(zone_names=[zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"]], new_type="interface")
    for row in (STATOR, ROTOR):
        modify.make_periodic(zone_name=zones[f"{row}_periodic_left"], shadow_zone_name=zones[f"{row}_periodic_right"],
                             rotate_periodic=True, create=True, auto_translation=True)
    configure_numerics(settings, options)

    omega = point["swirl_sign"] * point["rpm"] * 2 * np.pi / 60
    rotor_frame = settings.setup.cell_zone_conditions.fluid[names[ROTOR]].reference_frame
    rotor_frame.frame_motion = True
    rotor_frame.mrf_omega.value = omega
    create_turbo_interface(solver, f"stator_rotor_{kind}", zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"], kind)
    configure_boundaries(settings, zones, state, point, options)

    settings.mesh.check()
    settings.solution.initialization.initialization_type = "hybrid"
    settings.solution.run_calculation.iter_count = options["solution_strategy"][0]["iterations"]
    return {"cell_zones": names, "boundaries": zones, "rotor_omega_rad_s": omega, "interface": kind,
            "passages": {row.split("_fluid")[0]: count for row, count in passages.items()}}
