"""Fluent stage case: barotropic material, rotating rotor zone, mixing plane, periodicity and boundaries."""

import numpy as np

from ..common import MACH_EXPRESSION, boundary_types, configure_numerics

STATOR, ROTOR = "stator_1_fluid", "rotor_1_fluid"
BOUNDARIES = ("inlet", "outlet", "blade", "hub", "shroud", "periodic_left", "periodic_right")


def zone(types, row, name):
    return next(key for key in types if key.endswith(f"{row}_{name}"))


def create_mixing_plane(solver, name, zone1, zone2):
    """General turbo interface with the mixing-plane pitch change.

    Fluent 24.2 creates a working mixing plane only through the TUI command
    ``turbo-create`` (choice 2); the settings-API turbo interface is listed but
    leaves the interface zones unassigned at initialization.
    """
    settings = solver.settings
    settings.setup.turbo_models.enabled = True
    solver.execute_tui(f"/define/turbo-model/turbo-create {name} {zone1} () {zone2} () 2")
    if name not in settings.setup.mesh_interfaces.interface.get_object_names():
        raise RuntimeError(f"Fluent did not create the mixing-plane interface {name!r}.")


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
    names = {row: next(name for name in fluids.get_object_names() if name.startswith(row)) for row in (STATOR, ROTOR)}
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


def configure_stage(solver, mesh_file, state, expressions, point, options):
    """Import the stage CGNS and configure the complete, unsolved case.

    Order matters in Fluent 24.2: periodic zones before the mixing plane
    (otherwise ``make_periodic`` prompts interactively), the coupled scheme and
    the rotor frame motion before ``turbo-create``.
    """
    settings = solver.settings
    settings.file.import_.read(file_type="cgns-mesh", file_name=str(mesh_file))
    settings.mesh.check()
    names = configure_material(settings, expressions, options["material_name"])

    types = boundary_types(settings.setup.boundary_conditions)
    zones = {f"{row}_{name}": zone(types, row.split("_fluid")[0], name) for row in (STATOR, ROTOR) for name in BOUNDARIES}
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
    create_mixing_plane(solver, "stator_rotor_mixing_plane", zones[f"{STATOR}_outlet"], zones[f"{ROTOR}_inlet"])
    configure_boundaries(settings, zones, state, point, options)

    settings.mesh.check()
    settings.solution.initialization.initialization_type = "hybrid"
    settings.solution.run_calculation.iter_count = options["solution_strategy"][0]["iterations"]
    return {"cell_zones": names, "boundaries": zones, "rotor_omega_rad_s": omega}
