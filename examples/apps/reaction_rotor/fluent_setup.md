# Fluent setup of a stator + rotor stage: from the mesh to a runnable case

This is the manual version of what `turbo_moc.fluent.stage.case.configure_stage` does, step by step, in the same
order (Fluent 24.2 is sensitive to it). The numbers refer to the example stage in `output/stage/`
(`build_stage_mesh.py`: default MoC stator + subsonic reaction rotor, 5000 rpm). The last section lists the
differences when the meshes come from TurboGrid.

## 0. Inputs

| File | Built by | Contents |
| --- | --- | --- |
| `output/stage/mesh/stage.cgns` | `build_stage_mesh.py` (app jobs + project export) | Both rows, one fluid zone each, coordinates in m |
| `output/stage/mesh/README.txt` | same | Zones, boundaries, placement, interface plane, mesh quality |
| `output/stage/barotropy/fluent_expressions.txt` | `barotropic_model.py` | `barotropic_density`, `barotropic_speed_of_sound`, `barotropic_viscosity`, ... |
| `output/stage/barotropy/inlet_state.json` | same | Inlet total pressure, density and viscosity, fit range |

Example stage values: cyclopentane, p0 = 251.3 kPa, x0 = 0.025 (T0 = 353.04 K); stator exit pressure at design
95 kPa; barotropic fit down to 25.1 kPa; rotor 5000 rpm, rotation sign -1 about +z (omega = -523.6 rad/s).

## 1. Import and check the mesh

1. Fluent 3D, double precision, pressure-based solver (8 processes are enough for ~0.4 M cells).
2. File > Import > CGNS > Mesh: `stage.cgns`. The file is in metres; no scaling.
3. Domain > Mesh > Check. Expect two cell zones `stator_1_fluid` and `rotor_1_fluid`, and 14 face zones named
   `<row>_fluid.<row>_<patch>` with patch = inlet, outlet, blade, hub, shroud, periodic_left, periodic_right.
   All are imported as walls.

## 2. Zone types

| Zone | Type |
| --- | --- |
| `stator_1_fluid.stator_1_inlet` | pressure-inlet |
| `rotor_1_fluid.rotor_1_outlet` | pressure-outlet |
| `stator_1_fluid.stator_1_outlet`, `rotor_1_fluid.rotor_1_inlet` | interface |
| blades, hubs, shrouds | wall (default) |

In the GUI: Boundary Conditions > select the zone > Type. PyFluent:
`settings.mesh.modify_zones.zone_type(zone_names=[...], new_type="pressure-inlet")`.

## 3. Periodicity (before the row interface)

For each row, make `<row>_periodic_left` / `<row>_periodic_right` a rotational periodic pair
(Domain > Mesh > Make Periodic, or in PyFluent as the pipeline does:
`settings.mesh.modify_zones.make_periodic(zone_name=left, shadow_zone_name=right, rotate_periodic=True,
create=True, auto_translation=True)`). The app writes matching periodic nodes (error < 1e-13 mm), so a conformal
periodic pair is created. Do this before step 6, or Fluent asks interactively when the mixing plane is built.

## 4. Models and material

1. General: pressure-based, steady, operating pressure 0 Pa, no gravity. Energy off (barotropic model).
2. Viscous: k-omega SST.
3. Named expressions: copy every expression from `fluent_expressions.txt` (Parameters & Customization > Expressions,
   name = the line above each definition). Add `Mach = VelocityMagnitude / barotropic_speed_of_sound`.
4. Create a fluid material (e.g. `cyclopentane_barotropic`): density = expression `barotropic_density`,
   viscosity = expression `barotropic_viscosity`. Assign it to both cell zones.

## 5. Rotating rotor zone

Cell Zone Conditions > `rotor_1_fluid` > Frame Motion: on, axis +z through the origin,
omega = sign x rpm x 2 pi / 60 = -523.6 rad/s for this stage. The sign makes the rotor move with the stator exit
swirl. The pipeline computes it with `turbo_moc.fluent.stage.design.swirl_sign`. If you flip it, the relative
inflow angle has the wrong sign and the rotor runs backwards.

## 6. Row interface

Enable Turbo models, then create the interface in the TUI (the settings-API version leaves the zones unassigned
in 24.2):

```
/define/turbo-model/turbo-create stator_rotor_mixing_plane stator_1_fluid.stator_1_outlet () rotor_1_fluid.rotor_1_inlet () 2
```

The last number is the interface type: 2 mixing plane, 1 frozen rotor without pitch change, 0 with pitch scaling.
A frozen rotor needs both rows to span the same sector: with 31 + 31 blades it works directly; otherwise copy
passages (`case.replicate_passages`). The rows are 2 mm apart, and the interface plane is at Z = 21 mm.

## 7. Boundary conditions

| Boundary | Setting |
| --- | --- |
| Stator inlet (pressure-inlet) | Gauge total pressure = initial gauge pressure = 251300 Pa; direction normal to boundary; turbulence intensity 5 % and viscosity ratio 10 (pipeline defaults) |
| Rotor outlet (pressure-outlet) | Static pressure: **your choice** (see below); backflow turbulence as at the inlet |
| Rotor shroud | Moving wall, absolute speed 0 (stationary casing in the rotating zone) |
| Other walls | No slip (hub and blades rotate with their zone) |

Outlet pressure: for vortex rotors the pipeline takes it from the rotor's design flow solution. The reaction rotor
is geometry only, so there is no design value. 95 kPa (the stator design exit pressure) gives zero rotor reaction.
Choose a lower value for the reaction you want, but stay above the barotropic fit limit (25.1 kPa).

## 8. Numerics, initialization and run

1. Coupled pressure-velocity, least-squares gradients, pseudo-time (global time step, automatic, conservative
   length scale), high-order-term relaxation on the flow variables. First order to start, then second order.
2. Residuals: Fluent's default scaled residuals. Normalized residuals stop 24.2 at the first renormalization in a
   reopened case.
3. Initialization: standard initialization at rest at the inlet total pressure. Then lower the outlet pressure in
   steps from p0 to the target (`setup.start_from_rest`). Two-phase density falls by orders of magnitude through the
   stage, so a hybrid initialization often reverses the inlet. For a subsonic rotor with a mild pressure ratio,
   hybrid initialization usually works too.
4. Monitors: mass flow at the inlet and outlet, rotor blade torque (moment about z), `Mach` on a midspan plane.

## 9. Differences when the meshes come from TurboGrid

TurboMoC's TurboGrid export (the "BladeGen" files: blade, hub and shroud curves per row) is geometry only. You mesh
each row in TurboGrid, export each mesh, and import them into Fluent. Steps 4, 5, 7 and 8 are identical; the
following change:

| Step | App mesh (`stage.cgns`) | TurboGrid meshes |
| --- | --- | --- |
| Import | One file, both rows already placed (axial gap, phase) | One file per row: import the first, then Mesh > Append; check the axial placement and the gap, and translate if needed |
| Units | Metres in the file | Check the export length unit; scale in Fluent if it is not m |
| Zone names | `<row>_fluid.<row>_<patch>`, which the pipeline scripts expect | TurboGrid names (INFLOW/OUTFLOW, HUB, SHROUD, BLADE, PER1/PER2 or similar, depending on the version and export settings); rename them or map them by hand. `case.configure_stage` needs a name map before it can be used |
| Periodicity | Matching nodes, conformal periodic pair | Usually matching (TurboGrid builds periodic surfaces), so the same `make-periodic`. If they don't match, use a non-conformal periodic interface |
| Interface sides | Separate outlet/inlet zones at the same plane, non-matching nodes | Same idea; make sure the two sides are at the same axial position and cover the same span |
| Tip clearance | None in this example (blade to shroud) | TurboGrid can mesh the clearance; then the shroud of the rotor needs the stationary-wall setting of step 7 as well |
| Mesh quality | Report in `mesh/README.txt` (min SICN per row) | TurboGrid's mesh analysis; Fluent Mesh > Check / Report Quality |
| Blade-row topology | Unstructured hex-dominant section, extruded along the span | Structured ATM topology (O-grid around the blade), usually higher quality at the leading and trailing edges |

The automated pipeline (`python -m turbo_moc.fluent.stage.setup case.yaml`) works with app meshes only. It also
needs a vortex rotor today, because it reads the outlet pressure from the rotor's flow solution.
