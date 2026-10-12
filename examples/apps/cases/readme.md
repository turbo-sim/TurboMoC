# App case configurations

These files initialize the current TurboMoC design app:

- `axial_stator.yaml`, `radial_stator.yaml`, `linear_stator.yaml`: a single stator.
- `rotor_only.yaml`: a standalone vortex rotor using Novec649.
- `stator_rotor.yaml`: a Cyclopentane stage with the rotor coupled to the stator exit.

Choose the working fluid once in `machine.fluid_name`, alongside `design_mode`
and `geometry_type`. Every row uses this fluid, including inactive rows when
reactivated. Each stator's `method_of_characteristics` and each rotor's
`rotor_design` specify their own inlet property pair and two values:

```yaml
machine:
  design_mode: stator_rotor
  geometry_type: axial
  fluid_name: Cyclopentane

rows:
- id: stator_1
  role: stator
  geometry:
    method_of_characteristics:
      inlet_property_pair: PQ_INPUTS
      inlet_property_1: 251300.0  # Stagnation pressure [Pa]
      inlet_property_2: 0.025     # Vapor quality [kg/kg]
      # Other row inputs follow in the full case files.
```

The two values follow the order of the selected CoolProp/JaxProp input pair,
using SI units. Stator values describe the stagnation state; manual rotor values
describe the relative stagnation state. A coupled rotor retains its manual pair
and values for later use, while its effective inlet comes from the stator exit.

For stator-to-rotor coupling, choose the speed type and value:

```yaml
inlet_coupling:
  mode: stator_exit
  rotational_speed:
    type: actual  # actual, blade_velocity_ratio, or specific
    value: 1000.0 # RPM for actual; dimensionless for the other modes
```

The ratio is `U/c0` at the rotor inlet midspan radius, with
`c0 = sqrt(2 * delta_h_is)`. Specific speed follows TurboDash's dimensionless
definition, `omega * sqrt(mass_flow / rho_out_s) / delta_h_is**0.75`.
Both modes reference expansion from the stator stagnation state to its back
pressure, rather than to the full stage outlet. Mass flow is a meanline estimate
from stator exit density, meridional velocity and annular area, without blade
blockage. The readout reports the resolved RPM, blade speed and reference data.
Change the numeric value to suit the selected units when changing speed type.
Historical `rpm` settings migrate to `rotational_speed: {type: actual, value: ...}`.
Speed affects coupled inlet conditions; manually specified relative inlets remain
independent of it.

Annular rows retain an editable `meridional_channel.num_blades`. After each
successful MoC or vortex solve, the app sets it to the ceiling of the natural
midspan blade count (at least two), then permits a manual override. The sidebar
shows the noninteger natural count, scaled native pitch and actual pitch deviation.
Changing radii/chords updates this reference without overwriting a manual count;
a new successful solve resets the count again.

Natural sizing uses the native solver pitch/chord ratio `k` and conformal chord
`C' = integral(dm/r)` at midspan: `N* = 2*pi/(k*C')`. Rounding or overriding the
count can change passage spacing; exact agreement with the solver pitch is not
guaranteed. Linear cascades show their scaled native pitch and no annular count.

Historical case files with row-level fluids and pressure/quality or
pressure/temperature inputs are migrated when loaded. If active manual rows
specify different fluids, set `machine.fluid_name` explicitly to resolve the
conflict. A coupled rotor inherits the upstream stator's fluid.

Run `python examples/apps/run_app.py` for the axial stator example. To select
another case, use `turbo_moc-app --config examples/apps/cases/<case>.yaml`.
