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

Historical case files with row-level fluids and pressure/quality or
pressure/temperature inputs are migrated when loaded. If active manual rows
specify different fluids, set `machine.fluid_name` explicitly to resolve the
conflict. A coupled rotor inherits the upstream stator's fluid.

Run `python examples/apps/run_app.py` for the axial stator example. To select
another case, use `turbo_moc-app --config examples/apps/cases/<case>.yaml`.
