# Theory

The method of characteristics (MoC) constructs a supersonic flow field and the
wall that supports it. For nozzle design, the objective is an expansion from
prescribed stagnation conditions to a specified exit pressure, with a uniform
velocity direction at the outlet. The resulting contour can be incorporated into
a stator blade. A related vortex-flow construction produces rotor blade sections
that turn an already supersonic inlet flow.

These methods are useful for preliminary design of turbines handling real gases
and liquid–vapor mixtures, including organic Rankine cycles, refrigeration, and
liquefaction systems. A one-dimensional calculation establishes the required
area ratio, but does not determine the wall shape or resolve the expansion waves
and their reflections. The MoC supplies this two-dimensional construction.

This guide describes the planar methods implemented in `moc`, with particular
attention to details that are condensed in the accompanying papers
{cite:p}`cioffiStators2026,cioffiNonideal2026`. The starting point is steady,
inviscid, irrotational flow with uniform stagnation enthalpy and entropy.
Liquid–vapor states are evaluated under the homogeneous equilibrium model
(HEM). The geometry is intended for subsequent CFD analysis. The papers compare
the equilibrium MoC with barotropic HEM simulations; this does not make the
geometry generator a viscous-flow or turbine-efficiency solver.

## Reading the guide

Start with the [notation and coordinate conventions](notation.md), then follow
the thermodynamic model through the characteristic equations to the geometry
construction. The rotor chapter uses the same thermodynamic and Prandtl–Meyer
relations as the nozzle chapters. Radial mapping is a geometric operation applied
after either blade section has been generated.

```{toctree}
:maxdepth: 1

notation
thermodynamic_model
isentropic_expansions
method_of_characteristics
nozzle_design
saturation_crossings
stator_design
rotor_design
radial_mapping
```

The [examples](../examples/index.md) provide executable designs and coordinate
exports. Parameter definitions are collected in the
[API reference](../api_reference.md).

```{note}
This is a first draft. Admonitions headed **Review note** identify source
inconsistencies or implementation details to check with the authors. Figure
placeholders describe the illustrations still needed; extracted figures retain
their original notation, which is reconciled in the captions.
```
