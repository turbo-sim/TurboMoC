# Method of Characteristics

`moc` is a Python package for designing supersonic nozzles and turbomachinery
blades using the method of characteristics. It combines real-fluid
thermodynamics with conventional and minimum-length nozzle solvers, geometry
parametrization, plotting tools, and an interactive web application.

The same workflow supports single-phase, saturated two-phase, and flashing
expansions under the Homogeneous Equilibrium Model. Solver results are exposed
as plain, JSON-safe data that can be inspected, plotted, or passed to the
stator and rotor geometry tools.

![Characteristics mesh for a supersonic nozzle](../images/characteristics_mesh.png)

## Documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc

Install the package, design a nozzle, inspect the result, and launch the app.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Guided applications of the solvers, geometry tools, plots, and interactive app.
:::

:::{grid-item-card} Theory
:link: theory/index
:link-type: doc

Mathematical background for the method of characteristics for nozzle and blade design.
:::

:::{grid-item-card} Bibliography
:link: references/bibliography
:link-type: doc

Browse the references cited throughout the documentation.
:::

:::{grid-item-card} Developer guide
:link: developer_guide
:link-type: doc

Install the package for development, build the documentation, and release new versions.
:::

:::{grid-item-card} API reference
:link: api_reference
:link-type: doc

Inspect the modules, classes, and functions provided by the Python package.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
examples/index
theory/index
references/bibliography
developer_guide
api_reference
```
