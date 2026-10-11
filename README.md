# TurboMoC

**From a real-fluid supersonic nozzle to a meshed turbine stage, in one Python package.**

TurboMoC designs supersonic nozzles with the method of characteristics (MoC) using real-fluid
thermodynamics, including two-phase flashing and condensing expansions under the Homogeneous
Equilibrium Model (HEM). The nozzle becomes a stator blade, a supersonic vortex rotor completes the
stage, and both rows are sized in 3D, assembled into axial or radial stages, meshed for CFD and
exported to CAD and Fluent. Every step is available from Python scripts and from an interactive app.

📦 **PyPI package**: [https://pypi.org/project/turbo_moc/](https://pypi.org/project/turbo_moc/)

📚 **Documentation**: [https://turbo-sim.github.io/TurboMoC/](https://turbo-sim.github.io/TurboMoC/)

🎓 **Tutorials**: [https://turbo-sim.github.io/TurboMoC/examples/index.html](https://turbo-sim.github.io/TurboMoC/examples/index.html)

![Axial stage, swept 3D passage mesh and radial stage generated with TurboMoC](docs/images/readme_3d.png)

## Features

### Nozzle design with the method of characteristics

- Conventional and minimum-length nozzles for single-phase and two-phase (HEM) flows.
- Real-fluid properties through [jaxprop](https://pypi.org/project/jaxprop/)/CoolProp, including
  wet inlets (vapor quality) and expansions crossing the saturation dome.
- Smooth sonic transition, kernel and reflex regions, with axis and wall property distributions.

![Characteristics mesh of a two-phase Cyclopentane nozzle and its expansion in the T-s diagram](docs/images/readme_nozzle.png)

### Blade design

- **Stator**: the nozzle wall becomes a B-spline blade with prescribed inlet/outlet metal angles,
  trailing-edge radius and throat, editable control point by control point.
- **Rotor**: supersonic vortex blades (Goldman's method, generalized to real gases), impulse or
  reacting, optionally coupled to the stator exit state.

![Stator blade derived from the nozzle wall and a supersonic vortex rotor blade](docs/images/readme_blades.png)

### 3D rows and stages

- Size the stator for a mass flow and blade count; the rotor shares its annulus.
- **Axial** rows wrap onto the annulus between hub and shroud; **radial** rows use an
  angle-preserving log-spiral map, with the blade count and outer radius chosen so that the
  radial passage is exactly the one designed in 2D.
- Assemble stator and rotor rows into a stage and export STEP/STL solids.

### Meshing for CFD

- **2D blade-to-blade meshes** with [Gmsh](https://gmsh.info/): structured O-H (TurboGrid-like)
  or quad-dominant unstructured passages, with inflation layers sized from a target y+ estimated
  on the actual expansion (mixture viscosity for two-phase flows).
- **3D meshes** of stator and rotor passages, axial or radial: the 2D section swept or extruded
  into hexahedra/prisms, matching the exported CAD solid, with y+-based endwall layers and
  rotational periodicity.
- CGNS (Fluent-ready) and Gmsh exports; automatic quality checks and repair of folded cells.

![Unstructured 2D passage mesh and the inflation layers at the trailing edge](docs/images/readme_mesh_2d.png)

### Interactive app and workflows

- A Dash app takes you through the pipeline: nozzle → stator → rotor → 3D sizing and meshing →
  stage assembly, with 3D views, and downloads that become available as each step is completed.
- Scripted workflows in [`examples/`](examples): nozzle, stator and rotor design, Gmsh meshing
  ([2D](examples/gmsh/readme.md), [3D](examples/gmsh_3D/readme.md)) and a
  [Fluent case setup](examples/fluent/readme.md).

## Installation instructions

TurboMoC requires Python 3.11, 3.12, or 3.13.

### Install from PyPI

Install with pip in your Python environment:

```bash
python -m pip install --upgrade "turbo_moc[cad]"
```

The `cad` extra installs CadQuery (STEP/STL export) and the app's 3D viewer. Gmsh is always
installed.

To install TurboMoC from PyPI into an existing [Poetry](https://python-poetry.org/docs/cli/#add)
project, use `poetry add` with bracket notation for optional extras:

```bash
poetry add "turbo_moc[cad]"
```

Include the `fluent` extra to install optional PyFluent support alongside CAD:

```bash
poetry add "turbo_moc[cad,fluent]"
```

Use `"turbo_moc[fluent]"` if you only need the Fluent extra. Poetry records the
selected extras in your project's dependencies and installs them from PyPI.

### Install from source

With [Poetry](https://python-poetry.org/docs/#installation), clone the repository and install the package with CAD support:

```bash
git clone https://github.com/turbo-sim/TurboMoC.git
cd TurboMoC
poetry install --extras cad
```

For CAD and Fluent support in the source checkout, use:

```bash
poetry install --extras "cad fluent"
```

The `fluent` extra installs the Python interface; Fluent must be installed and
licensed separately. See the [Fluent workflow README](examples/fluent/readme.md)
for the companion barotropy installation and example configuration.

Use `poetry run python` to run Python scripts in the Poetry environment.

## Quick start

Save the following as `quick_start.py` and run `python quick_start.py` (or `poetry run python quick_start.py` for a source installation). It designs a CO₂ expansion nozzle, prints the convergence status and exit Mach number, and displays the characteristics mesh with Matplotlib.

The conditions are representative of a transcritical CO₂ heat pump: the nozzle
expands CO₂ from a gas-cooler outlet at 35 °C and 1.2 times the critical pressure
(about 88.5 bar) to the saturation pressure at an evaporation temperature of 5 °C
(about 39.7 bar).

```python
import jaxprop as jxp
import matplotlib.pyplot as plt
from turbo_moc import design_nozzle, mpl

fluid = jxp.Fluid("CO2")
inlet_pressure = 1.2 * fluid.critical_point.p  # Pa
inlet_temperature = 273.15 + 35.0  # K; gas-cooler outlet at 35 °C
evaporation_temperature = 273.15 + 5.0  # K (5 °C)
back_pressure = fluid.get_state(jxp.QT_INPUTS, 0, evaporation_temperature).p

result = design_nozzle(
    fluid_name="CO2",
    P0=inlet_pressure,
    T0=inlet_temperature,  # K
    p_back=back_pressure,  # Pa; saturation pressure at 5 °C
    solver="conventional",
)

print(f"Converged: {result['converged']}")
print(f"Exit Mach number: {result['exit_M']:.4f}")

fig, ax = mpl.plot_characteristics_mesh(result, mirror=True)
plt.show()
```

See the [worked examples](https://turbo-sim.github.io/TurboMoC/examples/index.html)
for nozzle, stator, and rotor design, including plots and coordinate exports.

## Interactive app

Launch the app from your terminal:

```bash
turbo_moc-app
```

For a Poetry source installation, use `poetry run turbo_moc-app`. Open the local URL printed in
the terminal. The phases follow the design pipeline:

1. **Nozzle**: MoC design, property plots and the T-s diagram.
2. **Stator**: blade parametrization (2D), then sizing, 3D wrap, and 2D/3D meshing.
3. **Rotor**: vortex blade design (2D), then sizing, 3D wrap and 3D meshing.
4. **Cascade**: axial or radial stage assembly and STEP export.

Long computations (MoC, meshing) run in the background, and each download button is enabled once
its model is ready.

## Reproducing the figures

The figures above are generated from the package itself:

```bash
poetry run python docs/images/make_readme_figures.py
```
