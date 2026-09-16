# TurboMoC

TurboMoC is an open-source Python package for exploring nozzle and blade designs with real-fluid thermodynamics. It supports single-phase and two-phase expansions, including flashing and condensing flows, under the Homogeneous Equilibrium Model (HEM). Use Python scripts or the interactive app to define operating conditions, inspect the flow, and export geometries for CFD simulations.

📦 **PyPI package**: [https://pypi.org/project/turbo_moc/](https://pypi.org/project/turbo_moc/)

📚 **Documentation**: [https://turbo-sim.github.io/TurboMoC/](https://turbo-sim.github.io/TurboMoC/)

🎓 **Tutorials**: [https://turbo-sim.github.io/TurboMoC/examples/index.html](https://turbo-sim.github.io/TurboMoC/examples/index.html)



## Features

- Design conventional and minimum-length supersonic nozzles.
- Explore single-phase and two-phase flows using real-fluid properties.
- Generate stator and rotor blade geometries from nozzle designs.
- Visualize flow properties and compare designs using Python scripts or the interactive app.
- Export coordinates, figures, and stator STEP geometry for downstream workflows.

![Method of characteristics mesh for a supersonic nozzle](docs/images/characteristics_mesh.png)

![Stator blade parametrization](docs/images/stator_blade.png)

## Installation instructions

TurboMoC requires Python 3.11, 3.12, or 3.13. Install it with pip in your Python environment:

```bash
python -m pip install --upgrade "turbo_moc[cad]"
```

This also installs CadQuery for STEP geometry export and upgrades an existing TurboMoC installation.

To install from source with [Poetry](https://python-poetry.org/docs/#installation), clone the repository and install the package with CAD support:

```bash
git clone https://github.com/turbo-sim/TurboMoC.git
cd TurboMoC
poetry install --extras cad
```

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

For a Poetry source installation, use `poetry run turbo_moc-app`. Open the local URL printed in the terminal to explore nozzle designs, parametrize stator blades, compare results, and export plots and STEP geometry.
