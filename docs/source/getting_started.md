# Getting started

## Introduction

TurboMoC designs supersonic nozzles and turbomachinery geometry with a real-fluid
thermodynamic core supplied by `jaxprop`. It includes a conventional rounded-arc
solver and a sharp-corner minimum-length-nozzle solver. Both expose a common
high-level function and return the same JSON-safe result structure.

The toolkit supports single-phase, saturated two-phase, and subcooled-liquid
flashing or wet-to-dry inlet conditions under the Homogeneous Equilibrium Model.
Results can be visualized with Matplotlib or Plotly and used to parametrize
stator and rotor blades.

## Installation instructions

TurboMoC requires Python 3.11, 3.12, or 3.13. Install it with pip in your Python
environment:

```bash
python -m pip install --upgrade "turbo_moc[cad]"
```

This also installs CadQuery for STEP geometry export and upgrades an existing
TurboMoC installation.

### Install from source with Poetry

To install from source with [Poetry](https://python-poetry.org/docs/#installation),
clone the repository and install the package with CAD support:

```bash
git clone https://github.com/turbo-sim/TurboMoC.git
cd TurboMoC
poetry install --extras cad
```

Use `poetry run python` to run Python scripts in the Poetry environment.
For development tools and documentation builds, see the
[Developer guide](developer_guide.md#installation-for-developers).

### Verify the installation

```bash
python -c "import turbo_moc; print(turbo_moc.__version__)"
```

For a Poetry source installation, prefix this command with `poetry run`.

## Design a nozzle

The public `design_nozzle` function accepts ordinary Python values and returns a
flat dictionary suitable for serialization and plotting.

The conditions below are representative of a transcritical CO₂ heat pump: the
nozzle expands CO₂ from a gas-cooler outlet at 35 °C and 1.2 times the critical
pressure (about 88.5 bar) to the saturation pressure at an evaporation temperature
of 5 °C (about 39.7 bar).

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


## Launch the interactive app

```bash
turbo_moc-app
```

For a Poetry source installation, use `poetry run turbo_moc-app`.
Open the local URL printed in the terminal in a browser.

Continue to the [theory](theory/index.md), [examples](examples/index.md),
or [API reference](api_reference.md).
