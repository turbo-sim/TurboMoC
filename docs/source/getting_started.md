# Getting started

## Introduction

`moc` designs supersonic nozzles and turbomachinery geometry with a real-fluid
thermodynamic core supplied by `jaxprop`. It includes a conventional rounded-arc
solver and a sharp-corner minimum-length-nozzle solver. Both expose a common
high-level function and return the same JSON-safe result structure.

The toolkit supports single-phase, saturated two-phase, and subcooled-liquid
flashing or wet-to-dry inlet conditions under the Homogeneous Equilibrium Model.
Results can be visualized with Matplotlib or Plotly and used to parametrize
stator and rotor blades.

## Installation

`moc` supports Python 3.11 through 3.13. Once the package is released on PyPI,
install it with pip:

```bash
python -m pip install moc
```

For STEP export, include the optional CadQuery dependency:

```bash
python -m pip install "moc[cad]"
```

Until the first PyPI release, install directly from the repository:

```bash
python -m pip install git+https://github.com/turbo-sim/method_of_characteristics.git
```

For an editable installation to work on the source code, see the
[Developer guide](developer_guide.md#installation-for-developers).

Verify the installation:

```bash
python -c "import moc; print(moc.__version__)"
```

## Design a nozzle

The public `design_nozzle` function accepts ordinary Python values and returns a
flat dictionary suitable for serialization and plotting:

```python
from moc import design_nozzle

data = design_nozzle(
    fluid_name="nitrogen",
    P0=20e5,
    T0=113.0,
    p_back=2e5,
    solver="conventional",  # Use "mln" for a minimum-length nozzle.
    y_t=0.01,
    rho_t=0.2,
    rho_d=0.1,
)

wall = data["wall_final"]
```

Plot the characteristics mesh or nozzle contour from the same result:

```python
import moc

fig, ax = moc.mpl.plot_characteristics_mesh(data, mirror=True)
fig = moc.plotly.plot_nozzle_contour(data)
```

## Launch the interactive app

```bash
moc-app
```

Open `http://127.0.0.1:8050/` in a browser.

Continue to the [theory](theory/index.md), [examples](examples/index.md),
or [API reference](api_reference.md).
