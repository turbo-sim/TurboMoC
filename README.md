# moc

Method of Characteristics solver for supersonic nozzle and stator blade
design, built on a real-gas (CoolProp, via `jaxprop`) thermodynamic core.
Handles **single-phase, saturated two-phase, and subcooled-liquid
flashing / wet-to-dry inlets** under the Homogeneous Equilibrium Model
(HEM) assumption, on the same solver.

![MOC characteristics mesh](docs/images/characteristics_mesh.png)

## Features

- **Two solver kernels**: `ConventionalSolver` (rounded-arc, supports the
  flashing flat post-jump throat front) and `MOCSolverMLN`
  (minimum-length, sharp-corner) nozzles.
- **`design_nozzle(...)`**: one function, plain arguments in, flat
  JSON-safe dict out.
- **Stator blade parametrization**: fit a closed B-spline blade profile
  from the nozzle wall and export to STEP (`moc.geometry`).
- **Interactive Dash app** for nozzle design and blade parametrization,
  with `matplotlib`/`plotly` plotting and design comparison.

![Stator blade parametrization](docs/images/stator_blade.png)

## Install

```bash
conda env create -f environment.yaml
conda activate moc_env
pip install -e .
```

## Quick start

```python
from moc import design_nozzle

data = design_nozzle(
    fluid_name="nitrogen",
    P0=20e5, T0=113.0,      # subcooled-liquid (flashing) inlet
    p_back=2e5,
    solver="conventional",   # or "mln"
    y_t=0.01, rho_t=0.2, rho_d=0.1,
)

wall = data["wall_final"]    # {"x": [...], "y": [...], "p": [...], "M": [...], ...}
```

```python
import moc

fig, ax = moc.mpl.plot_characteristics_mesh(data, mirror=True)
fig = moc.plotly.plot_nozzle_contour(data)
```

## Interactive app

```bash
python examples/run_app_local.py
```

Two tabs: nozzle design (Phase 1) and stator blade parametrization
(Phase 2, consuming Phase 1's result), each with interactive plots,
design comparison ("pin as reference"), and plot/STEP export.

## Repo layout

- `src/moc/core`, `src/moc/solvers` -- solver kernels
- `src/moc/api.py` -- `design_nozzle`
- `src/moc/geometry` -- stator blade parametrization + CAD export
- `src/moc/plotting_mpl.py`, `src/moc/plotting_plotly.py` -- plotting
- `src/moc/app.py` -- Dash app
- `examples/` -- runnable scripts
