# Examples

Follow these tutorials to configure each design, interpret the generated
Matplotlib figures, and use the exported coordinates:

```{toctree}
:maxdepth: 1

nozzle_design
stator_design
rotor_design
```

Run these examples from the repository root after installing the package:

```bash
poetry run python examples/nozzle_design.py
poetry run python examples/stator_design.py
poetry run python examples/rotor_design.py
```

The nozzle example uses nitrogen at 20 bar with 10 K inlet subcooling, following
the flashing case in {cite:t}`cioffiNonideal2026`. The stator example uses
cyclopentane at 2.513 bar and an inlet quality of 0.025, following the design
conditions in {cite:t}`cioffiStators2026`. The rotor example retains a nitrogen
relative stagnation state at 20 bar and a vapor mass fraction of 0.5.
The [theory guide](../theory/index.md) explains the corresponding constructions.

The scripts display Matplotlib figures and save SVG figures and CSV coordinates
under `examples/output/nozzle_design/`, `examples/output/stator_design/`, and
`examples/output/rotor_design/`, respectively. Each script prints its absolute
output directory. Rerunning an example overwrites its own output files.

CSV headers identify the coordinate units: meters for the nozzle and rotor,
and millimeters for the stator blade. The stator also exports its source nozzle
wall in meters. The rotor's `blade.csv` contains the assembled blade, while
the `*_local.csv` files contain passage surfaces in their own local frames.

The tutorials embed the Python scripts directly and show saved SVG figures.
The documentation build does not run the solvers. After changing a design,
rerun its script and copy its SVG files from `examples/output/<script_name>/`
to `docs/source/examples/assets/<script_name>/` to refresh the tutorial figures.
