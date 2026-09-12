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

Each example starts from a two-phase nitrogen stagnation state at 20 bar with
a vapor mass fraction of 0.5 (in the relative frame for the rotor). The nozzle
example designs a nozzle alone; the stator example also builds a blade profile
from the nozzle wall; the rotor example designs a vortex-flow blade.

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
