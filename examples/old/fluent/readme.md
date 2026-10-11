# MoC stator mesh and barotropic Fluent simulation

This workflow builds a 2D stator passage, creates its barotropic material,
configures Fluent, runs first- and second-order stages, and exports static-pressure
and barotropic Mach-number PNGs with three periodic blade instances. Geometry and
mesh generation are shared with [../gmsh](../gmsh/readme.md).

## Installation and execution

From the method_of_characteristics repository root, install the optional
PyFluent dependency with Poetry's `fluent` extra:

```powershell
poetry install --extras fluent
```

This installs `ansys-fluent-core` without making it a mandatory dependency.
Then install the companion barotropy checkout into the Poetry environment and
run the workflow:

```powershell
poetry run python -m pip install -e ../barotropy
poetry run python examples/fluent/run_fluent_case.py
```

Fluent must be installed and licensed. The default selects Fluent 2024 R2,
2D double precision and a hidden GUI capable of rendering images. The processor
count is configurable through `fluent.processor_count`.
PyFluent is an optional `fluent` extra; barotropy uses the companion checkout.

There are no command-line arguments. Edit `stator_fluent.yaml`, or change
`CONFIG_FILE` near the start of `run_fluent_case.py` to select another YAML.
Paths resolve relative to that YAML, independently of the working directory.

## Configuration

`design_file: ../gmsh/stator_mesh.yaml` is the shared source for MoC inlet
conditions, nozzle design, blade/passage geometry, mesh parameters, named
boundaries and CGNS export. The Fluent workflow creates the mesh with viewers
disabled, regardless of the mesh-only example's viewer switches.

| Workflow setting | Purpose |
| --- | --- |
| `output_dir: output` | Root for the three output subfolders below |
| `moc_cache_file` / `force_moc_solve` | Reuse a matching nozzle design, or solve again |
| `save_geometry_figure` | Export the blade/passage construction PNG and SVG |
| `prepare_only: true` | Generate the mesh and barotropic model, then stop before Fluent |
| `run_case: false` | Create and save the configured Fluent case without initializing or solving |
| `run_case: true` | Create the case, solve using the strategy below, and save the final case, data and figures |
| `barotropic.pressure_ratio: 8.0` | Barotropic exit pressure = actual inlet pressure / 8 |
| `barotropic.polynomial_degree` | Polynomial degree within each pressure/phase segment |
| `fluent.solution_strategy` | List of dictionaries with `iterations`, `time_scale_factor` and `order: first` or `second` |
| `fluent.boundaries` | Actual mesh-zone names for inlet, outlet, blade and periodic sides |
| `fluent.residual_equations` | Equation names and their individual normalized residual tolerances |
| `fluent.plot_residuals: true` | Save `residuals.png` using `barotropy.plot_residuals()` |
| `fluent.high_order_relaxation_factor: 0.25` | Enable high-order term relaxation for all flow variables |
| `fluent.density_relaxation: 1.0` | Update density consistently with the pressure-dependent material |
| `fluent.print_transcript: true` | Stream Fluent console output, including each iteration's convergence table |
| `fluent.pressure_contour` / `fluent.mach_contour` | Enable each PNG and choose its filename and pixel dimensions |
| `fluent.periodic_instancing` | Enable native translational display copies; `repeats: 3` shows three blades |
| `reporting.level: INFO` | Workflow progress and summaries; `DEBUG` adds diagnostics, `WARNING` shows warnings/errors |
| `reporting.filename: workflow.log` | UTF-8 execution log under `output_dir`, replaced on each run |

### Execution reporting

Numbered sections follow configuration, the MoC design, blade geometry, mesh
generation, barotropic-model generation, Fluent setup, solving and contour export.
Preparation-only mode has five steps and case-creation mode has six. Each solver
stage announces its spatial order, timescale, iteration budget and stopping rule.
Stages advance directly without intermediate diagnostic reports. After the final
stage, the workflow checks pressure/property extrema, residuals and mass balance.
A final results section lists convergence, residuals and generated files.

Workflow messages are printed to the console and written immediately to
`output/workflow.log` with the same formatting. The file is flushed after every
record. Once logging starts, failures include a traceback in the log, and warnings
identify an unconverged final solution or pressures outside the fitted barotropic range.
All execution-time decorators and measurements have been removed.

Reporting uses Python's standard `logging` module through
[`../workflow_logging.py`](../workflow_logging.py). Only the entry point installs
handlers; imported Gmsh and barotropic helpers share them. This does not configure
logging for MoC, Gmsh, barotropy or PyFluent themselves.

With `fluent.print_transcript: true`, Fluent's native convergence table continues
to stream to the terminal. Its detailed native output is saved separately in
`output/fluent/stator_fluent.trn`, including when console streaming is disabled.
The workflow log contains stage headings and final results rather than a duplicate
of every Fluent iteration. MoC's native messages remain unchanged.

### Thermodynamics and boundaries

The barotropic model uses the **same actual inlet state as MoC**, including any
quality adjustment recorded in the solved nozzle. Pressure/quality is resolved
before passing inlet pressure and density to barotropy; a single-phase inlet
uses pressure/temperature.

The barotropic model's exit pressure is configured independently of the MoC
back pressure. With the current YAML, the model covers **251,300 to 31,412.5 Pa**,
while the Fluent pressure outlet remains **95,000 Pa**, from `nozzle.p_back`.
The model's minimum pressure must be strictly below the CFD outlet pressure;
both must exceed the fluid's triple-point pressure.

The workflow requires `calculation_type: equilibrium` and `efficiency: 1.0`
to follow the MoC inlet isentrope; other values are rejected during validation.
Piecewise polynomials preserve phase boundaries and have continuous values at
segment endpoints. Independent EOS samples check density, viscosity and speed
of sound to within 1% and check positive properties and compressibility.
Viscosity is held at its fitted endpoint outside the fitted range to avoid the
exponential extrapolation underflow found during troubleshooting.

Fluent imports metre-scaled CGNS coordinates (`length_scale: 0.001`), assigns
pressure inlet/outlet conditions, a stationary no-slip blade, and a conformal
translational periodic pair verified against the blade pitch. The inlet total
pressure is `nozzle.P0`; zero operating pressure makes gauge pressure equal
absolute pressure. Turbulence intensity and viscosity ratio are configurable.

### Solver and export

The solver is steady, pressure based and coupled, with SST k-omega turbulence
and the energy equation disabled. Density and viscosity use the barotropic
named expressions. Speed of sound, void fraction and vapour quality are
exported as diagnostic expressions; Fluent derives acoustic speed from its
pressure-dependent density expression.

Global pseudo time stepping uses conservative length scales, density
relaxation 1.0 and k/omega relaxation 0.5. The workflow runs **hybrid
initialization once before the first stage**. Subsequent stages continue from
the preceding solution without reinitializing. The saved setup case also selects
hybrid initialization; `run_case: false` does not execute initialization.
For the expression-based material, Fluent may print that it performs preliminary
standard initialization before hybrid initialization; this is part of Fluent's
hybrid procedure, which finishes before the first solver stage.

The current `fluent.solution_strategy` in `stator_fluent.yaml` is:

```yaml
solution_strategy:
  - {iterations: 200, time_scale_factor: 0.5, order: first}
  - {iterations: 200, time_scale_factor: 1.0, order: first}
  - {iterations: 200, time_scale_factor: 2.0, order: first}
  - {iterations: 200, time_scale_factor: 4.0, order: first}
  - {iterations: 200, time_scale_factor: 1.0, order: second}
  - {iterations: 1000, time_scale_factor: 3.0, order: second}
```

| Stage | Spatial order | Iterations | Timescale factor | Early convergence stop |
| --- | --- | --- | --- | --- |
| `stage_1` | First | Up to 200 | 0.5 | Enabled at normalized residuals of 1e-6 |
| `stage_2` | First | Up to 200 | 1.0 | Enabled at normalized residuals of 1e-6 |
| `stage_3` | First | Up to 200 | 2.0 | Enabled at normalized residuals of 1e-6 |
| `stage_4` | First | Up to 200 | 4.0 | Enabled at normalized residuals of 1e-6 |
| `stage_5` | Second | Up to 200 | 1.0 | Enabled at normalized residuals of 1e-6 |
| `stage_6` | Second | Up to 1000 | 3.0 | Enabled at normalized residuals of 1e-6 |

Each block's `order` switches pressure, density, momentum, k and omega together.
Every block stops as soon as all configured residual tolerances are met or its
maximum iteration count is reached, then advances to the next block. Convergence
in an earlier block does not skip subsequent blocks or the switch to second order.
The strategy can contain first- or second-order blocks in any sequence.
Fluent exceptions abort the run immediately. After the final block, the workflow
checks the overall transcript for numerical failures, computes pressure/property
extrema and inlet/outlet mass balance, and reads/plots the residual history.
Failed solution data are not saved. There are no intermediate diagnostic reports
or case/data checkpoints.

High-order term relaxation is enabled with factor 0.25 and Fluent's **Flow Variables
Only** selection, which includes the SST turbulence quantities. This follows the
[Fluent 2024 R2 HOTR controls](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_ug/flu_ug_sec_solve_choose_disc.html).

Residual equations and individual tolerances are defined under
`fluent.residual_equations`. Barotropy's `read_residual_file()` reads the overall
transcript for the CSV and summary. With `fluent.plot_residuals: true`, its
`plot_residuals()` function draws the residual history, saved as one PNG.

After solving, Fluent renders two contours over the entire fluid domain:

- `contour_pressure.png`: static pressure in Pa. Zero operating pressure makes this
  equal to absolute static pressure.
- `contour_mach.png`: the named expression `barotropic_mach_number`, defined as
  `VelocityMagnitude / barotropic_speed_of_sound`. This uses the acoustic-speed
  expression fitted by barotropy.

Native Periodic Instancing translates the displayed domain by minus the blade
pitch in x, with three repeats by default, matching the GUI panel. These are
graphics copies; the solver still computes one periodic passage. Set
`fluent.periodic_instancing.enabled: false` to display one passage. Both contour
definitions and the instancing settings are stored in the saved cases. The PNGs
show the final iterate, which may be unconverged.

## Outputs

With the current YAML, outputs are under `examples/fluent/output/`;
`output_dir` and `moc_cache_file` can change these locations.

The root `workflow.log` covers the entire execution, including preparation and
Fluent shutdown. Its filename is configurable under `reporting`.

| Folder | Files |
| --- | --- |
| `barotropy/` | Fluent expressions (safe text/JSON and raw comparison), inlet state, fit validation, phase diagram and property-fit figures |
| `gmsh/` | Nozzle cache, geometry PNG/SVG, CGNS mesh, mesh PNG, quality report HTML/JSON/CSV/PNG |
| `fluent/` | Final `stator_fluent.cas.h5` and matching `stator_fluent.dat.h5`, overall `stator_fluent.trn`, setup/run summaries, residual CSV and optional PNG, `contour_pressure.png`, `contour_mach.png` |

The table describes a complete run with the current enabled export settings.
Geometry figures, barotropic figures, mesh reports and each Fluent contour have
their own switches. `prepare_only: true` generates mesh/model artifacts and the
workflow log without launching Fluent; it takes precedence over `run_case` and
does not require PyFluent, but still requires barotropy and valid workflow settings.

The setup case is saved before initialization. After the final block and solution
checks, Fluent writes the final case and matching data together, replacing the
setup case at the same filename. The case retains the final block's discretization,
timescale and solver settings; the data contain the final solution. This also
happens when convergence stops the final block early. To view or resume the
solution in Fluent, read the case and its matching data file.
With `run_case: false`, this run produces only the setup case, setup summary
and overall transcript in this folder.

A run that reaches its iteration limit with valid fields also saves the final
case/data pair; `run_summary.json` reports `converged: false`. Solver failures
or invalid final fields prevent this final save. The summary records the saved
`case_file` and `data_file` paths after the paired write succeeds.

The run summary records requested and actual iterations for each stage, spatial
order, timescale factors, hybrid initialization, final residuals, pressure/property
extrema, inlet/outlet mass flows, mass imbalance and whether pressures exceed
the thermodynamic fit. The `diagnostics` object contains only the final checks.
Its `converged` flag requires all residuals at or below their configured
tolerances and relative mass imbalance below 0.1%. Completing the iteration
budget alone does not imply convergence. Failed stages retain the transcript
for diagnosis; failed solution data are not saved.

The pressure ratio is set by YAML rather than by the outlet boundary condition.
Local static pressure can leave the fitted range even when the outlet pressure
is inside it. Such cells use extrapolation or endpoint limits and set
`pressure_outside_fit: true` in the run summary; the independent EOS accuracy
checks apply within the fitted pressure range.

On repeat runs, generated files use the same names and are overwritten without
interactive prompts or timestamped case copies. Historical files from the earlier
workflow are not removed automatically. Mesh and property artifacts are regenerated.
The Fluent transcript is cleared before recording, so residual plots and iteration
counts include only the current run.

## Source files and checks

| File | Responsibility |
| --- | --- |
| `run_fluent_case.py` | YAML loading, shared mesh orchestration, Fluent configuration, strategy-based solve, periodic instancing and both PNGs |
| `barotropic_model.py` | Inlet resolution, model generation, piecewise fits, validation and safe expression export |
| `stator_fluent.yaml` | Workflow, thermodynamic range, solver, iteration and image settings |
| `../workflow_logging.py` | Shared console/file reporting, numbered sections, summaries and failure tracebacks |

For the fast mesh/Fluent automation smoke suite, run:

```powershell
poetry run pytest tests/test_stator_mesh.py tests/test_stator_fluent.py -q
```

The Fluent module checks inlet/fit consistency, case-only creation and cleanup,
hybrid startup, stage transitions after early convergence, final-only checks and
case/data saving, numerical failures and both contour exports. Fluent sessions and
the barotropy residual reader are mocked; no Fluent installation/license or companion
barotropy checkout is needed. The fit check uses jaxprop and real equilibrium EOS
states. These checks do not validate the live Fluent settings API or rendering.

For a full execution check, run the current example using the command above, then inspect
`setup_summary.json`, `run_summary.json`, the overall transcript and the images.

The Python files use Black formatting, descriptive helper functions, docstrings
and comments explaining thermodynamic and solver choices. To format them again,
install Black as a local development tool and run:

```powershell
poetry run python -m pip install black
poetry run python -m black examples/fluent
```

Black is not required to run the workflow.
