# TurboMoC examples

Each workflow folder contains a README, a runner and its own YAML file:

- `gmsh_2D/`: `run_mesh_generation.py`, `workflow_gmsh_stator_2D.yaml`.
- `gmsh_3D/`: `run_mesh_generation.py`, `workflow_gmsh_stator_3D.yaml`.
- `fluent_2D/`: `run_fluent_case.py`, `workflow_fluent_stator_2D.yaml`.
- `fluent_3D/`: `run_fluent_case.py`, `workflow_fluent_stator_3D.yaml`.

[App launchers](apps/readme.md) live in `apps/`; their sample machine-project
YAML files live exclusively in `apps/cases/`. MoC design demonstrations remain
under `moc_design/`. Numerical implementations live in `src/turbo_moc`.

Run any example script by its absolute path from any working directory.
The runners use `pathlib.Path(__file__).resolve().parent` to locate files
beside the script. `Path` handles Windows, macOS and Linux path separators;
these paths do not depend on the terminal's working directory.
No command-line arguments are required: edit its YAML or the `CONFIG_FILE`
assignment to choose another case. Each workflow writes to `output/` beside
its runner. The current app passes `apps/output/` explicitly to
`create_app(..., output_dir=OUTPUT_DIR)`; no output environment variable is needed.
Fluent `design_file`
and `output_dir` resolve relative to its YAML; the supplied output is `output`.
The standalone design demonstrations retain `examples/output/<design>/`.

```powershell
python examples/gmsh_2D/run_mesh_generation.py
python examples/gmsh_3D/run_mesh_generation.py
python examples/fluent_2D/run_fluent_case.py
python examples/fluent_3D/run_fluent_case.py
python examples/apps/run_app.py
python examples/apps/run_app_legacy.py
```

The [Fluent 3D workflow](fluent_3D/readme.md) prepares and solves one stationary
stator passage with rotational periodicity for axial/radial rows or translational
periodicity for linear cascades. It uses the existing 3D mesher and the same
barotropic model as Fluent 2D, adds hub/shroud walls and reports internal-section
pressure/Mach contours. Set `prepare_only: true` in its YAML to generate the mesh
and model without launching Fluent.

[Theory and migration notes](../z_information/readme.md) explain the algorithms
and source ownership. The preserved app and numerical implementations remain
under `turbo_moc.deprecated`. Workflows use the shared `turbo_moc.reporting`
module directly.
