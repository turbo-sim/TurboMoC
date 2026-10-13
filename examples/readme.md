# TurboMoC examples

Each workflow folder contains a README, a runner and its own YAML file:

- `gmsh_2D/`: `run_mesh_generation.py`, `workflow_gmsh_stator_2D.yaml`.
- `gmsh_3D/`: `run_mesh_generation.py`, `workflow_gmsh_stator_3D.yaml`.
- `fluent_2D/`: `run_fluent_case.py`, `workflow_fluent_stator_2D.yaml`.
- `fluent_3D/`: `run_fluent_case.py`, `workflow_fluent_stator_3D.yaml`.

MoC design demonstrations live under `moc_design/`. Follow the
[design tutorials](../docs/source/examples/index.md) to configure a nozzle,
stator or rotor, interpret its figures and use the exported coordinates.

Run the following commands from the repository root after `poetry install`.
The Gmsh and Fluent runners locate their YAML files beside the script and need
no command-line arguments. Edit the adjacent YAML or the runner's `CONFIG_FILE`
to choose another case. Follow each workflow's README for additional
dependencies and, for Fluent, installation and licensing requirements.

```powershell
poetry run python examples/moc_design/nozzle_design.py
poetry run python examples/moc_design/stator_design.py
poetry run python examples/moc_design/rotor_design.py
poetry run python examples/gmsh_2D/run_mesh_generation.py
poetry run python examples/gmsh_3D/run_mesh_generation.py
poetry run python examples/fluent_2D/run_fluent_case.py
poetry run python examples/fluent_3D/run_fluent_case.py
```

Each Gmsh workflow writes to `output/` beside its runner. Fluent paths, including
`output_dir` and the optional `design_file`, resolve relative to its YAML; the
supplied output directory is also `output/`. The design demonstrations write to
`examples/output/<design>/`.

The [Fluent 3D workflow](fluent_3D/readme.md) prepares and solves one stationary
stator passage with rotational periodicity for axial/radial rows or translational
periodicity for linear cascades. It uses the existing 3D mesher and the same
barotropic model as Fluent 2D, adds hub/shroud walls and reports internal-section
pressure/Mach contours. Set `prepare_only: true` in its YAML to generate the mesh
and model without launching Fluent.

The [theory guide](../docs/source/theory/index.md) explains the design methods.
See [workflow configuration](workflow_configuration.md) to reuse machine and row
settings across Gmsh and Fluent examples.
