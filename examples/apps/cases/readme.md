# Editable workflow configurations

The workflow directories contain only runners and READMEs. Their editable
YAML inputs are grouped here:

- `gmsh_2D/stator_mesh.yaml`: planar stator geometry and meshing.
- `gmsh_3D/axial_stator.yaml`, `radial_stator.yaml`, `linear_stator.yaml`:
  single-row project examples.
- `gmsh_3D/stator_rotor.yaml`, `rotor_only.yaml`: ordered-row planning examples;
  the new app's rotor numerical integration remains a placeholder.
- `fluent_2D/stator_fluent.yaml`: automated planar Fluent workflow.

Gmsh/Fluent runners default to these local files. Pass `--config` to select a
copy or another case. The current app defaults to its packaged axial resource;
pass `--config examples/apps/cases/<case>.yaml` to load an editable
case. Package defaults remain under `src/turbo_moc/data/` for installations
without the examples directory.

Fluent `design_file` and `output_dir` resolve relative to its YAML file. The
shipped Fluent case refers to `../gmsh_2D/stator_mesh.yaml` and writes to
`../../../turbo_moc_output/fluent_2D`, avoiding outputs among the inputs.
