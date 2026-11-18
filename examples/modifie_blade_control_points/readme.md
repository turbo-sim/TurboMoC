# Control-point and KL blade perturbation example

This example generates blade geometries from
`blade_design_baseline.yaml`. The YAML control-point displacements are applied
to the solved baseline blade first. Normalized control-point samples are then
generated and a KL perturbation is applied to each control-point geometry.

Run it from the repository root with:

```powershell
python examples\modifie_blade_control_points\control_points_random_field_perturbation.py
```

The script prints progress for configuration loading, nozzle solving, sample
generation, figure creation, and YAML export. It creates the following files in the local
`figures` directory:

- `baseline_vs_cp_samples.png` and `.svg`: YAML baseline and control-point samples.
- `cp_and_kl_samples.png` and `.svg`: control-point and CP+KL geometries.
- `sample_01_comparison.png` through `sample_05_comparison.png` and `.svg`:
  individual baseline/CP/CP+KL comparisons.

It also creates complete app-project YAML files in `generated_blades`:

- `blade_design_baseline_sample_01.yaml` through
  `blade_design_baseline_sample_05.yaml`.

Each file has the same structure and settings as `blade_design_baseline.yaml`,
with the corresponding control-point displacements and KL
`geometry_uncertainty` values.

It also creates `yaml_samples/sample_01.yaml` through
`yaml_samples/sample_05.yaml`. Each YAML file is a complete app project whose
blade geometry contains both that sample's control-point displacements and KL
`geometry_uncertainty` values.

The first control-point sample is neutral (`0.5` for every selected control
point). The remaining samples use random seed `42`; KL samples use seeds
starting at `90`. Control-point displacements use the bounds defined by the
script for each selected control point.
