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
generation, and figure creation. It creates the following files in the local
`figures` directory:

- `baseline_vs_cp_samples.png` and `.svg`: YAML baseline and control-point samples.
- `cp_and_kl_samples.png` and `.svg`: control-point and CP+KL geometries.
- `sample_01_comparison.png` through `sample_05_comparison.png` and `.svg`:
  individual baseline/CP/CP+KL comparisons.

The first control-point sample is neutral (`0.5` for every selected control
point). The remaining samples use random seed `42`; KL samples use seeds
starting at `90`. Control-point displacements are bounded by plus or minus
five percent of the blade pitch.
