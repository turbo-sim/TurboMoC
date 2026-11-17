# Blade with geometric uncertainty

This example builds a stator blade from `blade_design_baseline.yaml` and
generates Karhunen-Loeve (KL) geometric perturbations. The shared
`define_perturbed_blade` helper interpolates the random field onto the blade
curve and applies the perturbation as a signed fraction of the nominal blade
chord. The plotting helper writes both PNG and SVG figures.

Run it from the repository root with:

```powershell
python examples\blade_with_geometric_uncertainty\gaussina_random_field_perturbation.py
```

The script solves the nozzle, constructs the nominal blade using the YAML
geometry settings, generates 10 KL samples with `sigma0=2` and random seed
`42`, and saves:

```text
examples\blade_with_geometric_uncertainty\figures\geometric_uncertainty.png
examples\blade_with_geometric_uncertainty\figures\geometric_uncertainty.svg
```

The example requires the project dependencies and a working nozzle/property
configuration. Adjust `N_SAMPLES`, `SIGMA0`, or `RANDOM_SEED` in the script to
change the generated ensemble.
