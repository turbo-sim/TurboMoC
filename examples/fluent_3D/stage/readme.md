# Axial stator + rotor stage in Fluent 3D

This example designs the app's default axial stage with the rotor coupled to the
stator exit at the YAML speed. It runs the app's own jobs (blade designs, 3D
meshes, combined stage CGNS) and builds the barotropic material along the stator
inlet isentrope. Then it opens a configured, unsolved Fluent case with its GUI.

From the repository root:

```powershell
poetry run python examples/fluent_3D/stage/run_fluent_stage.py
```

Fluent 2024 R2 must be installed and licensed. The companion `barotropy`
checkout is required, as for the [stator workflow](../stator/readme.md).

## Fluent setup

| Item | Setting |
| --- | --- |
| Physics | Pressure-based steady, SST k-omega, energy off, operating pressure 0 Pa |
| Material | Barotropic density and viscosity expressions in both cell zones |
| Rotor zone | Frame motion about +z at the YAML speed, with the sign that matches the stator swirl |
| Stator/rotor coupling | General turbo interface, mixing-plane pitch change, halfway along the `axial_interspace_mm` gap (10 mm, half a chord) |
| Periodicity | Rotational periodic pair per row |
| Inlet | Stator total pressure, normal to the boundary |
| Outlet | Rotor exit static pressure from the app's stage statistics |
| Walls | No-slip; the rotor shroud is stationary in the absolute frame |
| Numerics | Coupled pseudo-transient, second order, 8 processes, hybrid initialization selected |

## Outputs

| Path | Content |
| --- | --- |
| `output/project.yaml` | App project that was run |
| `output/operating_point.json` | Rotor speed, relative inlet Mach and angle, exit pressure, stage summary |
| `output/mesh/` | `stage.cgns`, row meshes and the mesh report from the project export |
| `output/barotropy/` | Barotropic fits and figures |
| `output/fluent/` | `stage_fluent.cas.h5`, `setup_summary.json` and the transcript |
| `output/report/` | `report.pdf` (also `.html`, `.md`), `report.json` and `figures/` from the report script below |

## Report of the solved stage

After running the case in Fluent, save case and data as `output/fluent/stage_fluent.cas.h5`, then:

```powershell
poetry run python examples/fluent_3D/stage/report_fluent_stage.py
poetry run python examples/fluent_3D/stage/report_fluent_stage.py --plots-only
```

The first command reads the case in a headless Fluent session. It extracts the boundary and midspan
surfaces, Fluent's mass-flow and moment reports, the mesh quality and the residual history to
`output/report/data/`. `--plots-only` rebuilds the figures and tables from those files without Fluent.
Velocity, Mach number and isentropic quantities use the same barotropic fits as the Fluent material,
evaluated in Python, with H(p) = ∫dp/ρ as the isentropic head.

| Output | Content |
| --- | --- |
| `geometry.png` | 3D blading (pyvista) and meridional view with the 25/50/75 % spans |
| `midspan_mach_pressure.png` | Unwrapped midspan Mach number (stator absolute, rotor relative) and p/P₀ |
| `blade_loading.png` | Blade p/P₀ and isentropic Mach number at 25, 50 and 75 % span |
| `stator_exit_pitchwise.png` | Mach number, p/P₀ and flow angle along the pitch at the stator exit |
| `density_gradient_stator.png`, `density_gradient_rotor.png` | Numerical schlieren at midspan |
| `residuals.png` | Residual history from the run transcript |
| `report.pdf`, `report.html`, `report.md` | Four-page report: summary cards and highlights, stage performance and station tables against the MoC design, mesh, setup and convergence, and all figures |

## Code

Both runners are thin wrappers around [`turbo_moc.fluent.stage`](../../../src/turbo_moc/fluent/stage/):

| Module | Role |
| --- | --- |
| `design.py` | App jobs, stage CGNS export, MoC design point (`operating_point.json`) |
| `case.py` | Fluent case: material, rotor frame motion, mixing plane, periodicity, boundaries |
| `setup.py` | `main(config)` for `run_fluent_stage.py` |
| `extraction.py` | Headless Fluent: surfaces, mass-flow and moment reports, mesh quality, residuals |
| `analysis.py` | Kinematics, mass averages and stage performance from the extracted data |
| `figures.py` | Report figures at printed size |
| `document.py` | Report blocks rendered to Markdown, HTML and PDF |
| `report.py` | `main(config, plots_only)` for `report_fluent_stage.py` |

The Fluent barotropic expressions are evaluated in Python by `BarotropicClosure` in
[`barotropic_model.py`](../../../src/turbo_moc/fluent/barotropic_model.py).
