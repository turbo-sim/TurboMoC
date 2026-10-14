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
| Stator/rotor coupling | General turbo interface, mixing-plane pitch change |
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
