# App launchers

Run either script directly; no arguments are required:

```powershell
python examples/apps/run_app.py
python examples/apps/run_app_legacy.py
```

`run_app.py` opens the current Geometry generation/Mesh generation interface at
<http://127.0.0.1:8052>. Its `CONFIG_FILE` selects `cases/axial_stator.yaml`;
change that assignment to use another sample. Paths resolve relative to the
launcher, so execution works from another directory. Jobs write to `output/`
beside this launcher. The 3D viewer dependencies are installed by default.

The first sidebar controls are **Load configuration** and **Save configuration**.
Load a `.yaml` or `.yml` project to populate the machine and row settings.
Save downloads the current configuration as YAML, including settings for every
row. Invalid files leave the current settings intact.

`cases/` contains axial, radial and linear stator samples plus executable
rotor-only and stator/rotor cases. `rotor_only.yaml` starts with a manual
relative-frame inlet and the legacy vortex-method defaults. `stator_rotor.yaml`
uses a 3 mm axial blade interspace and couples the rotor inlet to the current
upstream stator result; compute that stator first. Its 1000 RPM sample speed
uses the rotor's 105 mm midspan inlet radius. Selecting a row changes the
geometry/mesh generation target while keeping the shared machine view.
These are the only app YAML samples; the package has Python initialization
defaults and accepts any caller-supplied project YAML.

`run_app_legacy.py` opens the preserved Nozzle/Stator/Rotor/Cascade interface at
<http://127.0.0.1:8050>. It retains its diagnostics, control-point editing,
sizing, CAD and rotor tools. Both launchers accept
`TURBO_MOC_EXAMPLE_SMOKE_TEST=1` for initialization without a server.

Installed package commands remain available separately: `turbo_moc-design-app`
(or `python -m turbo_moc.apps.app`) and `turbo_moc-app` for the preceding app.
The installed current command still accepts explicit configuration, port and
output arguments; these are not required by the example launchers.

See the [app guide](../../src/turbo_moc/apps/app/readme.md) and
[feature comparison](../../z_information/merging_summary.md).

Mesh generation automatically includes Fluent-compatible CGNS conversion.
Use **Download project (.zip)** beside the generation controls to download:

- The final `project.yaml` and a readable `README.txt`.
- Completed 3D CGNS meshes, grouped by blade row under `meshes/`.
- 2D CGNS meshes for linear cascades only.
- One `mesh_report.json` combining mesh statistics and quality results for all
  included meshes, and one `activity.log` combining their computation logs.

The button becomes available when a current mesh is ready and no job is active.
Outdated or incomplete meshes are omitted and identified in the README.
The archive uses available stator and rotor artifacts. Internal job files,
MoC/vortex results and derived-geometry caches are omitted.
Temporary MSH topology is removed after conversion.

Rotor inlet coupling reproduces the legacy local velocity triangle and relative
stagnation-state calculation. It does not model interspace losses, radial
transport, or a connected stator-rotor CFD interface. Axial spacing is the
minimum gap from the farthest upstream trailing edge to the rotor leading-edge
plane; radial spacing follows the supplied row inlet/outlet radii. Each row's
mesh remains an independent domain. See the [app guide](../../src/turbo_moc/apps/app/readme.md)
for input units, coupling assumptions and supported rotor settings.
