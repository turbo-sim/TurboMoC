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
upstream stator result. Check **Stator** and **Rotor** under **Apply calculations to**
and click **Compute blade design** to compute the stator first, then the rotor.
Its 1000 RPM sample speed
uses the rotor's 105 mm midspan inlet radius. Selecting a row changes the
quality report through **Inspect row** while keeping the shared
machine view. The calculation checkboxes separately control which rows run.
**Plot rows** checkboxes control which rows appear in geometry and mesh views;
hiding a row retains its calculation results. Meridional mesh projections show
checked rows with current 2D meshes together. Blade-to-blade geometry and mesh
also show every checked row in a shared midspan conformal chart, including the
physical interspace and angular phase. Solid meridional outlines show the blade extent;
dashed outlines show inlet/outlet flow-domain extensions. Hover over the compact
control strip for interspace details and guidance.
Each 3D view has a visibility checkbox and **Display** button for every cascade.
**Display** opens a panel beside the viewer for that cascade's displayed blade
count, surface visibility and colors. Geometry and mesh keep separate display
settings; **Copy from geometry** transfers the selected cascade's settings to
the mesh view. Display changes preserve the camera and do not alter the physical
blade count, YAML configuration or exported meshes. Hover over **Displayed blades**
for guidance.
These are the only app YAML samples; the package has Python initialization
defaults and accepts any caller-supplied project YAML.

`run_app_legacy.py` opens the preserved Nozzle/Stator/Rotor/Cascade interface at
<http://127.0.0.1:8050>. It retains its diagnostics, control-point editing,
sizing, CAD and rotor tools. Both launchers accept
`TURBO_MOC_EXAMPLE_SMOKE_TEST=1` for initialization without a server.

Installed package commands remain available separately: `turbo_moc-app`
(or `python -m turbo_moc.app`) and `turbo_moc-app-legacy` for the preceding app
(`python -m turbo_moc.legacy.app`).
The installed current command still accepts explicit configuration, port and
output arguments; these are not required by the example launchers.
Run `poetry install --only-root` after updating a Poetry checkout to refresh
the installed commands. Prefix either command with `poetry run` in that environment.

See the [app guide](../../src/turbo_moc/app/readme.md) and
[development checklist](../../docs/internal_information/app_development_checklist.md).

Mesh generation automatically includes Fluent-compatible CGNS conversion.
Use **Export project (.zip)** beside the generation controls to download:

- The final `project.yaml` and a readable `README.txt`.
- One top-level `stage.cgns` with a separate fluid zone for every active row.
- Completed 3D CGNS meshes, grouped by blade row under `meshes/`.
- 2D CGNS meshes for linear cascades only.
- One `mesh_report.json` combining mesh statistics and quality results for all
  included meshes, and one `activity.log` combining their computation logs.

The button becomes available when every active row has a current 3D mesh and
no job is active. A linear cascade can also export its current true 2D mesh
without a volume mesh; if both dimensions are available, `stage_2d.cgns` is added.
Export includes every active row regardless of calculation scope. Missing or
outdated annular row meshes block a partial-stage export. Internal job files,
MoC/vortex results and derived-geometry caches are omitted.
Temporary MSH topology is removed after conversion.

The two mesh-generation buttons run selected rows sequentially and retain
already-current meshes. Generate current 2D meshes before requesting 3D meshes.
Progress reports the current row, and the terminal retains live output from
each job. Completed rows update their views immediately; cancellation or a
failed row stops the remaining queue and retains earlier successful results.

Rotor inlet coupling reproduces the legacy local velocity triangle and relative
stagnation-state calculation. It does not model interspace losses, radial
transport, or a connected stator-rotor CFD interface. Axial spacing is the
minimum gap from the farthest upstream trailing edge to the rotor leading-edge
plane; radial spacing follows the supplied row inlet/outlet radii. Each row's
mesh remains an independent zone inside `stage.cgns`, with separate interface
faces. Set up their interface coupling and the rotor reference frame in the
CFD solver. See the [app guide](../../src/turbo_moc/app/readme.md)
for input units, coupling assumptions and supported rotor settings.
