# Droplet impingement: 3D stage CFD

Scripts that build, run and post-process a 3D steady RANS simulation of a supersonic axial turbine stage in Ansys
Fluent, used to study the flow that liquid droplets meet when they reach the rotor. The stage pairs:

- a stator designed with the method of characteristics (TurboMoC), 44 blades;
- an impulse rotor designed with TurboDash, 120 blades.

The two-phase working fluid is modelled with a barotropic closure: density, speed of sound and viscosity are
functions of pressure along the inlet isentrope (equilibrium model), so the energy equation is not solved. The rows
are coupled by a mixing plane. One passage per row is simulated with rotational periodicity, and the machine axis is
the x axis.

## Requirements

- TurboMoC installed in the active Python environment (`pip install -e .` from the repository root)
- Ansys Fluent 2024 R2 with PyFluent, Ansys CFX (for `cfx5gtmconv`), and `barotropy` for the barotropic model
- Ansys TurboGrid meshes of the stator and rotor passage (not in the repository)

## Inputs

The only manual step is to put the TurboGrid meshes `stator.gtm` and `rotor.gtm` in `meshes/`.

- **Meshes.** Fluent cannot read `.gtm` files, so `01_setup_stage_case.py` converts each one to a CFX definition file
  (`.def`) with `cfx5gtmconv`. It does this whenever the `.def` is missing or older than its `.gtm`. The converter is
  taken from `PATH`, or else from the newest Ansys installation (`AWP_ROOT<version>`).
- **Design inputs** ship in `inputs/`:
  - `stator_moc_config.yaml`: TurboMoC stator design; sets the inlet state of the barotropic model;
  - `rotor_turbodash.yaml`: TurboDash meanline design; compared against the CFD in the report.
- **Row placement.** Each TurboGrid mesh has its own axial origin. After the rotor is appended, it is translated
  axially so its inlet plane sits `AXIAL_GAP` (default 0) downstream of the stator outlet plane. The setup stops if
  either interface is not a plane x = const.

Folders, in `settings.py`, each with an environment-variable override:

| Folder | Environment variable | Default | Contents |
| --- | --- | --- | --- |
| `MESH_DIR` | `DROPLET_MESH_DIR` | `meshes/` | `.gtm` meshes and their `.def` conversions |
| `RUN_DIR` | `DROPLET_RUN_DIR` | `~/Documents/fluent_runs/paper3_stage` | Fluent case, data and transcripts; keep it outside cloud-synced folders, whose sync locks Fluent files |
| `REPORT_DIR` | `DROPLET_REPORT_DIR` | `RUN_DIR/report` | report and figures |

Meshes and Fluent files are ignored by git.

## Workflow

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `01_setup_stage_case.py <no_gui\|gui> [processors]` | Converts the meshes, builds the barotropic model, imports both meshes, renames the TurboGrid zones, translates the rotor so its inlet meets the stator outlet, and sets up materials, rotation, periodicity, the mixing plane and the boundary conditions. Writes `stage_fluent.cas.h5` and `setup_summary.json` (with the applied rotor translation). |
| 2 | `02_open_case_gui.py [processors]` | Opens the case in the Fluent GUI for initialisation and the run. |
| 3 | `03_save_when_converged.py <serverinfo file>` | Optional: attaches to the open GUI session and saves case and data once it stops iterating. |
| 4 | `04_extract_results.py [processors]` | Reads the solved case headless and writes surface data (blade-to-blade cuts at 10/50/90 % span, axial cuts, walls) and Fluent reports to `RUN_DIR/report/data`. |
| 5 | `05_make_report.py` | Report (Markdown, HTML, PDF) and figures: stage performance against the TurboDash meanline, spanwise rotor-inlet kinematics, wetness, and the impact angle of droplets that lag the vapour. |

The rotor speed, the outlet pressure and the axial gap between the rows are constants at the top of
`01_setup_stage_case.py`.
