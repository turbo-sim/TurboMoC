# TurboMoC application implementation

This directory owns the browser UI and its local job service. Shared numerical
code lives in [../geometry/](../geometry/) and [../meshing/](../meshing/).
Case YAML and launchers live in [../../gmsh_3D/](../../gmsh_3D/readme.md).
The [theory guide](../../theory_guide.md) and [roadmap](../../roadmap.md) live
at the `example_meshing/` root.

| File | Responsibility |
| --- | --- |
| `application.py` | Dash factory, layout, controls and callbacks. |
| `workflow.py` | Stage identities, selected-result handoff, derived-geometry reuse and worker launch/poll/cancel. |
| `worker.py` | Isolated numerical execution with live progress and logs. |
| `preview.py` | Smooth, capped geometry tessellation and display normals. |
| `plots.py` | Labeled 2D engineering views and quality histograms. |
| `assets/studio.js` | Browser VTK scenes, rotational/translational display copies and mapper updates. |
| `assets/mesh_edges.js` | Reused display triangles and antialiased cell-edge shader. |
| `assets/studio.css` | App layout and styling. |

The numerical owners are [geometry_2d.py](../geometry/geometry_2d.py),
[geometry_3d.py](../geometry/geometry_3d.py),
[meridional_channel.py](../geometry/meridional_channel.py),
[linear_cascade.py](../geometry/linear_cascade.py),
[moc.py](../geometry/moc.py), [meshing_2d.py](../meshing/meshing_2d.py),
[meshing_3d.py](../meshing/meshing_3d.py), [quality.py](../meshing/quality.py)
and [inspection.py](../meshing/inspection.py).
`MeridionalChannel`, the linear-cascade coordinate adapter and
`Design.meridional_channel` express the shared section-to-physical model. The app uses the same reference/volume/export stages as the
CLI, without importing historical folders or a second numerical implementation.

The UI has two persistent workspaces, **Geometry generation** and **Mesh generation**, with
settings grouped by row. The sidebar shows every active row in a collapsible
group, with geometry or meshing settings chosen by the workspace. The selected
row supplies section plots, generation actions and the quality report. Shared physical views
assemble available rows. Machine mode activates stator, rotor or both, while
the independent geometry type chooses linear, axial or radial mapping.
Default row labels are Stator and Rotor; workspace tabs have no numbers,
and the viewer slider reads Displayed blades. Stable YAML IDs remain unchanged.
Linear cascade is unavailable for the combined mode; mixed flow is outside
the current interface. Rotor conventional/vortex settings remain explicit
planning inputs, with numerical actions disabled.

[configuration_project.py](../configuration_project.py) owns the ordered rows,
stable IDs, machine-mode validation and conversion to existing flat numerical
inputs. Each row's scientific job references are retained independently when
selection changes. Geometry and mesh changes invalidate that row's affected
stages; switching the selected row is a view change. The terminal identifies
the row and stage being computed.

Editing retains the last validated project independently of currently
invalid input text. An invalid field can remain visible while the user
corrects it; it does not erase the other validated settings or publish a new
numerical configuration. If a structural row/machine change discards invalid
edits, the app reports that explicitly. Scientific jobs require a valid
current configuration rather than silently using earlier valid settings.
Dynamic axial interspace controls are also checked against the selected
row/family so obsolete controls cannot overwrite a newly selected model.

Assembly planning is distinct from row-local scientific meshes.
`geometry.axial_interspace_mm` on an axial rotor specifies the nonnegative
minimum upstream TE-to-rotor LE-plane gap when an upstream row is active.
[configuration_project.py](../configuration_project.py) resolves it through
`assembly_placement` and reports adjacent-row spacing through
`assembly_interspaces`. The upstream extent uses the greater hub/tip chord.
A paired rotor's derived relative position takes priority over a literal
manual display offset; the interspace is inactive for rotor-only mode.

Radial rows remain positioned by their radii. Their directed gap is positive
for clearance and negative for overlap for either inflow or outflow, with no
extra interspace input. The shared meridional view includes planned rotor
channel outlines, not fabricated rotor blades. Angular phase rotates annular
displays. These assembly transforms leave independent row meshes unchanged;
physical row interfaces and solver reference frames remain future integration.

The header links to
[Sustainable Thermal Power, research group at DTU Construct](https://thermalpower.dtu.dk/).
The former Show all settings control is removed; settings follow their
workspace and active row groups.

The Mesh generation workspace combines 2D section/grid views and the 3D boundary/report
view, while keeping generation and export actions separate. Meridional/grid
captions retain the mapped first-layer-height and span-spacing estimates,
distinguishing conformal targets from physical sizes. Linear rows
support true 2D export and physical 3D extrusion, including an optional one-cell
span. Annular 2D downloads retain their explicit conformal-reference meaning.

The dense reference mesh uses Plotly `Scattergl` for its many unique cell
edges. When its row has no current artifact, `plots.section_mesh_placeholder`
keeps an empty GL trace rather than replacing the figure with a different
renderer. This clears the visible mesh while retaining the existing scene and
its contexts. The more modest meridional grid uses SVG `Scatter`, avoiding an
unnecessary second GL scene. Twenty empty/restore cycles, including title,
axis and `uirevision` changes, retained the same two internal canvases and
exactly two GL contexts with no errors or warnings. These display changes
leave scientific meshes and exported connectivity untouched.

Every **Compute MoC** action solves afresh. Its successful job writes
`moc_result.json`; downstream stages read only the explicitly selected job.
There is no MoC cache lookup or reuse policy. The fitted-template and derived
geometry stores retain rapid design iteration for that selected result. A new
MoC job changes downstream identities even when its input values are identical.

Dependencies remain managed by `pyproject.toml`: Dash/dash-vtk for the interface,
NumPy/SciPy for geometry, Gmsh for meshing, VTK/PyVista for volume inspection,
diskcache for derived-geometry reuse and PyYAML for configuration. Quality
calculations are independent array operations; inspection keeps optional VTK
filters behind a separate numerical adapter. Common logging uses the
package-scoped [../logging.py](../logging.py).

Run `python -m example_meshing.gmsh_3D.run_mesh_app` from the repository root.
The minimal launcher accepts `--config path/to/project.yaml` and `--port 8053`;
`create_app(config_file=None)` initializes the selected project. Rotor-only
files open the planning interface without generated geometry. YAML initialization
does not reopen sessions or adopt earlier scientific artifacts.
Workers launch as `example_meshing.src.app.worker`. Each job retains
`project.yaml` for the submitted machine/rows and `configuration.yaml` for
the selected flat numerical row, so project state and numerical inputs remain
reviewable separately. Job submission rejects inactive rows and unsupported
rotor providers before launching a worker. [../paths.py](../paths.py)
resolves case/output directories independently of the working directory.
Jobs write beneath `example_meshing/gmsh_3D/output/app/`; browser assets come
from this directory's `assets/`. Restart after Python edits because workers
verify the submitted source revision.

The final implementation passes **110 tests** in 74.32 s, including current
project/validation state, interspaces, rotor planning outlines and existing
numerical/app/Fluent behavior. A real fresh axial/linear Edge workflow passed
both mesh stages and exports, row retention/rotor guards, gap diagnostics,
one-element extrusion, translated copies, live progress, cancellation,
invalid-edit preservation, physical-size captions and the header. Its final
audit recorded zero exceptions, log errors or WebGL-context warnings.

The 3D adapter owns a persistent React root inside each Dash viewport and
reuses installed dash-vtk components. Callback children return `no_update`;
one View and stable actor pool update data and hide unused actors without
Dash remounts. New datasets/copy counts fit; surface/width edits retain the
camera. A focused 50-edit audit also verified actual View/canvas and camera
stability without errors or warnings. No new frontend package/CDN is needed.
The [migration record](../../migration.md) records current and historical
checks, review artifacts, manual acceptance and remaining solver integration.
