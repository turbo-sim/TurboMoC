# MoC blades and conforming 3D passage meshes

Run `examples/gmsh_3D/run_mesh_generation.py` from the repository root. It has
no CLI arguments: select `axial_stator.yaml` or `radial_stator.yaml` using
`CONFIG_FILE` at the top of the script. Cache controls remain there; the YAML
`workflow` mapping controls figure saving, viewers and logging. Outputs stay in `examples/gmsh_3D/output/<case>/`.

```powershell
.venv/Scripts/python.exe examples/gmsh_3D/run_mesh_generation.py
```

The main script obtains a thermodynamic MoC nozzle solution, fits the stator
blade with the existing cubic B-spline construction, builds a smooth periodic
passage, meshes one section in Gmsh, and maps corresponding sections into 3D.
The cache stores only the MoC solution. Blade geometry and meshes are rebuilt
on every run; fitted blade coordinates are not cached.

## Interactive browser app

The local Dash app uses the same axial/radial YAML files, geometry functions,
MoC cache validation and mesh/export functions as the script:

```powershell
.venv/Scripts/python.exe -m pip install ".[mesh-viz]"
.venv/Scripts/python.exe examples/gmsh_3D/run_mesh_app.py
```

Open **http://127.0.0.1:8051**. The example also needs the companion Barotropy
package, as does the mesh-generation script. Select the axial or radial case
and expand the YAML groups in the left pane. Geometric parameters have sliders
and exact number inputs; sliders update on release. The port and default case
are globals at the top of `run_mesh_app.py`.

- **Compute MoC** loads a matching validated solution or solves the edited
  nozzle inputs. A matching existing cache is loaded on startup without a solve.
  Changing nozzle inputs requires another click; changing blade, passage or
  machine inputs rebuilds geometry using the existing MoC result.
- **m′–θ geometry** shows the fitted blade, periodic passage and smooth inlet/
  outlet extensions at midspan. **3D geometry** maps display tessellations of
  all seven surfaces into physical coordinates, without generating a CFD mesh.
- **Displayed blades** controls rotational copies from one blade to the complete
  row. Under **Mesh → PyVista → Surfaces**, select blade, hub, shroud, inlet,
  outlet and either periodic surface. The selection applies to both 3D tabs.
  Drag to rotate, scroll to zoom, and press **R** over the 3D view to fit it.
- **Create mesh** runs the existing conforming passage mesher and CGNS exporter
  in a background process. **3D mesh** displays the actual boundary faces of
  that volume mesh. **Mesh → PyVista → Show mesh edges** controls element lines.
  Geometry and surface selection remain usable while a background job runs.
- Download the mesh, CGNS file, input snapshot, summary and workflow log from
  the links below the view. **Download YAML** exports the edited configuration;
  editing in the app does not overwrite the example YAML files.

Each job uses `output/dash/<session>/<job>/`. The MoC solution stays on the
server; only surface data needed for display is sent to browser memory.
Fitted blade coordinates are not cached on disk. Meshing and view changes never
trigger a MoC solve. Editing geometry or mesh numerics marks an existing mesh
as an earlier snapshot until **Create mesh** is clicked again. Exported meshes
contain one physical periodic passage; extra displayed blades are copies.

This is a local app: Dash's diskcache background manager keeps expensive jobs
out of request handlers. The native Gmsh/PyVista window switches in `workflow`
are used by the standalone script; the app embeds its views. The browser viewer
uses [Dash VTK](https://dash.plotly.com/vtk), so no separate desktop viewer or
Dash plugin is needed. Its volume-mesh inspection currently shows exterior
surfaces, rather than internal cells or slicing. See the official
[background callback documentation](https://dash.plotly.com/background-callbacks)
for the worker model.

## Geometry inputs

All machine dimensions are **millimetres**. Nozzle inputs retain their existing
SI convention. `machine.num_blades` is the physical full-annulus blade count.

| Case | Independent inputs | Interpretation |
| --- | --- | --- |
| Axial | `hub_radius`, `blade_height`, `hub_chord`, `tip_chord`, `num_blades` | Tip radius = hub radius + height; chord varies linearly across span. Leading edges share axial position zero. Hub and shroud are cylindrical. |
| Radial | `inlet_radius`, `outlet_radius`, `blade_height`, `num_blades` | Blade radii at LE and TE. Inlet > outlet gives inflow; reverse them for outflow. Parallel endwalls are at z = 0 and z = height. |

These radii/chords locate the **blade**, not the extended mesh inlet/outlet.
`passage.inlet_chords` and `outlet_chords` extend the conformal domain beyond
the blade. Radial mesh-cap radii therefore differ from the specified blade
radii. Axial tapered meshes can have span-dependent inlet/outlet axial
stations, while their endwall radii remain constant: this is not flaring.
There is no tip clearance, hub fillet, spanwise twist, or prescribed lean.

## One geometry definition and one meshing core

Axial and radial YAML inputs are convenience interfaces. The runner calls
`geometry_from_machine()` once; its two short adapters return the same
`GeometryDefinition`: hub and shroud B-splines in `(r, z)` and a blade count.
`Channel` and `create_mesh()` consume this definition without branching on
the machine type. The MoC blade fitting, periodic passage construction,
volume sweep, mesh checks and exports are shared.

Both wall curves use a meridional station `u`: zero locates the blade leading
edge and one locates its trailing edge. The general stream surface is

```text
P(u, eta) = (1-eta)*hub(u) + eta*shroud(u),   P = (r, z)
m'(u, eta) = integral_0^u |dP/du|/r du
eta = 0 at hub; eta = 1 at shroud
```

The axial adapter produces straight vertical meridional walls at the hub/tip
radii, with their specified chord lengths. The radial adapter produces
straight radial walls at two axial heights. Radial inflow and outflow simply
reverse the direction of those curves. Chord, reference radius, conformal
scaling and cell orientation are derived from the curves rather than a label.

Custom curved walls can be supplied directly as SciPy `BSpline` objects in
Python. For example, the existing tapered axial description is equivalent to:

```python
definition = geo.GeometryDefinition(
    hub=geo.meridional_line((100, 0), (100, 40)),
    shroud=geo.meridional_line((120, 0), (120, 32)),
    num_blades=12,
)
channel = geo.Channel(definition)
# Pass geometry=definition to msh.create_mesh(...).
```

Wall extensions follow their endpoint tangents. Straight walls use a single
analytic conformal integral/inverse for axial, radial and oblique directions.
Curved walls use numerical quadrature and an inverse ODE evaluated at all mesh
nodes. Corresponding hub/shroud stations must define a nonfolded channel with
positive radii. The blade still uses the shared MoC template with isotropic
conformal scaling; arbitrary spanwise lean/twist is not included. A curved-wall
mapping and a small swept mesh are checked, but a custom channel still needs
mesh-quality assessment. `mesh_summary.json` records the general wall curves.

## The m'-theta plane

Mueller and Verstraete (2017), Section 2.2 and Figure 6, describe stacking
blade-to-blade grids on surfaces of revolution using a conformal mapping.
The dimensionless meridional coordinate is

```text
dm = sqrt(dz^2 + dr^2)
m' = integral(dm/r)
dl^2 = r^2 * (dm'^2 + dtheta^2)
```

We use `q = (Rref*theta, -Rref*m')` in Gmsh, with clockwise theta and a
downstream-negative-y convention matching the 2D stator example. The physical
right-handed azimuth is `-theta`. The reference radius is the midspan radius
for axial blades and sqrt(inlet_radius*outlet_radius) for radial blades.

Both source coordinates receive **one isotropic scale**. Chordwise
`xi = -q_y / (Rref*blade_mprime_extent)` is 0–1 over the blade, but angular
coordinates are not independently normalized: that would alter blade angles.
This conformal chord fraction differs from the wall-curve station `u` in
general; their relationship is obtained by inverting the meridional integral.
The prescribed physical chord/radii determine the conformal chord. Passage
pitch is independently `2*pi*Rref/num_blades`; changing blade count changes
spacing/solidity, not the fitted MoC profile. The new passage is recentered
using the existing clearance optimization and smooth cubic inlet/outlet blends.
An impossible pitch is rejected rather than silently squeezing the blade.

With global **Z as rotation axis**, the inverse maps are:

```text
theta = -q_x/Rref
mprime = -q_y/Rref
X = r*cos(theta)
Y = r*sin(theta)

Axial: r = hub_radius + eta*blade_height
       Z = r*mprime
       blade_mprime_extent = interpolated_chord(eta)/r

Radial: r = inlet_radius*exp(sign*mprime)
        sign = sign(outlet_radius-inlet_radius)
        Z = eta*blade_height
        blade_mprime_extent = abs(log(outlet_radius/inlet_radius))
```

Sections are scaled isotropically in the conformal plane to maintain
their metal angles and the wall curves' LE-to-TE extent. A positive-weight harmonic
displacement moves the core to the new outer boundaries. Blade and near-wall
collar nodes are prescribed, so smoothing cannot remove the inflation layers.
Every section retains the same connectivity. Core deformation is applied
whenever the conformal chord changes across span, irrespective of the machine
type. Excessive deformation that inverts cells is rejected. With the radial
adapter the conformal chord is constant, so the reference plane is reused and
extruded between parallel endwalls.

## Mesh controls and physical layer sizes

`mesh.numerics` groups the existing 2D algorithm, size, curvature,
recombination and `inflation_layers` controls. Two extra parameters set the
span distribution: `span_layers` counts cells across the height, and
`span_stretch` uses symmetric tanh clustering near both endwalls. Zero gives
uniform spacing; larger values increase clustering. There are L+1 sections
for L layers. This controls endwall refinement separately from blade inflation.

Mesh lengths are specified in the **reference conformal plane**. Physical
local scale is `r/Rref`. Thus a radial first-layer height approximately becomes
`first_height*r/Rref`, varying along the blade. For axial blades with uniform
chord, its physical size is approximately independent of span. With taper,
the blade collar is scaled with the chord, so physical heights also vary with
`chord(eta)/midspan_chord`. Curvature and Gmsh merging can alter local heights.
These are not promises of constant physical 3D wall-normal spacing or y+.
At tapered blade/endwall junctions, section normals also differ from 3D normals.

Quadrilaterals become hexahedra; remaining triangles become prisms. The code
checks cell Jacobians and SICN, outward boundary orientation, complete exterior
coverage and matching periodic nodes before CGNS export. `mesh_summary.json`
records quality, nominal layer heights, span stations and the rotation matrix.
Assess the recorded quality and refine resolution for the intended CFD use;
the example is not a mesh-convergence study.

## Boundaries, periodicity and export

| Physical group | Meaning |
| --- | --- |
| `blade` | Blade wall |
| `hub`, `shroud` | Lower/upper span endwalls; planar for radial, cylindrical for axial |
| `inlet`, `outlet` | Extended passage caps |
| `periodic_left`, `periodic_right` | Conforming rotational periodic surfaces |
| `fluid_domain` | One fluid volume |

`periodic_right` is the rotation of `periodic_left` by **-2*pi/N about Z**.
Their shared section topology gives identical surface connectivity and a
complete node bijection, including their intersections with other patches.
Gmsh's periodic transform is master-to-slave; CGNS stores reciprocal
current-to-donor maps, so `right_to_left` has angle +2*pi/N.

The `.msh` file contains physical groups and Gmsh periodic metadata. The
CGNS 3.4/ADF file contains a 3D base/zone, one fluid volume element section,
seven named surface sections and face-centered BC ranges, metre units, and
reciprocal periodic vertex lists/rotation metadata. Physical coordinates are
multiplied by `mesh.cgns.length_scale` (0.001 for mm -> m).

The exporter adapts ParaBlade's audited 3D exporter without requiring ParaBlade
at runtime. It uses TurboMoC's existing `io.cgns_adf` container implementation.
Boundary types are controlled in the YAML. Rotational periodic pairing remains
a Fluent setup step; CGNS metadata does not replace CFD solver setup. Fluent
has not been launched by this example.

## Python visualization

`mesh.matplotlib.num_blades` controls display copies, not physical pitch or
exported passage count. The run saves an actual reference-plane mesh figure
and a 3D surface mesh view in PNG/SVG. Extra blades are rigid rotations of
the generated mesh by one pitch. Blade interiors are white, flow surfaces
use `domain_color`, and mesh edges use `mesh_color`; opacity is always one.
All Matplotlib figures use `barotropy.set_plot_options()` and the YAML DPI.

For interactive Python inspection, install the optional extra:

```powershell
.venv/Scripts/python.exe -m pip install ".[mesh-viz]"
```

The case YAML contains all viewer and workflow controls:

```yaml
workflow:
  save_figures: true
  show_gmsh: true       # Also requires mesh.show_gui: true.
  show_pyvista: true
  log_level: INFO

mesh:
  # ... existing mesh controls ...
  pyvista:
    show_mesh_edges: true
    surfaces:
      blade: true
      hub: true
      shroud: false
      inlet: false
      outlet: false
      periodic_left: false
      periodic_right: false
```

`run_mesh_generation.py` opens the selected surfaces in two PyVista panels:
**one passage** and **the full turbine row**. Enable only `blade` for a blade-only
view. Both panels use `mesh.pyvista.surfaces` and `show_mesh_edges`; surfaces
have distinct colors and a legend. Hub/shroud represent flow-domain patches,
rather than solid turbine parts. Drag to rotate, scroll to zoom, or Shift+drag
to pan. Each panel has an independent camera. The saved physical blade count
and rotational pitch determine the row.

With `workflow.save_figures: true`, Matplotlib saves its PNG/SVG figures.
When PyVista is also enabled, the initial two-panel view is saved separately
as `pyvista_mesh.png` using an offscreen render before opening the interactive
window. Closing that window with its X button cannot cancel the saved image.
The PyVista image is a pixel-based screenshot, not a Matplotlib DPI export.
Set `workflow.show_pyvista: false` for runs without PyVista. Both
`workflow.show_gmsh` and `mesh.show_gui` must be true to open Gmsh.

For inspection of an existing case without remeshing, call
`meshing_source.view_mesh_pyvista()` with its output directory and the YAML's
`mesh.pyvista` mapping.

## Understanding mesh and viewer warnings

The Blossom recombination algorithm pairs triangles into quadrilaterals.
An odd triangle count prevents perfect pairing, so Gmsh can retain triangles;
those become prisms in the swept mesh. The warning alone does not establish
that cells are invalid, and the exporter supports both hexahedra and prisms.
It does mean that a fully quadrilateral/hexahedral mesh is not guaranteed.

Positive Jacobians and SICN are necessary checks, rather than a CFD-quality
certificate. For example, a minimum SICN of 0.0051 deserves inspection of the
worst cells: anisotropic inflation cells and distorted cells can both have low
condition-based quality. Assess cell angles, aspect ratios and the intended
solver requirements before using the mesh for CFD. Viewer-close/screenshot
warnings concern rendering, not geometry or mesh export; the separate
offscreen preview avoids their previous cause.

## Source organization and references

- `run_mesh_generation.py`: configuration, nozzle cache and workflow logging.
- `run_mesh_app.py`: YAML editor, background jobs and browser views;
  `app_assets/` contains its styling and client-side VTK scene assembly.
- `geometry_source.py`: thin input adapters, general hub/shroud definition,
  conformal mapping, blade fitting, display tessellation,
  centering and corresponding boundary locations.
- `meshing_source.py`: reference mesh, core deformation, volume assembly,
  checks, CGNS export and Python mesh views.

The construction follows Mueller and Verstraete, *CAD Integrated Multipoint
Adjoint-Based Optimization of a Turbocharger Radial Turbine* (2017),
[Section 2.2 / Figure 6](https://doi.org/10.3390/ijtpp2030014), and ParaBlade's
`projects/mesh_examples/3D`, `3D_v2`, and `3D_v3` implementations. This example
uses Gmsh recombination rather than the paper's elliptic structured grid solver.
API references: [Gmsh periodic meshes](https://www.gmsh.info/doc/texinfo/gmsh.html#gmsh_002fmodel_002fmesh_002fsetPeriodic),
[CGNS periodic transforms](https://cgns.org/cgns-archives/CGNS_docs_current/midlevel/connectivity.html),
[PyVista mesh construction](https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.html).
