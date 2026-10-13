# TurboMoC app user guide

Design a stator and rotor, inspect their meshes, and export a complete project.

This tutorial follows the supplied {download}`stator–rotor example <../../examples/apps/cases/stator_rotor.yaml>`. It introduces the essential controls as you use them, from the inlet state to a three-dimensional mesh. You can also follow the stator-only branch below.

TurboMoC calculates a stator profile using the **method of characteristics (MoC)** and a rotor profile using a **vortex-flow design method**. It then constructs blade passages and generates meshes for subsequent analysis. These are design calculations: the app does **not** run a CFD simulation or predict viscous losses, efficiency, or off-design performance.

## 1. Quick reference

For a first run, keep the example settings and follow this sequence:

1. Start the app and click **Load configuration**. Choose `examples/apps/cases/stator_rotor.yaml`.
2. Leave **Calculate → All rows** and **Configure display > Blade rows > All rows**.
3. In **Geometry generation**, click **Compute blade design**. Wait for both rows to finish.
4. Open **Mesh generation** and click **Generate 2D mesh**. Inspect the blade surfaces and trailing edges.
5. Click **Generate 3D mesh**. Inspect the volume mesh and **Mesh quality report**.
6. Click **Export project (.zip)**. Keep the archive together with your design records.

Each calculation depends on the preceding stage. After changing inputs, regenerate the affected stages before exporting. **Save configuration** separately preserves your settings.

Jump to [startup](#2-start-the-app-and-load-the-example), [stator inputs](#3-understand-the-stator-settings), [rotor inputs](#4-understand-the-rotor-and-row-connection), [2D meshing](#6-generate-and-inspect-the-2d-mesh), [3D meshing](#7-generate-and-explore-the-3d-mesh), [quality reports](#8-read-the-mesh-quality-report), or [export](#9-save-settings-and-export-the-project).

## 2. Start the app and load the example

Use Python **3.11–3.13** and [Poetry](getting_started.md). From the repository root, install the package and launch the app:

```bash
poetry install
poetry run python examples/apps/run_app.py
```

Run `poetry install` when setting up the environment. On subsequent launches, use only the `poetry run` command. Open [http://127.0.0.1:8052/](http://127.0.0.1:8052/) and keep the terminal running while using the app. Its initial case is an axial stator; load the tutorial file explicitly:

1. Click **Load configuration** at the top of the sidebar.
2. Select `examples/apps/cases/stator_rotor.yaml`.
3. Check **Design mode: Stator + Rotor**, **Machine type: Axial flow**, and **Working fluid: Cyclopentane**.

![App overview showing configuration loading, machine setup, calculation scope, and the two workspaces.](assets/app_user_guide/01-load-and-navigate.png)

*Interface orientation, shown here with a completed design. Start by loading the case and checking machine setup, then follow the three calculation buttons in order.*

The sidebar contains inputs; the main area contains results. Expand **Stator** or **Rotor** to inspect each row's settings. Drag the divider beside the sidebar for more room. Commit an edited numerical value by pressing Enter or moving focus out of the field.

Three selectors have different purposes:

| Control | What it controls |
| --- | --- |
| **Design mode** | Which rows belong to the active machine and therefore to the required export. |
| **Calculate** | Which active rows the next calculation processes. **All rows** runs them sequentially in machine order. |
| **Configure display > Blade rows** | Check the rows to display and edit together. This selection is shared across geometry and mesh plots; calculation and export scope remain independent. |

## 3. Understand the stator settings

In **Geometry generation**, expand **Stator**. A stator is the stationary blade row: it expands and turns the flow before it enters the rotor. The MoC calculation provides the supersonic expansion contour used to construct its blade profile.

### Inlet state and expansion

Under **Method of characteristics**, check the following:

| Setting | Example value | Meaning |
| --- | --- | --- |
| **Inlet stagnation property pair** | `PQ` | Defines the inlet state using pressure and vapor quality. |
| Pressure | `251300` Pa | Stagnation pressure: the pressure associated with bringing the flow to rest isentropically. |
| Vapor quality | `0.025` | Vapor mass fraction: 0.025 means 2.5% vapor by mass. |
| **p back** | `95000` Pa | Target exit static pressure for the nozzle design. |
| **solver** | `conventional` | The selected MoC construction. |
| **n** | `15` | Resolution of the initial characteristic discretization. |

Leave **use true sauer line**, the geometric construction parameters, and marching controls unchanged initially. They affect nozzle construction; they are not mesh settings.

**Watch the units:** pressure is in Pa and temperature in K. The MoC inputs **y t**, **rho t**, and **rho d** are in **metres**; `rho t` and `rho d` are geometric radii, not densities. Blade/channel lengths and mesh lengths below use **millimetres**. The nozzle contour is scaled when forming the final blade, so its computational throat size is not the final machine passage size.

### Channel dimensions and blade shape

Under **Meridional channel definition**, the example specifies a **hub radius** of `100` mm, **blade height** of `10` mm, and **hub chord** and **tip chord** of `20` mm. The hub is the inner endwall; the shroud lies at radius 110 mm. Equal hub and tip chords give equal axial extents at both ends of the blade.

Under **Blade geometry definition**, the main controls are:

| Setting | Example value | Meaning |
| --- | --- | --- |
| **metal angle in** | `0`° | Inlet camberline direction. |
| **metal angle out** | `65`° | Camberline direction at the throat; also the angle used for stator-to-rotor inlet transfer. |
| **r trailing** | `0.5` mm | Rounded trailing-edge radius in the source blade template, before scaling to the channel chord. |
| **inlet opening ratio** | `1.3` | Pitch divided by passage inlet opening; influences the convergent blade section. |

Keep the spline and point-count controls unchanged for this walkthrough.

<img src="assets/app_user_guide/02-stator-settings.png" alt="Stator settings showing the inlet property pair, expansion inputs, and channel dimensions." width="440">

*Thermodynamic inputs determine the expansion; channel and blade controls determine the physical blade row.*

The saved **Actual number of blades** starts at `31` for both rows. After each successful blade-design calculation, the app sets it to the rounded-up **Natural number of blades (midspan)**. This natural count comes from the designed pitch-to-chord ratio and machine dimensions. You may override the integer afterward to change passage spacing; another explicit blade-design calculation applies the suggested count again. Use **Displayed blades** in the plot controls when you only want to show more or fewer copies.

### Flow domain and extensions

Expand **Flow domain definition** to control the passage around the blade. These settings are available for both stator and rotor:

| Control | What to choose |
| --- | --- |
| **Domain shift mode** | Keep **auto** initially, or select **manual** to position the periodic boundaries yourself. |
| **Domain shift value [pitch]** | Pitchwise shift as a fraction of pitch; used only in manual mode. |
| **Number of points** | Sampling of the domain curves. Mesh resolution is controlled separately in **Mesh generation**. |
| **Inlet / Outlet → Extension type** | Choose **Smooth B-spline** or **Straight line** independently at each end. The defaults are a smooth inlet and a straight outlet. |
| **Inlet / Outlet → Extension length [chords]** | Distance to the inlet or outlet, expressed in reference chord lengths. |
| **Inlet / Outlet → Axial tangent / Blade tangent** | Dimensionless tangent-handle lengths that shape a smooth extension. They have no effect on a straight extension. |

A straight extension continues the blade-reference tangent to the inlet or outlet plane. Changing extension type can move where the periodic boundaries meet that plane; inspect the resulting domain before meshing. For a connected stator–rotor pair, the app derives the two internal extension lengths from the inter-row gap, as described below.

Saved YAML files use the same nested **Inlet** and **Outlet** groups. Older flat flow-domain settings are converted when loaded. If extension types were omitted, choose **Smooth B-spline** at both ends to reproduce the earlier all-smooth construction.

## 4. Understand the rotor and row connection

Expand **Rotor**. The rotor is the moving row, so its design uses velocity and Mach number relative to the blades.

Under **Rotor inlet state**, retain **Inlet conditions → Couple to stator exit**, **Rotational speed type → Actual speed [rpm]**, and speed `1000` rpm.

Read-only relative stagnation pressure, temperature, inlet Mach number, and inlet flow angle are initially pending. They become available after the current stator calculation converges. The app combines the stator exit velocity with the rotor blade speed at its inlet midspan radius: 105 mm for this example.

This coupling transfers the stator exit state ideally. It does not calculate flow or losses in the gap. Saved manual rotor pressure, temperature, Mach number, and angle are fallback inputs used only with **Specify manually**.

Under **Vortex blade design**, the main controls are:

| Setting | Example value | Meaning |
| --- | --- | --- |
| **M outlet** | `1.5` | Relative outlet Mach number. |
| **M lower** | `1.05` | Constant Mach number on the pressure-side arc. |
| **M upper** | `2.0` | Constant Mach number on the suction-side arc. |
| **num points** | `60` | Resolution of the transition arcs. |
| **le te ratio** | `25` | Target pitch divided by trailing-edge radius; larger values give a smaller relative edge radius. |

The outlet flow angle is derived from mass continuity. The vortex method requires a **supersonic relative inlet**, and the app also rejects negative relative inlet angles. If changing speed produces a warning, correct the speed or inlet conditions before calculating.

<img src="assets/app_user_guide/03-rotor-settings.png" alt="Rotor settings showing stator-exit coupling, rotational speed, and vortex design inputs." width="440">

*Read-only inlet quantities are calculated from the upstream stator; they are not additional values to type.*

Under **Axial interspace**, `axial interspace mm = 3` places the rotor leading-edge plane 3 mm after the farthest stator trailing edge. With the example's 20 mm chords, the stator occupies axial positions 0–20 mm and the rotor 23–43 mm.

The flow domains meet halfway across this gap, at **21.5 mm**. Under **Flow domain definition**, internal extension lengths are derived readouts. The outer stator inlet and rotor outlet lengths remain editable. Saved stator `outlet.extension_length` and rotor `inlet.extension_length` values apply when those rows are used alone; extension type and tangent controls remain available in each group.

Keep matching hub radii and blade heights for this axial pair. The interface construction requires a positive gap and consistent endwalls.

## 5. Compute and inspect the blade design

1. Set **Calculate → All rows** and **Configure display > Blade rows > All rows**.
2. Click **Compute blade design** once.
3. Follow the job status and progress bar. The stator runs first; its result supplies the coupled rotor inlet.
4. Expand **Terminal · live activity** for calculation details or to investigate an error.

Wait until the queue finishes before meshing. **Cancel job** stops the active calculation and remaining queued work; previously successful results are retained.

In **Blade-to-blade and meridional channel geometry**, inspect blade profiles and the passage between neighboring blades. The meridional view shows radius against axial position, making channel dimensions and the inter-row gap easy to check.

![Three-dimensional blade geometry with blue stator blades, orange rotor blades, and the Configure display panel.](assets/app_user_guide/04-stage-geometry.png)

*Three display copies of each row: stator in blue and rotor in orange. Endwalls are hidden here to expose the blades; use the meridional view to check their alignment.*

### Choose rows, blade copies, and surfaces

The **Configure display** panel sits to the right of each 3D plot. Expand **Blade rows** to check Stator, Rotor, or **All rows**. Expand **Displayed blades** to set each selected row's count with its own slider or number box. For an annular row, the range is one to that row's actual blade count; a linear cascade offers up to 25 display copies.

Surface controls apply to all checked rows. Select **Blade only** or **All surfaces**, or check individual boundaries. When selected rows have different settings, the controls show the first row's settings; an edit applies to every selected row. Blade-copy counts remain independent. **Reset display** restores each selected row's display defaults.

Default colors identify the surfaces:

| Surface | Default color |
| --- | --- |
| Stator blades | Blue |
| Rotor blades | Orange |
| Hub and shroud | Gray |
| Inlet and outlet | Red |
| Both periodic sides | Green |

Click a color swatch to customize it. Geometry and mesh views keep separate display settings. Changing copies or appearance does not change the physical blade count or exported mesh.

Both hub and shroud appear as smooth, continuous supports in the geometry view, including beneath the blade ends. Hide either surface to see the blades clearly. The mesh view shows the actual mesh boundaries.

### Orient the camera

Drag in a 3D plot to rotate; use the mouse wheel or drag with the right mouse button to zoom. The **orientation cube** at the bottom-left follows the camera. Click a face for an axis-aligned view, a beveled edge for a diagonal view, or a corner for an isometric view. Clicking **+X**, for example, looks from positive X toward the model. Surrounding arrows orbit in **30°** steps or roll the image in **15°** steps.

**HOME** restores the default orientation: slightly oblique with Z pointing right and Y up for an axial turbine, or looking along the Z shaft axis at the X–Y disk for a radial turbine. **ISO** selects an isometric view. In annular views, **Z follows the shaft**, **+Y follows positive-speed rotor motion at the reference passage**, and X completes the right-handed frame. Reversing the rotor speed does not flip the axes.

Camera shortcuts preserve zoom and the point you are looking at. With the cube focused, use the **arrow keys** to orbit, **Q/E** to roll, **Home** for the default orientation, or **I** for ISO. Geometry and mesh cameras are independent; changing the view does not change exported coordinates.

### Optional: stop at a standalone stator

To create a stator-only project, change **Design mode → Stator only**, then follow the mesh and export steps below. It is simplest to choose this immediately after loading the example, before calculating.

You can also change mode after inspecting the combined design. This restores the stator's standalone outlet extension, so regenerate its **2D** and **3D** meshes. Follow the enabled buttons if a prerequisite needs recalculating.

Choosing **Calculate → Stator** while retaining **Stator + Rotor** is useful for inspecting one row, but does not enable stator-only export: every active row must have a current mesh.

## 6. Generate and inspect the 2D mesh

Open **Mesh generation**, expand the desired row, and inspect **2D mesh · reference conformal mm**. An annular machine is first meshed in a flattened reference passage, then mapped through the physical channel to construct its 3D mesh. Reference-plane lengths are not universally equal to physical distances throughout the final annulus.

The blade-to-blade plot uses **θ [rad, clockwise]** and **−m′ [dimensionless]**; its axes are not physical millimetres. The meridional pane uses **z** and **r** in mm. It projects the section-boundary stations through the prescribed span grid and can appear before a 3D mesh has been generated. Solid outlines show blade extents; dashed outlines show the flow-domain extensions.

Both example rows use:

| Setting | Value | Purpose |
| --- | --- | --- |
| **blade size** | `0.25` mm | Target blade spacing; smaller values give finer surface resolution. |
| **domain size** | `1.0` mm | Target spacing away from the blade. |
| **curvature elements** | `50` | Curvature-based resolution; increasing it refines tightly curved regions. |
| **recombine** | **Yes** | Requests quadrilateral recombination. |
| **recombine minimum quality** | `0.1` | Minimum quality for recombined quadrilaterals; poor candidates remain triangles. |
| **inflation layers → enabled** | **Yes** | Adds thin cells next to the blade wall. |
| **first height** | `0.03` mm | Requested first wall-normal layer height. |
| **growth ratio** | `1.2` | Successive nominal layers grow by a factor of 1.2. |
| **thickness** | `1.0` mm | Allowance for the inflation stack. |

Leave algorithm and smoothing controls unchanged initially. With **Calculate → All rows**, click **Generate 2D mesh** and wait for both rows.

![The section mesh and meridional span preview in the Mesh generation workspace.](assets/app_user_guide/05-section-mesh.png)

*Inspect the whole passage before zooming into its most demanding regions.*

**Recombine minimum quality** accepts values from 0 to 1. Increasing it can leave more triangles where a quadrilateral would have poor shape. This is Gmsh's recombination quality, not a skewness limit; inspect the final mesh as well. The default is `0.1`, including when omitted from older YAML files. Use `0.01` if you need the former Gmsh default.

### Inspect the inflation layers

Use **Configure display > Blade rows** to check only **Stator** or **Rotor** and isolate that row. Hover over the 2D plot to reveal its toolbar, then use box zoom to select a relatively straight blade surface. Double-click the plot to reset the view. Look for thin wall-adjacent cells, increasing spacing through the inflation region, and a transition toward the coarser passage mesh.

![Close-up of stator wall-adjacent inflation layers and their transition into the core mesh.](assets/app_user_guide/06-inflation-closeup.png)

*First height controls the smallest wall-normal spacing; growth ratio and thickness control how it expands away from the blade.*

Thickness is an allowance, not a promise of an identical stack everywhere: curvature and nearby boundaries can shorten or merge layers. First height is a meshing input, not a verified CFD wall-resolution target such as a prescribed wall y-plus.

### Inspect the trailing edge

Zoom into the rounded trailing edge. Look for adequate points around the curve, orderly inflation cells around the turn, and a smooth transition into the wake region.

![Close-up of stator trailing-edge curvature resolution and surrounding mesh cells.](assets/app_user_guide/07-trailing-edge-closeup.png)

*The close-up reveals how the rounded geometry is represented, which can be difficult to judge from the whole passage.*

Distinguish shape from resolution: **r trailing** or the rotor's **le te ratio** changes the blade geometry; **curvature elements** and **blade size** change its mesh representation. After a sizing edit, regenerate the 2D mesh before generating the 3D mesh.

## 7. Generate and explore the 3D mesh

Under **Span discretization**, retain **span layers = 25** and **span stretch = 1.5**. Span layers divide the distance from hub to shroud. Positive stretch clusters stations toward both endwalls; zero gives uniform spacing. This is separate from blade-wall inflation.

Click **Generate 3D mesh**. The app maps and sweeps the section mesh through the channel, checks the resulting cells, and prepares the report.

![Three-dimensional stator and rotor passage meshes with mesh edges visible.](assets/app_user_guide/08-volume-mesh.png)

*The 3D viewer shows boundary surfaces of the volume mesh. Blade and hub surfaces expose the spanwise discretization and the relationship between row passages.*

In **Configure display**, select **Blade rows** and adjust each row's **Displayed blades** count. One copy makes local details easier to inspect; several copies show the annular arrangement. Counts apply in all three **Display modes**:

| Display mode | Use it to |
| --- | --- |
| **Boundary surfaces** | Inspect the passage shape, individual boundaries, and mesh edges. |
| **Mesh quality filter** | Locate the worst cells or cells beyond a quality threshold. |
| **Spanwise layer** | Inspect one complete cell layer at a time, from hub to shroud. |

### Boundary surfaces

Turn **Mesh edges** on and show **Blade** and **Hub** first. Hide **Shroud** or other outer surfaces if they obscure the interior. Add **Inlet**, **Outlet**, and periodic boundaries to understand the passage. Surface checkboxes and color pickers follow the same controls as the geometry view.

### Locate poor-quality cells

Choose **Mesh quality filter**, then **Metric → Skewness**, **Orthogonality**, or **Edge aspect ratio**. Under **Select cells**, choose:

- **Worst N:** enter the **Number of cells** to show. The app selects that many original cells per row, up to the number available.
- **Threshold:** show cells at or above the chosen skewness or aspect ratio, or at or below the chosen orthogonality. The field label shows the comparison direction.

The selected cells are colored green for better quality through yellow to red for worse quality. A single colorbar inside **Configure display** shows the scale. Skewness and orthogonality use their 0–1 ranges, with opposite good-to-bad directions. Edge aspect ratio uses a logarithmic scale from 1 to a rounded upper value covering the selected meshes.

Expand **Transparent surfaces** to choose the faint gray boundaries that help locate the cells within the passage. All surfaces are checked initially; uncheck any that obscure the region of interest. This selection is independent of **Boundary surfaces**. Displayed copies repeat the same selected cells; they do not add cells to the quality statistics.

### Inspect the interior one layer at a time

Choose **Spanwise layer** and a metric. Move the slider, or use **Previous** and **Next**, to step from hub to shroud. The caption shows the layer number and its midpoint as a percentage of span, for example **Layer 2 of 25 · 3.4% span**.

This mode shows the complete cell layer, without the quality filter, so you can follow changes through the interior. Colors describe the original 3D cells. With several rows selected, the app aligns their layers by span position even if their layer counts differ. The slider and caption follow the first selected row. **Transparent surfaces** remains available for context.

Check smooth spanwise progression and the intended interface location. Coincident interface positions do not imply matching nodes or a connected fluid solution.

Changing only **span layers** or **span stretch** requires a new 3D mesh. Display colors, surfaces, copies, and mesh edge widths do not require remeshing.

## 8. Read the mesh quality report

Expand **Mesh quality report**. Set **Report → Total** for a stage overview, then select **Stator** or **Rotor** for an individual row.

![Mesh quality report with row status, cell counts, quality metrics, and histograms.](assets/app_user_guide/09-mesh-quality.png)

*Start with availability and current status, then inspect the weakest cells and metric distributions.*

Read the report in this order:

1. **Status:** every active row should be **Current**. Reports can still show older meshes; heed **Outdated** or **Unavailable** labels.
2. **Volume cells / Volume nodes:** counts describe one generated passage per row. Displayed blade copies do not multiply statistics. Combined node counts sum rows without merging interface nodes.
3. **Minimum signed inverse condition number (SICN):** values near zero indicate poor or degenerate shape; negative values indicate inversion. Thin inflation cells can have low positive SICN.
4. **Distributions and extremes:** use the interpretation below to identify regions needing closer inspection.

| Metric | Interpretation |
| --- | --- |
| **Skewness** | Maximum equiangular face skewness per cell; 0 is ideal and 1 is degenerate. Inspect the maximum and high-value tail. |
| **Orthogonality** | Face-normal/centroid-connector alignment; 1 is ideal. Inspect the minimum and low-value tail. |
| **Edge aspect ratio** | Longest cell edge divided by shortest; 1 means equal edge lengths. Thin inflation cells naturally have larger ratios. |

Tables include **Min**, **Mean**, **Max**, and **P95**. P95 is the 95th percentile: it describes the high-value tail even for orthogonality, where low values are the concern. Histograms show whether an extreme affects a few cells or much of the mesh.

The total report counts cells with **skewness > 0.8** and **orthogonality < 0.2**. These are inspection flags, not universal acceptance criteria. Individual reports include **Worst cells · original Gmsh IDs** for locating problematic elements. Hover over metric labels for definitions; the app's definitions may differ from a CFD solver's metrics.

Use **Mesh quality filter** to locate cells highlighted by the report, then inspect nearby layers before changing mesh settings. Successful mesh generation does not establish suitability for a particular CFD calculation. Cell counts and quality values depend on your current settings; the figures illustrate how to read the results rather than target values to reproduce.

## 9. Save settings and export the project

Click **Save configuration** to download the current YAML, including adjusted physical blade counts and the nested inlet/outlet settings. Displayed copies, colors, camera positions, and inspection selections are not saved. Loading the YAML restores inputs; recompute the design and meshes when reopening it.

For this axial case, **Export project (.zip)** becomes available when every active row has a completed, current **3D mesh** and no job is running. A 2D annular reference mesh alone is not the final physical CFD deliverable. Neither **Calculate** nor the **Blade rows** display selection narrows export scope.

![Project export control after the active row meshes are ready.](assets/app_user_guide/10-export.png)

*Export after reviewing current mesh status. Keep settings and reports with the meshes.*

The archive contains:

| File | Purpose |
| --- | --- |
| `project.yaml` | Final configuration for reproducing the design. |
| `stage.cgns` | Complete active machine, with a separate fluid zone for each row. |
| `meshes/stator_1/`, `meshes/rotor_1/` | Individual row CGNS exports. |
| `mesh_report.json` | Summaries, quality data, interfaces, and effective coupled rotor inputs. |
| `activity.log` | Logs of the included meshes and preceding calculations. |
| `README.txt` | Contents, coordinates, boundaries, and downstream-use notes. |

The example's **CGNS mesh output → length scale = 0.001** converts millimetres to metres. Multirow meshes include assembly placement. Each row contributes **one periodic passage**, regardless of displayed copies.

In a CFD solver, check units and assign inlet/outlet conditions, walls, periodic pairing, rotor motion, and the stator–rotor interface. The archive preserves separate interface sides; it does not select a mixing-plane, sliding-mesh, or other flow-interface model. Consult its `README.txt` for import details. True 2D CGNS export is available for linear cascades; axial and radial exports use physical 3D meshes.

## 10. Make changes and recover from common problems

Change one group of settings at a time, using the example as your baseline.

| Situation | Next step |
| --- | --- |
| Calculation button disabled | Read validation messages and hover over the **Calculate** control for its prerequisite note. Correct invalid inputs or complete the preceding stage. |
| Coupled rotor inlet pending | Complete the current stator design before running the rotor alone, or use **All rows** for blade design. |
| Rotor inlet warning | Adjust speed or inlet conditions until the supported relative inlet is recovered. |
| Interface or clearance error | Check positive gap, matching endwalls, and space for the inflation thickness. |
| Outdated mesh | Regenerate the affected chain. A new explicit blade solve invalidates downstream meshes even with unchanged inputs. |
| Export unavailable | Check all active rows, not only the visible/calculated row, have current 3D meshes. |
| Calculation fails | Read **Terminal · live activity**, correct the reported input, and rerun the affected stage. |
| No cells visible in the quality filter | Check **Blade rows**, then increase **Number of cells** or relax the threshold. Switch to **Boundary surfaces** to see the complete passage. |
| Interior hidden by gray surfaces | Open **Transparent surfaces** and uncheck the boundaries blocking your view. |
| “Application source changed” | Restart the app after a code update, reload your saved configuration, and recompute. |

For more background, see [nozzle design](theory/nozzle_design.md), [stator design](theory/stator_design.md), [notation and units](theory/notation.md), and [rotor design by the vortex-flow method](theory/rotor_design.md).

---

*Updated for the app on 2026-10-13 using `examples/apps/cases/stator_rotor.yaml`. The screenshots illustrate the original tutorial run; some control layouts predate the display updates described here. Save the exported configuration to record the settings used for your own run.*
