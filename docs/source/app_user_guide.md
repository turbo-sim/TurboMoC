# TurboMoC app user guide

Design a stator and rotor, inspect their meshes, and export a complete project.

This tutorial follows the supplied {download}`stator–rotor example <../../examples/apps/cases/stator_rotor.yaml>`. It introduces the essential controls as you use them, from the inlet state to a three-dimensional mesh. You can also follow the stator-only branch below.

TurboMoC calculates a stator profile using the **method of characteristics (MoC)** and a rotor profile using a **vortex-flow design method**. It then constructs blade passages and generates meshes for subsequent analysis. These are design calculations: the app does **not** run a CFD simulation or predict viscous losses, efficiency, or off-design performance.

## 1. Quick reference

For a first run, keep the example settings and follow this sequence:

1. Start the app and click **Load configuration**. Choose `examples/apps/cases/stator_rotor.yaml`.
2. Leave **Calculate → All rows** and **Show → All rows**.
3. In **Geometry generation**, click **Compute blade design**. Wait for both rows to finish.
4. Open **Mesh generation** and click **Generate 2D mesh**. Inspect the blade surfaces and trailing edges.
5. Click **Generate 3D mesh**. Inspect the volume mesh and **Mesh quality report**.
6. Click **Export project (.zip)**. Keep the archive together with your design records.

Each calculation depends on the preceding stage. After changing inputs, regenerate the affected stages before exporting. **Save configuration** separately preserves your settings.

Jump to [startup](#2-start-the-app-and-load-the-example), [stator inputs](#3-understand-the-stator-settings), [rotor inputs](#4-understand-the-rotor-and-row-connection), [2D meshing](#6-generate-and-inspect-the-2d-mesh), [3D meshing](#7-generate-and-explore-the-3d-mesh), [quality reports](#8-read-the-mesh-quality-report), or [export](#9-save-settings-and-export-the-project).

## 2. Start the app and load the example

Use Python **3.11–3.13**. From the repository root, activate your existing environment; for a Windows PowerShell environment named `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/apps/run_app.py
```

The installation command is needed when setting up the environment. The launcher opens the app at [http://127.0.0.1:8052/](http://127.0.0.1:8052/). Keep its terminal running while using the app. Its initial case is an axial stator; load the tutorial file explicitly:

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
| **Show** | Which rows appear in the views. It does not change calculation or export scope. |

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

The saved **Actual number of blades** starts at `31` for both rows. After each successful blade-design calculation, the app sets it to the rounded-up **Natural number of blades (midspan)**. This natural count comes from the designed pitch-to-chord ratio and machine dimensions. You may override the integer afterward to change passage spacing; another explicit blade-design calculation applies the suggested count again. In the illustrated run, the counts became **28 stator blades** and **85 rotor blades**.

## 4. Understand the rotor and row connection

Expand **Rotor**. The rotor is the moving row, so its design uses velocity and Mach number relative to the blades.

Under **Rotor inlet state**, retain **Inlet conditions → Couple to stator exit**, **Rotational speed type → Actual speed [rpm]**, and speed `1000` rpm.

Read-only relative stagnation pressure, temperature, inlet Mach number, and inlet flow angle are initially pending. They become available after the current stator calculation converges. The app combines the stator exit velocity with the rotor blade speed at its inlet midspan radius: 105 mm for this example.

In the illustrated run, these readouts were approximately **179302 Pa**, **340.94 K**, **Mach 1.253**, and **61.76°**. Use them to recognize the coupled state; they are calculated outputs, not values to enter manually.

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

The flow domains meet halfway across this gap, at **21.5 mm**. Under **Flow domain definition**, internal extensions are derived readouts. The outer stator inlet and rotor outlet extensions remain editable. Saved stator `outlet_chords` and rotor `inlet_chords` values apply when those rows are used alone.

Keep matching hub radii and blade heights for this axial pair. The interface construction requires a positive gap and consistent endwalls.

## 5. Compute and inspect the blade design

1. Set **Calculate → All rows** and **Show → All rows**.
2. Click **Compute blade design** once.
3. Follow the job status and progress bar. The stator runs first; its result supplies the coupled rotor inlet.
4. Expand **Terminal · live activity** for calculation details or to investigate an error.

Wait until the queue finishes before meshing. **Cancel job** stops the active calculation and remaining queued work; previously successful results are retained.

In **Blade-to-blade and meridional channel geometry**, inspect blade profiles and the passage between neighboring blades. The meridional view shows radius against axial position, making channel dimensions and the inter-row gap easy to check.

![Three-dimensional blade geometry with blue stator blades, orange rotor blades, and the Configure display panel.](assets/app_user_guide/04-stage-geometry.png)

*Three display copies of each row: stator in blue and rotor in orange. Endwalls are hidden here to expose the blades; use the meridional view to check their alignment.*

In **Three dimensional geometry**, drag to rotate; use the mouse wheel or drag with the right mouse button to zoom. **Configure display** lets you select a **Blade row**, change **Displayed blades**, show **Blade only** or **All surfaces**, and toggle individual boundaries. **Reset display** restores display defaults. These copies illustrate the cascade; they do not change physical blade count or exported mesh.

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
| **inflation layers → enabled** | **Yes** | Adds thin cells next to the blade wall. |
| **first height** | `0.03` mm | Requested first wall-normal layer height. |
| **growth ratio** | `1.2` | Successive nominal layers grow by a factor of 1.2. |
| **thickness** | `1.0` mm | Allowance for the inflation stack. |

Leave algorithm and smoothing controls unchanged initially. With **Calculate → All rows**, click **Generate 2D mesh** and wait for both rows.

![The section mesh and meridional span preview in the Mesh generation workspace.](assets/app_user_guide/05-section-mesh.png)

*Inspect the whole passage before zooming into its most demanding regions.*

### Inspect the inflation layers

Use **Show → Stator** or **Rotor** to isolate a row. Hover over the 2D plot to reveal its toolbar, then use box zoom to select a relatively straight blade surface. Double-click the plot to reset the view. Look for thin wall-adjacent cells, increasing spacing through the inflation region, and a transition toward the coarser passage mesh.

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

Use **Configure display** to select each row and turn **Mesh edges** on. Show **Blade** and **Hub** first; hide **Shroud** or other outer surfaces when they obscure the interior. Add **Inlet**, **Outlet**, and periodic boundaries to understand the passage. Choose one displayed blade for detail or several to see the annular arrangement.

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

For reference, the illustrated run produced:

| Row | Volume cells | Volume nodes | Minimum SICN | Maximum skewness | Minimum orthogonality |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stator | 212,475 | 222,768 | 0.03127 | 0.9251 | 0.7398 |
| Rotor | 132,950 | 147,316 | 0.1013 | 0.9048 | 0.3009 |
| Combined | 345,425 | 370,084 | 0.03127 | 0.9251 | 0.3009 |

There were **107 cells with skewness above 0.8** and **none with orthogonality below 0.2**. The high-skewness cells warrant inspection even though mesh generation completed. These example results depend on settings and implementation; they do not establish suitability for a particular CFD calculation.

## 9. Save settings and export the project

Click **Save configuration** to download the current YAML, including adjusted blade counts. Loading it later restores inputs; it does not restore the browser session or completed results. Recompute the design and meshes when reopening it.

For this axial case, **Export project (.zip)** becomes available when every active row has a completed, current **3D mesh** and no job is running. A 2D annular reference mesh alone is not the final physical CFD deliverable. Neither **Calculate** nor **Show** narrows export scope.

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
| “Application source changed” | Restart the app after a code update, reload your saved configuration, and recompute. Jobs check that the app and worker use the same source version. |

For more background, see [nozzle design](theory/nozzle_design.md), [stator design](theory/stator_design.md), [notation and units](theory/notation.md), and [rotor design by the vortex-flow method](theory/rotor_design.md).

---

*Prepared on 2026-10-13 using `examples/apps/cases/stator_rotor.yaml`. Screenshots and example results were captured from a source snapshot dated 2026-10-13; framing and visible surfaces may differ between figures. Computed blade counts and mesh results depend on solver output and implementation. Save the exported configuration to record the settings used for your run.*
