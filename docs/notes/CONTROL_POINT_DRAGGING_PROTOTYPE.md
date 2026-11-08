# Local nurbspy control-point editor prototype

The stator control-point pane embeds the **local sibling nurbspy viewer**. The centre blade has draggable handles; two adjacent blades show the current passage without extra control points. Copies update live at the configured production pitch, expressed in the native editing plane. For annular rows this respects the configured blade count rather than assuming the original MoC pitch. The viewer stays orthographic in x–y, with pan, zoom, and **Fit passage**.

A single pinned pane contains point selection, X/Y coordinates, displacement readouts, and **Reset point**. Its view options are limited to Grid, Control polygon, Original outline, and Adjacent blades. Scientific colouring, Frenet–Serret frames, weights, z editing, and 3D controls are omitted. The first and last two points remain fixed and cannot join a selection. Status updates use a fixed-height footer, and editing or changing view options preserves the camera and drawing area. The native viewer's redundant frame and heading are removed from this mode.

A small TurboMoC adapter imports nurbspy's renderer, interaction controls, native geometry model, and binary evaluation path. The initial prototype made no nurbspy changes; the subsequent configurable-preview addition lives in nurbspy's shared sampling layer. The adapter currently depends on the local viewer API, including some implementation methods, so this is a local prototype rather than a pinned distribution dependency. The sibling checkout is imported directly; `pyproject.toml` does not declare a nurbspy dependency.

The 8 November 2026 adapter revision uses the nurbspy selection/history and automatic-sampling API introduced through `51ef25d`, following the original prototype's `8c3e963`. It reuses native multi-selection, edit-history transactions, automatic display sampling, and HTTP packet decoding. TurboMoC supplies only its editing restrictions, passage context, and applied displacement inputs.

## Run

From this repository in Git Bash:

```bash
PYTHONPATH=src poetry run python -m turbo_moc.app --port 8052
```

The sibling `../nurbspy/src` is selected automatically. If the checkout is elsewhere, set `TURBO_MOC_NURBSPY` to its repository root or `src` directory before starting. Use an environment with the app and local nurbspy's numerical dependencies installed; the tested environment is this repository's Python 3.13 `.venv`.

Open the app, generate the stator geometry, then choose **Modify stator control points**. Drag a point or enter precise x/y coordinates in the pinned pane. Coordinates and displacement readouts use millimetres. **Apply** commits the draft to the project; **Save configuration** downloads its YAML. **Cancel** discards the draft. **Reset point** and **Reset all** return the selected point or entire draft to the unedited MoC-derived baseline, even when opening an already edited case; Apply is needed to commit a reset. Displayed coordinates use fewer digits for readability while the stored model retains full precision; confirming an untouched field does not round the geometry.

**Shift-click** adds or removes a point from the selection. **Box select**, or Shift-drag over the drawing, selects several editable points. Drag a selected point to move the group together. The X/Y fields, displacement readouts, and **Reset point** refer to the active point shown in the selector. Click empty space to clear the selection; this does not change the geometry.

**Undo** and **Redo** use nurbspy's session-local history. Each completed drag, numeric edit, Reset point, or Reset all is one step; camera changes and selection changes do not enter history. Keyboard shortcuts are Ctrl/Command+Z and Ctrl/Command+Shift+Z when focus is in the editor outside an input. Escape cancels an in-progress drag. History is discarded when the editing session closes and is never saved in YAML.

An invalid full edit is rejected and the previous shape is restored without adding a history step. The footer reports the rejection. Apply is available once a valid full evaluation is restored; a preview alone cannot be applied.

## Saved geometry inputs

Free x–y movement uses a new, explicitly named field, leaving legacy scalar normal offsets readable:

```yaml
# Within rows[*].geometry.blade_geometry:
control_point_displacements:
  5: [0.006, -0.003]  # [dx, dy] / baseline chord scale
```

The reconstruction is $P_i=P_i^0+c_0[\delta x_i,\delta y_i]$, where $P_i^0$ and $c_0$ come from the unedited stator fit. The scale is the existing `blade_chord` definition: the diagonal of the sampled curve's axis-aligned bounding box, preserving TurboMoC's conventions. These are total baseline-relative displacements, never increments accumulated across drags or reloads. Omitted entries mean exactly zero; Reset saves an empty displacement mapping.

Existing `control_point_offsets` files continue to generate the same geometry. Opening and applying an unchanged legacy case preserves its scalar values. A deliberate free edit converts the case's applied modifications to XY vectors. Nonempty scalar and vector fields cannot be combined. Unchanged vector entries retain their stored precision; float32 drawing buffers never enter YAML.

Apply reconstructs the current baseline on the server and validates point locks, planarity, unit weights, knots/degree, and self-intersection. A changed MoC result or geometry identity rejects an obsolete editing session. Mesh/CAD/export identities continue through the existing project workflow. The control-point CSV includes appended `dx`/`dy` columns alongside the legacy scalar-offset column.

## Prototype limits

The editor supports free x–y movement of individual points and groups. Normal constraints are deferred. Protected endpoints, planarity, and fixed weights remain enforced on the server as well as in the editor.

Display sampling uses `quality="fine", preview_quality="full"`. Dragging reuses the fine display grid without switching to a coarse outline; a completed edit can refine the drawing and update its topology. This is a configuration of nurbspy's shared sampler, whose default remains `preview_quality="coarse"` for other viewers/editors. Adjacent blades share the central blade's drawing buffers at every quality. Drawing resolution does not change the float64 control-point model, saved displacements, or production geometry sampling.

Exact reproduction assumes unchanged generator inputs and implementation; changing CP count or parametrization still requires an explicit reset/rebase decision. Keep the local nurbspy revision/working tree with archived prototype cases until its viewer API is stabilized and pinned. Do not replace TurboMoC's baseline-relative YAML inputs with a nurbspy figure snapshot; the two formats serve different purposes.

Focused regression tests cover both parametrizations, YAML reconstruction, no-op/zero behavior, legacy compatibility, protected points, finite vectors, self-intersection, exports, and stale Apply requests. Browser verification uses a synthetic MoC wall to exercise the real app without rerunning the thermodynamic solver.

Verification on 8 November 2026: 64 focused editor, geometry/export, and surrounding-app Python regressions passed. The browser check in `tests/browser/check_control_point_editor.py` passed for normal and mirrored 30-point blades at 1600×1000 and 1200×900. It checks production passage pitch, shared live neighbor buffers, unchanged camera and canvas bounds, cursor tracking within 0.3 pixels, no-op numeric precision, group selection/dragging, protected endpoints, undo/redo for resets and group edits, invalid-edit rollback, compact layout, Apply, YAML download/upload, reconstruction, and Cancel. The normal-blade run additionally verifies Escape after an accepted drag preview. Both runs completed without browser errors or unexpected HTTP failures. Regenerated control points match the applied model within $10^{-13}$ mm. Screenshots, saved YAML, and verification records are in the ignored `output/control_point_ux/` and `output/control_point_ux_mirrored/` directories.
