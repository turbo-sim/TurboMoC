# moc

Method of Characteristics solvers for nozzle/stator design, covering
single-phase, saturated-two-phase, and subcooled-liquid-flashing inlet
conditions on the same real-gas (CoolProp-backed, via `jaxprop`) thermo core.

Two solver variants, sharing one core (`moc.core.classes`):

- `moc.ConventionalSolver` (`moc.solvers.conventional.MOCSolver`) --
  rounded-arc kernel. Supports the flashing flat post-jump throat front as
  well as ordinary single-phase and saturated-two-phase inlets.
- `moc.MOCSolverMLN` (`moc.solvers.mln.MOCSolverMLN`) -- minimum-length
  (sharp-corner) nozzle, centred expansion fan from the throat corner.

Both take a config dict and expose the same `build_paper_data()` /
`export_full_paper_data()` schema, so results from the two are directly
comparable.

## Install

Editable install for development:

```bash
pip install -e .
```

## Interactive app

```bash
conda activate moc_env
python examples/run_app_local.py      # or: moc-app, or: python -m moc.app
```

then open <http://127.0.0.1:8050/>. Two top-level tabs:

- **Phase 1: Nozzle design** -- fluid dropdown, inlet condition, target back
  pressure, solver choice (conventional/MLN), geometry, a Compute button and
  progress bar, and five plot tabs (contour, characteristics mesh, axis
  properties, wall properties, T-s diagram). `app.py` is structured after
  turbodash's `app.py`: one compute callback calling `design_nozzle` once,
  storing the result in a `dcc.Store`, and a second callback fanning it out
  into the plots. The one departure from turbodash's pattern: the compute
  callback runs `background=True` (via `DiskcacheManager`) with a real
  progress bar, since a solve here is 5-30s+ (see "Progress reporting"
  below) -- Compute-button-then-wait, not live recompute-on-slider-drag.
- **Phase 2: Stator blade** -- takes Phase 1's `wall_final` (via its
  `dcc.Store`) as the divergent-section suction side and builds a closed
  stator blade profile (`moc.geometry.parametrize_stator_blade`, see below).
  Metal angles, trailing-edge radius, leading-edge position, control-point
  count are plain form inputs; Compute is synchronous (no progress bar --
  this is closed-form geometry + one ODE integration + a B-spline fit, not
  an iterative solve). Includes a STEP export panel (2D face or extruded 3D
  solid) alongside the same "Download this plot" pattern as Phase 1.

**Plots are displayed with `moc.plotly`** in an interactive `dcc.Graph` --
zoom, pan, hover, and a native download-as-PNG toolbar button, same as
turbodash's own app. `moc.plotly`'s figures are styled (`plotting_plotly.
_apply_journal_style`) to read close to `moc.mpl`'s boxed-axes/gridline
look rather than Plotly's own default template, so switching between the
two doesn't mean switching visual language -- same colors either way, only
interactive-vs-static differs. Each tab also has its own "Download this
plot" button + format dropdown (PNG/SVG/PDF), separate from Plotly's own
toolbar button (PNG only): it re-renders the SAME figure with `moc.mpl` on
click, so an SVG/PDF download is a real vector file, not a rasterized PNG
relabeled -- display and file-export are genuinely two different renders
of the same data.

For a deployed (not local dev) server, `moc.app.server` is the underlying
Flask/WSGI app (`gunicorn moc.app:server`, matching turbodash's `Procfile`
convention) -- not set up here yet, local dev only for now.

**Comparing two designs:** run one, click "Pin as reference", change some
inputs, Compute again -- every plot overlays both (the pinned one dashed
and lighter). "Clear reference" drops it. This is `moc.plotly`'s
`reference=<data dict>` parameter (also in `moc.mpl`, so it works outside
the app too), not app-specific state -- see the next section.

## Quick start: the `design_nozzle` API

`moc.design_nozzle(...)` is the recommended entry point -- it takes plain
arguments (fluid name, operating point, geometry, solver choice) and
returns one flat, JSON-safe result dict. No `MOCSolver`/`FluidManager`
objects to manage. This is deliberately styled after
[turbodash](https://github.com/turbo-sim)'s `core.py` pattern: a UI layer
(Dash, a REST handler, a notebook) calls this one function per design
evaluation and fans the returned dict out into plots/tables -- the function
itself has no UI/IO concerns.

```python
from moc import design_nozzle

data = design_nozzle(
    fluid_name="nitrogen",
    P0=20e5, T0=113.0,       # subcooled-liquid (flashing) inlet
    p_back=2e5,               # design target: expand down to this back pressure
    solver="conventional",    # or "mln" for the minimum-length variant
    y_t=0.01, rho_t=0.2, rho_d=0.1,
)

print(data["converged"], data["exit_M"], data["runtime_s"])
wall = data["wall_final"]      # {"x": [...], "y": [...], "p": [...], "M": [...], ...}
axis = data["axis"]            # centerline property distributions
```

Inlet condition selection (same for both solvers):

- `Q0` omitted (default `nan`) + `T0` -- subcooled liquid (flashing) or
  single-phase superheated vapor, both via (P0, T0).
- `Q0` in `[0, 1]` -- saturated two-phase inlet, via (P0, Q0); `T0` is
  derived automatically.

`solver="mln"` requires `p_back` (used to compute the corner fan's target
turning angle). Invalid requests (missing target, `solver` typo, `Q0` out
of range, or a solve that fails outright) raise `moc.NozzleDesignError`
with a clear message rather than an internal traceback.

`moc.list_supported_fluids()` returns CoolProp's fluid list, for
populating a fluid-selection dropdown/control.

## Plotting

Two plotting modules, same four functions in each, both consuming the
`data` dict from `design_nozzle()` / `build_paper_data()` directly -- no
solver objects, mirroring turbodash's `plotting_mpl.py`/`plotting_plotly.py`
split (`td.mpl`/`td.plotly`):

- `moc.mpl` -- matplotlib, for static/paper-quality output.
- `moc.plotly` -- Plotly `go.Figure`, for the Dash app (Dash consumes these
  natively as a callback's `Output(..., "figure")`).

```python
import moc

data = moc.design_nozzle(fluid_name="nitrogen", P0=20e5, T0=113.0, p_back=2e5)

fig, ax = moc.mpl.plot_nozzle_contour(data)       # the wall shape
fig, ax = moc.mpl.plot_characteristics_mesh(data) # C+/C- characteristic fan
fig, axes = moc.mpl.plot_axis_properties(data)    # P (left) + M (right) vs x, centerline
fig, axes = moc.mpl.plot_wall_properties(data)    # same, along the wall

fig = moc.plotly.plot_nozzle_contour(data)        # same 4 functions, go.Figure
```

`plot_nozzle_contour`/`plot_characteristics_mesh` default `mirror=True`/
`False` respectively (the solver only computes one half of the axisymmetric
domain) -- pass `mirror=True` on the mesh plot too for a full both-halves
view. See `examples/plot_design.py` for a complete runnable example
(saves all 8 combinations to `examples/output/plots/`).

**Comparing two designs:** every one of the four functions also accepts
`reference=<second data dict>`, overlaid dashed/lighter alongside the
current one (same color scheme, so pressure stays orange/Mach stays green
in both, only current-vs-reference is distinguished by line style):

```python
data_A = moc.design_nozzle(fluid_name="nitrogen", P0=20e5, T0=113.0, p_back=2e5)
data_B = moc.design_nozzle(fluid_name="nitrogen", P0=20e5, T0=108.0, p_back=2e5)

fig, ax = moc.mpl.plot_axis_properties(data_B, reference=data_A)
```

`plot_characteristics_mesh` only overlays the reference's wall by default
(its full characteristic mesh gets visually busy fast) -- pass
`show_reference_mesh=True` for the full overlay. This is what backs the
app's "Pin as reference" button (see "Interactive app" above).

**`plot_ts_diagram(data, ...)`** -- the saturation dome (from `data['fluid_name']`,
cached per fluid since it's a real ~2s fluid-property computation the first
time) plus three highlighted points of the expansion: **Total** (stagnation,
`data['s0']`/`['T0_stagnation']`), **MoC start** (the first solved axis
point -- approximately the sonic/critical state Stage 1 seeds from), and
**Outlet** (the last axis point). MoC start and Outlet are connected with a
straight line; Total is not connected to either (it sits upstream of where
the solve actually starts, at ~zero velocity). Also supports `reference=`.

## Stator blade parametrization (`moc.geometry`)

`moc.geometry.parametrize_stator_blade(...)` builds a closed stator blade
profile from a MOC nozzle wall contour, following the same "plain args in,
flat JSON-safe dict out" pattern as `design_nozzle`. It takes the
divergent-section wall as the blade's suction side, integrates a
linear-metal-angle-blend camberline for the convergent section, builds the
pressure side from a mirrored/translated copy of the wall plus a cubic
Hermite spline bridging the leading edge, and fits a fixed-endpoint cubic
B-spline through the closed contour:

```python
import moc
from moc.geometry import parametrize_stator_blade

data = moc.design_nozzle(fluid_name="nitrogen", P0=20e5, T0=113.0, p_back=2e5)
wall = data["wall_final"]  # meters, SI -- design_nozzle's own units

blade = parametrize_stator_blade(
    wall["x"], wall["y"],
    metal_angle_in=0.0, metal_angle_out=70.0, r_trailing=0.5, n_cp=30,
)
fig = moc.plotly.plot_blade(blade)  # or moc.mpl.plot_blade(blade)
```

Note the unit boundary: `wall["x"]`/`["y"]` are in **meters** (matching
`design_nozzle`), but everything inside `blade` -- `suction`, `pressure`,
`blade_curve`, `control_points`, `trailing_edge`, `pitch`,
`throat_opening`, `inlet_opening` -- is in **millimeters** (`blade["units"]
== "mm"`), matching the original blade-parametrization script's working
units.

This is a direct port of an existing standalone script
(`1.1_stator_axial.py`, `utils/spline_gen.py`), verified to reproduce its
control points to floating-point precision (~1e-15 mm) on the same
reference input.

`moc.geometry.export_blade_step(blade, face_path, solid_path=None,
extrude_length=1.0)` exports the fitted contour to STEP -- a 2D face,
plus an extruded 3D solid if `solid_path` is given. Requires `cadquery`
(the `cad` extra in `pyproject.toml`, or `environment.yaml`'s
conda-forge dependency) -- not needed for `parametrize_stator_blade`
itself, only for CAD export.

## Progress reporting (for a UI, not a live-slider app)

A solve takes 5-30s+, so a UI built on `design_nozzle` should use an
explicit "Compute" button + progress bar, not live-recompute-on-slider-drag
-- every `run_stage_*` method (and `design_nozzle` itself) accepts
`progress_callback=<callable(fraction, stage_label)>`, called at the same
points the terminal progress bars already print from. `moc.progress.
make_progress_reporter` combines the four stages' local fractions into one
overall (percent, "Stage N of 4") stream, weighted so the bar's rate roughly
tracks where the time actually goes (Stage 3, the kernel marching, is the
dominant cost).

For Dash specifically, this is meant to plug into a `background=True`
callback (with e.g. `DiskcacheManager` -- no extra infra needed for a
single-machine/small-team deployment) so the 5-30s solve doesn't block the
app for other users, and `progress=[Output(...), Output(...)]` gets driven
live instead of the request just hanging:

```python
from moc import design_nozzle, make_progress_reporter

@app.callback(
    Output("result-store", "data"),
    Input("compute-btn", "n_clicks"),
    State("fluid-dropdown", "value"),
    ...,
    background=True,
    manager=background_callback_manager,
    progress=[Output("progress-bar", "value"), Output("progress-bar", "label")],
)
def run_design(set_progress, n_clicks, fluid_name, ...):
    reporter = make_progress_reporter(set_progress)
    return design_nozzle(fluid_name=fluid_name, ..., progress_callback=reporter)
```

## Caching across a design/optimization loop

`design_nozzle` already reuses a cached `FluidManager` internally
(keyed on `fluid_name, backend, P0, T0, Q0`) -- calling it repeatedly at a
fixed operating point while sweeping geometry (`y_t`, `rho_t`, `rho_d`, ...)
or switching between `solver="conventional"`/`"mln"` only pays the fluid
table's build cost once. Nothing extra to do for this at the call site.

## Lower-level usage

For direct control over solver stages (e.g. custom stage sequencing, or
inspecting intermediate `Sauer`/`IVP` fronts), use `MOCSolver`/
`MOCSolverMLN` directly -- `design_nozzle`'s body in `moc/api.py` is a
short, readable example of exactly this:

```python
import numpy as np
from moc import ConventionalSolver, get_fluid_manager

fm = get_fluid_manager("nitrogen", "HEOS", P0=20e5, T0=113.0, Q0=float("nan"))

config = {
    "y_t": 0.01, "n": 15, "rho_t": 0.2,
    "P0": 20e5, "Q0": float("nan"), "T0": 113.0,
    "tau": np.linspace(0, 90, 91), "rho_d": 0.1,
    "Noz_Mach": 1.8860, "P_back": 2e5,
    "fluid_name": "nitrogen", "delta_flow": 0.0, "n_reflex": float("nan"),
}

solver = ConventionalSolver(config, show_plot=False, fluid_manager=fm)
solver.run_stage_1_sauer()
solver.run_stage_2_ivp()
solver.run_stage_3_kernel()
solver.run_stage_4_reflex()
solver.export_full_paper_data("nozzle.json")
```
