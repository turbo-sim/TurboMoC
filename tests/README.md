# Test suite

Keep tests that protect a distinct numerical result, user workflow or failure
boundary. Avoid tests that freeze incidental implementation details, component
nesting, colours, labels or exact histogram bin counts.

## Retained app coverage

| Area | Tests protect |
| --- | --- |
| Configuration and controls | YAML load/save, validation, browser number conversion, preservation of row settings and read-only coupled rotor inputs |
| Calculation queues | Selected row order, stator-to-rotor handoff, prerequisites, failures, cancellation and publication of completed rows |
| Workflow | Artifact invalidation, worker provenance, independent submission/polling and navigation after successful meshing |
| Runtime | Sample-case startup, packaged assets, execution outside the checkout directory, separate output folders and Windows file locks |
| Views | Visibility of both rows, physical interspace and interface coordinates, closed blade previews and visualization updates without remeshing |
| Project export | Current CGNS files, multi-zone packaging, provenance and rejection of stale or incomplete exports |

The geometry, meshing, interface and CGNS tests remain because they verify the
numerical work behind these app workflows. Real Gmsh checks use small synthetic
profiles to avoid repeatedly solving MoC. Axial, tapered axial, radial inflow,
radial outflow and general curved meridional mappings exercise different cases.
Existing scientific solver tests remain outside the app cleanup.

Tests whose subject is a preserved deprecated implementation live in
`tests/deprecated/`. These cover radial wrapping and CAD export, the old radial
mesh pipeline, stator/rotor sizing and wall-spacing estimates. Some import
compatibility aliases from `geometry` or `meshing`, which resolve to code under
`src/turbo_moc/deprecated/`. The default run excludes this folder. Current MoC,
rotor flow physics, shared blade fitting and canonical geometry/meshing tests
remain in the default suite.

`conftest.py` provides a small shared design fixture. `app_helpers.py` builds Dash
requests from registered callback metadata, so tests need not duplicate the
callback's positional input order.

## Removed or consolidated coverage

- Removed legacy-app tests: `test_app_rotor_mesh.py`, `test_app_stator_mesh.py`,
  `test_app_stator_mesh_3d.py`, `test_legacy_app_compatibility.py`, and the legacy
  app example-launch check. The legacy application itself remains available.
- Removed `test_deprecated_meshing.py`, which checked deprecated import plumbing.
- Removed temporary migration guards that checked module locations, scanned
  imports or compared scientific source files with Git HEAD. Retained a fresh
  process startup check to verify that the installed app works independently of
  historical example packages.
- Removed duplicate stage-invalidation checks, cosmetic layout assertions,
  recursive-versus-compiled spline implementation parity and exact histogram
  bin counts. Merged MoC progress forwarding into the fresh-solve test and digest
  checks into the browser-number regression.
- Regrouped useful checks from `test_gmsh_3d_v2.py`,
  `test_package_migration.py`, `test_meshing_consolidation.py` and
  `test_app_output_directory.py` into runtime, workflow, view, design/meshing and
  example tests. Those old test module names no longer describe the codebase.
- Removed tests of the former slice/worst-cell app inspection controls; retained
  quality metric and real mesh validation checks.

Plot assertions identify blade rows by name rather than by their position among
flow-domain traces. Essential behavior should survive adding a trace or changing
the layout.

## Running tests

From the repository root, run the current implementation suite:

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

Plain `pytest` uses the same default selection, excluding `tests/fluent/` and
`tests/deprecated/`. The default suite requires neither PyFluent, a Fluent
installation/license, CadQuery nor the companion barotropy package.

Fluent workflow and graphics checks are opt-in:

```powershell
poetry run pytest tests/fluent -q
```

These checks use mocked Fluent sessions and field requests; they do not launch
Fluent. They retain a real jaxprop EOS check for the barotropic fit.

Run the preserved deprecated
tests explicitly when changing that implementation or its compatibility aliases:

```powershell
.venv/Scripts/python.exe -m pytest tests/deprecated -q
```

To run the default, Fluent and deprecated suites together:

```powershell
.venv/Scripts/python.exe -m pytest tests -q -o norecursedirs=
```

The override removes the directory exclusion for that invocation only.

For changes limited to app behavior, start with the affected modules. For example:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_app_calculation_queue.py tests/test_app_workflow.py tests/test_app_views.py -q
```

`tests/run_tests.py` remains a small package/example smoke runner. It does not run
the complete suite. The deprecated CAD tests require CadQuery. Fluent workflow
tests use mocked solver sessions; a passing suite does not certify a licensed
Fluent import or simulation.

On Windows, use a short, fresh `--basetemp` path if the checkout path and job IDs
would otherwise exceed filesystem path limits. Pytest clears a supplied
`--basetemp`, so choose a directory reserved for disposable test files.

## Review verification

On 2026-10-11, the cleanup reduced collection from 414 to 332 tests. The final
complete run passed all 332; the optional CAD module was skipped because CadQuery
was unavailable. VTK emitted NumPy deprecation warnings. No application or
scientific implementation changes were needed. The subsequent folder separation
preserves those tests while excluding deprecated coverage from default runs:
308 current tests were collected by default, and 24 deprecated tests were available
explicitly, plus the optional CAD module. Collection was checked for each mode;
19 fast deprecated tests were rerun and passed after the move. The slower mesh
tests were already passing in the complete run before relocation.

The subsequent Fluent separation moves 72 checks into `tests/fluent/`, leaving
236 checks in the default suite. Ubuntu CI installs `libglu1-mesa`, the native
library required to import Gmsh. The planar mesh workflow uses jaxprop plotting
helpers and is tested with barotropy explicitly unavailable.

Verification on Windows with Python 3.13: all 236 default checks passed with
PyFluent, barotropy and CadQuery imports blocked. The 15 planar mesh checks and
72 opt-in Fluent checks also passed together. The default run reported 170
VTK/NumPy deprecation warnings; no test failed. The Ubuntu system-library fix
requires verification by the next GitHub Actions run.
