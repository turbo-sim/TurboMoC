# Developer guide

## Installation for developers

With Python 3.11, 3.12, or 3.13, Git, and
[Poetry](https://python-poetry.org/docs/#installation) installed,
clone the repository and install the package, development tools, and optional
CadQuery support:

```bash
git clone https://github.com/turbo-sim/TurboMoC.git
cd TurboMoC
poetry install --with dev --extras cad
```

Poetry installs the package in editable mode, so changes to the source under
`src/turbo_moc/` are available without reinstalling. The `dev` dependency group
includes documentation, testing, and release tools. The `cad` extra provides
CadQuery for STEP export. Scientific solvers live in `core`; geometry,
meshing, visualization, Fluent workflows, configuration, and the Dash app have
separate subpackages under `src/turbo_moc/`. Preserved workflows live in `legacy`.

JAX is used by the core solver and comes through the `jaxprop` dependency;
this package does not have a separate `jax` extra.

## Building the documentation

Run these commands from the repository root:

```bash
poetry install --with dev --extras cad
poetry run python docs/build_docs.py
```

This generates the API pages, builds the site, opens a browser, and watches
for changes at `http://127.0.0.1:8000`. For a single build:

```bash
poetry run python docs/build_docs.py --no-autobuild
```

The HTML entry point is `docs/_build/html/index.html`.

`docs/build_docs.py` is the shared build entry point. Package settings live in
`docs/conf.py`; all Markdown and reStructuredText documentation lives under
`docs/source/`. Edit narrative pages there. Files in `docs/source/api/` are
regenerated on every build, so edit Python docstrings to change the API.

To refresh the bibliography from the configured Zotero group before building,
set `ZOTERO_API_KEY` when the group is private and run:

```bash
poetry run python docs/build_docs.py --build-bib
```

## Running the tests

The smoke-test runner checks that the installed package exposes its basic
public interface and executes the Python design scripts under
`examples/moc_design/` and launchers under `examples/apps/`.

Run the package and design/app example smoke checks from the repository root:

```bash
poetry run python tests/run_tests.py
```

You can also invoke pytest directly or run one test module:

```bash
poetry run pytest tests/test_examples.py
```

The default suite includes app, geometry, meshing and solver checks, and excludes
the opt-in Fluent and deprecated folders:

```bash
poetry run pytest
```

It needs neither Fluent/PyFluent nor the companion barotropy package. On Ubuntu,
install Gmsh's native library with `sudo apt-get install libglu1-mesa` before
running mesh checks. GitHub Actions installs this dependency automatically.

For the fast Gmsh and opt-in Fluent workflow checks:

```bash
poetry run pytest tests/test_stator_mesh.py tests/fluent -q
```

It generates one small real mesh, checks its exported artifacts and cache behavior,
and tests Fluent orchestration with mocks. Figure rendering is mocked while mesh
generation and CGNS export remain real; full quality-report generation is skipped.
It uses no Fluent license and does not run a MoC solve or CFD iterations. A real EOS check covers
the barotropic fit.

The example checks set `TURBO_MOC_EXAMPLE_SMOKE_TEST=1` and use Matplotlib's
non-interactive backend. The nozzle and rotor calculations still run and their
figures are still constructed, but the checks do not start the Dash server or
open plot windows. The design examples still export coordinates and figures
to their own ignored `examples/output/` directories.

The package publishing workflow runs the default pytest suite on Ubuntu and
Windows for release tags and pull requests targeting `main`.

## Releasing a new version

From a clean `main` branch with the intended changes committed, bump the
version and push the automatically created release commit and tag:

```bash
poetry run bump-my-version bump patch
git push origin main --tags
```

Use `minor` or `major` instead of `patch` when appropriate. The bump
command calculates the new version, updates the configured version files,
creates a commit, and creates the corresponding release tag automatically.
GitHub Actions then:

- `deploy_docs.yaml` builds the documentation for pull requests and deploys
  it to GitHub Pages on updates to `main`.
- `publish_to_pipy.yml` runs the default test suite on Ubuntu and Windows for
  pull requests to `main` and version tags, then publishes tagged releases to
  PyPI after those checks pass.

The app-specific `test_and_deploy_app.yaml` workflow is currently disabled.

## Deploying the web application

Web application deployment is not implemented yet. Deployment instructions
will be added here when a hosting and release workflow is available.
