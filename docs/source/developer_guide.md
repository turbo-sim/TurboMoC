# Developer guide

## Installation for developers

With Git and [Poetry](https://python-poetry.org/docs/#installation) installed,
clone the repository and install the package and development tools using
Python 3.11 through 3.13:

```bash
git clone https://github.com/turbo-sim/method_of_characteristics.git
cd method_of_characteristics
poetry install --with dev
```

Poetry installs the package in editable mode, so changes to the source under
`src/moc/` are available without reinstalling. The `dev` dependency group
includes documentation and release tools. The main implementation areas are `core`,
`solvers`, `geometry`, `rotor`, plotting, and the Dash application.

JAX is used by the core solver and comes through the `jaxprop` dependency;
this package does not have a separate `jax` extra. CAD export is optional:

```bash
poetry install --with dev --extras cad
```

`cad` is a package **extra**, selectable by both pip and Poetry; `dev` is a
dependency **group** for working on the repository.

### Alternative: Conda environment

The supplied Conda environment includes CadQuery for STEP export. After
cloning the repository, run these commands from its root:

```bash
conda env create -f environment.yaml
conda activate moc_env
python -m pip install -e .
```

This installs the package in editable mode. To also install the documentation
tools into the active Conda environment, run `poetry install --with dev`.

## Building the documentation

Run these commands from the repository root:

```bash
poetry install --with dev
poetry run python docs/build_docs.py
```

This generates the API pages, builds the site, opens a browser, and watches
for changes at `http://127.0.0.1:8000`. For a single build:

```bash
poetry run python docs/build_docs.py --no-autobuild
```

The HTML entry point is `docs/_build/html/index.html`. The same commands work
with a virtual environment's Python instead of `poetry run python` when the
development dependencies are already installed.

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

The repository has a deliberately small smoke-test suite. It checks that the
installed package exposes its basic public interface and executes every Python
script under `examples/`.

Run the complete suite from the repository root:

```bash
poetry run python tests/run_tests.py
```

You can also invoke pytest directly or run one test module:

```bash
poetry run pytest tests/test_examples.py
```

The example checks set `MOC_EXAMPLE_SMOKE_TEST=1` and use Matplotlib's
non-interactive backend. The nozzle and rotor calculations still run and their
figures are still constructed, but the checks do not start the Dash server,
open plot windows, or write output files.

GitHub Actions runs the same smoke suite on Ubuntu and Windows for pushes and
pull requests targeting `main`.

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
  it to GitHub Pages.
- `test_and_deploy_app.yaml` runs the smoke tests on Ubuntu and Windows for
  pull requests and pushes to `main`.

PyPI publication is not currently automated.

## Deploying the web application

Web application deployment is not implemented yet. Deployment instructions
will be added here when a hosting and release workflow is available.
