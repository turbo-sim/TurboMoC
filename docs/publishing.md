# Publishing TurboMoC to PyPI

The workflow `.github/workflows/publish_to_pipy.yml` builds with setuptools,
checks distribution metadata, and tests the built wheel on Ubuntu and Windows
with Python 3.11, 3.12, and 3.13. Pull requests to `main` and manual workflow runs
validate without publishing. Pushing a matching `v*.*.*` tag publishes the tested
wheel and source distribution after all checks pass.

## One-time setup

1. Create a PyPI account, verify your email, and enable two-factor authentication.
2. Confirm you can publish under `turbo_moc`, the name in `pyproject.toml`.
   PyPI normalizes this to `turbo-moc`. If someone else owns the name, choose an
   available distribution name before releasing.
3. In GitHub **Settings > Environments**, create `pypi`. Restrict deployment to
   version tags (`v*`). Optionally add a required reviewer for manual approval.
4. For a new project, open <https://pypi.org/manage/account/publishing/> and add
   a **pending publisher** with these fields:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `turbo_moc` |
   | GitHub owner | `turbo-sim` |
   | Repository | `method_of_characteristics` |
   | Workflow filename | `publish_to_pipy.yml` |
   | Environment | `pypi` |

   For an existing project you own, add the same Trusted Publisher under that
   project's **Publishing** settings. A pending publisher creates the project
   on its first successful upload; it does not reserve the name.
5. Commit and push the workflow and package changes to `main`. Run the workflow
   manually from GitHub Actions to check the build and tests before tagging.

Trusted Publishing uses GitHub's identity token. No `PYPI_USERNAME`,
`PYPI_PASSWORD`, or API token secret is required.

## First release (0.1.0)

Ensure the release commit contains the complete `src/turbo_moc` package and that
`pyproject.toml`, `src/turbo_moc/__init__.py`, and `.bumpversion.toml` all specify
`0.1.0`. From the committed, tested revision on `main`:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin main
git push origin v0.1.0
```

Watch **Actions > Build, test, and publish TurboMoC to PyPI**. Approve the `pypi`
environment if you configured a reviewer. A GitHub Release page is optional;
the tag push triggers publication.

After success, verify in a fresh Python 3.11–3.13 environment:

```bash
python -m pip install turbo_moc==0.1.0
python -c "import turbo_moc; print(turbo_moc.__version__)"
turbo_moc-app
```

For optional STEP export, install `"turbo_moc[cad]==0.1.0"` instead. The workflow
tests the default installation; optional CadQuery export is not covered.

## Subsequent releases

With all changes committed and a clean working tree, use the existing bump
configuration to update both version strings and create a commit and tag:

```bash
poetry run bump-my-version bump patch
git push origin main
git push origin v0.1.1
```

Use `minor` or `major` as appropriate and push the specific tag created by the
command. If editing versions manually, also update `.bumpversion.toml`.
The workflow rejects a tag that differs from `v` plus the package version.
Already published distribution filenames cannot be reused; corrections after
publication need a new version.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [creating a project with a pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
