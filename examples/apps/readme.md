# App launchers

Run either launcher from the repository root after `poetry install`:

```powershell
poetry run python examples/apps/run_app.py
poetry run python examples/apps/run_app_legacy.py
```

`run_app.py` opens the current app at <http://127.0.0.1:8052>. Its
`CONFIG_FILE` selects `cases/axial_stator.yaml`; change that assignment to use
another sample. Configuration paths resolve relative to the launcher, and jobs
write to its adjacent `output/` directory. The 3D viewer dependencies are
installed by default.

Use the [app user guide](../../docs/source/app_user_guide.md) for configuration,
blade design, mesh generation, display controls and project export. The
[case notes](cases/readme.md) describe the supplied axial, radial, linear,
rotor-only and stator/rotor configurations.

`run_app_legacy.py` opens the preserved Nozzle/Stator/Rotor/Cascade interface at
<http://127.0.0.1:8050>. It retains its diagnostics, control-point editing,
sizing, CAD and rotor tools.

## Installed commands

The package commands support launching without an example script:

```powershell
poetry run turbo_moc-app --config examples/apps/cases/stator_rotor.yaml
poetry run turbo_moc-app-legacy
```

The equivalent module commands are `poetry run python -m turbo_moc.app` and
`poetry run python -m turbo_moc.legacy.app`. The current app also accepts
explicit port and output arguments. Run `poetry install --only-root` after
updating a checkout to refresh the installed commands.

Both example launchers accept `TURBO_MOC_EXAMPLE_SMOKE_TEST=1` for initialization
without starting a server.
