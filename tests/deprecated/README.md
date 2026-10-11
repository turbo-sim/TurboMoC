# Preserved deprecated implementation tests

These tests cover code retained under `src/turbo_moc/deprecated/`, sometimes
through public compatibility aliases. They remain available for maintenance of
the legacy implementation and comparison with the current pipeline.

| Module | Subject |
| --- | --- |
| `test_radial.py` | Historical radial conformal mapping and handedness |
| `test_radial_mesh.py` | Historical stator/rotor mesh wrapping and extrusion |
| `test_radial_solid.py` | Radial mapping and CAD exports based on that mapping |
| `test_sizing.py` | Historical stator rescaling and mass-flow sizing |
| `test_rotor_sizing.py` | Historical rotor rescaling and pitch sizing |
| `test_wall_spacing.py` | Historical first-layer estimates and inflation utilities |

Default pytest collection excludes this directory through `norecursedirs` in
`pyproject.toml`. An explicit path opts in, without editing the configuration:

```powershell
# Deprecated implementation only
.venv/Scripts/python.exe -m pytest tests/deprecated -q

# Current and deprecated implementations
.venv/Scripts/python.exe -m pytest tests -q -o norecursedirs=
```

CadQuery is optional; CAD tests are skipped when it is unavailable. Legacy-app
tests removed in the earlier cleanup are not recreated here.
