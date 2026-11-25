# Subsonic reaction rotor (TurboDash camberline + modified NACA thickness)

Geometry-only rotor blade type `reaction` in the TurboMoC studio (`rotor_design.blade_type: reaction`). It needs no
vortex solve and no supersonic inlet. A second geometry-only type, `constant_passage`, builds the Baines
constant-passage blade (previously only in `examples/moc_design/example_baines_blade.py`) from the blade angle and
the pitch/chord ratio alone.

## Case

Axial rotor: meridional chord 15 mm, mean radius 50 mm (hub 45 mm), span 10 mm, 24 blades.
TurboDash convention: metal angles 0 deg -> 60 deg. Maximum full thickness 0.12 c at 0.30 c, leading-edge radius 0.01 c,
trailing-edge thickness 0.015 c, 8 deg trailing-edge wedge, finite cut-off.

| Case | Half-turning location | Turning concentration | Case file |
| --- | --- | --- | --- |
| baseline | 0.52 | 1.2 | `reaction_rotor_baseline.yaml` |
| shifted | 0.65 | 2.0 | `reaction_rotor_shifted.yaml` |

## Commands

```
python examples/apps/reaction_rotor/run_reaction_rotor.py all   # blade -> 2D reference mesh -> 3D volume + CGNS
python examples/apps/reaction_rotor/verify_cgns.py              # reopen the CGNS files and audit them
python -m pytest tests/test_reaction_blade.py                   # geometry, conventions, studio workflow
```

`run_reaction_rotor.py` writes each case file, reloads it with `projects.load_project`, and runs the studio's own
jobs (`flow.start_job` -> worker subprocess). The volume job reloads the persisted reference section. Each case writes
`output/<case>/summary.json`, and the audit writes `output/cgns_audit.json`.

## Conventions

- Angles are converted once, in `core/reaction_blade.py`. TurboDash's counterclockwise azimuth maps to the TurboMoC
  solver frame with y = -theta, so beta_TurboMoC = -beta_TurboDash (0/60 -> 0/-60). The pressure side is the lower wall.
- Thickness inputs are fractions of the meridional chord. The section is built at unit chord and mapped to the row's
  chord by one similarity transform (`geometry/rotor_blade.py`). The same `Design` feeds the preview, the passage,
  the 2D mesh and the 3D mesh.
- Pitch: annular rows take it from the blade count. `pitch_to_chord` is used only by linear cascades.
- TurboDash is an optional dependency. Its two leaf functions are imported lazily: `geometry.turning_angle` and
  `thickness_models.full_thickness` (family `naca`). `build_section` on TurboDash main still evaluates
  `denton_thickness`, so it is not used.
