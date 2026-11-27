# Subsonic reaction rotor: parametrization and thickness distribution

The notes cover the geometry-only rotor blade `blade_type: reaction` in TurboMoC, which is the blade we agreed to use for the subsonic rotor. The code is in `src/turbo_moc/core/reaction_blade.py`, and the example stage is in this folder (`build_stage_mesh.py`).

## 1. Section construction

The section is built at unit meridional (axial) chord. One similarity transform then scales it to the row's chord.

1. **Camberline angle.** The metal angle runs from the inlet metal angle to the outlet metal angle along the chord. It follows TurboDash's `turning_angle` law, interpolated on the angle. Two parameters shape the law:
   - `turning_location`: the fraction of the chord where half of the turning is done. 0.5 is symmetric; a larger value puts the turning in the rear (aft-loaded).
   - `turning_concentration`: how strongly the turning concentrates around that location. 1 is close to uniform; larger values give a sharper bend.
2. **Camberline.** The camberline is the integral of the slope tan(beta) along the chord.
3. **Thickness.** The modified NACA law of TurboDash (`thickness_models.full_thickness`, family `naca`) gives the full thickness. Half of it is laid on each side along the camberline normal. The slope at the endpoints is passed to the law, so the leading-edge radius and the trailing-edge wedge angle are physical, measured normal to the camberline.
4. **Trailing edge.** The trailing edge is a finite cut-off normal to the camberline, of thickness `trailing_edge_thickness`.

## 2. Inputs and agreed values

All thicknesses are fractions of the meridional chord.

| Input | Default | Meaning |
| --- | --- | --- |
| `inlet_metal_angle_deg` | 0 | Inlet metal angle, TurboDash convention |
| `outlet_metal_angle_deg` | 60 | Outlet metal angle, TurboDash convention |
| `turning_location` | 0.52 | Half-turning location along the chord |
| `turning_concentration` | 1.2 | Concentration of the turning |
| `maximum_thickness` | 0.12 | Maximum full thickness |
| `maximum_thickness_location` | 0.30 | Position of the maximum thickness |
| `leading_edge_radius` | 0.01 | Leading-edge radius |
| `trailing_edge_thickness` | 0.015 | Trailing-edge thickness (cut-off) |
| `trailing_edge_wedge_angle_deg` | 8 | Trailing-edge wedge angle |
| `solidity_chord` | 1.0494 | True chord / pitch: sets the natural blade count, as in TurboDash |
| `pitch_to_chord` | 0.8727 | Only for linear cascades |

Two cases are compared in `README.md`. The baseline uses half-turning location 0.52 and concentration 1.2; the shifted case uses 0.65 and 2.0, for an aft-loaded blade.

## 3. Blade count and pitch

In an annular row, the pitch comes from the blade count, 2πr/N. The natural blade count follows TurboDash's true-chord solidity:

    s/c = 1 / (sigma * cos(0.5 (kappa_in + kappa_out)))

Here sigma is `solidity_chord`, and the cosine term converts the meridional chord to the true chord through the mean stagger. Take sigma, or the pitch-to-chord ratio, from your meanline design.

## 4. Degree of reaction as the rotor outlet condition

In the app (Stator + Rotor, rotor inlet coupled to the stator exit), the rotor outlet can be set with **Degree of reaction**. From the stator exit state (h1, s1) and the total enthalpy h00:

    h2 = (h1 - R h00) / (1 - R),     p_back = p(h2, s1)

Only the rotor outlet pressure changes; the stator is untouched. The resulting back-pressure is the value to impose at the Fluent outlet (`setup_fluent_case.py --p-out`).

## 5. Conventions

- TurboDash measures azimuth counterclockwise. The TurboMoC solver frame uses y = −θ, so β_TurboMoC = −β_TurboDash (0/60 becomes 0/−60). The pressure side is the lower wall.
- One design feeds the preview, the passage, the 2D reference mesh and the 3D volume mesh, so all of them show the same blade.
- TurboDash is an optional dependency: it must be installed or on the path to build a reaction blade.

## 6. Files in this folder

| File | Purpose |
| --- | --- |
| `run_reaction_rotor.py`, `reaction_rotor_*.yaml` | Rotor-only examples (blade, 2D mesh, 3D mesh, CGNS) |
| `build_stage_mesh.py` | MoC stator + reaction rotor: stage mesh `output/stage/mesh/stage.cgns` |
| `barotropic_model.py` | Barotropic fluid model (Fluent expressions) |
| `fluent_setup.md` | Manual Fluent setup, step by step |
| `setup_fluent_case.py` | Automatic Fluent setup from the mesh (use after doing it once by hand) |
| `postprocess_3d.py` | 3D post-processing with PyVista from the case and data files |
