"""Stator (MoC) + subsonic reaction rotor (TurboDash camberline + NACA): stage mesh through the app jobs.

    python examples/apps/reaction_rotor/build_stage_mesh.py

Uses the Fluent stage pipeline's own design step (turbo_moc.fluent.stage.design.prepare): app jobs for both rows
(blade design, 2D reference mesh, 3D volume mesh), then the app's project export. Writes
output/stage/mesh/stage.cgns (both rows, one zone each, row-prefixed boundaries), the export report and the
project YAML. The rotor inlet is coupled to the stator exit, so its inlet metal angle follows beta_rel.
"""
from pathlib import Path

from turbo_moc.fluent.stage import design

HERE = Path(__file__).resolve().parent
WORKFLOW = {
    "geometry_type": "axial",
    "rpm": 5000.0,                 # Subsonic relative inflow: beta_rel ~ 35 deg, M_rel ~ 0.75 (default stator)
    "axial_interspace_mm": 2.0,
    "rows": {"rotor_1": {"geometry": {"rotor_design": {"blade_type": "reaction"}},
                         # Thin trailing edge (0.015 c): a thinner inflation stack than the vortex-rotor default
                         "mesh": {"cgns": {"filename": "rotor_mesh_3d.cgns"},
                                  "numerics": {"blade_size": 0.2, "domain_size": 0.8, "span_layers": 20,
                                               "inflation_layers": {"first_height": 0.012, "growth_ratio": 1.2,
                                                                    "thickness": 0.4}}}}},
}

if __name__ == "__main__":
    out = HERE / "output" / "stage"
    project, registry = design.prepare(WORKFLOW, out)
    print("stage mesh:", sorted(p.name for p in (out / "mesh").iterdir()))
