"""Run the same staged 3D geometry/mesh/export workflow used by TurboMoC."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example_meshing.src.meshing.meshing_3d import main
from example_meshing.src.paths import GMSH_3D_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GMSH_3D_DIR / "axial_stator.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row", help="Stable row ID in a project YAML (for example stator_1).")
    args = parser.parse_args()
    main(args.config, args.output, row_id=args.row)
