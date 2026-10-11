"""Run the consolidated 2D Gmsh workflow; geometry/mesh settings live in YAML."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example_meshing.src.meshing.meshing_2d import main, CONFIG_FILE, OUTPUT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_true", help="Skip the geometry construction figure.")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--show-gmsh", action="store_true")
    args = parser.parse_args()
    main(args.config, args.output,
         save_figures=not args.no_figures, generate_mesh=not args.geometry_only,
         show_gmsh=args.show_gmsh)
