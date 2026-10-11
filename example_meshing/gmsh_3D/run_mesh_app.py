"""Launch TurboMoC; its implementation lives in example_meshing/src/app/."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example_meshing.src.app.application import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch TurboMoC")
    parser.add_argument("--config", type=Path, help="Machine-project YAML to open")
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()
    create_app(args.config).run(host="127.0.0.1", port=args.port, debug=False)
