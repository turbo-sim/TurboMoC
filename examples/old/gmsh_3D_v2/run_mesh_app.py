"""Launch TurboMoC; the application implementation lives in app/."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.gmsh_3D_v2.app.application import create_app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8052, debug=False)
