"""
run_app_local.py -- local dev entry point for the TurboMoC Dash app.

Run:
    conda activate turbo_moc_env
    python examples/run_app_local.py

Then open http://127.0.0.1:8050/ in a browser. Equivalent to running
`turbo_moc-app` (the console-script entry point) or `python -m turbo_moc.app` directly.

Set TURBO_MOC_EXAMPLE_SMOKE_TEST=1 to validate the app without starting the server.
"""

import os
import sys
from pathlib import Path

# Running this file puts examples/ first on sys.path, where the examples/gmsh
# folder would shadow the real gmsh package that turbo_moc.meshing imports.
# Module level on purpose: the app's background worker re-imports this file.
_EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _EXAMPLES_DIR]

from turbo_moc.app import app  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("TURBO_MOC_EXAMPLE_SMOKE_TEST") == "1":
        if app.layout is None or not callable(app.run):
            raise RuntimeError("The Dash application was not initialized correctly.")
        print("Dash application initialized successfully.")
    else:
        app.run(debug=True)
