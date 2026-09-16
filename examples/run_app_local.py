"""
run_app_local.py -- local dev entry point for the TurboMoC Dash app.
Mirrors turbodash's demos/run_app_local.py.

Run:
    conda activate turbo_moc_env
    python examples/run_app_local.py

Then open http://127.0.0.1:8050/ in a browser. Equivalent to running
`turbo_moc-app` (the console-script entry point) or `python -m turbo_moc.app` directly.

Set TURBO_MOC_EXAMPLE_SMOKE_TEST=1 to validate the app without starting the server.
"""

import os

from turbo_moc.app import app

if __name__ == "__main__":
    if os.environ.get("TURBO_MOC_EXAMPLE_SMOKE_TEST") == "1":
        if app.layout is None or not callable(app.run):
            raise RuntimeError("The Dash application was not initialized correctly.")
        print("Dash application initialized successfully.")
    else:
        app.run(debug=True)
