"""
run_app_legacy.py -- launch the preserved TurboMoC Dash app.

Run:
    conda activate turbo_moc_env
    python examples/apps/run_app_legacy.py

Automatically opens http://127.0.0.1:8050/ in a browser. Equivalent to running
`turbo_moc-app-legacy` (the console-script entry point) or `python -m turbo_moc.legacy.app` directly.

Set TURBO_MOC_EXAMPLE_SMOKE_TEST=1 to validate the app without starting the server.
"""

import os
import webbrowser
from threading import Timer

from turbo_moc.legacy.app import app

if __name__ == "__main__":
    if os.environ.get("TURBO_MOC_EXAMPLE_SMOKE_TEST") == "1":
        if app.layout is None or not callable(app.run):
            raise RuntimeError("The Dash application was not initialized correctly.")
        print("Dash application initialized successfully.")
    else:
        # Open only in the reloader process to avoid duplicate browser tabs.
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()
        app.run(debug=True)
