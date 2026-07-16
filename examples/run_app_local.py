"""
run_app_local.py -- local dev entry point for the moc Dash app.
Mirrors turbodash's demos/run_app_local.py.

Run:
    conda activate moc_env
    python examples/run_app_local.py

Then open http://127.0.0.1:8050/ in a browser. Equivalent to running
`moc-app` (the console-script entry point) or `python -m moc.app` directly.
"""

from moc.app import app

if __name__ == "__main__":
    app.run(debug=True)
