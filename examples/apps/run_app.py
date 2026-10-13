"""Launch TurboMoC with a local example case and local output folder."""
import os
import webbrowser
from pathlib import Path
from threading import Timer

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "cases" / "axial_stator.yaml"
OUTPUT_DIR = HERE / "output"

if __name__ == "__main__":
    from turbo_moc.app import create_app
    app = create_app(CONFIG_FILE, output_dir=OUTPUT_DIR)
    if os.environ.get("TURBO_MOC_EXAMPLE_SMOKE_TEST") == "1":
        if app.server.test_client().get("/_dash-layout").status_code != 200:
            raise RuntimeError("The TurboMoC design app did not initialize.")
        print("TurboMoC design application initialized successfully.")
    else:
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8052/")).start()
        app.run(port=8052, debug=False)
