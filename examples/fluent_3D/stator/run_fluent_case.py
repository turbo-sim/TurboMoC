"""Run this example using its adjacent YAML; no command-line arguments needed."""
from pathlib import Path
from turbo_moc.fluent.workflow_3d import main

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_stator_3D.yaml"

if __name__ == "__main__":
    main(CONFIG_FILE)
