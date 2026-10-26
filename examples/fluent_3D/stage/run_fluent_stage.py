"""Run this example using its adjacent YAML; no command-line arguments needed."""
from pathlib import Path
from turbo_moc.fluent.stage.setup import main

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_stage.yaml"

if __name__ == "__main__":
    main(CONFIG_FILE)
