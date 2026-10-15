"""Report the solved stage using the adjacent YAML; ``--plots-only`` reuses the extracted data without Fluent."""
import sys
from pathlib import Path
from turbo_moc.fluent.stage.report import main

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_fluent_stage.yaml"

if __name__ == "__main__":
    main(CONFIG_FILE, plots_only="--plots-only" in sys.argv[1:])
