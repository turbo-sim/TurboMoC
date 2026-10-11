"""Run this example using its adjacent YAML; no command-line arguments needed."""
from pathlib import Path
from turbo_moc.meshing.meshing_2d import main

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "workflow_gmsh_stator_2D.yaml"
OUTPUT_DIR = HERE / "output"

if __name__ == "__main__":
    main(CONFIG_FILE, OUTPUT_DIR)
