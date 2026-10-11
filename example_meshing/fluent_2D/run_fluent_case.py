"""Run the consolidated Fluent workflow; all solver settings live in YAML."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example_meshing.src.fluent.workflow_2d import main, CONFIG_FILE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    main(parser.parse_args().config)
