"""Run the 2D Fluent control-point perturbation workflow."""

import argparse
from pathlib import Path

from turbo_moc.fluent.workflow_2d_cp_perturbation import main

EXAMPLE_DIR = Path(__file__).resolve().parent
BASELINE_FILE = EXAMPLE_DIR / "base_line_fluent_stator_2D.yaml"
CASE_FILE = EXAMPLE_DIR / "cases" / "case_1" / "case_1.yaml"


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_positional", nargs="?", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE_FILE)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.baseline, args.case or args.case_positional or CASE_FILE, args.output)


if __name__ == "__main__":
    cli()
