"""Run every shipped example in non-interactive smoke-test mode."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((PROJECT_ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script):
    env = os.environ.copy()
    env["MOC_EXAMPLE_SMOKE_TEST"] = "1"
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"{script.name} failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
