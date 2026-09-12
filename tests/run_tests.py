#!/usr/bin/env python3
"""Run the repository's small pytest smoke suite."""

from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
TEST_FILES = [
    TESTS_DIR / "test_package.py",
    TESTS_DIR / "test_examples.py",
]


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-vv", *(str(path) for path in TEST_FILES)]))
