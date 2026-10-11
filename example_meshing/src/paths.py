"""Resolve case/output locations independently of launchers and working directory."""
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SOURCE_DIR.parent
PROJECT_DIR = EXAMPLE_DIR.parent
GMSH_2D_DIR = EXAMPLE_DIR / "gmsh_2D"
GMSH_3D_DIR = EXAMPLE_DIR / "gmsh_3D"
FLUENT_2D_DIR = EXAMPLE_DIR / "fluent_2D"
