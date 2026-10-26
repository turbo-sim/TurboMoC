"""Run canonical examples in non-interactive smoke-test mode."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(
    path for folder in ("moc_design", "apps")
    for path in (PROJECT_ROOT / "examples" / folder).glob("*.py")
    if path.name != "run_app_legacy.py"
)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script):
    env = os.environ.copy()
    env["TURBO_MOC_EXAMPLE_SMOKE_TEST"] = "1"
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


def test_workflow_runners_resolve_local_files_from_another_directory(tmp_path, monkeypatch):
    import runpy
    import yaml
    from turbo_moc.meshing import mesh_2d as meshing_2d, mesh_3d as meshing_3d
    from turbo_moc.fluent import workflow_2d, workflow_3d
    from turbo_moc.fluent.stage import setup as stage_setup
    monkeypatch.chdir(tmp_path)
    for family, module, runner in (
        ("gmsh_2D", meshing_2d, "run_mesh_generation.py"),
        ("gmsh_3D", meshing_3d, "run_mesh_generation.py"),
        ("fluent_2D", workflow_2d, "run_fluent_case.py"),
        ("fluent_3D/stator", workflow_3d, "run_fluent_case.py"),
        ("fluent_3D/stage", stage_setup, "run_fluent_stage.py"),
    ):
        calls = []
        monkeypatch.setattr(module, "main", lambda *args: calls.append(args))
        folder = PROJECT_ROOT / "examples" / family
        runpy.run_path(str(folder / runner), run_name="__main__")
        assert len(calls) == 1
        config = calls[0][0]
        assert config.is_absolute() and config.parent == folder and config.is_file()
        if len(calls[0]) == 2:
            assert calls[0][1] == folder / "output"
        else:
            assert yaml.safe_load(config.read_text())["output_dir"] == "output"
