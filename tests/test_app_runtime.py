"""App startup, output isolation, live cancellation and Windows file-lock handling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from turbo_moc.apps.app import create_app, workflow, worker
from turbo_moc.apps.app import workflow as flow
from app_helpers import components

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/turbo_moc"
CASES = {"axial_stator": ROOT / "examples/apps/cases/axial_stator.yaml"}


@pytest.mark.parametrize("filename,mode,family,roles", [
    ("axial_stator", "stator", "axial", ["stator"]),
    ("radial_stator", "stator", "radial", ["stator"]),
    ("linear_stator", "stator", "linear", ["stator"]),
    ("rotor_only", "rotor", "axial", ["rotor"]),
    ("stator_rotor", "stator_rotor", "axial", ["stator", "rotor"]),
])
def test_sample_projects_build_the_requested_workspaces(filename, mode, family, roles):
    from turbo_moc.apps.app import create_app
    from turbo_moc.configuration_project import active_rows
    cases = ROOT / "examples/apps/cases"

    application = create_app(cases / f"{filename}.yaml")
    response = application.server.test_client().get("/_dash-layout")
    assert response.status_code == 200
    layout = components(response.get_json())
    project = layout["base-project"]["data"]
    assert project["machine"] == {"design_mode": mode, "geometry_type": family}
    assert [row["role"] for row in active_rows(project)] == roles
    assert layout["design-mode"]["value"] == mode
    assert layout["machine-family"]["value"] == family
    assert [option["value"] for option in layout["selected-row"]["options"]] == [
        row["id"] for row in active_rows(project)]
    client = application.server.test_client()
    assert client.get("/_dash-dependencies").status_code == 200
    assert client.get("/").status_code == 200
    for asset in ("studio.css", "studio.js", "mesh_edges.js"):
        assert client.get(f"/assets/{asset}").status_code == 200


def test_jobs_use_the_installed_worker_module_and_writable_output(monkeypatch, tmp_path_factory):
    from turbo_moc.apps.app import workflow
    from turbo_moc.configuration import read_configuration

    launches = []

    def launch(command, **options):
        launches.append((command, options))
        return SimpleNamespace(pid=123456)

    # Two 32-character IDs already lengthen job paths; a short test-directory
    # name keeps this checkout's long Windows path below the platform limit.
    output = tmp_path_factory.mktemp("jobs")
    monkeypatch.setattr(workflow, "OUTPUT", output)
    monkeypatch.setattr(workflow, "PROCESSES", {})
    monkeypatch.setattr(workflow.subprocess, "Popen", launch)
    config = read_configuration(CASES["axial_stator"])
    handle = workflow.start_job("a" * 32, "nozzle", config, {})
    folder = workflow.directory(handle)
    assert folder.is_relative_to(output) and not folder.is_relative_to(PACKAGE)
    command, options = launches[0]
    assert command[:4] == [sys.executable, "-u", "-m", "turbo_moc.apps.app.worker"]
    assert Path(command[4]) == folder / "request.json"
    assert options["cwd"] == folder
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert request["config"] == config and request["handle"]["row_id"] == "stator_1"
    assert (folder / "configuration.yaml").is_file()
    assert (folder / "status.json").is_file() and (folder / "activity.log").is_file()


def test_canonical_app_initializes_without_example_packages(tmp_path):
    # Blocking the old namespaces catches indirect implementation dependencies,
    # including imports hidden in app creation or default configuration loading.
    code = """
import importlib.abc
from pathlib import Path
import sys
class RejectExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('examples', 'example_meshing'):
            raise AssertionError('Canonical dependency on examples: ' + fullname)
sys.meta_path.insert(0, RejectExamples())
from turbo_moc.apps.app import create_app, worker, workflow
from turbo_moc.geometry import geometry_2d, geometry_3d, meridional_channel
from turbo_moc.meshing import meshing_2d, meshing_3d
from turbo_moc.fluent import workflow_2d, workflow_3d
output = Path(sys.argv[1]).resolve()
assert workflow.OUTPUT == Path.cwd() / 'output'
assert not output.is_relative_to(Path(workflow.__file__).resolve().parents[2])
application = create_app(output_dir=output)
with application.server.app_context():
    assert workflow.output_directory() == output
client = application.server.test_client()
assert client.get('/_dash-layout').status_code == 200
assert client.get('/assets/mesh_edges.js').status_code == 200
assert not any(name.split('.')[0] in ('examples', 'example_meshing') for name in sys.modules)
print('canonical package boundary verified')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "generated")], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "canonical package boundary verified" in result.stdout


def test_progress_poll_survives_a_transient_windows_snapshot_lock(monkeypatch, tmp_path_factory):
    from turbo_moc.apps.app import workflow

    monkeypatch.setattr(workflow, "OUTPUT", tmp_path_factory.mktemp("poll"))
    monkeypatch.setattr(workflow, "PROCESSES", {})
    handle = {"session": "a" * 32, "job": "b" * 32}
    folder = workflow.directory(handle)
    folder.mkdir(parents=True)
    workflow.write_json(folder / "status.json", {"state": "running", "fraction": 0.5})
    (folder / "activity.log").write_text("Live MoC progress", encoding="utf-8")
    original = Path.read_text
    blocked = []

    def read(path, *args, **kwargs):
        if path.name == "status.json" and len(blocked) < 2:
            blocked.append(path)
            raise PermissionError("Windows sharing violation during replacement")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    status, activity = workflow.poll_job(handle)
    assert len(blocked) == 2
    assert status == {"state": "running", "fraction": 0.5}
    assert activity == "Live MoC progress"


def test_apps_use_separate_output_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("TURBO_MOC_OUTPUT_DIR", str(tmp_path / "unused"))
    reference = {"session": "a" * 32, "job": "b" * 32}
    first = create_app(output_dir=tmp_path / "first")
    second = create_app(output_dir=tmp_path / "second")
    for app, name in ((first, "first"), (second, "second"), (first, "first")):
        with app.server.app_context():
            assert workflow.directory(reference) == (
                tmp_path / name / reference["session"] / reference["job"])


def test_worker_recovers_output_root_from_request(tmp_path, monkeypatch):
    root = tmp_path / "custom output"
    request_path = root / ("a" * 32) / ("b" * 32) / "request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps({"config": {"workflow": {"log_level": "INFO"}}}))
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path / "wrong")
    monkeypatch.chdir(tmp_path)

    def check_root(path, request):
        assert workflow.output_directory() == root
        assert path == request_path

    monkeypatch.setattr(worker, "_run_job", check_root)
    worker.run(request_path)


def test_live_worker_and_cancellation_keep_artifacts(tmp_path, monkeypatch):
    # Run a tiny deliberately long subprocess rather than mocking termination.
    import subprocess
    import sys
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    reference = {"session": "a" * 32, "job": "b" * 32, "started": time.time()}
    folder = flow.directory(reference)
    folder.mkdir(parents=True)
    flow.write_json(folder / "status.json", {"state": "running", "message": "Native operation",
                                          "fraction": None, "started": reference["started"]})
    (folder / "previous_mesh.cgns").write_text("good artifact")
    with (folder / "activity.log").open("wb") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-c",
            "import time; print('live heartbeat',flush=True); time.sleep(30)"], stdout=stream)
    flow.PROCESSES[(reference["session"], reference["job"])] = process
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            state, log = flow.poll_job(reference)
            if "heartbeat" in log:
                break
            time.sleep(0.05)
        assert "live heartbeat" in log and state["state"] == "running"
        flow.cancel_job(reference)
        state, _ = flow.poll_job(reference)
        assert state["state"] == "cancelled"
        assert (folder / "previous_mesh.cgns").read_text() == "good artifact"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        flow.PROCESSES.pop((reference["session"], reference["job"]), None)



def test_atomic_publication_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("temporary sharing violation")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    flow.write_json(tmp_path / "status.json", {"state": "complete"})
    assert len(attempts) == 3
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"
