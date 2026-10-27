"""Folders and files shared by the droplet-impingement stage scripts; folders can be overridden with environment variables.

MESH_DIR   TurboGrid stator.gtm / rotor.gtm; 01_setup_stage_case.py converts them to .def next to them (not in git)
RUN_DIR    Fluent case, data, transcripts and report data; keep it outside cloud-synced folders (sync locks Fluent files)
REPORT_DIR report.md/.html/.pdf and figures
"""
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _folder(variable, default):
    return Path(os.environ.get(variable, default)).expanduser()


MESH_DIR = _folder("DROPLET_MESH_DIR", HERE / "meshes")
RUN_DIR = _folder("DROPLET_RUN_DIR", Path.home() / "Documents" / "fluent_runs" / "paper3_stage")
REPORT_DIR = _folder("DROPLET_REPORT_DIR", RUN_DIR / "report")
CASE_NAME = "stage_fluent.cas.h5"

# Design inputs, shipped with the example
STATOR_CONFIG = HERE / "inputs" / "stator_moc_config.yaml"   # TurboMoC stator design: inlet state for the barotropic model
ROTOR_DESIGN = HERE / "inputs" / "rotor_turbodash.yaml"      # TurboDash meanline design, compared in the report


def cfx5gtmconv():
    """The TurboGrid-to-CFX-definition converter: on PATH, else in the newest Ansys installation (AWP_ROOT<version>)."""
    found = shutil.which("cfx5gtmconv")
    if found:
        return found
    for variable in sorted((name for name in os.environ if name.startswith("AWP_ROOT")), reverse=True):
        for product in ("CFX", "TurboGrid", "CFD-Post"):
            candidate = Path(os.environ[variable]) / product / "bin" / "cfx5gtmconv.exe"
            if candidate.exists():
                return str(candidate)
    raise FileNotFoundError("cfx5gtmconv not found: add the Ansys CFX bin folder to PATH.")
