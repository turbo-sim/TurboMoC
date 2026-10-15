"""Stage report contracts that need neither Fluent nor the app pipeline."""

import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from turbo_moc.fluent.barotropic_model import BarotropicClosure
from turbo_moc.fluent.stage import analysis, design, document, report

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "examples/fluent_3D/stage"
FIELDS = analysis.FIELDS


def test_closure_evaluates_fluent_syntax_and_integrates_the_head():
    expressions = {
        "barotropic_density": "IF(AbsolutePressure >= 1e5 [Pa], 4.0 [kg m^-3], 2.0 [kg m^-3] + 0*exp(0))",
        "barotropic_speed_of_sound": "100 [m s^-1] + 0*AbsolutePressure",
    }
    closure = BarotropicClosure(expressions, p_min=2e4, p_max=2e5, samples=20001)
    assert closure.density([5e4, 1.5e5]) == pytest.approx([2.0, 4.0])
    assert closure.speed_of_sound(3e4) == pytest.approx(100.0)
    drop = closure.head(1.5e5) - closure.head(5e4)
    assert drop == pytest.approx(5e4 / 2.0 + 5e4 / 4.0, rel=1e-3)
    assert closure.pressure_from_head(closure.head(7e4)) == pytest.approx(7e4, rel=1e-6)


def test_closure_rejects_python_outside_the_expression_namespace():
    closure = BarotropicClosure({"barotropic_density": "1.0 + 0*AbsolutePressure"}, 1e4, 1e5, samples=10)
    closure.sources["barotropic_density"] = "__import__('os')"
    with pytest.raises(NameError):
        closure.density(5e4)


def test_kinematics_follow_the_design_sign_convention():
    xyz = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    swirl = -1.0
    tangential = np.array([[0.0, -30.0], [30.0, 0.0]])
    values = {"x-velocity": tangential[:, 0], "y-velocity": tangential[:, 1], "z-velocity": np.full(2, 40.0),
              "relative-x-velocity": tangential[:, 0], "relative-y-velocity": tangential[:, 1] + 0.0,
              "relative-z-velocity": np.full(2, 40.0)}
    parts = analysis.Kinematics(omega=-200.0, swirl_sign=swirl, flow_sign=1.0).components(xyz, values)
    assert parts["c_t"] == pytest.approx([30.0, 30.0])
    assert parts["u"] == pytest.approx([20.0, 20.0])
    assert parts["c"] == pytest.approx([50.0, 50.0])
    assert np.degrees(np.arctan2(parts["c_t"], parts["c_m"])) == pytest.approx([36.8699] * 2, rel=1e-4)


def uniform_outlet(n=8, rho=2.0, w=40.0, swirl=30.0, p=5e4):
    theta = np.linspace(0, 0.2, n)
    centroids = np.column_stack([0.1 * np.cos(theta), 0.1 * np.sin(theta), np.zeros(n)])
    e_theta = np.column_stack([-np.sin(theta), np.cos(theta)])
    arrays = {"s|normals": np.tile([0.0, 0.0, 1e-4], (n, 1)), "s|centroids": centroids,
              "s|face|absolute-pressure": np.full(n, p), "s|face|density": np.full(n, rho),
              "s|face|x-velocity": swirl * e_theta[:, 0], "s|face|y-velocity": swirl * e_theta[:, 1],
              "s|face|z-velocity": np.full(n, w)}
    for axis in "xyz":
        arrays[f"s|face|relative-{axis}-velocity"] = arrays[f"s|face|{axis}-velocity"]
    return arrays


def test_mass_averages_of_a_uniform_surface():
    model = BarotropicClosure({"barotropic_density": "2.0 + 0*AbsolutePressure",
                               "barotropic_speed_of_sound": "100.0 + 0*AbsolutePressure"}, 1e4, 1e5, samples=200)
    kinematics = analysis.Kinematics(omega=0.0, swirl_sign=1.0, flow_sign=1.0)
    result = analysis.mass_averages(uniform_outlet(), "s", model, kinematics, rotating=True)
    assert result["mass_flow"] == pytest.approx(8 * 2.0 * 40.0 * 1e-4)
    assert result["p"] == pytest.approx(5e4)
    assert result["M"] == pytest.approx(0.5)
    assert result["alpha"] == pytest.approx(np.degrees(np.arctan2(30.0, 40.0)))
    assert result["backflow_fraction"] == 0.0


def test_registry_is_rebuilt_from_the_latest_session(tmp_path):
    for session, row, action in (("old", "stator_1", "nozzle"), ("new", "stator_1", "nozzle"),
                                 ("new", "rotor_1", "volume")):
        job = tmp_path / session / f"{row}_{action}"
        job.mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({"reference": {"session": session, "row_id": row,
                                                                   "action": action}}))
    os.utime(tmp_path / "old" / "stator_1_nozzle" / "result.json", (1, 1))
    registry = design.registry_from_jobs(tmp_path)
    assert registry == {"stator_1": {"nozzle": {"session": "new", "row_id": "stator_1", "action": "nozzle"}},
                        "rotor_1": {"volume": {"session": "new", "row_id": "rotor_1", "action": "volume"}}}


def test_markdown_and_html_number_tables_and_figures_continuously(tmp_path):
    from PIL import Image
    (tmp_path / "figures").mkdir()
    Image.new("RGB", (400, 200), "white").save(tmp_path / "figures/a.png", dpi=(200, 200))
    table = ("table", "Caption", ["Quantity", "Value"], None, [["Power", "-1.5"]], "numeric-1")
    blocks = [("title", "Title", "Subtitle", [("Fluid", "MM")]), table,
              ("columns", [table], [("figure", ["figures/a.png"], "Residuals.")], 0.6),
              ("figure", ["figures/a.png"], "Second figure.")]
    markdown = document.render_markdown(blocks)
    assert [markdown.count(f"**Table {i}.**") for i in (1, 2)] == [1, 1]
    assert [markdown.count(f"**Figure {i}.**") for i in (1, 2)] == [1, 1]
    page = document.render_html(blocks, tmp_path)
    assert "<b>Table 2.</b>" in page and "<b>Figure 2.</b>" in page
    assert "−1.5" in page


def test_report_runner_passes_the_adjacent_yaml_and_the_plots_only_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(report, "main", lambda config, plots_only=False: calls.append((config, plots_only)))
    monkeypatch.setattr(sys, "argv", ["report_fluent_stage.py", "--plots-only"])
    runpy.run_path(str(STAGE / "report_fluent_stage.py"), run_name="__main__")
    assert calls == [(STAGE / "workflow_fluent_stage.yaml", True)]
