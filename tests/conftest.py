"""Small synthetic design shared by app and real-meshing checks."""
from pathlib import Path

import numpy as np
import pytest

@pytest.fixture
def small_design():
    from turbo_moc.configuration.row import read_configuration

    case = Path(__file__).resolve().parents[1] / "examples/apps/cases/axial_stator.yaml"
    config = read_configuration(case)
    config["meridional_channel"]["num_blades"] = 12
    config["blade_geometry"].update(r_trailing=0.1, num_baseline_points=100, num_curve_points=100, n_cp=15)
    config["flow_domain"]["inlet"]["extension_length"] = 0.5
    config["flow_domain"]["outlet"]["extension_length"] = 0.5
    config["flow_domain"]["number_of_points"] = 40
    config["mesh"]["numerics"].update(terminal_output=False, blade_size=0.7, domain_size=3,
        curvature_elements=16, smoothing_steps=1, span_layers=3, span_stretch=0.8)
    config["mesh"]["numerics"]["inflation_layers"].update(first_height=0.03, growth_ratio=1.4, thickness=0.15)
    x = np.linspace(0, 0.01, 80)
    nozzle = {"converged": True, "wall_final": {
        "x": x.tolist(), "y": (0.002 + 0.001 * (x / x[-1]) ** 2).tolist()}}
    return nozzle, config
