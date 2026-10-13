"""Focused speed conversions and reference-state checks; no blade solves."""
from copy import deepcopy
from math import cos, pi, radians, sqrt

import jaxprop as jxp
import pytest

from turbo_moc.configuration import project as projects
from turbo_moc.app import create_app
from turbo_moc.app.rotor_speed import resolve_speed, stator_reference


def test_speed_modes_resolve_to_the_same_rpm_and_blade_speed():
    radius, drop, density, mass_flow = 100., 50000., 2., 5.
    omega = 1000 * 2 * pi / 60
    ratio = omega * radius / 1000 / sqrt(2 * drop)
    specific = omega * sqrt(mass_flow / density) / drop ** .75
    for kind, value in (("actual", 1000.), ("blade_velocity_ratio", ratio), ("specific", specific)):
        speed = resolve_speed({"type": kind, "value": value}, radius,
            enthalpy_drop=drop, exit_density=density, mass_flow=mass_flow)
        assert speed["rpm"] == pytest.approx(1000.)
        assert speed["U"] == pytest.approx(omega * radius / 1000)


def test_legacy_rpm_migrates_without_mutating_the_saved_case():
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    coupling = project["rows"][1]["geometry"]["inlet_coupling"]
    coupling.pop("rotational_speed")
    coupling["rpm"] = -1234.
    before = deepcopy(project)
    migrated = projects.validate_project(projects.normalize_project(project))
    assert migrated["rows"][1]["geometry"]["inlet_coupling"]["rotational_speed"] == {
        "type": "actual", "value": -1234.}
    assert project == before


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_reference_uses_stator_back_pressure_and_exit_continuity(family):
    project = projects.default_project(family)
    project["machine"]["fluid_name"] = "Air"
    config = projects.row_configuration(project, "stator_1")
    config["method_of_characteristics"].update(
        inlet_property_pair="PT_INPUTS", inlet_property_1=500000., inlet_property_2=400., p_back=100000.)
    fluid = jxp.Fluid("Air")
    inlet = fluid.get_state(jxp.PT_INPUTS, 500000., 400.)
    outlet = fluid.get_state(jxp.PSmass_INPUTS, 100000., inlet.s)
    nozzle = {"wall_final": {"rho": [float(outlet.rho)], "T": [float(outlet.T)], "M": [1.4]}}
    result = stator_reference(config, nozzle, 65.)
    assert result["enthalpy_drop"] == pytest.approx(float(inlet.h - outlet.h))
    assert result["exit_density"] == pytest.approx(float(outlet.rho))
    channel = config["meridional_channel"]
    area = (pi * ((channel["hub_radius"] + channel["blade_height"]) ** 2 - channel["hub_radius"] ** 2)
            if family == "axial" else 2 * pi * channel["outlet_radius"] * channel["blade_height"]) * 1e-6
    assert result["mass_flow"] == pytest.approx(float(outlet.rho) * 1.4 * float(outlet.a) * cos(radians(65.)) * area)


def test_speed_value_label_matches_the_selected_mode():
    app = create_app()
    label = next(value["callback"].__wrapped__ for key, value in app.callback_map.items()
                 if key.startswith('{"row":["MATCH"],"type":"rotor-speed-label"}.children'))
    assert "rpm" in label("actual")
    assert "U/c₀" in label("blade_velocity_ratio")
    assert "Specific speed" in label("specific")
