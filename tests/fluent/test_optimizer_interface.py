import pytest


def test_optimizer_report_contains_flow_and_isentropic_efficiency():
    from turbo_moc.fluent.common import optimizer_report

    report = optimizer_report(
        {
            "mass-inlet": -2.0,
            "mass-outlet": 1.99,
            "flow-angle-inlet": 0.0,
            "flow-angle-outlet": 65.0,
            "velocity-magnitude-outlet": 40.0,
        },
        {
            "isentropic_reference": {
                "inlet_enthalpy": 1000.0,
                "inlet_entropy": 10.0,
                "outlet_pressure": 100000.0,
                "outlet_enthalpy_isentropic": 200.0,
                "velocity_out_isentropic": 50.0,
            }
        },
    )

    assert report["mass_flow_inlet"] == pytest.approx(2.0)
    assert report["flow_angle_outlet_deg"] == pytest.approx(65.0)
    assert report["objective_isentropic_efficiency"] == pytest.approx(0.64)
    assert report["isentropic_efficiency_status"] == "available"


def test_optimizer_report_explicitly_marks_missing_velocity_diagnostics():
    from turbo_moc.fluent.common import optimizer_report

    report = optimizer_report({}, {})

    assert report["isentropic_efficiency_status"] == "unavailable"
    assert "velocity diagnostics" in report["isentropic_efficiency_error"]
