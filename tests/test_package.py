"""Basic checks that the installed package exposes its public interface."""

import turbo_moc


def test_package_version_and_public_api():
    assert turbo_moc.__version__

    expected_names = {
        "FluidManager",
        "InternalPoint",
        "ConventionalSolver",
        "MOCSolverMLN",
        "design_nozzle",
        "list_supported_fluids",
        "NozzleDesignError",
        "mpl",
        "plotly",
    }
    assert expected_names <= set(turbo_moc.__all__)
    assert all(hasattr(turbo_moc, name) for name in expected_names)


def test_supported_fluids_are_available():
    fluids = turbo_moc.list_supported_fluids()
    assert fluids
    assert any(name.lower() == "nitrogen" for name in fluids)
