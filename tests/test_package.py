"""Basic checks that the installed package exposes its public interface."""

import moc


def test_package_version_and_public_api():
    assert moc.__version__

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
    assert expected_names <= set(moc.__all__)
    assert all(hasattr(moc, name) for name in expected_names)


def test_supported_fluids_are_available():
    fluids = moc.list_supported_fluids()
    assert fluids
    assert any(name.lower() == "nitrogen" for name in fluids)
