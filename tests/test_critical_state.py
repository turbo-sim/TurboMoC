"""Regression checks for subcooled-liquid throat calculations."""

import jaxprop as jxp
import numpy as np
import pytest

from turbo_moc.core.classes import jax_solve_critical_state


@pytest.mark.parametrize("fluid_name", ["CO2", "nitrogen"])
def test_subcooled_liquid_critical_state(fluid_name):
    fluid = jxp.FluidJAX(fluid_name)
    pressure = 20e5
    temperature = fluid.fluid.get_state(jxp.PQ_INPUTS, pressure, 0).T - 10.0
    inlet = fluid.get_state(jxp.PT_INPUTS, pressure, temperature)

    throat = jax_solve_critical_state(fluid, inlet.h, inlet.s)

    assert fluid.fluid.triple_point_liquid.p < throat.p < pressure
    assert 0 < throat.Q < 1
    assert throat.h < inlet.h
    assert np.isfinite(throat.a) and throat.a > 0
    assert float(throat.s) == pytest.approx(float(inlet.s), rel=1e-6)
