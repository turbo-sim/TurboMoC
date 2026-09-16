"""
Internal helper: a cached jxp.Fluid per fluid name, for T-s diagram
plotting. Shared by turbo_moc.plotting_mpl and turbo_moc.plotting_plotly so the fluid
isn't reconstructed and the saturation line isn't recomputed on every
single plot call -- jaxprop's compute_saturation_line does real
fluid-property evaluations (~2s for a 60-point dome), not free, and the
dome only depends on the fluid, not on the design. jxp.Fluid itself caches
its saturation line internally (self.sat_liq/self.sat_vap) once computed,
so reusing the same Fluid instance across calls is what actually avoids
the recompute -- get_fluid() below is that cache.

turbo_moc.plotting_mpl draws directly with the cached Fluid's own
plot_phase_diagram(...) (jaxprop's native T-s/phase-diagram plotter, same
one used across the rest of this codebase's jaxprop-based scripts) instead
of hand-rolled dome lines. turbo_moc.plotting_plotly can't call into a
matplotlib-drawing method, so get_saturation_dome() below still hands it
plain saturation-line coordinates (computed via the same cached Fluid).
"""

import jaxprop as jxp
from jaxprop.coolprop.fluid_properties import compute_saturation_line

_fluid_cache = {}


def get_fluid(fluid_name):
    if fluid_name not in _fluid_cache:
        _fluid_cache[fluid_name] = jxp.Fluid(fluid_name)
    return _fluid_cache[fluid_name]


def get_saturation_dome(fluid_name, n_points=60):
    fluid = get_fluid(fluid_name)
    if fluid.sat_liq is None or fluid.sat_vap is None:
        fluid.sat_liq, fluid.sat_vap = compute_saturation_line(fluid, N=n_points)
    liq, vap = fluid.sat_liq, fluid.sat_vap
    return {
        "liq_s": [float(v) for v in liq.s], "liq_T": [float(v) for v in liq.T],
        "vap_s": [float(v) for v in vap.s], "vap_T": [float(v) for v in vap.T],
        "crit_s": float(fluid.critical_point.s), "crit_T": float(fluid.critical_point.T),
    }
