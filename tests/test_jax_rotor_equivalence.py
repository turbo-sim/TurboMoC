"""The optional JAX rotor (turbo_moc.jax_rotor) must stay identical to the NumPy reference (turbo_moc.core).

Fails when core.vortex_blade / core.rotor_starting change and the JAX copy does not (or the reverse). Four inlet
states that exercise the different paths: two-phase (P0, Q0) as in the TurboExpo stages, superheated vapour,
liquid 0.02 K below saturation (coupling-nudge phase), and a wet-to-dry Novec649 isentrope where M(V) is not
monotone (first-root rule of the Mach inversion). Full matrix: worktree TurboMoC_jax, tests/jax_rotor/VERIFICATION.md.
"""
import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from turbo_moc.core import rotor_starting as ref_starting  # noqa: E402
from turbo_moc.core import vortex_blade as ref_vb  # noqa: E402
from turbo_moc.core.thermo_characteristics import get_fluid_manager  # noqa: E402
from turbo_moc.jax_rotor import vortex_blade as jax_vb  # noqa: E402
from turbo_moc.jax_rotor.isentrope import JaxIsentrope  # noqa: E402
from turbo_moc.jax_rotor.starting import StartingModel  # noqa: E402

TOL_GEOMETRY = 1e-10  # relative to the chord (measured: 4e-14)
TOL_GRADIENT = 1e-5   # AD vs central difference of the reference (measured: 1.7e-8)


def _state(fluid, kind):
    pc, pt = CP.PropsSI("pcrit", fluid), CP.PropsSI("ptriple", fluid)
    P0 = max(0.03 * pc, 3.0 * pt) if kind != "wet-to-dry" else 0.3 * pc
    tsat = CP.PropsSI("T", "P", P0, "Q", 0, fluid)
    return {"two-phase": dict(P0_rel=P0, T0_rel=tsat, Q0_rel=0.5),
            "vapour": dict(P0_rel=P0, T0_rel=tsat + 20.0, Q0_rel=float("nan")),
            "liquid": dict(P0_rel=P0, T0_rel=tsat - 0.02, Q0_rel=float("nan")),
            "wet-to-dry": dict(P0_rel=P0, T0_rel=tsat, Q0_rel=0.5)}[kind]


def _design(manager):
    """Impulse blade with elliptic edges inside the table's Mach range (the TurboExpo edge recipe)."""
    V = np.asarray(manager._nu_V_table)
    M = V / manager._spline_a_of_V(V)
    m_in = max(1.3, M[0] + 0.25)
    assert m_in + 0.35 < M.max() - 0.02, "test state has no usable Mach range"
    return dict(M_inlet=m_in, M_outlet=m_in, M_lower=m_in - 0.15, M_upper=m_in + 0.35, beta_inlet=55.0, num_points=60,
                le_te_ratio=59.088, le_ellipse_ratio=3.2, te_ellipse_ratio=5.0)


CASES = [("Cyclopentane", "two-phase"), ("Cyclopentane", "vapour"), ("Water", "liquid"), ("Novec649", "wet-to-dry")]


@pytest.fixture(scope="module", params=CASES, ids=[f"{f}-{k}" for f, k in CASES])
def case(request):
    fluid, kind = request.param
    state = _state(fluid, kind)
    manager = get_fluid_manager(fluid, "HEOS", state["P0_rel"], state["T0_rel"], state["Q0_rel"])
    design = _design(manager)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the wet-to-dry state warns about the ambiguous inversion, by design
        reference = ref_vb.design_rotor_vortex_blade(fluid_name=fluid, **state, **design)
    return fluid, state, design, JaxIsentrope.from_manager(manager), reference


def test_blade_geometry_matches_reference(case):
    fluid, state, design, iso, reference = case
    blade = jax_vb.design_rotor_vortex_blade(iso, **design)
    chord = abs(reference["chord"])
    for group in ("pressure", "suction", "blade"):
        for axis in ("x", "y"):
            err = np.max(np.abs(np.asarray(blade[group][axis]) - np.asarray(reference[group][axis]))) / chord
            assert err < TOL_GEOMETRY, f"{group}.{axis}: {err:.1e}"
    for key in ("pitch", "vortex_pitch", "solidity", "beta_outlet"):
        assert abs(float(blade[key]) - reference[key]) <= TOL_GEOMETRY * max(abs(reference[key]), 1.0), key


def test_starting_check_matches_reference(case):
    fluid, state, design, iso, reference = case
    f, p0, _ = jax_vb.make_design_function(iso, design)
    check = StartingModel(fluid, **state).check(f(p0), p0["M_inlet"], p0["beta_inlet"])
    expected = ref_starting.starting_check(dict(reference, **state))
    for key in ("throat_ratio", "kantrowitz_ratio", "isentropic_ratio", "margin"):
        assert abs(float(check[key]) - expected[key]) < 1e-9 * abs(expected[key]), key
    assert bool(check["starts"]) == expected["starts"]


def test_gradient_matches_reference_differences(case):
    """d pitch / d M_upper and d M_lower (smooth outputs): AD vs central differences of the NumPy reference."""
    fluid, state, design, iso, _ = case
    f, p0, _ = jax_vb.make_design_function(iso, design)
    for var in ("M_upper", "M_lower"):
        ad = float(jax.grad(lambda x: f({**p0, var: x})["pitch"])(p0[var]))
        h = 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            up = ref_vb.design_rotor_vortex_blade(fluid_name=fluid, **state, **{**design, var: design[var] + h})
            down = ref_vb.design_rotor_vortex_blade(fluid_name=fluid, **state, **{**design, var: design[var] - h})
        fd = (up["pitch"] - down["pitch"]) / (2 * h)
        assert abs(ad - fd) < TOL_GRADIENT * max(abs(fd), 1e-6), f"{var}: AD {ad} vs FD {fd}"


@pytest.mark.parametrize("ratio", ["kantrowitz_ratio", "isentropic_ratio"])
def test_ratio_derivative_matches_reference(ratio):
    """d/dM of the Kantrowitz and isentropic ratios (spline on the reference values) vs a Richardson-extrapolated
    central difference of the reference (steps 1e-3, 2e-3: truncation O(h^4), root-solve noise ~1e-8)."""
    state = _state("Cyclopentane", "two-phase")
    model = StartingModel("Cyclopentane", **state)
    reference = getattr(ref_starting, ratio)
    for mach in (1.3, 1.7):
        ad = float(jax.grad(getattr(model, ratio))(mach))
        d = lambda h: (reference(model.isentrope, mach + h) - reference(model.isentrope, mach - h)) / (2 * h)
        richardson = (4 * d(1e-3) - d(2e-3)) / 3
        assert abs(ad - richardson) < 1e-6 * abs(richardson), f"{ratio} at M {mach}: {ad} vs {richardson}"
