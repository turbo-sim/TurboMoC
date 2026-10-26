"""Build and validate a barotropic model at the MoC inlet state."""

import json
import re
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from turbo_moc.reporting import display_path, get_logger, summary
from turbo_moc.configuration.fluid_inputs import solver_parameters

logger = get_logger("fluent.barotropy")

# Thermodynamic and transport properties exported as Fluent named expressions.
PROPERTY_VARIABLES = (
    "density",
    "viscosity",
    "speed_of_sound",
    "void_fraction",
    "vapor_quality",
)
POSITIVE_PROPERTIES = ("density", "viscosity", "speed_of_sound")
MAX_FIT_RELATIVE_ERROR = 0.01


class BarotropicClosure:
    """The exported Fluent expressions evaluated in Python, plus the isentropic head H(p) = ∫ dp/ρ.

    Post-processing then uses exactly the density and speed of sound the solver
    used, instead of refitting or calling the equation of state again.
    """

    def __init__(self, expressions, p_min, p_max, samples=4000):
        self.sources = {name: re.sub(r"\s*\[[^\]]*\]", "", text).replace("IF(", "where(")
                        for name, text in expressions.items()}
        grid = np.geomspace(0.8 * p_min, 1.05 * p_max, samples)
        inverse = 1.0 / self.density(grid)
        self._p = grid
        self._head = np.concatenate([[0.0], np.cumsum(0.5 * (inverse[1:] + inverse[:-1]) * np.diff(grid))])

    @classmethod
    def from_file(cls, expressions_file, p_min, p_max):
        expressions = json.loads(Path(expressions_file).read_text(encoding="utf-8"))
        return cls(expressions, p_min, p_max)

    def evaluate(self, name, pressure):
        pressure = np.asarray(pressure, dtype=float)
        return np.asarray(eval(self.sources[name], {"__builtins__": {}, "where": np.where, "exp": np.exp,
                                                    "AbsolutePressure": pressure}), dtype=float)

    def density(self, pressure):
        return self.evaluate("barotropic_density", pressure)

    def speed_of_sound(self, pressure):
        return self.evaluate("barotropic_speed_of_sound", pressure)

    def head(self, pressure):
        return np.interp(pressure, self._p, self._head)

    def pressure_from_head(self, head):
        return np.interp(head, self._head, self._p)


def positive(value, name):
    """Reject invalid physical/settings values before starting external tools."""
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def resolve_inlet_state(barotropy, nozzle_parameters, nozzle):
    """Preserve the quality actually used by MoC, including its optional Q0 nudge.

    Saturated pressure/temperature alone cannot identify a two-phase inlet.
    Resolve pressure/quality first, then pass its density to BarotropicModel.
    """
    fluid = barotropy.Fluid(
        nozzle_parameters["fluid_name"],
        backend=nozzle_parameters.get("backend", "HEOS"),
    )
    actual = nozzle.get("meta", {})
    # MoC metadata is authoritative, including any quality nudge. Resolve the
    # editable input pair only when it must supply missing stagnation values.
    parameters = nozzle_parameters
    actual_quality = actual.get("Q0")
    actual_two_phase = actual_quality is not None and np.isfinite(actual_quality)
    if not {"P0", "Q0"}.issubset(actual) or (not actual_two_phase and "T0" not in actual):
        parameters = solver_parameters(nozzle_parameters)
    p0 = float(actual.get("P0", parameters.get("P0", np.nan)))
    q0 = actual.get("Q0", parameters.get("Q0"))
    if q0 is not None and np.isfinite(q0):
        if not 0 <= q0 <= 1:
            raise ValueError("The MoC inlet quality must be in [0, 1].")
        state = fluid.get_state(barotropy.PQ_INPUTS, p0, float(q0))
    else:
        state = fluid.get_state(
            barotropy.PT_INPUTS,
            p0,
            positive(actual.get("T0", parameters.get("T0", np.nan)), "nozzle.T0"),
        )
        q0 = None
    return fluid, state, q0


def generate_barotropic_model(barotropy, config, nozzle, settings, output_dir):
    """Fit the inlet isentrope and export validated, usable Fluent expressions.

    The fit extends to inlet pressure / pressure_ratio independently of the CFD
    back pressure. Return expressions and inlet/outlet properties for Fluent.
    """
    logger.info("  Resolve the actual MoC inlet state.")
    fluid, inlet, quality = resolve_inlet_state(barotropy, config["nozzle"], nozzle)
    from turbo_moc.configuration.fluid_inputs import exit_pressure
    p_back = exit_pressure(config["nozzle"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # The model pressure range is independent of the MoC/CFD back pressure.
    triple_limit = float(fluid.triple_point_liquid.p)
    if p_back <= triple_limit:
        raise ValueError(
            "The outlet pressure must exceed the fluid's triple-point pressure."
        )
    p_min = float(inlet.p) / settings["pressure_ratio"]
    if p_min <= triple_limit:
        raise ValueError(
            "The barotropic exit pressure must exceed the EOS triple-point limit."
        )
    if p_min >= p_back:
        raise ValueError(
            "The barotropic minimum pressure must be below the back pressure."
        )
    model = barotropy.BarotropicModel(
        fluid_name=config["nozzle"]["fluid_name"],
        backend=config["nozzle"].get("backend", "HEOS"),
        p_in=float(inlet.p),
        rho_in=float(inlet.rho),
        p_out=p_min,
        efficiency=settings["efficiency"],
        calculation_type=settings["calculation_type"],
        **({"blending_onset": settings["blending_onset"], "blending_width": settings["blending_width"],
            "HEOS_solver": settings.get("HEOS_solver", "hybr")}
           if settings["calculation_type"] == "blending" else {}),
        ODE_solver=settings["ODE_solver"],
        ODE_tolerance=settings["ODE_tolerance"],
        polynomial_degree=settings["polynomial_degree"],
        polynomial_format="horner",
        polynomial_variables=list(PROPERTY_VARIABLES),
        output_dir=str(output_dir),
    )
    summary(
        logger,
        "Thermodynamic conditions",
        {
            "Fluid": config["nozzle"]["fluid_name"],
            "Inlet pressure": f"{inlet.p:,.1f} Pa",
            "Inlet temperature": f"{inlet.T:.6f} K",
            "Inlet vapor quality": (
                quality if quality is not None else "Single-phase inlet"
            ),
            "CFD outlet pressure": f"{p_back:,.1f} Pa",
            "Barotropic minimum pressure": f"{p_min:,.1f} Pa",
        },
    )
    logger.info("  Solve the expansion and fit the property polynomials.")
    model.solve()
    model.fit_polynomials()
    if settings["calculation_type"] == "blending":
        logger.info("  Keep barotropy's blended polynomials (equilibrium/metastable blending).")
        fit_report = {"segments": len(model.poly_fitter.poly_breakpoints) - 1, "max_relative_error": {},
                      "blending_onset": settings["blending_onset"], "blending_width": settings["blending_width"],
                      "reference": "barotropy blended equilibrium/metastable expansion"}
    else:
        logger.info("  Refine the fit and validate against independent EOS samples.")
        fit_report = fit_wide_isentrope(
            barotropy,
            model.poly_fitter,
            fluid,
            float(inlet.s),
            settings["polynomial_degree"],
            settings.get("segment_pressure_ratio", 2.0),
        )
    summary(
        logger,
        "Polynomial fit validation",
        {
            "Pressure segments": fit_report["segments"],
            **{
                f"Maximum {name.replace('_', ' ')} error": f"{error:.3%}"
                for name, error in fit_report["max_relative_error"].items()
            },
        },
    )
    logger.info("  Export the validated Fluent property expressions.")
    model.export_expressions_fluent(output_dir=str(output_dir))
    expressions = barotropy.read_expression_file(
        str(output_dir / "fluent_expressions.txt")
    )
    missing = {f"barotropic_{var}" for var in PROPERTY_VARIABLES} - expressions.keys()
    if missing or any(not definition for definition in expressions.values()):
        raise RuntimeError(f"Incomplete barotropic expressions: {sorted(missing)}")
    # Density must increase with absolute pressure for positive compressibility.
    # Viscosity stays at its endpoint outside the fit, rather than
    # using barotropy's exponential branch which underflows for viscosity.
    mu_min, mu_max = model.poly_fitter.evaluate_polynomial(
        [p_min, float(inlet.p)], "viscosity"
    )
    expressions["barotropic_viscosity"] = bounded_viscosity_expression(
        expressions["barotropic_viscosity"], p_min, float(inlet.p), mu_min, mu_max
    )
    expression_file = output_dir / "fluent_expressions.txt"
    (output_dir / "fluent_expressions_raw.txt").write_text(
        expression_file.read_text(), encoding="utf-8"
    )
    expression_file.write_text(
        "FLUENT expressions loaded by run_fluent_case.py\n\n"
        + "".join(
            f"{name}\n{definition}\n\n" for name, definition in expressions.items()
        ),
        encoding="utf-8",
    )
    (output_dir / "fluent_expressions_safe.json").write_text(
        json.dumps(expressions, indent=2), encoding="utf-8"
    )
    (output_dir / "fit_validation.json").write_text(
        json.dumps(fit_report, indent=2), encoding="utf-8"
    )
    p = np.geomspace(p_min, float(inlet.p), 5000)
    for var in POSITIVE_PROPERTIES:
        values = model.poly_fitter.evaluate_polynomial(p, var)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise RuntimeError(
                f"The {var} polynomial has nonpositive or nonfinite values; revise the fit."
            )
        if var == "density" and np.any(np.diff(values) <= 0):
            raise RuntimeError(
                "The density fit has nonpositive compressibility; revise the polynomial fit."
            )
    if settings["save_figures"]:
        logger.info("  Export the phase diagram and property-fit figures.")
        model.poly_fitter.plot_phase_diagram(
            fluid=fluid, var_x="s", var_y="T", savefig=True, showfig=False
        )
        for var in PROPERTY_VARIABLES:
            model.poly_fitter.plot_polynomial_and_error(
                var=var, savefig=True, showfig=False
            )
    state = {
        "fluid_name": config["nozzle"]["fluid_name"],
        "P0": float(inlet.p),
        "T0": float(inlet.T),
        "rho0": float(inlet.rho),
        "mu0": float(mu_max),
        "s0": float(inlet.s),
        "Q0": quality,
        "p_back": p_back,
        "p_min": p_min,
    }
    state["barotropic_exit_pressure"] = p_min
    state["rho_back"] = float(
        model.poly_fitter.evaluate_polynomial([p_back], "density")[0]
    )
    state["mu_back"] = float(
        model.poly_fitter.evaluate_polynomial([p_back], "viscosity")[0]
    )
    (output_dir / "inlet_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    logger.info("  Barotropic model saved: %s", display_path(output_dir))
    return expressions, state


def bounded_viscosity_expression(definition, p_min, p_max, mu_min, mu_max):
    """Hold transport properties at endpoints outside the thermodynamic fit."""
    return (
        f"IF(AbsolutePressure < {p_min:.17g} [Pa], {mu_min:.17g} [Pa s], "
        f"IF(AbsolutePressure > {p_max:.17g} [Pa], {mu_max:.17g} [Pa s], ({definition})))"
    )


PHASE_INSET = 1e-6
BOUNDARY_INSET = 2e-3


def _two_phase(barotropy, fluid, pressure, entropy):
    return bool(fluid.get_state(barotropy.PSmass_INPUTS, float(pressure), entropy).to_dict()["is_two_phase"])


def phase_boundaries(barotropy, fluid, entropy, p_low, p_high, samples=400):
    """Pressures where the isentrope enters or leaves the two-phase region."""
    pressures = np.geomspace(p_high, p_low, samples)
    phases = [_two_phase(barotropy, fluid, value, entropy) for value in pressures]
    result = []
    for (high, low), (upper, lower) in zip(zip(pressures[:-1], pressures[1:]), zip(phases[:-1], phases[1:])):
        if upper == lower:
            continue
        for _ in range(60):
            middle = 0.5 * (high + low)
            if _two_phase(barotropy, fluid, middle, entropy) == upper:
                high = middle
            else:
                low = middle
        result.append(0.5 * (high + low))
    return result


def _with_phase_boundaries(barotropy, fluid, entropy, breakpoints, p_in):
    """Barotropy's normalized breakpoints with exact phase boundaries replacing nearby ones, and those boundaries."""
    breakpoints = [float(value) for value in breakpoints]
    boundaries = [value / p_in for value in phase_boundaries(
        barotropy, fluid, entropy, min(breakpoints) * p_in, max(breakpoints) * p_in)]
    kept = [value for value in breakpoints
            if all(abs(value - boundary) > 1e-2 * boundary for boundary in boundaries)
            or value in (breakpoints[0], breakpoints[-1])]
    return sorted(set(kept + boundaries), reverse=True), boundaries


def fit_wide_isentrope(barotropy, fitter, fluid, entropy, degree, segment_pressure_ratio=2.0):
    """Refine barotropy's phase segments for an extended pressure range.

    Fit directly sampled equilibrium EOS states, retaining phase boundaries.
    Endpoint corrections make neighbouring polynomial values continuous.
    Independent midpoint samples check accuracy, positivity and compressibility.
    """
    # Barotropy expresses pressure as p / p_in. Preserve phase boundaries, then
    # subdivide each segment so its highest/lowest pressure ratio is at most
    # segment_pressure_ratio (2 by default; less where density changes steeply,
    # e.g. just inside the dome from the saturated liquid).
    breakpoints, boundaries = _with_phase_boundaries(barotropy, fluid, entropy, fitter.poly_breakpoints, fitter.p_in)
    edges = []
    for high, low in zip(breakpoints[:-1], breakpoints[1:]):
        count = max(1, int(np.ceil(np.log(high / low) / np.log(segment_pressure_ratio))))
        edges.extend(np.geomspace(high, low, count + 1)[:-1])
    edges.append(breakpoints[-1])
    handles = {var: [] for var in PROPERTY_VARIABLES}
    errors = {var: 0.0 for var in PROPERTY_VARIABLES}
    for high, low in zip(edges[:-1], edges[1:]):
        # A phase flash (REFPROP in particular) can land on the wrong side of
        # a boundary within a few pascals: sample further inside the segment there.
        inset_low, inset_high = ((BOUNDARY_INSET if edge in boundaries else PHASE_INSET) * (high - low)
                                 for edge in (low, high))
        normalized_pressure = np.linspace(low + inset_low, high - inset_high, max(33, 4 * degree + 1))
        states = [
            fluid.get_state(
                barotropy.PSmass_INPUTS, float(pressure * fitter.p_in), entropy
            ).to_dict()
            for pressure in normalized_pressure
        ]
        # Validate at midpoints that were not used in the fit.
        check_pressure = (normalized_pressure[:-1] + normalized_pressure[1:]) / 2
        check_states = [
            fluid.get_state(
                barotropy.PSmass_INPUTS, float(pressure * fitter.p_in), entropy
            ).to_dict()
            for pressure in check_pressure
        ]
        for variable in PROPERTY_VARIABLES:
            property_values = np.array([state[variable] for state in states])
            polynomial = Polynomial.fit(
                normalized_pressure, property_values, degree
            ).convert()
            # Correct both endpoints exactly with a linear offset. Adjacent
            # segments then agree on their common thermodynamic state.
            correction_low = property_values[0] - polynomial(low)
            correction_high = property_values[-1] - polynomial(high)
            correction_slope = (correction_high - correction_low) / (high - low)
            polynomial += Polynomial(
                [correction_low - low * correction_slope, correction_slope]
            )

            reference_values = np.array([state[variable] for state in check_states])
            fitted_values = polynomial(check_pressure)
            relative_error = float(
                np.max(
                    np.abs(fitted_values - reference_values)
                    / np.maximum(np.abs(reference_values), 1e-8)
                )
            )
            errors[variable] = max(errors[variable], relative_error)
            if variable in POSITIVE_PROPERTIES:
                if (
                    not np.isfinite(fitted_values).all()
                    or np.any(fitted_values <= 0)
                    or relative_error > MAX_FIT_RELATIVE_ERROR
                ):
                    raise RuntimeError(
                        f"Inaccurate {variable} fit on {low*fitter.p_in:g}..{high*fitter.p_in:g} Pa: "
                        f"{relative_error:.2%}"
                    )
            if variable == "density" and np.any(
                polynomial.deriv()(check_pressure) <= 0
            ):
                raise RuntimeError("Density fit has nonpositive compressibility.")
            handles[variable].append(polynomial)
    fitter.poly_breakpoints = edges
    fitter.poly_handles = handles
    return {
        "segments": len(edges) - 1,
        "max_relative_error": errors,
        "p_min": float(fitter.p_out),
        "p_max": float(fitter.p_in),
        "reference": "Independent equilibrium EOS samples at the MoC inlet entropy",
    }
