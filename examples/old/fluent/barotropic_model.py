"""Build and validate a barotropic model at the MoC inlet state."""

import json

import numpy as np
from numpy.polynomial import Polynomial

from examples.workflow_logging import display_path, get_logger, summary

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
    p0 = float(actual.get("P0", nozzle_parameters["P0"]))
    q0 = actual.get("Q0", nozzle_parameters.get("Q0"))
    if q0 is not None and np.isfinite(q0):
        if not 0 <= q0 <= 1:
            raise ValueError("The MoC inlet quality must be in [0, 1].")
        state = fluid.get_state(barotropy.PQ_INPUTS, p0, float(q0))
    else:
        state = fluid.get_state(
            barotropy.PT_INPUTS,
            p0,
            positive(nozzle_parameters.get("T0", np.nan), "nozzle.T0"),
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
    p_back = float(config["nozzle"]["p_back"])
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
    logger.info("  Refine the fit and validate against independent EOS samples.")
    fit_report = fit_wide_isentrope(
        barotropy,
        model.poly_fitter,
        fluid,
        float(inlet.s),
        settings["polynomial_degree"],
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


def fit_wide_isentrope(barotropy, fitter, fluid, entropy, degree):
    """Refine barotropy's phase segments for an extended pressure range.

    Fit directly sampled equilibrium EOS states, retaining phase boundaries.
    Endpoint corrections make neighbouring polynomial values continuous.
    Independent midpoint samples check accuracy, positivity and compressibility.
    """
    # Barotropy expresses pressure as p / p_in. Preserve phase boundaries, then
    # subdivide each segment so its highest/lowest pressure ratio is at most 2.
    # This limits polynomial oscillation across a wide thermodynamic range.
    edges = []
    for high, low in zip(fitter.poly_breakpoints[:-1], fitter.poly_breakpoints[1:]):
        count = max(1, int(np.ceil(np.log2(high / low))))
        edges.extend(np.geomspace(high, low, count + 1)[:-1])
    edges.append(fitter.poly_breakpoints[-1])
    handles = {var: [] for var in PROPERTY_VARIABLES}
    errors = {var: 0.0 for var in PROPERTY_VARIABLES}
    for high, low in zip(edges[:-1], edges[1:]):
        normalized_pressure = np.linspace(low, high, max(33, 4 * degree + 1))
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
