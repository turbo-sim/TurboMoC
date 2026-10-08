"""
turbo_moc.geometry.sizing -- couple a reference 2D cascade stator blade
(parametrize_stator_blade/_semi's output) to a real annulus + mass-flow
target, producing an actual sized stator (r_hub, r_shroud, N_blades, and a
rescaled blade geometry).

Two consistency equations, in this module's own mm/m^3/kg/s unit convention
(matching parametrize_stator_blade's "units": "mm" and the existing cascade
mdot_choked calc in app.py):

    pitch:      k * pitch_ref = 2*pi*r_hub / N_blades
    mass flow:  mdot = rho_star * a_star * (k * throat_ref) * H * N_blades * 1e-6

rho_star/a_star come from the ORIGINAL Phase 1 nozzle solve's throat (M=1)
point (wall_final index 0), via compute_throat_sonic_state -- never from a
fresh design_nozzle() call at a different scale (the MOC solver is not
numerically scale-invariant; rescaling must be pure post-hoc multiplication
of the already-solved reference geometry, see rescale_stator_blade).
"""

import numpy as np
import jaxprop as jxp

__all__ = [
    "compute_throat_sonic_state",
    "rescale_stator_blade",
    "size_stator_mode_a",
    "size_stator_mode_b",
    "rescale_rotor_blade",
    "size_rotor_from_pitch",
]

# Point-array fields holding length data (mm) -- multiply both x and y by k.
_LENGTH_POINT_KEYS = ("suction", "pressure", "blade_curve", "trailing_edge")
# Scalar fields holding length data (mm) -- multiply by k.
_LENGTH_SCALAR_KEYS = (
    "pitch", "throat_opening", "inlet_opening",
    "axial_chord_convergent", "axial_chord_divergent", "r_trailing",
)
# Left untouched: knots (normalized B-spline parameter), metal_angle_in/out
# (degrees), mirror (bool), units (string), n_cp/variant if present.


def compute_throat_sonic_state(nozzle_result: dict) -> tuple[float, float]:
    """(rho_star, a_star) at the Phase 1 nozzle's throat/sonic point
    (wall_final index 0), using the design's own fluid_name + local
    (rho, T). DmassT_INPUTS, not PT_INPUTS: the throat state can sit
    arbitrarily close to the saturation curve (e.g. wet-steam/two-phase
    designs), where (p, T) alone is degenerate -- CoolProp raises
    "Saturation pressure ... is within 1e-4% of given p" because phase
    can't be resolved from pressure and temperature at saturation.
    rho and T together are never degenerate this way."""
    wall = nozzle_result["wall_final"]
    rho_star = float(wall["rho"][0])
    T_star = float(wall["T"][0])
    fluid = jxp.Fluid(nozzle_result["fluid_name"])
    a_star = float(fluid.get_state(jxp.DmassT_INPUTS, rho_star, T_star).a)
    return rho_star, a_star


def rescale_stator_blade(blade_data: dict, k: float) -> dict:
    """New blade dict (does not mutate input) with every length field
    (point arrays + scalar lengths + control_points) multiplied by k;
    angles/counts/booleans/strings/knots are carried over unchanged. Pure
    post-hoc multiplication of an already-solved reference shape -- never
    re-runs parametrize_stator_blade at a different scale."""
    new_data = dict(blade_data)
    for key in _LENGTH_POINT_KEYS:
        pts = blade_data.get(key)
        if pts is not None:
            new_data[key] = {
                "x": (np.asarray(pts["x"], dtype=float) * k).tolist(),
                "y": (np.asarray(pts["y"], dtype=float) * k).tolist(),
            }
    for key in _LENGTH_SCALAR_KEYS:
        value = blade_data.get(key)
        if value is not None:
            new_data[key] = float(value) * k
    control_points = blade_data.get("control_points")
    if control_points is not None:
        new_data["control_points"] = (np.asarray(control_points, dtype=float) * k).tolist()
    new_data["sizing_k"] = float(k)
    return new_data


def size_stator_mode_a(blade_data: dict, nozzle_result: dict,
                        mdot: float, n_blades: int, r_hub: float) -> dict:
    """Case A ('cycle-level'): mdot, n_blades, r_hub given; k and H derived
    in closed form from pitch + mass-flow consistency. No iteration, no
    re-invocation of design_nozzle at a different scale."""
    pitch_ref = float(blade_data["pitch"])
    throat_ref = float(blade_data["throat_opening"])
    rho_star, a_star = compute_throat_sonic_state(nozzle_result)

    k = (2.0 * np.pi * r_hub / n_blades) / pitch_ref
    H = mdot / (rho_star * a_star * k * throat_ref * n_blades * 1e-6)
    r_shroud = r_hub + H

    return {
        "mode": "A", "k": float(k), "r_hub": float(r_hub), "r_shroud": float(r_shroud),
        "H": float(H), "n_blades": int(n_blades), "mdot": float(mdot),
        "rho_star": rho_star, "a_star": a_star,
        "scaled_blade": rescale_stator_blade(blade_data, k),
    }


def size_stator_mode_b(blade_data: dict, nozzle_result: dict,
                        r_hub: float, H: float, n_blades: int, mdot: float) -> dict:
    """Case B ('full meanline'): r_hub, H, n_blades, mdot ALL given (e.g.
    from a turbodash YAML sizing). Computes k two independent ways and
    reports their mismatch as a diagnostic -- a large residual means the 2D
    blade SHAPE (metal_angle_out/solidity) needs redesigning, not just
    rescaling. scaled_blade is built from k_pitch: pitch consistency is the
    hard geometric constraint (N_blades copies of the blade must tile the
    annulus exactly), so it is treated as ground truth for the exported
    shape; the mass-flow side is the diagnostic that tells you whether that
    shape is also aerodynamically consistent."""
    pitch_ref = float(blade_data["pitch"])
    throat_ref = float(blade_data["throat_opening"])
    rho_star, a_star = compute_throat_sonic_state(nozzle_result)

    k_pitch = (2.0 * np.pi * r_hub / n_blades) / pitch_ref
    k_massflow = mdot / (rho_star * a_star * throat_ref * H * n_blades * 1e-6)
    residual = (k_pitch - k_massflow) / k_pitch

    return {
        "mode": "B", "k_pitch": float(k_pitch), "k_massflow": float(k_massflow),
        "residual": float(residual), "r_hub": float(r_hub), "r_shroud": float(r_hub + H),
        "H": float(H), "n_blades": int(n_blades), "mdot": float(mdot),
        "rho_star": rho_star, "a_star": a_star,
        "scaled_blade": rescale_stator_blade(blade_data, k_pitch),
    }


def rescale_rotor_blade(rotor_data: dict, k: float) -> dict:
    """Pure post-hoc multiplication of design_rotor_vortex_blade's output
    by k -- only blade.x/y, pitch, chord: the fields the 3D wrap/export
    pipeline (wrap_blade_annular's source="rotor" branch reads ONLY
    blade_data["blade"]) and this module's own results table actually use.
    The rotor's other returned substructures (pressure/suction/mach_lines/
    blade_lower/upper) are left out entirely -- not needed downstream, and
    not meaningfully rescaled here. Doesn't mutate the input."""
    blade = rotor_data["blade"]
    return {
        "blade": {
            "x": [v * k for v in blade["x"]],
            "y": [v * k for v in blade["y"]],
        },
        "pitch": float(rotor_data["pitch"]) * k,
        "chord": float(rotor_data["chord"]) * k,
        "beta_inlet": rotor_data["beta_inlet"], "beta_outlet": rotor_data["beta_outlet"],
        "units": "mm", "sizing_k": float(k),
    }


def size_rotor_from_pitch(rotor_data: dict, r_hub: float, r_shroud: float, n_blades: int) -> dict:
    """k derived from pitch consistency alone -- unlike the stator, no
    mass-flow equation is available at the 2D-cascade level for this
    tool's rotor method (design_rotor_vortex_blade has no mdot semantics).
    r_hub/r_shroud are expected to come from the STATOR's own sizing
    result (size_stator_mode_a/b): same annulus, axial stage, one
    through-flow passage -- n_blades is the rotor's own free choice,
    generally different from the stator's N."""
    pitch_ref = float(rotor_data["pitch"])
    k = (2.0 * np.pi * r_hub / n_blades) / pitch_ref
    return {
        "k": float(k), "r_hub": float(r_hub), "r_shroud": float(r_shroud),
        "n_blades": int(n_blades), "pitch_scaled": pitch_ref * k,
        "scaled_blade": rescale_rotor_blade(rotor_data, k),
    }
