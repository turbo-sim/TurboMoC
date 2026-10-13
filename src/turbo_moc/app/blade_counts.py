"""Natural passage pitch and annular blade count, before pitch overrides."""
from math import ceil, isfinite, pi

import numpy as np

from turbo_moc.geometry import passage_2d as planar
from turbo_moc.geometry.meridional_channel import MeridionalChannel, geometry_from_machine


def native_passage(config, solution, *, template=None):
    """Extract pitch and meridional chord in the solver's own length units."""
    if "rotor_design" in config:
        pitch = float(solution["pitch"])
        chord = float(np.ptp(solution["blade"]["x"]))
    else:
        if not solution.get("converged"):
            raise ValueError("A converged stator solution is required for natural blade sizing.")
        if template is None:
            from turbo_moc.geometry.stator_blade import parametrize_stator_blade
            wall = solution["wall_final"]
            template = parametrize_stator_blade(wall["x"], wall["y"], **config["blade_geometry"])
        pitch = float(template["pitch"])
        chord = float(np.ptp(planar.blade_contour(template)[:, 1]))
    if not all(isfinite(x) and x > 0 for x in (pitch, chord)):
        raise ValueError("Natural blade sizing requires a positive native pitch and chord.")
    return {"pitch": pitch, "chord": chord}


def sizing(config, native):
    """Size at midspan using the same conformal chord as the blade adapters."""
    ratio = native["pitch"] / native["chord"]
    dimensions = config["meridional_channel"]
    if dimensions["type"] == "linear":
        return {"native": native, "pitch_chord": ratio,
                "natural_pitch_mm": ratio * dimensions["meridional_chord"],
                "natural_num_blades": None, "suggested_num_blades": None}
    mapping = MeridionalChannel(geometry_from_machine(dimensions))
    count = 2 * pi / (ratio * mapping.prime_chord(.5))
    return {"native": native, "pitch_chord": ratio,
            "natural_pitch_mm": ratio * mapping.reference_chord,
            "natural_num_blades": count, "suggested_num_blades": max(2, ceil(count))}


def from_reference(config, reference):
    """Use only the selected current solve; reuse the stator fitting cache."""
    from turbo_moc.app import workflow as flow
    solution = flow.load_nozzle(reference, config)
    template = None if "rotor_design" in config else flow.get_stator_template(config, reference, solution)
    return sizing(config, native_passage(config, solution, template=template))
