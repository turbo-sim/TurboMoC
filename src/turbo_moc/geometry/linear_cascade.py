"""Physical planar cascades using the same blade-to-blade mesh as annular rows.

The numerical plane retains pitchwise q_x and downstream-negative q_y.
Unlike the annular conformal chart, both coordinates are physical lengths:
X = -q_y, Y = q_x, Z = eta * blade_height. No reference radius is needed.
"""

from dataclasses import dataclass

import numpy as np

from . import geometry_2d as planar
from .meridional_channel import scale_blade, scale_flow_domain, validate_machine


@dataclass(frozen=True)
class LinearCascadeDefinition:
    meridional_chord: float
    blade_height: float
    pitch: float

    def to_dict(self):
        return {
            "type": "linear",
            "meridional_chord_mm": self.meridional_chord,
            "blade_height_mm": self.blade_height,
            "pitch_mm": self.pitch,
        }


class LinearCascadeMapping:
    """Identity blade sections extruded between two parallel span endwalls."""

    is_linear = True
    # (q_x,q_y) -> (Y,-X) preserves planar winding; increasing eta adds +Z.
    orientation = 1

    def __init__(self, definition):
        if not isinstance(definition, LinearCascadeDefinition):
            raise TypeError("LinearCascadeMapping requires a LinearCascadeDefinition.")
        lengths = np.asarray(
            [definition.meridional_chord, definition.blade_height, definition.pitch]
        )
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("Linear cascade chord, height and derived pitch must be positive.")
        self.definition = definition
        self.reference_chord = float(definition.meridional_chord)
        self.blade_height = float(definition.blade_height)
        self.pitch = float(definition.pitch)
        self.characteristic_length = float(lengths.max())

    def section_scale(self, eta):
        """Every span station has the same physical profile and pitch."""
        return 1.0

    def map_points(self, q, eta):
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all() or not np.isfinite(eta):
            raise ValueError("Linear cascade mapping requires finite coordinates.")
        span = np.full(q.shape[:-1], eta * self.blade_height)
        return np.stack((-q[..., 1], q[..., 0], span), axis=-1)

    def meridional_points(self, u, eta, *, derivative=False):
        """Return (span height, streamwise distance) for the meridional plot."""
        u = np.asarray(u, dtype=float)
        if derivative:
            tangent = np.full_like(u, self.reference_chord)
            return np.stack((np.zeros_like(u), tangent), axis=-1)
        span = np.full_like(u, eta * self.blade_height)
        return np.stack((span, u * self.reference_chord), axis=-1)

    def periodic_transform(self):
        """Homogeneous translation from periodic_left to periodic_right."""
        transform = np.eye(4)
        transform[1, 3] = self.pitch
        return transform


def build_linear_section(nozzle, config, template):
    """Scale the fitted blade and its MoC-derived pitch together to the chord.

    Preserving pitch/chord avoids introducing an independent, unstated solidity
    choice. The sole automatic passage-placement search remains in 2D.
    """
    dimensions = config["machine"]
    validate_machine(dimensions)
    contour = planar.blade_contour(template)
    origin = contour[np.argmax(contour[:, 1])]
    scale = dimensions["meridional_chord"] / np.ptp(contour[:, 1])
    passage = config["passage"]
    domain = planar.generate_flow_domain(
        template,
        nozzle["wall_final"],
        inlet_chords=passage["inlet_chords"],
        outlet_chords=passage["outlet_chords"],
        smooth=passage.get("smooth", {}),
        domain_shift_mode=passage.get("domain_shift_mode", "auto"),
        domain_shift=passage.get("domain_shift", 0),
    )
    blade = scale_blade(template, scale, origin)
    curves = scale_flow_domain(domain, scale, origin)
    curves["blade_curve"] = planar.xy(blade["blade_curve"])
    curves["trailing_edge"] = planar.xy(blade["trailing_edge"])
    mapping = LinearCascadeMapping(LinearCascadeDefinition(
        float(dimensions["meridional_chord"]),
        float(dimensions["blade_height"]),
        float(blade["pitch"]),
    ))
    return mapping, blade, curves
