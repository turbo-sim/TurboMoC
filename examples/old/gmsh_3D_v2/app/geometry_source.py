"""One reference placement and a general hub/shroud conformal mapping.

V2 deliberately changes passage placement: optimize at eta=0.5 once, carry
that reference-plane translation across span, and preserve angular pitch.
There are no axial/radial branches in section construction or visualization.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from examples.gmsh import geometry_source as planar
from examples.gmsh_3D.geometry_source import (
    Channel, GeometryDefinition, corresponding_boundary, geometry_from_machine,
    meridional_line, scale_blade, span_stations, validate_machine,
)
from examples.gmsh_3D import geometry_source as original
from turbo_moc.geometry import parametrize_stator_blade
from .configuration import to_shared


@dataclass
class Design:
    """Canonical reference blade and passage, before any mesh exists."""
    channel: Channel
    blade: dict
    curves: dict

    def section(self, eta):
        """Scale shape/chord while retaining pitch and the reference translation."""
        scale = self.channel.section_scale(float(eta))
        if np.isclose(scale, 1, rtol=1e-12, atol=1e-12):
            return self.blade, self.curves
        blade = scale_blade(self.blade, scale, np.zeros(2))
        blade["pitch"] = self.channel.pitch
        shift = np.array([self.curves["shift_mm"], 0.0])
        curves = dict(self.curves)
        # Unshift the centerline before scaling, then restore the ONE physical
        # reference-plane shift. Scaling both periodic sides directly would
        # change pitch and break rotational conformity.
        for name in (
            "passage_reference", "convergent", "divergent",
            "inlet_extension", "outlet_extension", "inlet_control_points",
            "outlet_control_points", "inlet_spline", "outlet_spline",
            "convergent_spline",
        ):
            value = self.curves.get(name)
            if isinstance(value, BSpline):
                curves[name] = BSpline(value.t, (value.c - shift) * scale + shift, value.k)
            elif isinstance(value, np.ndarray):
                curves[name] = (value - shift) * scale + shift
        center = curves["passage_reference"]
        curves["periodic_left"] = center - [self.channel.pitch / 2, 0]
        curves["periodic_right"] = center + [self.channel.pitch / 2, 0]
        curves["inlet"] = np.array([curves["periodic_left"][0], curves["periodic_right"][0]])
        curves["outlet"] = np.array([curves["periodic_left"][-1], curves["periodic_right"][-1]])
        curves["blade_curve"] = planar.xy(blade["blade_curve"])
        curves["trailing_edge"] = planar.xy(blade["trailing_edge"])
        require_containment(blade, curves)
        return blade, curves


def require_containment(blade, curves):
    """Cheap full-contour containment; volume generation also checks cell quality."""
    contour = planar.blade_contour(blade)
    center = curves["passage_reference"][::-1]
    center_x = np.interp(contour[:, 1], center[:, 1], center[:, 0])
    margin = blade["pitch"] / 2 - np.abs(contour[:, 0] - center_x)
    if margin.min() <= 0:
        raise ValueError(
            "A span section does not fit the fixed reference passage placement. "
            "Reduce blade count/chord/taper or adjust the reference shift."
        )
    return float(margin.min())


def build_design(nozzle, config, *, definition=None, template=None):
    """Fit once and perform the sole auto-centering search at the reference span."""
    channel = Channel(definition or geometry_from_machine(config["meridional_channel"]))
    wall = nozzle["wall_final"]
    if template is None:
        template = parametrize_stator_blade(wall["x"], wall["y"], **config["blade_geometry"])
    blade, curves = original.build_section(nozzle, to_shared(config), channel, template=template)
    design = Design(channel, blade, curves)
    # Endpoint checks provide a quick early error; every sweep station is checked
    # again during meshing. This does not claim a proof over continuous span.
    design.section(0.0)
    design.section(1.0)
    return design


def encode(value):
    """JSON-safe scientific artifact format, without loading arbitrary pickle."""
    if isinstance(value, BSpline):
        return {"spline": {"t": value.t.tolist(), "c": value.c.tolist(), "k": value.k}}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"spline"}:
            item = value["spline"]
            return BSpline(item["t"], item["c"], item["k"])
        if set(value) == {"array"}:
            return np.asarray(value["array"], dtype=float)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def design_data(design):
    return {
        "geometry": design.channel.definition.to_dict(),
        "blade": encode(design.blade), "curves": encode(design.curves),
    }


def load_design(data):
    definition = data["geometry"]
    walls = [
        BSpline(definition[name]["knots"], definition[name]["control_points_rz_mm"],
                definition[name]["degree"])
        for name in ("hub", "shroud")
    ]
    return Design(Channel(GeometryDefinition(*walls, definition["num_blades"])),
                  decode(data["blade"]), decode(data["curves"]))


def section_data(design):
    result = {
        name: (design.curves[name] / design.channel.reference_radius).tolist()
        for name in ("periodic_left", "periodic_right", "inlet", "outlet",
                     "passage_reference", "inlet_extension", "outlet_extension")
    }
    result["blade"] = (planar.blade_contour(design.blade) /
                       design.channel.reference_radius).tolist()
    return result


def meridional_grid(design, numerics, section=None, *, samples=45):
    """Projection of actual periodic-boundary stations swept across actual eta.

    Before a reference mesh exists, use sampled station locations. Afterwards
    every displayed streamwise station comes from its periodic_left mesh nodes.
    This is a meridional grid projection, not a second independent CFD mesh.
    """
    eta = span_stations(numerics)
    ref = design.curves
    if section is None:
        q = ref["periodic_left"]
        y = np.linspace(q[0, 1], q[-1, 1], samples)
        q = np.column_stack((np.interp(y, q[::-1, 1], q[::-1, 0]), y))
    else:
        ids = np.unique(section["boundaries"]["periodic_left"])
        q = section["xy"][ids]
        q = q[np.argsort(-q[:, 1])]
    rows = []
    for value in eta:
        _, target = design.section(float(value))
        moved = corresponding_boundary(q, "periodic_left", ref, target)
        u = design.channel.parameter_from_prime(-moved[:, 1] / design.channel.reference_radius,
                                                float(value))
        rows.append(design.channel.meridional_points(u, float(value)))
    rz = np.asarray(rows)
    # Mapped in-surface blade-height estimates, not physical 3D normal heights.
    heights = []
    h0 = numerics["inflation_layers"]["first_height"]
    for value in (0.0, 0.5, 1.0):
        radius = design.channel.meridional_points(np.array([0.0, 0.5, 1.0]), value)[:, 0]
        heights.extend((h0 * design.channel.section_scale(value) *
                        radius / design.channel.reference_radius).tolist())
    spacing = np.linalg.norm(np.diff(rz, axis=0), axis=2)
    return {
        "rz": rz.tolist(), "eta": eta.tolist(),
        "basis": "reference mesh boundary stations" if section is not None else "sampled geometry stations",
        "span_layers": len(eta) - 1, "streamwise_stations": len(q),
        "min_span_spacing_mm": float(spacing.min()),
        "max_span_spacing_mm": float(spacing.max()),
        "estimated_first_height_mm": [min(heights), max(heights)],
    }

