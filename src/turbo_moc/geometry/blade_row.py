"""One reference placement with an annular mapping or linear extrusion.

Reference passage placement: optimize at eta=0.5 once, carry
that reference-plane translation across span, and preserve the chosen pitch.
Axial and radial convenience dimensions share the same annular mapping.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline

from turbo_moc.geometry import passage_2d as planar
from .meridional_channel import (
    MeridionalChannel, GeometryDefinition, corresponding_boundary, geometry_from_machine,
    meridional_line, scale_blade, span_stations, validate_machine,
)
from . import meridional_channel as meridional_geometry
from .linear_cascade import LinearCascadeMapping, LinearCascadeDefinition, build_linear_section
from .flow_interfaces import apply_interfaces
from turbo_moc.geometry.stator_blade import blade_self_intersects, parametrize_stator
from turbo_moc.configuration.row import to_shared


@dataclass
class Design:
    """Canonical reference blade and passage, before any mesh exists."""
    meridional_channel: MeridionalChannel | LinearCascadeMapping
    blade: dict
    curves: dict
    assembly: dict = field(default_factory=dict)
    flow_domain: dict = field(default_factory=dict)

    def section(self, eta):
        """Scale shape/chord while retaining pitch and the reference translation."""
        scale = self.meridional_channel.section_scale(float(eta))
        if (np.isclose(scale, 1, rtol=1e-12, atol=1e-12)
                and (not self.assembly.get("interfaces") or float(eta) == .5)):
            return self.blade, self.curves
        blade = scale_blade(self.blade, scale, np.zeros(2))
        blade["pitch"] = self.meridional_channel.pitch
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
            "blade_reference", "blade_reference_spline", "blade_reference_guides",
        ):
            value = self.curves.get(name)
            if isinstance(value, BSpline):
                curves[name] = BSpline(value.t, (value.c - shift) * scale + shift, value.k)
            elif isinstance(value, np.ndarray):
                curves[name] = (value - shift) * scale + shift
        center = curves["passage_reference"]
        curves["periodic_left"] = center - [self.meridional_channel.pitch / 2, 0]
        curves["periodic_right"] = center + [self.meridional_channel.pitch / 2, 0]
        curves["inlet"] = np.array([curves["periodic_left"][0], curves["periodic_right"][0]])
        curves["outlet"] = np.array([curves["periodic_left"][-1], curves["periodic_right"][-1]])
        curves["blade_curve"] = planar.xy(blade["blade_curve"])
        curves["trailing_edge"] = planar.xy(blade["trailing_edge"])
        # A shared interface is a physical plane/cylinder, not a fixed chord
        # fraction. Rebuild its extension after the section's blade is scaled.
        curves = apply_interfaces(self.meridional_channel, blade, curves,
                                  self.assembly, self.flow_domain, float(eta))
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
    if not nozzle.get("converged"):
        raise ValueError("A converged MoC nozzle solution is required.")
    wall = nozzle["wall_final"]
    if template is None:
        template = parametrize_stator(wall["x"], wall["y"], **config["blade_geometry"])
    if blade_self_intersects(template):
        raise ValueError("The stator blade outline crosses itself (high exit angle): raise "
                         "blade_geometry.inlet_opening_ratio, e.g. to 1.6-2.")
    if config["meridional_channel"]["type"] == "linear" and definition is None:
        meridional_channel, blade, curves = build_linear_section(nozzle, to_shared(config), template)
    else:
        meridional_channel = MeridionalChannel(
            definition or geometry_from_machine(config["meridional_channel"])
        )
        blade, curves = meridional_geometry.build_section(
            nozzle, to_shared(config), meridional_channel, template=template
        )
    assembly = config.get("assembly", {})
    settings = config["flow_domain"]
    curves = apply_interfaces(meridional_channel, blade, curves, assembly, settings, .5)
    require_containment(blade, curves)
    design = Design(meridional_channel, blade, curves, assembly, settings)
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
        "geometry": design.meridional_channel.definition.to_dict(),
        "blade": encode(design.blade), "curves": encode(design.curves),
        "assembly": encode(design.assembly), "flow_domain": encode(design.flow_domain),
    }


def load_design(data):
    definition = data["geometry"]
    mapping = mapping_from_data(definition)
    return Design(mapping, decode(data["blade"]), decode(data["curves"]),
                  decode(data.get("assembly", {})), decode(data.get("flow_domain", {})))


def mapping_from_data(definition):
    """Reconstruct an explicit linear or annular mapping from its artifact."""
    if definition.get("type") == "linear":
        return LinearCascadeMapping(LinearCascadeDefinition(
            definition["meridional_chord_mm"],
            definition["blade_height_mm"],
            definition["pitch_mm"],
        ))
    walls = [
        BSpline(definition[name]["knots"], definition[name]["control_points_rz_mm"],
                definition[name]["degree"])
        for name in ("hub", "shroud")
    ]
    return MeridionalChannel(GeometryDefinition(*walls, definition["num_blades"]))


def section_data(design):
    mapping = design.meridional_channel
    length_scale = 1.0 if mapping.is_linear else mapping.reference_radius
    result = {
        name: (design.curves[name] / length_scale).tolist()
        for name in ("periodic_left", "periodic_right", "inlet", "outlet",
                     "passage_reference", "inlet_extension", "outlet_extension")
    }
    result["blade"] = (planar.blade_contour(design.blade) / length_scale).tolist()
    result["coordinate_basis"] = "linear_physical" if mapping.is_linear else "annular_conformal"
    result["coordinate_units"] = "mm" if mapping.is_linear else "dimensionless"
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
    mapping = design.meridional_channel
    for value in eta:
        _, target = design.section(float(value))
        moved = corresponding_boundary(q, "periodic_left", ref, target)
        if mapping.is_linear:
            u = -moved[:, 1] / mapping.reference_chord
        else:
            u = mapping.parameter_from_prime(-moved[:, 1] / mapping.reference_radius, float(value))
        rows.append(mapping.meridional_points(u, float(value)))
    rz = np.asarray(rows)
    # Mapped in-surface blade-height estimates, not physical 3D normal heights.
    heights = []
    h0 = numerics["inflation_layers"]["first_height"]
    if mapping.is_linear:
        heights = [h0]
    else:
        for value in (0.0, 0.5, 1.0):
            radius = mapping.meridional_points(np.array([0.0, 0.5, 1.0]), value)[:, 0]
            heights.extend((h0 * mapping.section_scale(value) * radius / mapping.reference_radius).tolist())
    spacing = np.linalg.norm(np.diff(rz, axis=0), axis=2)
    return {
        "rz": rz.tolist(), "eta": eta.tolist(),
        "basis": "reference mesh boundary stations" if section is not None else "sampled geometry stations",
        "span_layers": len(eta) - 1, "streamwise_stations": len(q),
        "min_span_spacing_mm": float(spacing.min()),
        "max_span_spacing_mm": float(spacing.max()),
        "estimated_first_height_mm": [min(heights), max(heights)],
        "coordinate_basis": "linear_streamwise_span" if mapping.is_linear else "annular_meridional",
        "axis_labels": (["Streamwise distance [mm]", "Span height [mm]"] if mapping.is_linear
                        else ["Axial coordinate [mm]", "Radius [mm]"]),
    }

