"""TurboMoC configuration names with a compatibility boundary for shared V1 code."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from examples.gmsh_3D.run_mesh_generation import validate_configuration as validate_shared

SECTIONS = {
    "nozzle": "method_of_characteristics",
    "machine": "meridional_channel",
    "blade": "blade_geometry",
    "passage": "flow_domain",
}
SMOOTH_FIELDS = (
    "inlet_axial_tangent", "inlet_blade_tangent",
    "outlet_blade_tangent", "outlet_axial_tangent", "num_points",
)
VIEW_DEFAULTS = {"section_edge_width": 0.35, "volume_edge_width": 0.35}


def normalize(config):
    """Accept old case files; publish only the current, flat configuration."""
    result = deepcopy(config)
    for old, new in SECTIONS.items():
        if old in result:
            if new in result:
                raise ValueError(f"Use either {old} or {new}, not both.")
            result[new] = result.pop(old)
    domain = result["flow_domain"]
    smooth = domain.pop("smooth", {})
    # Explicit flat values take precedence when migrating a partially edited file.
    for name, value in smooth.items():
        domain.setdefault(name, value)
    view = result["mesh"].setdefault("visualization", {})
    for name, value in VIEW_DEFAULTS.items():
        view.setdefault(name, value)
    return result


def to_shared(config):
    """Translate at the reuse boundary; V2 geometry/jobs retain readable names."""
    result = deepcopy(config)
    for old, new in SECTIONS.items():
        result[old] = result.pop(new)
    domain = result["passage"]
    domain["smooth"] = {name: domain.pop(name) for name in SMOOTH_FIELDS if name in domain}
    return result


def validate_configuration(config):
    validate_shared(to_shared(config))
    for name in VIEW_DEFAULTS:
        value = config["mesh"]["visualization"][name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"mesh.visualization.{name} must be positive.")
    return config


def read_configuration(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_configuration(normalize(yaml.safe_load(stream)))

