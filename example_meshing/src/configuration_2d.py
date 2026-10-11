"""Validate independently usable 2D design/mesh cases."""
from pathlib import Path
import numpy as np
import yaml
from .paths import GMSH_2D_DIR
CONFIG_FILE = GMSH_2D_DIR / "stator_mesh.yaml"

def read_configuration(config_file=CONFIG_FILE):
    """Read and validate the nozzle, blade, passage and mesh sections of the YAML."""
    from .geometry import geometry_2d as geo
    from .meshing import meshing_2d as msh

    with Path(config_file).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {
        "nozzle",
        "blade",
        "passage",
        "mesh",
    }:
        raise ValueError(
            "Configuration must contain nozzle, blade, passage, and mesh mappings."
        )
    if not all(isinstance(value, dict) for value in config.values()):
        raise ValueError("Each configuration section must be a mapping.")
    passage = config["passage"]
    for key in ("inlet_chords", "outlet_chords"):
        if not np.isfinite(passage[key]) or passage[key] <= 0:
            raise ValueError(f"passage.{key} must be finite and positive.")
    geo.validate_domain_shift(
        passage.get("domain_shift_mode", "manual"), passage.get("domain_shift", 0.0)
    )
    geo.validate_flow_domain_settings(passage.get("smooth", {}))

    msh.validate_mesh_settings(config["mesh"])
    return config
