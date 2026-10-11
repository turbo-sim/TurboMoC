"""Machine projects and row-scoped adapters for the existing numerical APIs.

A project describes an ordered collection of blade rows. Numerical stages still
receive one flat row configuration, so an assembly view never implies a coupled
mesh or a rotor implementation that does not yet exist.
"""
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import yaml

from .configuration import normalize, validate_configuration, validate_mesh_configuration, VIEW_DEFAULTS
from .paths import GMSH_3D_DIR


GEOMETRY_SECTIONS = (
    "method_of_characteristics", "meridional_channel", "blade_geometry", "flow_domain",
)
DESIGN_MODES = ("stator", "rotor", "stator_rotor")
GEOMETRY_TYPES = ("linear", "axial", "radial")
MODE_ROLES = {"stator": ("stator",), "rotor": ("rotor",),
              "stator_rotor": ("stator", "rotor")}
PLACEMENT_DEFAULTS = {"streamwise_offset_mm": 0.0, "angular_phase_deg": 0.0}


def get_row(project, row_id):
    """Return the row with a stable identifier, independently of list position."""
    for row in project["rows"]:
        if row["id"] == row_id:
            return row
    raise ValueError(f"Unknown blade row {row_id!r}.")


def active_rows(project):
    """Return participating rows in machine order; keep inactive settings intact."""
    roles = MODE_ROLES[project["machine"]["design_mode"]]
    return [row for row in project["rows"] if row["role"] in roles]


def normalize_project(config):
    """Copy a project, or wrap a historical flat case as a single stator row."""
    if not isinstance(config, dict):
        raise ValueError("The project configuration must be a mapping.")
    if "rows" in config:
        project = deepcopy(config)
        rows = project.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("rows must be an ordered list.")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each blade row must be a mapping.")
            row.setdefault("placement", deepcopy(PLACEMENT_DEFAULTS))
            if row.get("role") == "rotor":
                geometry = row.setdefault("geometry", {})
                if isinstance(geometry, dict):
                    geometry.setdefault("parametrization_type", "conventional")
                    if geometry.get("meridional_channel", {}).get("type") == "axial":
                        geometry.setdefault("axial_interspace_mm", 0.0)
                if isinstance(row.get("mesh"), dict):
                    view = row["mesh"].setdefault("visualization", {})
                    for name, value in VIEW_DEFAULTS.items():
                        view.setdefault(name, value)
            elif isinstance(row.get("geometry"), dict) and isinstance(row.get("mesh"), dict):
                # Reuse the existing name/visualization migration at row scope.
                try:
                    flat = normalize({**row["geometry"], "mesh": row["mesh"],
                                      "workflow": project.get("workflow", {})})
                except KeyError as exc:
                    raise ValueError(f"Row {row.get('id', '?')}: missing geometry section {exc}.") from exc
                row["geometry"] = {name: value for name, value in flat.items()
                                   if name not in ("mesh", "workflow")}
                row["mesh"] = flat["mesh"]
        return project
    flat = normalize(config)
    if set(flat) != {*GEOMETRY_SECTIONS, "mesh", "workflow"}:
        raise ValueError("A historical row requires geometry sections, mesh and workflow only.")
    return {"machine": {"design_mode": "stator",
                        "geometry_type": flat["meridional_channel"]["type"]},
            "rows": [{"id": "stator_1", "role": "stator",
                      "geometry": {name: flat[name] for name in GEOMETRY_SECTIONS},
                      "mesh": flat["mesh"], "placement": deepcopy(PLACEMENT_DEFAULTS)}],
            "workflow": flat["workflow"]}


def row_configuration(project, row_id):
    """Adapt one implemented row to the flat, independent meshing contract."""
    row = get_row(project, row_id)
    if row["role"] != "stator":
        raise ValueError("Rotor geometry and meshing integration is pending; rotor jobs are unavailable.")
    return deepcopy({**row["geometry"], "mesh": row["mesh"],
                     "workflow": project["workflow"]})


def validate_project(project):
    """Validate project structure and each row without inventing rotor physics."""
    if not isinstance(project, dict) or set(project) != {"machine", "rows", "workflow"}:
        raise ValueError("A machine project requires machine, rows and workflow mappings.")
    machine = project["machine"]
    if not isinstance(machine, dict) or set(machine) != {"design_mode", "geometry_type"}:
        raise ValueError("machine requires design_mode and geometry_type.")
    mode, family = machine["design_mode"], machine["geometry_type"]
    if mode not in DESIGN_MODES:
        raise ValueError(f"machine.design_mode must be one of {DESIGN_MODES}.")
    if family not in GEOMETRY_TYPES:
        raise ValueError(f"machine.geometry_type must be one of {GEOMETRY_TYPES}.")
    if mode == "stator_rotor" and family == "linear":
        raise ValueError("The stator + rotor configuration does not support a linear cascade.")
    if not isinstance(project["workflow"], dict):
        raise ValueError("workflow must be a mapping.")
    workflow = project["workflow"]
    for name in ("save_figures", "show_gmsh", "show_pyvista"):
        if type(workflow.get(name)) is not bool:
            raise ValueError(f"workflow.{name} must be true or false.")
    if workflow.get("log_level") not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("workflow.log_level must be DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    rows = project["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty ordered list.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"id", "role", "geometry", "mesh"} <= set(row):
            raise ValueError("Each row requires id, role, geometry and mesh.")
        if set(row) - {"id", "role", "geometry", "mesh", "placement"}:
            raise ValueError(f"Unknown fields in row {row.get('id', '?')}.")
        row_id = row["id"]
        if not isinstance(row_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", row_id):
            raise ValueError("Row IDs must start with a letter and contain letters, digits, _ or -.")
        if row_id in seen:
            raise ValueError(f"Duplicate blade row ID {row_id!r}.")
        seen.add(row_id)
        if row["role"] not in ("stator", "rotor"):
            raise ValueError(f"Unknown blade row role {row['role']!r}.")
        if not isinstance(row["geometry"], dict) or not isinstance(row["mesh"], dict):
            raise ValueError(f"Row {row_id}: geometry and mesh must be mappings.")
        meridional_channel = row["geometry"].get("meridional_channel")
        if not isinstance(meridional_channel, dict) or meridional_channel.get("type") != family:
            raise ValueError(f"Row {row_id}: meridional_channel.type must match machine.geometry_type.")
        if row["role"] == "stator":
            if set(row["geometry"]) != set(GEOMETRY_SECTIONS):
                raise ValueError(f"Row {row_id}: stator geometry requires {GEOMETRY_SECTIONS}.")
            validate_configuration(row_configuration(project, row_id))
        else:
            required = {"parametrization_type", "meridional_channel"}
            optional = {"axial_interspace_mm"} if family == "axial" else set()
            if not required <= set(row["geometry"]) or set(row["geometry"]) - required - optional:
                raise ValueError(f"Row {row_id}: rotor geometry contains only planning parameters.")
            if row["geometry"]["parametrization_type"] not in ("conventional", "vortex"):
                raise ValueError("Rotor parametrization_type must be conventional or vortex.")
            from .geometry.meridional_channel import validate_machine
            validate_machine(meridional_channel)
            if not isinstance(row["mesh"].get("numerics"), dict):
                raise ValueError(f"Row {row_id}: mesh.numerics must be a mapping.")
            validate_mesh_configuration(row["mesh"])
            if "axial_interspace_mm" in row["geometry"]:
                gap = row["geometry"]["axial_interspace_mm"]
                if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not np.isfinite(gap) or gap < 0:
                    raise ValueError(f"Row {row_id}: axial_interspace_mm must be finite and non-negative.")
        placement = row.get("placement", PLACEMENT_DEFAULTS)
        if not isinstance(placement, dict) or set(placement) != set(PLACEMENT_DEFAULTS):
            raise ValueError(f"Row {row_id}: placement requires {tuple(PLACEMENT_DEFAULTS)}.")
        for name, value in placement.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Row {row_id}: placement.{name} must be finite.")
        if family != "axial" and placement["streamwise_offset_mm"] != 0:
            raise ValueError("Display streamwise placement is available only for axial rows.")
    roles = {row["role"] for row in rows}
    if not set(MODE_ROLES[mode]) <= roles:
        raise ValueError(f"The {mode} configuration is missing a required blade row.")
    return project


def load_project(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_project(normalize_project(yaml.safe_load(stream)))


def _stator_template(family):
    # Read YAML directly so this helper also works after case files adopt projects.
    case_file = GMSH_3D_DIR / f"{family}_stator.yaml"
    if family == "linear" and not case_file.exists():
        case_file = GMSH_3D_DIR / "axial_stator.yaml"
    with case_file.open(encoding="utf-8") as stream:
        project = normalize_project(yaml.safe_load(stream))
    row = deepcopy(next(row for row in project["rows"] if row["role"] == "stator"))
    if family == "linear" and row["geometry"]["meridional_channel"]["type"] != "linear":
        axial = row["geometry"]["meridional_channel"]
        row["geometry"]["meridional_channel"] = {
            "type": "linear", "meridional_chord": axial["hub_chord"],
            "blade_height": axial["blade_height"]}
    return row, deepcopy(project["workflow"])


def default_project(family="axial"):
    if family not in GEOMETRY_TYPES:
        raise ValueError(f"Unknown machine geometry type {family!r}.")
    row, workflow = _stator_template(family)
    row["id"], row["role"] = "stator_1", "stator"
    row["placement"] = deepcopy(PLACEMENT_DEFAULTS)
    return {"machine": {"design_mode": "stator", "geometry_type": family},
            "rows": [row], "workflow": workflow}


def set_design_mode(project, mode):
    """Change participation, retaining all previously configured row data."""
    if mode not in DESIGN_MODES:
        raise ValueError(f"Unknown design mode {mode!r}.")
    result = deepcopy(project)
    if mode == "stator_rotor" and result["machine"]["geometry_type"] == "linear":
        raise ValueError("Choose an axial or radial machine before enabling stator + rotor.")
    family = result["machine"]["geometry_type"]
    for role in MODE_ROLES[mode]:
        if any(row["role"] == role for row in result["rows"]):
            continue
        template, _ = _stator_template(family)
        row_id = f"{role}_1"
        while any(row["id"] == row_id for row in result["rows"]):
            row_id += "_1"
        if role == "rotor":
            template["geometry"] = {
                "parametrization_type": "conventional",
                "meridional_channel": template["geometry"]["meridional_channel"]}
            if family == "axial":
                template["geometry"]["axial_interspace_mm"] = 0.0
        template.update(id=row_id, role=role, placement=deepcopy(PLACEMENT_DEFAULTS))
        if role == "stator":
            # When a rotor-only planning file gains its stator, place that new
            # row upstream while retaining the order of every existing row.
            upstream = next((index for index, row in enumerate(result["rows"])
                             if row["role"] == "rotor"), len(result["rows"]))
            result["rows"].insert(upstream, template)
        else:
            result["rows"].append(template)
    result["machine"]["design_mode"] = mode
    return result


def set_machine_type(project, family):
    """Replace only channel-family inputs; preserve all other row settings."""
    if family not in GEOMETRY_TYPES:
        raise ValueError(f"Unknown machine geometry type {family!r}.")
    if project["machine"]["design_mode"] == "stator_rotor" and family == "linear":
        raise ValueError("The stator + rotor configuration does not support a linear cascade.")
    result = deepcopy(project)
    template, _ = _stator_template(family)
    defaults = template["geometry"]["meridional_channel"]
    for row in result["rows"]:
        previous = row["geometry"]["meridional_channel"]
        if previous["type"] == family:
            continue
        meridional_channel = deepcopy(defaults)
        for name in meridional_channel:
            if name != "type" and name in previous:
                meridional_channel[name] = previous[name]
        # A chord is a length in both linear and axial families. Radial chord
        # follows the supplied radii and is not silently invented from it.
        if family == "linear" and "hub_chord" in previous:
            meridional_channel["meridional_chord"] = previous["hub_chord"]
        elif family == "axial" and "meridional_chord" in previous:
            meridional_channel["hub_chord"] = previous["meridional_chord"]
            meridional_channel["tip_chord"] = previous["meridional_chord"]
        row["geometry"]["meridional_channel"] = meridional_channel
        if row["role"] == "rotor":
            if family == "axial":
                row["geometry"].setdefault("axial_interspace_mm", 0.0)
            else:
                row["geometry"].pop("axial_interspace_mm", None)
        if family != "axial":
            row.setdefault("placement", deepcopy(PLACEMENT_DEFAULTS))["streamwise_offset_mm"] = 0.0
    result["machine"]["geometry_type"] = family
    return result


def update_row_placement(project, row_id, **values):
    result = deepcopy(project)
    placement = get_row(result, row_id).setdefault("placement", deepcopy(PLACEMENT_DEFAULTS))
    if set(values) - set(PLACEMENT_DEFAULTS):
        raise ValueError("Unknown row placement setting.")
    placement.update(values)
    return result


def _axial_row_origins(project):
    """Resolve ordered LE planes without changing the individual row geometry.

    The farthest TE is z_LE + max(c_hub, c_tip). Specifying the next gap from
    that plane ensures that taper cannot make the minimum clearance smaller
    than the user's input. A downstream literal display offset is ignored when
    its position is defined by this gap, avoiding two competing position inputs.
    """
    origins = {}
    previous = None
    for row in active_rows(project):
        origin = row.get("placement", PLACEMENT_DEFAULTS)["streamwise_offset_mm"]
        gap = row["geometry"].get("axial_interspace_mm")
        if gap is None and row["role"] == "rotor":
            # Historical rotor planning rows omitted the field; their canonical
            # normalized representation supplies the zero-clearance default.
            gap = 0.0
        if previous is not None and gap is not None:
            meridional_channel = previous["geometry"]["meridional_channel"]
            trailing_extent = max(meridional_channel["hub_chord"], meridional_channel["tip_chord"])
            origin = origins[previous["id"]] + trailing_extent + gap
        origins[row["id"]] = float(origin)
        previous = row
    return origins


def assembly_placement(project, row_id):
    """Return resolved display placement; independent row meshes remain local.

    Axial interspaces determine downstream row origins in an assembly. Radial
    row locations come directly from their radii, so no radial translation is
    added. A solitary row retains its legacy axial display offset.
    """
    row = get_row(project, row_id)
    placement = deepcopy(row.get("placement", PLACEMENT_DEFAULTS))
    family = project["machine"]["geometry_type"]
    if family == "axial":
        placement["streamwise_offset_mm"] = _axial_row_origins(project).get(
            row_id, placement["streamwise_offset_mm"])
    else:
        placement["streamwise_offset_mm"] = 0.0
    return placement


def assembly_interspaces(project):
    """Report signed clearances for adjacent active rows in machine order.

    Axial values measure the minimum upstream TE to downstream LE-plane gap.
    Radial values follow the upstream flow direction: s*(r_next_LE-r_TE), with
    s=sign(r_TE-r_LE). Positive means clearance for inflow and outflow alike;
    negative means overlap. Planning overlaps are reported rather than rejected.
    These values describe blade envelopes, not connected solver interfaces.
    """
    family = project["machine"]["geometry_type"]
    if family == "linear":
        return []
    rows = active_rows(project)
    origins = _axial_row_origins(project) if family == "axial" else {}
    result = []
    for upstream, downstream in zip(rows, rows[1:]):
        upstream_channel = upstream["geometry"]["meridional_channel"]
        downstream_channel = downstream["geometry"]["meridional_channel"]
        if family == "axial":
            trailing = origins[upstream["id"]] + max(
                upstream_channel["hub_chord"], upstream_channel["tip_chord"])
            leading = origins[downstream["id"]]
            gap = leading - trailing
            prescribed = ("axial_interspace_mm" in downstream["geometry"] or
                          downstream["role"] == "rotor")
            direction = "axial"
        else:
            trailing = upstream_channel["outlet_radius"]
            leading = downstream_channel["inlet_radius"]
            sign = np.sign(trailing - upstream_channel["inlet_radius"])
            gap = float(sign * (leading - trailing))
            prescribed = False
            direction = "inflow" if sign < 0 else "outflow"
        result.append({"upstream_id": upstream["id"], "downstream_id": downstream["id"],
                       "geometry_type": family, "gap_mm": float(gap), "prescribed": prescribed,
                       "upstream_trailing_edge_mm": float(trailing),
                       "downstream_leading_edge_mm": float(leading), "flow_direction": direction})
    return result
