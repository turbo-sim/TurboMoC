"""Machine projects and row-scoped adapters for the existing numerical APIs.

A project describes an ordered collection of blade rows. Numerical stages still
receive one flat row configuration. Flow-state coupling is explicit; an assembly
view does not imply a connected stator-rotor solver mesh.
"""
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import yaml

from turbo_moc.configuration.row import normalize, validate_configuration, VIEW_DEFAULTS
from turbo_moc.configuration.rotor import default_axial_interspace_mm, normalize_rotor_geometry, validate_rotor_configuration
from turbo_moc.configuration.fluid_inputs import INLET_FIELDS, inlet_values


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


def upstream_stator(project, row_id):
    """Nearest preceding active stator; stable IDs do not encode machine order."""
    upstream = None
    for row in active_rows(project):
        if row["id"] == row_id:
            return upstream
        if row["role"] == "stator":
            upstream = row
    return None


def _normalize_fluid_inputs(project):
    """Migrate row fluids and historical stagnation inputs to machine settings."""
    machine = project.get("machine")
    if not isinstance(machine, dict):
        raise ValueError("machine must be a mapping.")
    if "fluid_name" not in machine:
        roles = MODE_ROLES.get(machine.get("design_mode"), ("stator", "rotor"))
        names = []
        upstream_fluid = None
        for row in project["rows"]:
            if row.get("role") not in roles:
                continue
            geometry = row.get("geometry", {})
            if not isinstance(geometry, dict):
                continue
            section = "rotor_design" if row["role"] == "rotor" else "method_of_characteristics"
            inputs = geometry.get(section, {})
            if not isinstance(inputs, dict):
                continue
            fluid = inputs.get("fluid_name")
            if row["role"] == "stator":
                upstream_fluid = fluid
            elif upstream_fluid and geometry.get("inlet_coupling", {}).get("mode") == "stator_exit":
                # A coupled rotor's historical manual fluid was not used.
                continue
            if isinstance(fluid, str) and fluid.strip():
                names.append(fluid)
        if not names:
            raise ValueError("Specify the shared working fluid at machine.fluid_name.")
        if len(set(names)) > 1:
            import CoolProp.CoolProp as CP
            identities = {tuple(CP.AbstractState("HEOS", name).fluid_names()) for name in names}
            if len(identities) > 1:
                raise ValueError("Legacy rows specify different working fluids; set machine.fluid_name explicitly.")
        machine["fluid_name"] = names[0]
    for row in project["rows"]:
        rotor = row.get("role") == "rotor"
        section = "rotor_design" if rotor else "method_of_characteristics"
        geometry = row.get("geometry", {})
        inputs = geometry.get(section) if isinstance(geometry, dict) else None
        if not isinstance(inputs, dict):
            continue
        try:
            inputs.update(inlet_values(inputs, rotor=rotor))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Row {row.get('id', '?')}: cannot read inlet properties: {exc}") from exc
        for name in ("fluid_name", "P0", "T0", "Q0", "P0_rel", "T0_rel", "Q0_rel"):
            inputs.pop(name, None)
    return project


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
                    row["geometry"] = normalize_rotor_geometry(geometry)
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
        return _normalize_fluid_inputs(project)
    flat = normalize(config)
    if set(flat) != {*GEOMETRY_SECTIONS, "mesh", "workflow"}:
        raise ValueError("A historical row requires geometry sections, mesh and workflow only.")
    return normalize_project({"machine": {"design_mode": "stator",
                        "geometry_type": flat["meridional_channel"]["type"]},
            "rows": [{"id": "stator_1", "role": "stator",
                      "geometry": {name: flat[name] for name in GEOMETRY_SECTIONS},
                      "mesh": flat["mesh"], "placement": deepcopy(PLACEMENT_DEFAULTS)}],
            "workflow": flat["workflow"]})


def row_configuration(project, row_id):
    """Return numerical row inputs with the shared fluid, before coupling."""
    row = get_row(project, row_id)
    # Assembly metadata is derived separately, never persisted over user inputs.
    geometry = {key: value for key, value in row["geometry"].items()
                if key != "axial_interspace_mm"}
    config = deepcopy({**geometry, "mesh": row["mesh"], "workflow": project["workflow"]})
    section = "rotor_design" if row["role"] == "rotor" else "method_of_characteristics"
    config[section]["fluid_name"] = project["machine"]["fluid_name"]
    return config


def validate_project(project):
    """Validate row inputs independently of computed upstream design artifacts."""
    if not isinstance(project, dict) or set(project) != {"machine", "rows", "workflow"}:
        raise ValueError("A machine project requires machine, rows and workflow mappings.")
    machine = project["machine"]
    if not isinstance(machine, dict) or set(machine) != {"design_mode", "geometry_type", "fluid_name"}:
        raise ValueError("machine requires design_mode, geometry_type and fluid_name.")
    if not isinstance(machine["fluid_name"], str) or not machine["fluid_name"].strip():
        raise ValueError("machine.fluid_name must be a non-empty fluid name.")
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
        section = "rotor_design" if row["role"] == "rotor" else "method_of_characteristics"
        inputs = row["geometry"].get(section, {})
        if isinstance(inputs, dict):
            if "fluid_name" in inputs:
                raise ValueError(f"Row {row_id}: define the working fluid only at machine.fluid_name.")
            if not set(INLET_FIELDS) <= inputs.keys() or {
                    "P0", "T0", "Q0", "P0_rel", "T0_rel", "Q0_rel"} & inputs.keys():
                raise ValueError(f"Row {row_id}: specify inlet_property_pair, inlet_property_1 and inlet_property_2.")
        meridional_channel = row["geometry"].get("meridional_channel")
        if not isinstance(meridional_channel, dict) or meridional_channel.get("type") != family:
            raise ValueError(f"Row {row_id}: meridional_channel.type must match machine.geometry_type.")
        if row["role"] == "stator":
            if set(row["geometry"]) != set(GEOMETRY_SECTIONS):
                raise ValueError(f"Row {row_id}: stator geometry requires {GEOMETRY_SECTIONS}.")
            validate_configuration(row_configuration(project, row_id))
        else:
            required = {"parametrization_type", "meridional_channel", "rotor_design",
                        "inlet_coupling", "flow_domain"}
            optional = {"axial_interspace_mm"} if family == "axial" else set()
            if not required <= set(row["geometry"]) or set(row["geometry"]) - required - optional:
                raise ValueError(f"Row {row_id}: rotor geometry requires {sorted(required)}.")
            validate_rotor_configuration(row_configuration(project, row_id))
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


# UI initialization defaults are values, not paths to example YAML files.
# Callers may supply any project file; installed apps need no examples checkout.
_DEFAULT_STATOR_ROW = {'id': 'stator_1',
 'role': 'stator',
 'geometry': {'method_of_characteristics': {'fluid_name': 'Cyclopentane',
                                            'P0': 251300.0,
                                            'Q0': 0.025,
                                            'p_back': 95000.0,
                                            'solver': 'conventional',
                                            'use_true_sauer_line': True,
                                            'y_t': 0.01,
                                            'n': 15,
                                            'rho_t': 0.2,
                                            'rho_d': 0.1,
                                            'delta_flow': 0.0,
                                            'tau_max': 90.0,
                                            'tau_step': 1.0},
              'meridional_channel': {'type': 'axial',
                                     'hub_radius': 100.0,
                                     'blade_height': 10.0,
                                     'hub_chord': 20.0,
                                     'tip_chord': 20.0,
                                     'num_blades': 31},
              'blade_geometry': {'metal_angle_in': 0.0,
                                 'metal_angle_out': 65.0,
                                 'r_trailing': 0.5,
                                 'inlet_opening_ratio': 1.3,
                                 'n_cp': 30,
                                 'num_baseline_points': 500,
                                 'num_curve_points': 500},
              'flow_domain': {'inlet_chords': 1.5,
                              'outlet_chords': 2.0,
                              'domain_shift_mode': 'auto',
                              'domain_shift': 0.0,
                              'inlet_axial_tangent': 0.35,
                              'inlet_blade_tangent': 0.35,
                              'outlet_blade_tangent': 0.15,
                              'outlet_axial_tangent': 0.55,
                              'num_points': 120}},
 'mesh': {'show_gui': False,
          'numerics': {'terminal_output': True,
                       'verbosity': 2,
                       'blade_size': 0.25,
                       'domain_size': 1.0,
                       'curvature_elements': 50,
                       'algorithm': 8,
                       'recombine': True,
                       'recombination_algorithm': 1,
                       'smoothing_steps': 5,
                       'inflation_layers': {'enabled': True,
                                            'first_height': 0.03,
                                            'growth_ratio': 1.2,
                                            'thickness': 1.0},
                       'span_layers': 25,
                       'span_stretch': 1.5},
          'matplotlib': {'num_blades': 3,
                         'annotate_boundaries': True,
                         'mesh_color': 'orange',
                         'edge_color': 'black',
                         'domain_color': 'lightgrey',
                         'dpi': 500,
                         'rasterize_mesh': True},
          'visualization': {'section_edge_width': 0.35, 'volume_edge_width': 0.35},
          'pyvista': {'show_mesh_edges': True,
                      'surfaces': {'blade': True,
                                   'hub': True,
                                   'shroud': False,
                                   'inlet': False,
                                   'outlet': False,
                                   'periodic_left': False,
                                   'periodic_right': False}},
          'cgns': {'enabled': True,
                   'filename': 'stator_mesh_3d.cgns',
                   'length_scale': 0.001,
                   'boundary_types': {'blade': 'BCWallViscous',
                                      'inlet': 'BCInflowSubsonic',
                                      'outlet': 'BCOutflowSubsonic',
                                      'periodic_left': 'BCGeneral',
                                      'periodic_right': 'BCGeneral',
                                      'hub': 'BCWallViscous',
                                      'shroud': 'BCWallViscous'}}}}
_DEFAULT_WORKFLOW = {'save_figures': True, 'show_gmsh': True, 'show_pyvista': True, 'log_level': 'INFO'}


def _stator_template(family):
    row = deepcopy(_DEFAULT_STATOR_ROW)
    if family == "radial":
        row["geometry"]["meridional_channel"] = {'type': 'radial', 'inlet_radius': 100.0, 'outlet_radius': 80.0, 'blade_height': 8.0, 'num_blades': 12}
        row["mesh"]["numerics"].update({'blade_size': 0.6, 'domain_size': 2.0, 'curvature_elements': 40, 'inflation_layers': {'enabled': True, 'first_height': 0.03, 'growth_ratio': 1.2, 'thickness': 0.4}, 'span_layers': 16})
        row["mesh"]["matplotlib"]["dpi"] = 250
    elif family == "linear":
        row["geometry"]["meridional_channel"] = {'type': 'linear', 'meridional_chord': 20.0, 'blade_height': 10.0}
        row["mesh"]["numerics"].update(span_layers=8, span_stretch=0.0)
        row["mesh"]["cgns"]["filename"] = "linear_cascade_3d.cgns"
    return row, deepcopy(_DEFAULT_WORKFLOW)


def default_project(family="axial"):
    if family not in GEOMETRY_TYPES:
        raise ValueError(f"Unknown machine geometry type {family!r}.")
    row, workflow = _stator_template(family)
    row["id"], row["role"] = "stator_1", "stator"
    row["placement"] = deepcopy(PLACEMENT_DEFAULTS)
    return normalize_project({"machine": {"design_mode": "stator", "geometry_type": family},
            "rows": [row], "workflow": workflow})


def _adjacent_channel_defaults(channel, neighbour, *, downstream):
    """Give a newly created row compatible endwalls and a positive radial gap.

    Existing row inputs are never changed here. Ten percent of the neighbour's
    logarithmic radial chord gives a finite, positive default radius for both
    inflow and outflow; subsequent radial spacing is controlled by the radii.
    """
    channel["blade_height"] = neighbour["blade_height"]
    if channel["type"] == "axial":
        channel["hub_radius"] = neighbour["hub_radius"]
    elif channel["type"] == "radial":
        ratio = neighbour["outlet_radius"] / neighbour["inlet_radius"]
        if downstream:
            channel["inlet_radius"] = neighbour["outlet_radius"] * ratio ** 0.1
            channel["outlet_radius"] = channel["inlet_radius"] * ratio
        else:
            channel["outlet_radius"] = neighbour["inlet_radius"] / ratio ** 0.1
            channel["inlet_radius"] = channel["outlet_radius"] / ratio


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
        neighbour = next((row for row in result["rows"] if row["role"] != role), None)
        if neighbour is not None:
            _adjacent_channel_defaults(template["geometry"]["meridional_channel"],
                neighbour["geometry"]["meridional_channel"], downstream=role == "rotor")
        row_id = f"{role}_1"
        while any(row["id"] == row_id for row in result["rows"]):
            row_id += "_1"
        if role == "rotor":
            template["geometry"] = normalize_rotor_geometry({
                "meridional_channel": template["geometry"]["meridional_channel"]})
            template["mesh"]["cgns"]["filename"] = "rotor_mesh_3d.cgns"
            # Frontal-Delaunay is the conservative starting point for the
            # strongly curved rotor passage, including its natural-pitch
            # linear cascade. Existing rows retain their chosen algorithm.
            template["mesh"]["numerics"]["algorithm"] = 6
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
    return normalize_project(result)


def set_machine_type(project, family):
    """Replace only channel-family inputs; preserve all other row settings."""
    if family not in GEOMETRY_TYPES:
        raise ValueError(f"Unknown machine geometry type {family!r}.")
    if project["machine"]["design_mode"] == "stator_rotor" and family == "linear":
        raise ValueError("The stator + rotor configuration does not support a linear cascade.")
    result = deepcopy(project)
    template, _ = _stator_template(family)
    defaults = template["geometry"]["meridional_channel"]
    previous_row = None
    for row in result["rows"]:
        previous = row["geometry"]["meridional_channel"]
        if previous["type"] == family:
            previous_row = row
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
        if family == "radial" and previous_row is not None:
            # Only newly initialized radial dimensions are placed in sequence;
            # a repeated family selection preserves explicit radius settings.
            _adjacent_channel_defaults(meridional_channel,
                previous_row["geometry"]["meridional_channel"], downstream=True)
        row["geometry"]["meridional_channel"] = meridional_channel
        if row["role"] == "rotor":
            if family == "axial":
                row["geometry"].setdefault("axial_interspace_mm",
                    default_axial_interspace_mm(meridional_channel))
            else:
                row["geometry"].pop("axial_interspace_mm", None)
        if family != "axial":
            row.setdefault("placement", deepcopy(PLACEMENT_DEFAULTS))["streamwise_offset_mm"] = 0.0
        previous_row = row
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
            # normalized representation supplies the same chord-based default.
            gap = default_axial_interspace_mm(row["geometry"]["meridional_channel"])
        if previous is not None and gap is not None:
            meridional_channel = previous["geometry"]["meridional_channel"]
            trailing_extent = max(meridional_channel["hub_chord"], meridional_channel["tip_chord"])
            origin = origins[previous["id"]] + trailing_extent + gap
        origins[row["id"]] = float(origin)
        previous = row
    return origins


def assembly_placement(project, row_id):
    """Return physical assembly placement; preview arrays remain row-local.

    Axial interspaces determine downstream row origins in an assembly. Radial
    row locations come directly from their radii, so no radial translation is
    added. Export applies this placement for multirow projects. A solitary row
    retains its legacy axial display offset.
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


def shared_interfaces(project):
    """Resolve common physical boundaries in the gaps between active rows.

    A plane halfway across the minimum axial clearance remains outside a
    tapered blade at every span. Radial interfaces are cylinders halfway
    between the supplied TE and LE radii, for either flow direction. Current
    axial/radial families have constant endwall radii/heights: unequal endwalls
    need an explicit transition model and must not be silently joined.

    Keep this physical validation separate from validate_project, so incomplete
    edits remain visible and correctable in the app.
    """
    interfaces = []
    for gap in assembly_interspaces(project):
        upstream = get_row(project, gap["upstream_id"])
        downstream = get_row(project, gap["downstream_id"])
        up = upstream["geometry"]["meridional_channel"]
        down = downstream["geometry"]["meridional_channel"]
        pair = f"{upstream['id']} → {downstream['id']}"
        if gap["gap_mm"] <= 0:
            raise ValueError(f"{pair}: a shared interface requires a positive blade gap; "
                             "increase the axial interspace or separate the radial TE/LE radii.")
        family = gap["geometry_type"]
        matching = ("hub_radius", "blade_height") if family == "axial" else ("blade_height",)
        for name in matching:
            if not np.isclose(up[name], down[name], rtol=1e-10, atol=1e-9):
                raise ValueError(f"{pair}: shared interfaces require matching {name} "
                                 f"({up[name]:g} and {down[name]:g} mm). "
                                 "Set consistent meridional channel dimensions; endwall transitions are not implemented.")
        if family == "radial":
            up_sign = np.sign(up["outlet_radius"] - up["inlet_radius"])
            down_sign = np.sign(down["outlet_radius"] - down["inlet_radius"])
            if up_sign != down_sign:
                raise ValueError(f"{pair}: both radial rows must have the same flow direction.")
        interface = {
            "id": f"{upstream['id']}__{downstream['id']}",
            "upstream_id": upstream["id"], "downstream_id": downstream["id"],
            "geometry_type": family, "fraction": 0.5, "gap_mm": gap["gap_mm"],
            "coordinate_mm": 0.5 * (gap["upstream_trailing_edge_mm"] +
                                     gap["downstream_leading_edge_mm"]),
        }
        interfaces.append(interface)
    return interfaces


def row_assembly(project, row_id):
    """Resolve one row's placement and internal caps for a numerical job."""
    if row_id not in {row["id"] for row in active_rows(project)}:
        raise ValueError(f"Blade row {row_id!r} is not active in this machine project.")
    interfaces = shared_interfaces(project)
    if not interfaces:
        return None
    caps = {}
    for interface in interfaces:
        if interface["upstream_id"] == row_id:
            caps["outlet"] = interface
        if interface["downstream_id"] == row_id:
            caps["inlet"] = interface
    return {"row_id": row_id, "placement": assembly_placement(project, row_id),
            "interfaces": caps}
