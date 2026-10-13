"""Row-aware TurboMoC geometry and meshing workspaces."""
from copy import deepcopy
import base64
import binascii
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import time
import yaml
from uuid import uuid4

import dash_vtk  # Register the browser bundle before constructing scenes.
import jaxprop as jxp
from dash import ALL, MATCH, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from dash.exceptions import MissingCallbackContextException
from flask import abort, send_from_directory

from turbo_moc.app import workflow as flow, plots, nozzle_views, design_statistics, project_export, calculations, rotor_inlet, blade_counts, display_panel, mesh_reporting, inspection_panel
from turbo_moc.meshing import mesh_3d as mesh
from turbo_moc.configuration import project as project_model
from turbo_moc.configuration.rotor import default_axial_interspace_mm, SPEED_TYPES
from turbo_moc.api import list_supported_fluids
from turbo_moc.configuration.fluid_inputs import INLET_FIELDS, OUTLET_FIELDS, PROPERTIES, inlet_values, pair_properties, property_label
from turbo_moc.visualization.geometry_preview import preview_geometry
from turbo_moc.visualization import plotly as nozzle_plots
from turbo_moc.geometry import nozzle as nozzle_geometry

ACTIONS = ("nozzle", "reference", "volume", "cancel")
MACHINE_FLUID_ID = {"type": "parameter", "row": "", "path": "fluid_name"}
ROTOR_INLET_FIELDS = {
    "fluid_name": "Fluid",
    "P0_rel": "Relative stagnation pressure [Pa]",
    "T0_rel": "Relative stagnation temperature [K]",
    "M_inlet": "Relative inlet Mach number",
    "beta_inlet": "Relative inlet flow angle [deg]",
}


@lru_cache(maxsize=1)
def fluid_options():
    return list_supported_fluids()


def fluid_dropdown(identifier, value):
    # CoolProp aliases in existing files remain selectable alongside its full list.
    options = sorted(set(fluid_options()) | {value})
    return dcc.Dropdown(id=identifier, options=options, value=value,
        clearable=False, searchable=True)


def fluid_inlet_controls(parameters, prefix, row_id, *, rotor=False):
    values = inlet_values(parameters, rotor=rotor)
    pair = values["inlet_property_pair"]
    result = [html.Label([html.Span("Relative inlet stagnation property pair" if rotor else
            "Inlet stagnation property pair"), dcc.Dropdown(
            id={"type": "parameter", "row": row_id, "path": f"{prefix}.inlet_property_pair"},
            options=[{"label": f"{name.removesuffix('_INPUTS')} — " + " / ".join(
                PROPERTIES[prop][0] for prop in pair_properties(name)), "value": name}
                for name in jxp.INPUT_PAIRS], value=pair, clearable=False)],
            className="control fluid-selector")]
    for index in (1, 2):
        result.append(html.Label([html.Span(property_label(pair, index),
            id={"type": "fluid-property-label", "row": row_id, "section": prefix, "index": index}),
            dcc.Input(id={"type": "parameter", "row": row_id,
                "path": f"{prefix}.inlet_property_{index}"},
                value=values[f"inlet_property_{index}"], type="number", debounce=True,
                step="any")], className="control fluid-property"))
    return result


def moc_controls(parameters, row_id, exclude=()):
    return [*fluid_inlet_controls(parameters, "method_of_characteristics", row_id),
        *controls({name: value for name, value in parameters.items()
            if name not in {"fluid_name", "P0", "T0", "Q0", *INLET_FIELDS, *OUTLET_FIELDS, *exclude}},
            "method_of_characteristics", row_id)]


NOZZLE_SETUP_FIELDS = {
    "delta_flow": ("Nozzle type", [{"label": "Planar", "value": 0.0},
                                   {"label": "Axisymmetric", "value": 1.0}]),
    "solver": ("Nozzle contour", [{"label": "Conventional", "value": "conventional"},
                                  {"label": "Minimum length", "value": "mln"}]),
}


def nozzle_setup(project):
    if project["machine"]["design_mode"] != "nozzle":
        return []
    row = project_model.active_rows(project)[0]
    parameters = row["geometry"]["method_of_characteristics"]
    return [html.Label([html.Span(label), dcc.Dropdown(
        id={"type": "parameter", "row": row["id"], "path": f"method_of_characteristics.{name}"},
        value=parameters[name], options=options, clearable=False, searchable=False)], className="control")
        for name, (label, options) in NOZZLE_SETUP_FIELDS.items()]


OUTLET_OPTIONS = [{"label": "Static pressure", "value": "pressure"},
                  {"label": "Mach number", "value": "mach"}]


ROTOR_OUTLET_OPTIONS = [{"label": "Relative exit Mach number", "value": "mach"},
                        {"label": "Static backpressure", "value": "pressure"}]


def outlet_controls(parameters, row_id, section="method_of_characteristics"):
    rotor = section == "rotor_design"
    condition = parameters.get("outlet_condition", "mach" if rotor else "pressure")
    kind = "rotor-outlet-field" if rotor else "outlet-field"

    def field(name, label, key):
        return html.Div(html.Label([html.Span(label), dcc.Input(
            id={"type": "parameter", "row": row_id, "path": f"{section}.{key}"},
            value=parameters.get(key), type="number", debounce=True, step="any")], className="control"),
            id={"type": kind, "row": row_id, "condition": name},
            style={} if condition == name else {"display": "none"})

    title = ("Relative frame: the rotor's relative stagnation state is expanded isentropically "
             "to the backpressure, which sets the relative exit Mach number." if rotor else
             "Absolute frame: the inlet stagnation state is expanded isentropically; "
             "an exit Mach number is converted to the corresponding static pressure.")
    return group("Outlet", [html.Label([html.Span("Outlet condition"), dcc.Dropdown(
            id={"type": "parameter", "row": row_id, "path": f"{section}.outlet_condition"},
            value=condition, options=ROTOR_OUTLET_OPTIONS if rotor else OUTLET_OPTIONS,
            clearable=False, searchable=False)], className="control", title=title),
        field("pressure", "Exit static pressure [Pa]", "p_back"),
        field("mach", "Relative exit Mach number [-]" if rotor else "Exit Mach number [-]",
              "M_outlet" if rotor else "exit_mach")])


def number_field(row_id, path, label, value, kind=None):
    widget = html.Label([html.Span(label), dcc.Input(
        id={"type": "parameter", "row": row_id, "path": path}, value=value, type="number",
        debounce=True, step=1 if path.rsplit(".", 1)[-1] in flow.INTEGER_KEYS else "any")], className="control")
    if kind is None:
        return widget
    return html.Div(widget, id={"type": "nozzle-kind", "row": row_id, "field": f"{kind}:{path}"})


def nozzle_mesh_sidebar(row, settings):
    identifier = row["id"]
    return html.Details([html.Summary("Nozzle mesh"),
        group("Structured 2D mesh", [
            number_field(identifier, "mesh.nozzle.axial_cells", "Axial cells", settings["axial_cells"]),
            number_field(identifier, "mesh.nozzle.radial_cells", "Radial cells (axis to wall)", settings["radial_cells"]),
            number_field(identifier, "mesh.nozzle.wall_growth_ratio", "Wall growth ratio [-]",
                         settings["wall_growth_ratio"]),
            html.Small("Cell height grows by this ratio from the wall towards the axis; 1 gives uniform cells.")]),
        group("3D sweep", [
            number_field(identifier, "mesh.nozzle.circumferential_cells", "Circumferential cells (multiple of 4)",
                         settings["circumferential_cells"], "axisymmetric"),
            number_field(identifier, "mesh.nozzle.depth_cells", "Depth cells", settings["depth_cells"], "planar"),
            html.Small("Axisymmetric: full revolution with an O-grid core (all hexahedra). "
                       "Planar: full-height section extruded through the slab depth.")]),
        group("CGNS boundary types", controls(settings["boundary_types"], "mesh.nozzle.boundary_types", identifier)),
    ], open=True, className="settings mesh row-settings", **{"data-row": identifier})


def cad_controls(row, family):
    identifier, settings = row["id"], row["mesh"]["cad"]
    fields = [number_field(identifier, "mesh.cad.span_sections", "Span sections", settings["span_sections"]),
              number_field(identifier, "mesh.cad.disk_thickness", "Disk thickness [mm]", settings["disk_thickness"]),
              number_field(identifier, "mesh.cad.axial_margin", "Disk margin [chord fraction]", settings["axial_margin"]),
              number_field(identifier, "mesh.cad.root_overlap", "Blade root overlap [span fraction]",
                           settings["root_overlap"])]
    if family == "linear":
        fields.append(number_field(identifier, "mesh.cad.linear_blades", "Blades on the base plate",
                                   settings["linear_blades"]))
    else:
        fields.append(html.Label([html.Span("Fluid domain"), dcc.Dropdown(
            id={"type": "parameter", "row": identifier, "path": "mesh.cad.fluid_annulus"},
            options=[{"label": "One periodic passage", "value": False}, {"label": "Full annulus", "value": True}],
            value=settings["fluid_annulus"], clearable=False, searchable=False)], className="control"))
    return group("CAD export", [*fields, html.Small(
        "The disk is offset from the hub surface by its thickness and extends past the blade by the margin; "
        "blade roots are embedded in the disk by the overlap so the fused solid is one body.")])


def row_label(row):
    # A single stage needs concise labels; stable IDs still identify saved jobs
    # and allow independently named rows when multistage support is introduced.
    return {"stator_1": "Stator", "rotor_1": "Rotor"}.get(
        row["id"], row["id"].replace("_", " ").title())


def read_uploaded_project(contents):
    """Decode and validate a browser upload before changing any settings."""
    try:
        _, encoded = contents.split(",", 1)
        content = base64.b64decode(encoded, validate=True).decode("utf-8-sig")
        return project_model.validate_project(project_model.normalize_project(yaml.safe_load(content)))
    except (ValueError, TypeError, KeyError, UnicodeError, binascii.Error, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration: {exc}") from exc


def actual_blade_count(value, row_id):
    return html.Label([html.Span("Actual number of blades"), dcc.Input(
        id={"type": "parameter", "row": row_id, "path": "meridional_channel.num_blades"},
        value=value, type="number", min=2, step=1, debounce=True)], className="control",
        title="Each successful blade solve sets the actual count to ceil(natural count). "
              "You can then override it; changing the count changes passage spacing.")


def natural_blade_box(row_id):
    return html.Div("Compute blade design to obtain the natural pitch.",
        id={"type": "natural-blade-sizing", "row": row_id}, className="computed-inlet")


def meridional_controls(dimensions, row_id):
    fields = controls(dimensions, "meridional_channel", row_id)
    if dimensions["type"] == "linear":
        fields.append(natural_blade_box(row_id))
    return fields


def controls(value, prefix, row_id, *, hidden_paths=()):
    """Stable row IDs separate field ownership from display order."""
    result = []
    for name, item in value.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            result.append(html.Details([html.Summary(name.replace("_", " ").capitalize()),
                *controls(item, path, row_id, hidden_paths=hidden_paths)], open=True,
                className="nested-field-group"))
            continue
        if path == "meridional_channel.type":
            continue
        if path == "meridional_channel.num_blades":
            result.extend([html.Div(actual_blade_count(item, row_id),
                id={"type": "actual-blade-count", "row": row_id}), natural_blade_box(row_id)])
            continue
        identifier = {"type": "parameter", "row": row_id, "path": path}
        if name == "fluid_name":
            widget = fluid_dropdown(identifier, item)
        elif path == "blade_geometry.parametrization":
            widget = dcc.Dropdown(id=identifier, options=[
                {"label": "Symmetric (mirrored nozzle wall)", "value": "full"},
                {"label": "Asymmetric (circular-arc pressure side)", "value": "semi"}],
                value=item, clearable=False, searchable=False)
        elif name == "extension_type" and prefix.startswith("flow_domain."):
            widget = dcc.Dropdown(id=identifier, options=[
                {"label": "Smooth B-spline", "value": "bspline"},
                {"label": "Straight line", "value": "straight"}],
                value=item, clearable=False)
        elif isinstance(item, bool):
            widget = dcc.Dropdown(id=identifier, options=[
                {"label": "Yes", "value": True}, {"label": "No", "value": False}],
                value=item, clearable=False)
        elif name in ("solver", "domain_shift_mode", "parametrization_type"):
            options = {"solver": ["conventional", "mln"],
                "domain_shift_mode": ["auto", "manual"],
                "parametrization_type": ["conventional", "vortex"]}[name]
            widget = dcc.Dropdown(id=identifier, options=options, value=item, clearable=False)
        else:
            widget = dcc.Input(id=identifier, value=item,
                type="number" if isinstance(item, (int, float)) else "text", debounce=True,
                step=1 if name in flow.INTEGER_KEYS else "any")
        title = f"{row_id}.{path}"
        if name == "extension_type" and prefix.startswith("flow_domain."):
            title = ("Smooth B-spline turns from the blade direction to the axial direction. "
                     "Straight line continues the blade-reference tangent to the cap plane. "
                     "Tangent handle settings apply only to the B-spline. "
                     "Defaults: smooth inlet, straight outlet.")
        if path == "mesh.numerics.recombine_minimum_quality":
            widget.min, widget.max = 0, 1
            title = ("Minimum Gmsh quality for recombined quadrilaterals (0–1; default 0.1). "
                     "Higher values can retain more triangles. This is not a maximum skewness "
                     "limit; inspect the final 2D and 3D mesh quality.")
        label = "Blade parametrization" if path == "blade_geometry.parametrization" else name.replace("_", " ")
        if path == "blade_geometry.parametrization":
            title = ("Symmetric: the pressure side mirrors the MoC nozzle wall. "
                     "Asymmetric: only the suction side follows the MoC wall; the pressure side is one circular arc.")
        if prefix.startswith("flow_domain"):
            label = {"extension_length": "Extension length [chords]",
                     "domain_shift_value": "Domain shift value [pitch]",
                     "axial_tangent": "Axial tangent [-]",
                     "blade_tangent": "Blade tangent [-]"}.get(name, label.capitalize())
            if name == "extension_length":
                title = "Extension distance divided by the row's reference chord; the existing chord-based convention is unchanged."
            elif name == "number_of_points":
                title = "Curve sampling points, not mesh nodes."
                widget.min = 3
        result.append(html.Label([html.Span(label), widget],
            title=title, className="control", style={"display": "none"} if path in hidden_paths else {}))
    return result


def group(title, children):
    return html.Details([html.Summary(title), *children], open=True, className="field-group")


def rotor_inlet_controls(geometry, row_id, combined):
    """Keep manual values mounted while coupled values use separate read-only cells.

    This preserves the user's independent inlet state when toggling coupling,
    and prevents calculated values from feeding back into editable settings.
    """
    coupling = geometry["inlet_coupling"]
    coupled = coupling["mode"] == "stator_exit"
    manual = fluid_inlet_controls(geometry["rotor_design"], "rotor_design", row_id, rotor=True)
    for name, label in ROTOR_INLET_FIELDS.items():
        if name in ("fluid_name", "P0_rel", "T0_rel"):
            continue
        manual.append(html.Label([html.Span(label), dcc.Input(
            id={"type": "parameter", "row": row_id, "path": f"rotor_design.{name}"},
            value=geometry["rotor_design"][name],
            type="text" if name == "fluid_name" else "number", debounce=True,
            step="any")], className="control"))
    return group("Rotor inlet state", [
        html.Label([html.Span("Inlet conditions"), dcc.Dropdown(
            id={"type": "parameter", "row": row_id, "path": "inlet_coupling.mode"},
            options=[{"label": "Specify manually", "value": "manual"},
                {"label": "Couple to stator exit", "value": "stator_exit", "disabled": not combined}],
            value=coupling["mode"], clearable=False)], className="control"),
        html.Div([
            html.Label([html.Span("Rotational speed type"), dcc.Dropdown(
                id={"type": "parameter", "row": row_id, "path": "inlet_coupling.rotational_speed.type"},
                options=[{"label": label, "value": name} for name, label in SPEED_TYPES.items()],
                value=coupling["rotational_speed"]["type"], clearable=False)], className="control fluid-selector"),
            html.Label([html.Span(SPEED_TYPES[coupling["rotational_speed"]["type"]],
                id={"type": "rotor-speed-label", "row": row_id}), dcc.Input(
                id={"type": "parameter", "row": row_id, "path": "inlet_coupling.rotational_speed.value"},
                value=coupling["rotational_speed"]["value"], type="number", debounce=True, step="any")],
                className="control"),
            html.Small("Dimensionless speeds use the stator back pressure as their reference. Specific speed uses an estimated stator mass flow."),
            ],
            id={"type": "rotor-coupling-speed", "row": row_id},
            style={} if coupled else {"display": "none"}),
        html.Div(id={"type": "rotor-inlet-warning", "row": row_id}),
        html.Div(manual, id={"type": "rotor-manual-inlet", "row": row_id},
            style={"display": "none"} if coupled else {}),
        html.Div(id={"type": "rotor-coupled-inlet", "row": row_id}, className="computed-inlet",
            style={} if coupled else {"display": "none"}),
        html.Small("Inlet quantities use the rotor-relative frame; the flow angle is measured from the meridional direction."),
    ])


def internal_domain_sides(project, row_id):
    """Identify internal boundaries without requiring a currently valid gap."""
    rows = project_model.active_rows(project)
    index = next(i for i, row in enumerate(rows) if row["id"] == row_id)
    return {side for side, internal in (("inlet", index > 0),
        ("outlet", index < len(rows) - 1)) if internal}


def flow_domain_controls(project, row):
    """Retain solitary-row values while shared boundaries derive their extent.

    Hidden controls remain mounted so switching to a single row restores its
    saved extension, instead of writing the derived assembly value into YAML.
    """
    internal = internal_domain_sides(project, row["id"])
    children = controls(row["geometry"]["flow_domain"], "flow_domain", row["id"],
        hidden_paths={f"flow_domain.{side}.extension_length" for side in internal})
    children.append(html.Div(id={"type": "shared-domain", "row": row["id"]},
        className="computed-inlet"))
    return group("Flow domain definition", children)


def shared_domain_readout(project, row_id):
    """Show physical distances, not the inactive conformal chord fractions."""
    assembly = project_model.row_assembly(project, row_id)
    if not assembly:
        return []
    row = project_model.get_row(project, row_id)
    channel = row["geometry"]["meridional_channel"]
    origin = assembly["placement"]["streamwise_offset_mm"]
    result = []
    for side, interface in assembly["interfaces"].items():
        coordinate = interface["coordinate_mm"]
        axial = interface["geometry_type"] == "axial"
        label = f"{side.title()} interface {'z' if axial else 'radius'} [mm]"
        result.append(html.Label([html.Span(label), dcc.Input(value=coordinate,
            type="number", disabled=True)], className="control computed-control"))
        distances = []
        for fraction, name in ((0, "Hub"), (0.5, "Midspan"), (1, "Tip")):
            if axial:
                chord = (1 - fraction) * channel["hub_chord"] + fraction * channel["tip_chord"]
                distance = origin - coordinate if side == "inlet" else coordinate - origin - chord
            else:
                distance = abs(coordinate - channel[f"{side}_radius"])
            distances.append(html.Label([html.Span(f"{name} extension [mm]"),
                dcc.Input(value=round(distance, 10), type="number", disabled=True)],
                className="control computed-control"))
        result.extend(distances)
    result.append(html.Small("Internal boundaries meet at the shared mid-gap interface. "
        "These physical extensions are derived; saved chord fractions apply when the row is used alone. "
        "External inlet/outlet extensions remain editable."))
    return result


def nozzle_mode(project):
    return project["machine"]["design_mode"] == "nozzle"


def sidebar(project):
    result = []
    family = project["machine"]["geometry_type"]
    if nozzle_mode(project):
        row = project_model.active_rows(project)[0]
        parameters = row["geometry"]["method_of_characteristics"]
        settings = row["mesh"]["nozzle"]
        return [html.Details([html.Summary("Nozzle"), group("Method of characteristics",
            moc_controls(parameters, row["id"], (*NOZZLE_SETUP_FIELDS, "p_back"))),
            outlet_controls(parameters, row["id"]),
            group("Nozzle geometry", [
                number_field(row["id"], "mesh.nozzle.convergent_length", "Convergent length [m]",
                             settings["convergent_length"]),
                number_field(row["id"], "mesh.nozzle.planar_depth", "Slab depth [m]",
                             settings["planar_depth"], "planar"),
                number_field(row["id"], "mesh.nozzle.wall_thickness", "Wall thickness, CAD body [m]",
                             settings["wall_thickness"])])],
            open=True, className="settings design row-settings", **{"data-row": row["id"]}),
            nozzle_mesh_sidebar(row, settings)]
    for row in project_model.active_rows(project):
        identifier, geometry = row["id"], row["geometry"]
        if row["role"] == "stator":
            fields = [group(title, moc_controls(geometry[name], identifier, ("p_back",))
                if name == "method_of_characteristics" else meridional_controls(geometry[name], identifier)
                if name == "meridional_channel" else controls(geometry[name], name, identifier)) for name, title in (
                ("method_of_characteristics", "Method of characteristics"),
                ("meridional_channel", "Linear cascade definition" if family == "linear"
                     else "Meridional channel definition"),
                ("blade_geometry", "Blade geometry definition"))]
            fields.insert(1, outlet_controls(geometry["method_of_characteristics"], identifier))
            fields.append(flow_domain_controls(project, row))
        else:
            fields = [rotor_inlet_controls(geometry, identifier,
                    project["machine"]["design_mode"] == "stator_rotor"),
                group("Vortex blade design", controls({k: v for k, v in geometry["rotor_design"].items()
                    if k not in {*ROTOR_INLET_FIELDS, *INLET_FIELDS, "outlet_condition", "p_back", "M_outlet"}},
                    "rotor_design", identifier)),
                outlet_controls(geometry["rotor_design"], identifier, "rotor_design"),
                group("Linear cascade definition" if family == "linear" else "Meridional channel definition",
                    meridional_controls(geometry["meridional_channel"], identifier)),
                flow_domain_controls(project, row)]
            if family == "axial" and project["machine"]["design_mode"] == "stator_rotor":
                fields.insert(0, group("Axial interspace", [*controls({"axial_interspace_mm":
                    geometry.get("axial_interspace_mm", default_axial_interspace_mm(
                        geometry["meridional_channel"]))}, "", identifier),
                    html.Small("Minimum axial clearance from the upstream trailing edge to this leading-edge plane [mm]. "
                        "New or missing values default to 10% of the rotor midspan axial chord; saved gaps remain unchanged.")]))
        placement = row.get("placement", {})
        visible = {k: v for k, v in placement.items() if
            (k == "streamwise_offset_mm" and family == "axial") or
            (k == "angular_phase_deg" and family != "linear")}
        if row["role"] == "rotor" and project["machine"]["design_mode"] == "stator_rotor":
            visible.pop("streamwise_offset_mm", None)
        if visible:
            title = "Assembly placement" if len(project_model.active_rows(project)) > 1 else "Assembly placement · display only"
            fields.append(group(title, controls(visible, "placement", identifier)))
        fields.append(cad_controls(row, family))
        result.append(html.Details([html.Summary(row_label(row)), *fields],
            open=False, className="settings design row-settings", **{"data-row": identifier}))
        config = project_model.row_configuration(project, identifier)
        numerics = config["mesh"]["numerics"]
        section = {k: v for k, v in numerics.items() if k not in ("span_layers", "span_stretch")}
        mesh_fields = [
                group("2D mesh · physical mm" if family == "linear" else "2D mesh · reference conformal mm",
                      controls(section, "mesh.numerics", identifier)),
                group("Span discretization", controls({k: numerics[k] for k in
                    ("span_layers", "span_stretch")}, "mesh.numerics", identifier)),
                group("Mesh visualization", controls(config["mesh"]["visualization"],
                    "mesh.visualization", identifier)),
                html.Small("Edge widths are display pixels; they do not require remeshing."),
                group("CGNS mesh output", controls({k: v for k, v in config["mesh"]["cgns"].items()
                    if k != "enabled"}, "mesh.cgns", identifier)),
            ]
        result.append(html.Details([html.Summary(row_label(row)), *mesh_fields],
            open=False, className="settings mesh row-settings", **{"data-row": identifier}))
    return result


def empty(title):
    return plots.figure(title, "", "")


def viewport(name):
    return html.Div([html.Div(id=f"{name}-scene", className="vtk-host"),
        html.Small(id=f"{name}-note", style={"display": "none"})], className="vtk-frame")


def row_controls(project, row):
    options = calculations.scope_options(project)
    return html.Div([
        html.H3("Calculation and display", className="sidebar-box-title"),
        html.Label([html.Span("Calculate"),
            dcc.Dropdown(id="calculation-scope", clearable=False, searchable=False,
                value="all", options=[{"label": "All rows", "value": "all"}, *options])],
            id="calculation-controls", className="control",
            title="Selected rows calculate sequentially in machine order. Display selection is independent."),
        # Internal workflow state; report and display configuration have their
        # own independent selectors.
        html.Div([dcc.Checklist(id="plotted-rows", value=[item["value"] for item in options], options=options),
            dcc.Dropdown(id="selected-row", clearable=False, value=row["id"], options=options)],
            style={"display": "none"}),
    ], id="interspace-note", className="scope-settings")


@lru_cache(maxsize=8)
def reference_artifact(session, job):
    return mesh.load_section(flow.directory({"session": session, "job": job}))


@lru_cache(maxsize=8)
def volume_artifact(session, job):
    folder = flow.directory({"session": session, "job": job})
    return [json.loads((folder / name).read_text(encoding="utf-8")) for name in
        ("browser_surfaces.json", "mesh_summary.json", "mesh_report.json")]


def selected_references(state, registry):
    return (registry or {}).get((state or {}).get("row_id"), {})


def ready_rows(state, registry):
    """Yield current row profiles, resolving coupled inlet dependencies first."""
    for row in project_model.active_rows(state["project"]):
        try:
            config = flow.resolved_configuration(state["project"], row["id"], registry)
        except (ValueError, OSError, RuntimeError):
            continue
        reference = (registry or {}).get(row["id"], {}).get("nozzle")
        keys = flow.keys(config, reference)
        if flow.current(reference, keys["nozzle"]):
            yield row, config, reference, keys


# These are derived display datasets, never cached MoC solutions.
_PREVIEWS = {}
def cached_preview(key, design):
    if key not in _PREVIEWS:
        if len(_PREVIEWS) >= 8:
            _PREVIEWS.pop(next(iter(_PREVIEWS)))
        _PREVIEWS[key] = preview_geometry(design)
    return _PREVIEWS[key]


def report_panel(summary, report):
    cells = sum(summary["cells"].values())
    rows = [("Volume cells", f"{cells:,}"), ("Volume nodes", f"{summary['volume_nodes']:,}"),
            ("Span layers", len(summary["span_stations"]) - 1),
            (mesh_reporting.SICN_LABEL, f"{summary['quality']['min_sicn']:.4g}"),
            ("3D time [s]", f"{summary['elapsed_s']:.2f}")]
    table = html.Table([html.Tbody([html.Tr([html.Td(k,
        title=mesh_reporting.SICN_TIP if k == mesh_reporting.SICN_LABEL else None), html.Td(str(v))])
        for k, v in rows])], className="report-table")
    result = [table, mesh_reporting.metric_table(report["metrics"], report["definitions"]),
        mesh_reporting.histogram_row(report["metrics"], report["definitions"])]
    result += [html.Details([html.Summary("Worst cells · original Gmsh IDs"),
        *[html.P(n + ": " + ", ".join(f"{c['cell_id']} ({c['value']:.3g})" for c in cells))
          for n, cells in report["worst_cells"].items()]])]
    return result


def cad_card(title, status, report, reference):
    base = f"/artifacts/{reference['session']}/{reference['job']}/"
    rows = [html.Tr([html.Td(name.replace("_", " ").capitalize()), html.Td(item["solids"]),
                     html.Td("yes" if item["valid"] else "no"), html.Td("yes" if item["watertight"] else "no"),
                     html.Td(f"{item['volume_mm3']:.6g}")]) for name, item in report["shapes"].items()]
    checks = [html.Li(f"{name.replace('_', ' ')}: {value:.5f}") for name, value in report["volume_checks"].items()
              if not name.endswith("mm3")]
    links = [html.A(path, href=base + path, className="download-link") for path in report["files"].values()]
    previews = [html.Img(src=base + name.replace(".step", ".png"), className="cad-preview")
                for key, name in report["files"].items() if key != "assembly_parts"]
    return html.Section([
        html.Div([html.Div([html.H4(title), html.Small(f"STEP · {report['units']} · {report['frame']}")]),
                  html.Span(status.title(), className=f"status-badge {status}")], className="stats-card-header"),
        html.Table([html.Thead(html.Tr([html.Th(text) for text in ("Shape", "Solids", "Valid", "Watertight",
                                                                   "Volume [mm³]")])), html.Tbody(rows)],
                   className="stats-table stations"),
        html.Ul(checks, className="stats-notes"),
        html.Div(links, className="cad-links"),
        html.Div(previews, className="cad-previews"),
        html.Details([html.Summary("Printability notes"),
                      html.Ul([html.Li(text) for text in report["printability"]], className="stats-notes")]),
    ], className="stats-card")


def create_app(config_file=None, output_dir=None):
    app = Dash(__name__, assets_folder=str(flow.APP_DIR / "assets"),
        title="TurboMoC", suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("{%favicon%}",
        f'<link rel="icon" type="image/svg+xml" href="{app.get_asset_url("turbomoc-balanced-atlantic.svg")}">')
    app.server.config["TURBO_MOC_OUTPUT_DIR"] = (
        Path(output_dir).resolve() if output_dir is not None else flow.OUTPUT)

    @app.server.route("/artifacts/<session>/<job>/<filename>")
    def download(session, job, filename):
        folder = flow.directory({"session": session, "job": job})
        if Path(filename).name != filename or Path(filename).suffix.lower() not in (
            ".json", ".yaml", ".cgns", ".log", ".step", ".png"):
            abort(404)
        if Path(filename).suffix.lower() == ".cgns":
            status_file = folder / "status.json"
            if not status_file.exists() or flow.read_json(status_file).get("state") != "complete":
                abort(404)
        return send_from_directory(folder, filename, as_attachment=True)

    def layout():
        project = project_model.load_project(config_file) if config_file is not None else project_model.default_project()
        row = project_model.active_rows(project)[0]
        config = project_model.row_configuration(project, row["id"])
        stores = [dcc.Store(id=n) for n in ("config", "job", "job-completion", "launch-busy",
            "geometry-data", "geometry-key", "volume-data", "mesh-width")]
        return html.Div([*stores, dcc.Store(id="session", data=uuid4().hex),
            dcc.Store(id="loaded-project"), dcc.Download(id="configuration-download"),
            dcc.Download(id="project-download"),
            dcc.Store(id="base-project", data=project), dcc.Store(id="row-references", data={}),
            dcc.Store(id="applied-blade-count-jobs", data={}),
            dcc.Interval(id="poll", interval=750),
            html.Header([html.H2("TurboMoC"), html.A([
                html.Strong("Sustainable Thermal Power"),
                html.Span("Research group at DTU Construct")],
                href="https://thermalpower.dtu.dk/", target="_blank", rel="noopener noreferrer",
                className="authorship", id="authorship")]),
            html.Div([html.Aside([
                html.Div([
                    html.H3("Configuration files", className="sidebar-box-title"),
                    html.Div([
                        dcc.Upload(id="configuration-upload", accept=".yaml,.yml", multiple=False,
                            children=html.Button("Load configuration", id="load-configuration")),
                        html.Button("Save configuration", id="save-configuration", n_clicks=0),
                    ], className="configuration-actions"),
                    html.Small(id="configuration-message"),
                ], className="configuration-settings"),
                html.Div([
                    html.H3("Machine setup", className="sidebar-box-title"),
                    html.Label([html.Span("Design mode"), dcc.Dropdown(id="design-mode", clearable=False,
                        value=project["machine"]["design_mode"], options=[
                            {"label": "Nozzle only", "value": "nozzle"},
                            {"label": "Stator only", "value": "stator"},
                            {"label": "Rotor only", "value": "rotor"},
                            {"label": "Stator + Rotor", "value": "stator_rotor"}])], className="control"),
                    html.Label([html.Span("Machine type"), dcc.Dropdown(id="machine-family", clearable=False,
                        value=project["machine"]["geometry_type"], options=[
                            {"label": "Linear cascade", "value": "linear"},
                            {"label": "Axial flow", "value": "axial"},
                            {"label": "Radial flow", "value": "radial"}])], className="control",
                        id="machine-family-control"),
                    html.Div(nozzle_setup(project), id="nozzle-setup"),
                    html.Label([html.Span("Working fluid"), fluid_dropdown(
                        MACHINE_FLUID_ID, project["machine"]["fluid_name"])],
                        className="control"),
                    html.Small(id="machine-note"),
                ], className="project-settings"),
                row_controls(project, row),
                html.Div(id="settings", children=sidebar(project)),
            ], id="sidebar"), html.Div(id="sidebar-divider", className="sidebar-divider", role="separator",
                tabIndex=0, title="Drag to resize sidebar", **{"aria-label": "Resize sidebar",
                    "aria-orientation": "vertical", "aria-controls": "sidebar"}), html.Main([
                dcc.Tabs(id="workspace", value="design", children=[
                    dcc.Tab(label="Geometry generation", value="design"),
                    dcc.Tab(label="Mesh generation", value="mesh")]),
                html.Div(id="validation", className="message"),
                html.Div([html.Div([html.Button(label, id=name, n_clicks=0,
                    disabled=name=="cancel") for name, label in zip(ACTIONS,
                    ("Compute blade design",
                     "Generate 2D mesh", "Generate 3D mesh", "Cancel job"))],
                    className="actions"), html.Div([html.Span(id="spinner"), html.Span("Ready", id="job-status"),
                    html.Progress(id="job-progress", max=1, value=0)], className="job-strip"),
                    html.Button("Export project (.zip)", id="download-project", n_clicks=0,
                        disabled=True, title=project_export.NO_MESH)], className="action-bar"),
                html.Div(id="job-error", className="message"),
                html.Div(id="project-download-message", className="message"),
                html.Div([dcc.Loading(html.Div([
                    html.Section([
                        html.Div([
                        html.Details([html.Summary("Nozzle geometry"), html.Div([
                            html.Div([dcc.Graph(id="nozzle-section", figure=empty("Nozzle cross section"),
                                config={"displaylogo": False}, className="engineering-plot")], className="pane"),
                            html.Div([dcc.Graph(id="nozzle-meridional", figure=empty("Meridional geometry"),
                                config={"displaylogo": False}, className="engineering-plot")], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Axis and wall distributions"), html.Div([
                            html.Div([dcc.Graph(id="nozzle-axis", figure=empty("Axis properties"),
                                config={"displaylogo": False}, style={"height": "300px"})], className="pane"),
                            html.Div([dcc.Graph(id="nozzle-wall", figure=empty("Wall properties"),
                                config={"displaylogo": False}, style={"height": "300px"})], className="pane"),
                        ], className="split")], open=False, className="plot-group"),
                        ], id="nozzle-views", style={"display": "none"}),
                        html.Div([
                        html.Details([html.Summary("Blade-to-blade and meridional channel geometry"), html.Div([
                            html.Div([dcc.Graph(id="design-section", figure=empty("Selected blade row"),
                                config={"displaylogo": False}, className="engineering-plot"), html.Small(id="design-basis", style={"display": "none"})], className="pane"),
                            html.Div([dcc.Graph(id="meridional_channel", figure=empty("Machine geometry"),
                                config={"displaylogo": False}, className="engineering-plot"), html.Small(id="assembly-basis", style={"display": "none"})], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Three dimensional geometry"),
                            *display_panel.view(project, "geometry", viewport("geometry"))],
                            open=True, className="plot-group", id="geometry-details"),
                        html.Details([html.Summary("Design statistics"),
                            html.Div(id="design-statistics", className="design-statistics")],
                            open=False, className="plot-group"),
                        ], id="blade-views"),
                        html.Details([html.Summary("CAD export (STEP)"),
                            html.Div([html.Button("Generate CAD", id="cad", n_clicks=0, disabled=True),
                                html.Small("STEP solids in millimetres. Blade rows: fluid domain, blade, and disk-and-blade "
                                           "assembly as separate parts and fused. Nozzle: fluid domain and nozzle body.")],
                                className="cad-actions"),
                            html.Div(id="cad-view", className="design-statistics")],
                            open=False, className="plot-group"),
                    ], id="design-panel"),
                    html.Section([
                        html.Div([
                        html.Details([html.Summary("Meridional and cross-plane mesh"), html.Div([
                            html.Div([dcc.Graph(id="nozzle-mesh-2d", figure=empty("Structured 2D mesh"),
                                config={"displaylogo": False}, className="engineering-plot")], className="pane"),
                            html.Div([dcc.Graph(id="nozzle-mesh-cross", figure=empty("Cross-plane grid"),
                                config={"displaylogo": False}, className="engineering-plot")], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Three dimensional mesh"),
                            html.Div([html.Div(html.Div(id="nozzle-mesh-3d", className="vtk-host"), className="vtk-frame"),
                                nozzle_views.display_panel()], className="display-viewport"),
                            html.Div(id="nozzle-mesh-3d-info", className="message")],
                            open=True, className="plot-group"),
                        html.Details([html.Summary("Mesh quality report"),
                            html.Aside(id="nozzle-mesh-report", children="Generate a mesh to see its report.",
                                className="report")], open=True, className="plot-group mesh-quality"),
                        ], id="nozzle-mesh-views", style={"display": "none"}),
                        html.Div([
                        html.Details([html.Summary("Blade-to-blade and meridional channel mesh"), html.Div([
                            html.Div([dcc.Graph(id="reference-plot", figure=plots.section_mesh_placeholder("2D reference mesh"),
                                config={"displaylogo": False}, className="engineering-plot"), html.Small(id="reference-basis", style={"display": "none"})], className="pane"),
                            html.Div([dcc.Graph(id="meridional-plot", figure=empty("Span grid projection"),
                                config={"displaylogo": False}, className="engineering-plot"), html.Small(id="meridional-basis", style={"display": "none"})], className="pane"),
                        ], className="split")], open=True, className="plot-group"),
                        html.Details([html.Summary("Three dimensional mesh"),
                            *display_panel.view(project, "mesh", viewport("mesh")),
                            html.Div(id="volume-status", className="message")], open=True, className="plot-group", id="mesh-details"),
                        html.Details([html.Summary("Mesh quality report"),
                                html.Div([html.Label("Report", htmlFor="report-scope"), dcc.Dropdown(id="report-scope", value="all",
                                    options=[{"label": "Total", "value": "all"}, *calculations.scope_options(project)],
                                    clearable=False, searchable=False)], className="report-controls"),
                                html.Aside(id="report", children="Generate a 3D mesh to see its report.", className="report")],
                                open=True, className="plot-group mesh-quality", id="mesh-quality-details"),
                        ], id="blade-mesh-views"),
                    ], id="mesh-panel"),
                ], id="views"), type="circle", delay_show=300)],
                    className="content-scroll"),
                html.Details([html.Summary("Terminal · live activity"), html.Pre(id="activity",
                    children="No background job yet.")], open=False, className="terminal", id="terminal"),
            ])], className="shell")], id="app")
    app.layout = layout
    for pane in ("geometry", "mesh"):
        display_panel.register(app, pane)
    inspection_panel.register(app)

    @app.callback(Output({"type": "natural-blade-sizing", "row": ALL}, "children"),
        Input("config", "data"), Input("row-references", "data"),
        State({"type": "natural-blade-sizing", "row": ALL}, "id"))
    def natural_blade_sizing(state, registry, identifiers):
        readouts = []
        for identifier in identifiers:
            try:
                if not state or not state.get("valid"):
                    raise ValueError("Correct the settings to update natural blade sizing.")
                row_id = identifier["row"]
                config = flow.resolved_configuration(state["project"], row_id, registry)
                reference = (registry or {}).get(row_id, {}).get("nozzle")
                natural = blade_counts.from_reference(config, reference)
                count = natural["natural_num_blades"]
                fields = [html.Label([html.Span("Natural number of blades (midspan)"),
                    dcc.Input(value=round(count, 5) if count is not None else "N/A — linear cascade",
                        type="text", disabled=True)], className="control computed-control"),
                    html.Label([html.Span("Natural pitch [mm]"),
                        dcc.Input(value=round(natural["natural_pitch_mm"], 5), type="number", disabled=True)],
                        className="control computed-control")]
                if count is not None:
                    actual = config["meridional_channel"]["num_blades"]
                    deviation = 100 * (count / actual - 1)
                    tooltip = (f"Suggested integer count: {natural['suggested_num_blades']}. "
                        f"Actual pitch differs from natural pitch by {deviation:+.2f}% at midspan.")
                    fields[0].title = tooltip
                    fields[1].title = tooltip
                readouts.append(fields)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                readouts.append(html.Small(str(exc)))
        return readouts

    @app.callback(Output({"type": "actual-blade-count", "row": ALL}, "children"),
        Output("applied-blade-count-jobs", "data"), Input("row-references", "data"),
        State("config", "data"), State({"type": "actual-blade-count", "row": ALL}, "id"),
        State("applied-blade-count-jobs", "data"), prevent_initial_call=True)
    def apply_natural_blade_counts(registry, state, identifiers, applied):
        applied = deepcopy(applied or {})
        controls = []
        for identifier in identifiers:
            row_id = identifier["row"]
            reference = (registry or {}).get(row_id, {}).get("nozzle")
            job = [reference.get("session"), reference.get("job")] if reference else None
            if not job or applied.get(row_id) == job or not state or not state.get("valid"):
                controls.append(no_update)
                continue
            try:
                config = flow.resolved_configuration(state["project"], row_id, registry)
                natural = blade_counts.from_reference(config, reference)
                count = natural["suggested_num_blades"]
                controls.append(actual_blade_count(count, row_id) if count is not None else no_update)
                applied[row_id] = job
            except (ValueError, KeyError, TypeError, OSError, RuntimeError):
                # Obsolete or failed solves must not replace an editable count.
                controls.append(no_update)
        return controls, applied

    @app.callback(Output({"type": "rotor-speed-label", "row": MATCH}, "children"),
        Input({"type": "parameter", "row": MATCH, "path": "inlet_coupling.rotational_speed.type"}, "value"))
    def speed_value_label(kind):
        return SPEED_TYPES[kind]

    @app.callback(Output(MACHINE_FLUID_ID, "options"), Output(MACHINE_FLUID_ID, "value"),
        Input("base-project", "data"))
    def machine_fluid(project):
        value = project["machine"]["fluid_name"]
        return sorted(set(fluid_options()) | {value}), value

    for section in ("method_of_characteristics", "rotor_design"):
        @app.callback(
            Output({"type": "fluid-property-label", "row": MATCH, "section": section, "index": 1}, "children"),
            Output({"type": "fluid-property-label", "row": MATCH, "section": section, "index": 2}, "children"),
            Input({"type": "parameter", "row": MATCH, "path": f"{section}.inlet_property_pair"}, "value"))
        def inlet_property_labels(pair):
            return property_label(pair, 1), property_label(pair, 2)

    @app.callback(Output("calculation-scope", "options"), Output("calculation-scope", "value"),
        Output("report-scope", "options"), Output("report-scope", "value"),
        Input("base-project", "data"), State("calculation-scope", "value"), State("report-scope", "value"))
    def calculation_scope(project, scope, report):
        options = calculations.scope_options(project)
        available = {"all", *[item["value"] for item in options]}
        scope, report = [value if value in available else "all" for value in (scope, report)]
        return ([{"label": "All rows", "value": "all"}, *options], scope,
            [{"label": "Total", "value": "all"}, *options], report)

    @app.callback(Output("geometry-display-row", "options"), Output("geometry-display-row", "value"),
        Output("mesh-display-row", "options"), Output("mesh-display-row", "value"),
        Output("plotted-rows", "options"), Output("plotted-rows", "value"),
        Input("base-project", "data"), Input("geometry-display-row", "value"), Input("mesh-display-row", "value"),
        State("plotted-rows", "value"), State("plotted-rows", "options"))
    def displayed_rows(project, geometry, mesh, previous, old_options):
        options = calculations.scope_options(project)
        available = [item["value"] for item in options]
        if ctx.triggered_id == "base-project" or not ctx.triggered_id:
            old = [item["value"] for item in old_options or []]
            selected = available if set(previous or []) == set(old) else [r for r in available if r in (previous or [])]
            if previous and not selected:
                selected = available
        else:
            chosen = mesh if ctx.triggered_id == "mesh-display-row" else geometry
            selected = display_panel.selected_rows(chosen, previous, available)
        checked = (["all"] if selected == available else []) + selected
        choices = [{"label": "All rows", "value": "all"}, *options]
        return choices, checked, choices, checked, options, selected

    @app.callback(Output("loaded-project", "data"), Output("configuration-download", "data"),
        Output("configuration-message", "children"),
        Output("row-references", "data", allow_duplicate=True), Output("session", "data"),
        Output("job", "data", allow_duplicate=True),
        Input("configuration-upload", "contents"), Input("save-configuration", "n_clicks"),
        State("config", "data"),
        State("base-project", "data"), State("job", "data"), State("job-completion", "data"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), prevent_initial_call=True)
    def configuration_file(contents, clicks, state, base, handle, completed, values, identifiers):
        unchanged = (no_update,) * 3
        try:
            if ctx.triggered_id == "configuration-upload":
                if not contents:
                    return (no_update,) * 6
                if handle and completed != handle["job"]:
                    raise ValueError("Wait for the active job or cancel it before loading a configuration.")
                project = read_uploaded_project(contents)
                return project, no_update, "", {}, uuid4().hex, None
            project = project_model.validate_project(flow.edited_project(
                state["project"] if state and state.get("project") else base, values, identifiers))
            if state and not state.get("valid"):
                raise ValueError("Correct the invalid settings before saving the configuration.")
            saved = project_model.nozzle_case(project) if nozzle_mode(project) else project
            download = {"content": yaml.safe_dump(saved, sort_keys=False, allow_unicode=True),
                "filename": f"turbo_moc_{datetime.now():%Y-%m-%d_%H-%M-%S}.yaml", "type": "text/yaml"}
            return no_update, download, "Configuration saved.", *unchanged
        except (ValueError, TypeError, KeyError) as exc:
            return no_update, no_update, str(exc), *unchanged

    @app.callback(Output("base-project", "data"), Output("settings", "children"),
        Output("selected-row", "options"), Output("selected-row", "value"),
        Output("machine-family", "options"), Output("machine-family", "value"),
        Output("design-mode", "value"), Output("nozzle-setup", "children"),
        Output("machine-note", "children"),
        Input("design-mode", "value"), Input("machine-family", "value"),
        State("config", "data"), State("base-project", "data"), State("selected-row", "value"),
        Input("loaded-project", "data"))
    def structure(mode, family, state, base, selected, loaded=None):
        # Even an invalid edit state retains its last validated project. A mode
        # change must preserve those accepted edits when hiding a blade row.
        project = deepcopy(state["project"] if state and state.get("project") else base)
        if loaded is not None and ctx.triggered_id == "loaded-project":
            project = deepcopy(loaded)
            mode, family = project["machine"]["design_mode"], project["machine"]["geometry_type"]
            state = None
        note = ("Invalid entries discarded; last valid settings retained."
                if state and not state.get("valid") else "")
        if mode == "stator_rotor" and family == "linear":
            family = "axial"
            note += " Stator + Rotor uses an annular machine; Axial flow selected."
        # Change the family first: a combined project cannot validate as linear.
        if mode != "stator_rotor":
            project = project_model.set_design_mode(project, mode)
        project = project_model.set_machine_type(project, family)
        project = project_model.set_design_mode(project, mode)
        rows = project_model.active_rows(project)
        identifiers = [r["id"] for r in rows]
        selected = selected if selected in identifiers else identifiers[0]
        options = [{"label": label, "value": value, "disabled": mode=="stator_rotor" and value=="linear"}
            for label, value in (("Linear cascade", "linear"), ("Axial flow", "axial"), ("Radial flow", "radial"))]
        return (project, sidebar(project), [{"label": row_label(r), "value": r["id"]} for r in rows],
                selected, options, family, mode, nozzle_setup(project), note)

    @app.callback(Output("config", "data"), Output("validation", "children"),
        Input({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        Input("base-project", "data"), Input("row-references", "data"), Input("selected-row", "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), State("config", "data"))
    def edit(values, base, registry, row_id, identifiers, previous=None):
        try:
            # A loaded project replaces the controls. Values from the previous
            # sidebar may still accompany this update; retain the uploaded values.
            try:
                replaced = ctx.triggered_id == "base-project"
            except MissingCallbackContextException:
                replaced = False
            project = project_model.validate_project(
                deepcopy(base) if replaced else flow.edited_project(base, values, identifiers))
            row = project_model.get_row(project, row_id)
            state = {"project": project, "row_id": row_id, "role": row["role"], "valid": True,
                     "supported": True, "config": project_model.row_configuration(project, row_id)}
            try:
                config = flow.resolved_configuration(project, row_id, registry)
            except (ValueError, OSError, RuntimeError) as exc:
                # Missing or stale upstream results are a workflow dependency,
                # not an invalid project: retain editable settings and explain
                # why rotor execution is temporarily unavailable.
                state.update(supported=False, dependency_error=str(exc))
                return state, str(exc)
            state.update(config=config, **flow.keys(config, (registry or {}).get(row_id, {}).get("nozzle")))
            return state, ""
        except (ValueError, TypeError, KeyError) as exc:
            # Leave the raw input mounted so it can be corrected, but retain
            # the last accepted row settings for later structural changes.
            accepted = deepcopy(previous["project"] if previous and previous.get("project") else base)
            return {"project": accepted, "row_id": row_id, "valid": False, "supported": False}, str(exc)

    @app.callback(Output({"type": "shared-domain", "row": ALL}, "children"),
        Input("config", "data"), State({"type": "shared-domain", "row": ALL}, "id"))
    def shared_domain_display(state, identifiers):
        values = []
        for identifier in identifiers:
            try:
                if not state or not state.get("valid"):
                    values.append([])
                    continue
                values.append(shared_domain_readout(state["project"], identifier["row"]))
            except (ValueError, KeyError, TypeError) as exc:
                values.append(html.Small(f"Cannot derive the shared boundary: {exc}"))
        return values

    @app.callback(
        Output({"type": "rotor-manual-inlet", "row": ALL}, "style"),
        Output({"type": "rotor-coupled-inlet", "row": ALL}, "children"),
        Output({"type": "rotor-coupled-inlet", "row": ALL}, "style"),
        Output({"type": "rotor-coupling-speed", "row": ALL}, "style"),
        Output({"type": "rotor-inlet-warning", "row": ALL}, "children"),
        Input("config", "data"), Input("row-references", "data"),
        State({"type": "rotor-manual-inlet", "row": ALL}, "id"))
    def rotor_inlet_display(state, registry, identifiers):
        manual_styles, computed_values, computed_styles, speed_styles = [], [], [], []
        inlet_alerts = []
        for identifier in identifiers:
            row_id = identifier["row"]
            try:
                row = project_model.get_row(state["project"], row_id)
            except (KeyError, TypeError, ValueError):
                manual_styles.append({})
                computed_values.append([])
                computed_styles.append({"display": "none"})
                speed_styles.append({"display": "none"})
                inlet_alerts.append([])
                continue
            coupled = row["geometry"]["inlet_coupling"]["mode"] == "stator_exit"
            manual_styles.append({"display": "none"} if coupled else {})
            computed_styles.append({} if coupled else {"display": "none"})
            speed_styles.append({} if coupled else {"display": "none"})
            values, message = {}, (
                "Computed from the current stator exit and rotor speed. "
                "Ideal transfer of the stator exit state; no interspace-flow/loss model.")
            warnings = rotor_inlet.inlet_warnings(row["geometry"]) if not coupled else []
            if coupled:
                try:
                    if not state.get("valid"):
                        raise ValueError("Correct the invalid settings to update the coupled inlet state.")
                    resolved = flow.resolved_configuration(state["project"], row_id, registry)
                    values = resolved["rotor_design"]
                    source = resolved["rotor_source"]
                    message += (f" Inlet midspan radius: {source['inlet_radius_mm']:.4g} mm; "
                                f"blade speed: {source['triangle']['U']:.4g} m/s.")
                    speed = source.get("rotational_speed")
                    if speed:
                        message += f" Resolved speed: {speed['rpm']:.5g} rpm."
                        if "reference_pressure" in speed:
                            message += (f" Stator back-pressure reference: {speed['reference_pressure']:.5g} Pa; "
                                        f"c₀: {speed['c0']:.5g} m/s; estimated mass flow: {speed['mass_flow']:.5g} kg/s.")
                    warnings = rotor_inlet.inlet_warnings(resolved)
                except (ValueError, OSError, RuntimeError) as exc:
                    message = str(exc)
            inlet_alerts.append([html.Div(warning, className="inlet-warning", role="alert")
                                 for warning in warnings])
            computed_values.append([
                html.Label([html.Span(label), dcc.Input(value=values.get(name), disabled=True,
                    type="text" if name == "fluid_name" else "number", placeholder="Pending")],
                    className="control computed-control",
                    title="Calculated from stator exit; switch to manual inlet conditions to edit.")
                for name, label in ROTOR_INLET_FIELDS.items()] + [html.Small(message)])
        return manual_styles, computed_values, computed_styles, speed_styles, inlet_alerts

    @app.callback(Output("sidebar", "className"), Output("design-panel", "style"), Output("mesh-panel", "style"),
        Input("workspace", "value"))
    def workspace(name):
        return (f"active-{name}",
            {} if name == "design" else {"display": "none"}, {} if name == "mesh" else {"display": "none"})

    @app.callback(Output("machine-family-control", "style"), Output("interspace-note", "style"),
        Input("design-mode", "value"))
    def nozzle_sidebar(mode):
        hidden = {"display": "none"} if mode == "nozzle" else {}
        return hidden, hidden

    @app.callback(Output({"type": "nozzle-kind", "row": MATCH, "field": ALL}, "style"),
        Input({"type": "parameter", "row": MATCH, "path": "method_of_characteristics.delta_flow"}, "value"),
        State({"type": "nozzle-kind", "row": MATCH, "field": ALL}, "id"))
    def nozzle_kind_fields(delta, identifiers):
        kind = "axisymmetric" if delta else "planar"
        return [{} if identifier["field"].startswith(kind + ":") else {"display": "none"}
                for identifier in identifiers]

    @app.callback(Output({"type": "outlet-field", "row": MATCH, "condition": "pressure"}, "style"),
        Output({"type": "outlet-field", "row": MATCH, "condition": "mach"}, "style"),
        Input({"type": "parameter", "row": MATCH, "path": "method_of_characteristics.outlet_condition"}, "value"))
    def outlet_fields(condition):
        hidden = {"display": "none"}
        return ({}, hidden) if condition != "mach" else (hidden, {})

    @app.callback(Output({"type": "rotor-outlet-field", "row": MATCH, "condition": "pressure"}, "style"),
        Output({"type": "rotor-outlet-field", "row": MATCH, "condition": "mach"}, "style"),
        Input({"type": "parameter", "row": MATCH, "path": "rotor_design.outlet_condition"}, "value"))
    def rotor_outlet_fields(condition):
        hidden = {"display": "none"}
        return ({}, hidden) if condition == "pressure" else (hidden, {})

    @app.callback(Output("nozzle-views", "style"), Output("blade-views", "style"),
        Output("nozzle-mesh-views", "style"), Output("blade-mesh-views", "style"),
        Input("config", "data"))
    def nozzle_visibility(state):
        shown = bool(state and state.get("project") and nozzle_mode(state["project"]))
        nozzle, blade = ({}, {"display": "none"}) if shown else ({"display": "none"}, {})
        return nozzle, blade, nozzle, blade

    @app.callback(Output("nozzle-axis", "figure"), Output("nozzle-wall", "figure"),
        Input("config", "data"), Input("row-references", "data"))
    def nozzle_report(state, registry):
        blank = (empty("Axis properties"), empty("Wall properties"))
        if not state or not state.get("valid") or not nozzle_mode(state["project"]):
            return blank
        row = project_model.active_rows(state["project"])[0]
        try:
            config = flow.resolved_configuration(state["project"], row["id"], registry)
            data = flow.load_nozzle((registry or {}).get(row["id"], {}).get("nozzle"), config)
        except (ValueError, OSError, RuntimeError, KeyError):
            return blank
        return tuple(figure.update_layout(showlegend=False) for figure in (
            nozzle_plots.plot_axis_properties(data), nozzle_plots.plot_wall_properties(data)))

    def nozzle_shape(state, registry):
        row = project_model.active_rows(state["project"])[0]
        config = flow.resolved_configuration(state["project"], row["id"], registry)
        refs = (registry or {}).get(row["id"], {})
        data = flow.load_nozzle(refs.get("nozzle"), config)
        parameters = config["method_of_characteristics"]
        return config, refs, data, nozzle_geometry.contour(data, parameters, config["mesh"]["nozzle"])

    @app.callback(Output("nozzle-section", "figure"), Output("nozzle-meridional", "figure"),
        Input("config", "data"), Input("row-references", "data"))
    def nozzle_geometry_views(state, registry):
        blank = (empty("Nozzle cross section"), empty("Meridional geometry"))
        if not state or not state.get("valid") or not nozzle_mode(state["project"]):
            return blank
        try:
            config, _, _, shape = nozzle_shape(state, registry)
        except (ValueError, OSError, RuntimeError, KeyError):
            return blank
        return (nozzle_views.section_figure(shape),
                nozzle_views.meridional_figure(shape, config["mesh"]["nozzle"]["planar_depth"]))

    @app.callback(Output("nozzle-mesh-2d", "figure"), Output("nozzle-mesh-cross", "figure"),
        Output("nozzle-mesh-3d-info", "children"), Output("nozzle-mesh-report", "children"),
        Input("config", "data"), Input("row-references", "data"))
    def nozzle_mesh_views(state, registry):
        blank = (empty("Structured 2D mesh"), empty("Cross-plane grid"))
        if not state or not state.get("valid") or not nozzle_mode(state["project"]):
            return *blank, "", ""
        row = project_model.active_rows(state["project"])[0]
        refs = (registry or {}).get(row["id"], {})
        try:
            config = flow.resolved_configuration(state["project"], row["id"], registry)
            keys = flow.keys(config, refs.get("nozzle"))
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            return *blank, str(exc), ""
        figure, cross, note, report = *blank, "Generate the 3D mesh.", []
        settings = config["mesh"]["nozzle"]
        width = config["mesh"]["visualization"]["section_edge_width"]
        reference = refs.get("reference")
        if reference:
            try:
                data = nozzle_views.load(reference, "nozzle_mesh_2d.json")
                summary = data["summary"]
                figure = nozzle_views.mesh_2d_figure(flow.directory(reference), summary["axisymmetric"], max(width, 0.6))
                cross = nozzle_views.cross_figure(summary, settings, max(width, 0.6))
                status = [] if flow.current(reference, keys["reference"]) else [html.Div(
                    "Outdated 2D mesh: regenerate for the current settings.", className="message")]
                report += [html.H3("2D mesh"), *status, *nozzle_views.report_2d(summary, data["report"])]
            except (OSError, ValueError, KeyError) as exc:
                report.append(html.Div(f"2D mesh unavailable: {exc}", className="message"))
        volume = refs.get("volume")
        if volume:
            try:
                summary = nozzle_views.load(volume, "nozzle_mesh_3d.json")
                note = ("" if flow.current(volume, keys["volume"]) else
                        "Outdated 3D mesh: regenerate for the current settings.")
                status = [html.Div(note, className="message")] if note else []
                report += [html.H3("3D mesh"), *status,
                           *nozzle_views.report_3d(summary, nozzle_views.load(volume, "mesh_report.json"))]
            except (OSError, ValueError, KeyError) as exc:
                note = f"3D mesh unavailable: {exc}"
        return figure, cross, note, report or "Generate a mesh to see its report."

    @app.callback(Output("nozzle-mesh-3d", "children"), Output("nozzle-display-surfaces", "options"),
        Input("config", "data"), Input("row-references", "data"),
        Input("nozzle-display-surfaces", "value"), Input("nozzle-display-edges", "value"))
    def nozzle_mesh_scene(state, registry, surfaces, edges):
        if not state or not state.get("valid") or not nozzle_mode(state["project"]):
            return None, no_update
        row = project_model.active_rows(state["project"])[0]
        volume = (registry or {}).get(row["id"], {}).get("volume")
        if not volume:
            return None, no_update
        try:
            scene, available = nozzle_views.mesh_3d_scene(flow.directory(volume), surfaces or [], bool(edges))
        except (OSError, ValueError, KeyError):
            return None, no_update
        return scene, [{"label": name.title(), "value": name} for name in available]

    @app.callback(Output("nozzle-display-surfaces", "value"),
        Input({"type": "nozzle-display-preset", "action": ALL}, "n_clicks"), prevent_initial_call=True)
    def nozzle_display_preset(_):
        return (list(nozzle_views.SURFACE_COLORS) if ctx.triggered_id["action"] == "all"
                else list(nozzle_views.DEFAULT_SURFACES))

    @app.callback(Output("design-statistics", "children"),
        Input("config", "data"), Input("row-references", "data"))
    def statistics(state, registry):
        if not state or not state.get("valid") or nozzle_mode(state["project"]):
            return ""
        return design_statistics.panel(state["project"], registry)

    @app.callback(Output("cad-view", "children"), Input("config", "data"), Input("row-references", "data"))
    def cad_view(state, registry):
        if not state or not state.get("valid"):
            return ""
        cards = []
        for row in project_model.active_rows(state["project"]):
            refs = (registry or {}).get(row["id"], {})
            reference = refs.get("cad")
            if not reference:
                continue
            try:
                config = flow.resolved_configuration(state["project"], row["id"], registry)
                status = "current" if flow.current(reference, flow.keys(config, refs.get("nozzle"))["cad"]) else "outdated"
                report = json.loads((flow.directory(reference) / "cad_report.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                cards.append(html.Div(f"{row_label(row)}: CAD export unavailable ({exc}).", className="message"))
                continue
            cards.append(cad_card(row_label(row) if not nozzle_mode(state["project"]) else "Nozzle",
                                  status, report, reference))
        return cards or html.Small("Generate CAD to export STEP files.", className="message")

    @app.callback(Output("mesh-width", "data"), Input("config", "data"))
    def display_settings(state):
        if not state or not state.get("valid"):
            return no_update
        config = state.get("config")
        return config["mesh"]["visualization"]["volume_edge_width"] if config else 0.35

    @app.callback(Output("geometry-data", "data"), Output("geometry-key", "data"),
        Output("design-section", "figure"), Output("meridional_channel", "figure"),
        Output("design-basis", "children"), Output("assembly-basis", "children"),
        Input("config", "data"), Input("row-references", "data"),
        Input("plotted-rows", "value"), State("geometry-key", "data"))
    def geometry(state, registry, visible=None, previous_key=None):
        if not state or not state.get("valid") or nozzle_mode(state["project"]):
            return None, None, empty("Selected blade row"), empty("Machine geometry"), "", ""
        ready = list(ready_rows(state, registry))
        active = project_model.active_rows(state["project"])
        visible = {r["id"] for r in active} if visible is None else set(visible)
        placements = {r["id"]: project_model.assembly_placement(state["project"], r["id"]) for r in active}
        chart_rows = [(r["id"], r["geometry"]["meridional_channel"], placements[r["id"]]) for r in active]
        planning = [(row_label(r), r["geometry"]["meridional_channel"], placements[r["id"]])
            for r in active if r["id"] in visible and r["id"] not in {item[0]["id"] for item in ready}]
        try:
            interfaces = project_model.shared_interfaces(state["project"])
            interface_error = ""
        except ValueError as exc:
            interfaces, interface_error = [], str(exc)
        data_key = flow.digest([(r["id"], k["geometry"], placements[r["id"]]) for r, _, _, k in ready])
        interfaces = [item for item in interfaces if {item["upstream_id"], item["downstream_id"]} <= visible]
        key = flow.digest([sorted(visible), chart_rows, planning, interfaces, interface_error, [(r["id"], k["geometry"], placements[r["id"]])
            for r, _, _, k in ready]])
        if key == previous_key:
            return (no_update,) * 6
        datasets, views, errors, section_views, basis_notes = [], [], [], [], []
        basis = "Compute blade designs for the plotted rows to display their geometry."
        for row, config, reference, keys in ready:
            try:
                design = flow.get_design(config, reference)
                dataset = deepcopy(cached_preview(keys["geometry"], design))
                dataset.update(id=row["id"], role=row["role"], label=row_label(row), placement=placements[row["id"]],
                    geometry_type=config["meridional_channel"]["type"])
                datasets.append(dataset)
                if row["id"] in visible:
                    views.append((row_label(row), design, placements[row["id"]]))
                if row["id"] in visible:
                    section_views.append((row["id"], row_label(row), design, None, 0))
                    method = "vortex-method" if row["role"] == "rotor" else "MoC"
                    basis = (f"Physical linear cascade section [mm]; uniformly scaled {method} blade and pitch."
                        if dataset.get("is_linear") else
                        f"Reference conformal section: {method} coordinates scaled to the selected meridional chord.")
                    if row["role"] == "rotor":
                        pitch_ratio = design.blade["pitch"] / design.blade["natural_pitch"]
                        basis += (f" Relative outlet angle: {design.blade['metal_angle_out']:.3g} deg; "
                                  f"reference pitch: {design.blade['pitch']:.3g} mm "
                                  f"(vortex natural pitch: {design.blade['natural_pitch']:.3g} mm; "
                                  f"applied/natural ratio: {pitch_ratio:.3g}).")
                        if abs(pitch_ratio - 1) > 0.01:
                            basis += (" Blade count/chord change the vortex design pitch; the original cascade "
                                      "flow solution then serves as a profile design reference.")
                    basis_notes.append(f"{row_label(row)}: {basis}")
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(f"{row_label(row)}: {exc}")
        section, _ = plots.assembly_section_figure(state["project"], section_views)
        if section_views:
            basis = ("Shared midspan conformal chart: m′ starts at the first active row's leading edge; "
                "physical interspace and angular phase are included. " if len(active) > 1 else "") + " ".join(basis_notes)
        elif not visible:
            basis = "Select a row under Configure display > Blade rows to display its geometry."
        note = ("Shared machine view. Adjacent row domains meet at the marked physical interface; "
                "3D CGNS meshes use this assembly placement. Each row remains a separate mesh; "
                "solver interface coupling is configured in the CFD solver." if interfaces else
                "Shared machine view. A single-row mesh uses its row-local coordinates.")
        if interface_error:
            note = f"Shared interface unavailable: {interface_error}"
        if planning:
            note += " Pale envelopes show prescribed blade extents awaiting a current blade design."
        note += " Solid outlines: blade extent; dashed outlines: inlet/outlet flow-domain extensions."
        if errors:
            note += " " + " ".join(errors)
        return {"key": data_key, "rows": datasets}, key, section, plots.assembly_meridional_figure(
            views, planning, interfaces=interfaces), basis, note

    @app.callback(Output("reference-plot", "figure"), Output("meridional-plot", "figure"),
        Output("reference-basis", "children"), Output("meridional-basis", "children"),
        Input("config", "data"), Input("row-references", "data"),
        Input("plotted-rows", "value"), State("reference-plot", "figure"))
    def mesh_setup(state, registry, visible=None, previous=None):
        if not state or not state.get("valid") or nozzle_mode(state["project"]):
            return plots.section_mesh_placeholder("Selected-row 2D mesh"), empty("Span grid projection"), "Select a valid blade row.", ""
        active = project_model.active_rows(state["project"])
        visible = {r["id"] for r in active} if visible is None else set(visible)
        available = []
        for row in active:
            if row["id"] not in visible:
                continue
            try:
                config = (state["config"] if row["id"] == state["row_id"] and state.get("supported") else
                    flow.resolved_configuration(state["project"], row["id"], registry))
                refs = (registry or {}).get(row["id"], {})
                reference = refs.get("reference")
                key = flow.keys(config, refs.get("nozzle"))["reference"]
                if flow.current(reference, key):
                    available.append((row, config, reference, key))
            except (ValueError, OSError, RuntimeError):
                continue
        # The shared projection contains only rows with a current 2D mesh.
        # Shift all row charts into one assembly m'-theta frame, using the same
        # physical interspace as the meridional view. Never mutate mesh nodes.
        chart_rows = [(r["id"], r["geometry"]["meridional_channel"],
            project_model.assembly_placement(state["project"], r["id"])) for r in active]
        render_key = flow.digest([sorted(visible), chart_rows, [(r["id"], ref["job"], key,
            cfg["mesh"]["numerics"]["span_layers"], cfg["mesh"]["numerics"]["span_stretch"],
            project_model.assembly_placement(state["project"], r["id"])) for r, cfg, ref, key in available]])
        widths = {r["id"]: cfg["mesh"]["visualization"]["section_edge_width"] for r, cfg, _, _ in available}
        meta = (previous or {}).get("layout", {}).get("meta", {})
        if meta.get("mesh_key") == render_key:
            if meta.get("widths") == widths:
                return (no_update,) * 4
            section_patch, span_patch = Patch(), Patch()
            for row_id, index in meta.get("section_traces", {}).items():
                section_patch["data"][index]["line"]["width"] = widths[row_id]
            for row_id, indices in meta.get("span_traces", {}).items():
                for index in indices:
                    span_patch["data"][index]["line"]["width"] = widths[row_id]
            section_patch["layout"]["meta"]["widths"] = widths
            return section_patch, span_patch, no_update, no_update
        basis = "Generate an up-to-date 2D mesh for a plotted row." if visible else "Select a row under Configure display > Blade rows to display its mesh."
        entries, section_entries, basis_notes = [], [], []
        for row, config, reference, _ in available:
            section, design = reference_artifact(reference["session"], reference["job"])
            width = widths[row["id"]]
            entries.append((row["id"], row_label(row), design, config["mesh"]["numerics"], section,
                project_model.assembly_placement(state["project"], row["id"]), width))
            section_entries.append((row["id"], row_label(row), design, section, width))
            if config["mesh"]["numerics"]["inflation_layers"]["enabled"]:
                basis_notes.append((row["id"], row_label(row)))
        section_plot, section_indices = plots.assembly_section_figure(state["project"], section_entries, mesh=True)
        span_plot, indices, details = plots.assembly_meridional_mesh_figure(entries)
        section_plot.update_layout(meta={"mesh_key": render_key, "widths": widths,
            "span_traces": indices, "section_traces": section_indices})
        if entries:
            basis = ("Actual physical 2D mesh [mm]; translational periodicity." if entries[0][2].meridional_channel.is_linear else
                "Actual reference meshes in a shared midspan conformal chart; assembly interspace and angular phase included.")
            for row_id, label in basis_notes:
                lower, upper = details[row_id]["estimated_first_height_mm"]
                basis += f" {label}: sampled mapped first-layer height {lower:.3g}–{upper:.3g} mm."
            if basis_notes:
                basis += " These in-surface estimates are not measured 3D wall-normal distances."
        span_basis = ("Projected reference-boundary nodes and span lines for plotted rows with current 2D meshes; "
            "a sweep preview, not a separate volume mesh. Solid: blade extent; dashed: inlet/outlet domain.") if entries else ""
        for row_id, label, *_ in entries:
            data = details[row_id]
            span_basis += (f" {label}: {data['span_layers']} span layers; spacing "
                f"{data['min_span_spacing_mm']:.3g}–{data['max_span_spacing_mm']:.3g} mm.")
        return section_plot, span_plot, basis, span_basis

    @app.callback(Output("volume-data", "data"), Output("report", "children"),
        Input("config", "data"), Input("row-references", "data"), Input("report-scope", "value"))
    def mesh_view(state, registry, scope="all"):
        if not state or not state.get("valid"):
            return None, "Correct the configuration before inspecting the mesh."
        if nozzle_mode(state["project"]):
            return None, ""
        datasets, entries, issues = [], [], []
        selected_report = "Generate a 3D mesh for this row to see its report."
        rows = project_model.active_rows(state["project"])
        for row in rows:
            reference = (registry or {}).get(row["id"], {}).get("volume")
            if not reference:
                issues.append((row_label(row), "not generated"))
                continue
            try:
                surfaces, summary, report = volume_artifact(reference["session"], reference["job"])
                definition = summary.get("geometry", {})
                linear = definition.get("type") == "linear"
                # Browser surfaces stay row-local. A saved assembly records
                # the one transform also used by its CGNS export. In particular,
                # do not reposition a stale mesh using newly edited gap values.
                assembly = summary.get("assembly") or {}
                placement = assembly.get("placement") or project_model.assembly_placement(
                    state["project"], row["id"])
                dataset = {"id": row["id"], "role": row["role"], "label": row_label(row), "surfaces": surfaces,
                    "is_linear": linear, "placement": placement,
                    "geometry_type": display_panel.geometry_family(definition, state["project"]["machine"]["geometry_type"]),
                    "job": reference["job"], "session": reference["session"],
                    "span_stations": summary.get("span_stations", [])}
                if linear:
                    dataset["pitch_mm"] = definition["pitch_mm"]
                else:
                    cfg = project_model.row_configuration(state["project"], row["id"])
                    dataset["num_blades"] = definition.get("num_blades", cfg["meridional_channel"]["num_blades"])
                datasets.append(dataset)
                status = mesh_reporting.mesh_status(state["project"], row, registry)
                dataset["status"] = status
                if scope == "all":
                    arrays = mesh_reporting.metric_arrays(str(flow.directory(reference) / "volume_data.npz"))
                    entries.append({"id": row["id"], "label": row_label(row), "status": status,
                        "summary": summary, "report": report, "arrays": arrays})
                elif row["id"] == scope:
                    selected_report = [html.H3(row_label(row)), *([html.Div(
                        "Outdated mesh: regenerate for the current settings.", className="message")]
                        if status != "Current" else []), *report_panel(summary, report)]
            except (OSError, ValueError, KeyError) as exc:
                issues.append((row_label(row), f"unavailable: {exc}"))
                if row["id"] == scope:
                    selected_report = str(exc)
        if scope == "all":
            selected_report = mesh_reporting.total_panel(
                [{"id": row["id"], "label": row_label(row)} for row in rows], entries, issues)
        key = flow.digest([(d["id"], d["job"], d["placement"]) for d in datasets])
        return {"key": key, "rows": datasets}, selected_report

    @app.callback(Output("job", "data"), Output("job-error", "children"),
        *[Input(action, "n_clicks") for action in ACTIONS], Input("cad", "n_clicks"),
        State("config", "data"), State("session", "data"), State("job", "data"),
        State("job-completion", "data"), State("row-references", "data"),
        State("calculation-scope", "value"), prevent_initial_call=True,
        running=[(Output("launch-busy", "data"), True, False)])
    def launch(*arguments):
        state, session, handle, completed, registry, scope = arguments[-6:]
        action = ctx.triggered_id
        running = bool(handle and completed != handle["job"])
        if action == "cancel":
            if running:
                calculations.cancel(handle)
            return no_update, ""
        if running:
            return no_update, "Wait for the active row job or cancel it first."
        if not state or not state.get("valid"):
            return no_update, "Correct the invalid settings before calculating."
        try:
            handle = calculations.start(session, state["project"], scope if scope is not None else state["row_id"], action, registry)
            return (handle, "") if handle else (no_update, "All selected meshes are already current.")
        except (ValueError, OSError, RuntimeError) as exc:
            return no_update, str(exc)

    @app.callback(Output("row-references", "data"), Output("job-completion", "data"),
        Output("job-status", "children"), Output("job-progress", "value"), Output("activity", "children"),
        Output("spinner", "className"), Output("workspace", "value"),
        Output("selected-row", "value", allow_duplicate=True),
        Output("job", "data", allow_duplicate=True), Input("poll", "n_intervals"), Input("job", "data"),
        State("job-completion", "data"), State("row-references", "data"), State("base-project", "data"),
        State("plotted-rows", "value"),
        prevent_initial_call=True)
    def poll(_, handle, completed, registry, project, visible=None):
        if not handle or completed == handle["job"]:
            return (no_update,) * 9
        if calculations.cancellation_requested(handle):
            calculations.cancel(handle)
        status, log = flow.poll_job(handle)
        done = status["state"] != "running"
        refs, destination, selected, next_handle = no_update, no_update, no_update, None
        if status["state"] == "complete":
            result = json.loads((flow.directory(handle) / "result.json").read_text(encoding="utf-8"))
            refs = deepcopy(registry or {})
            row_id = handle.get("row_id", "stator_1")
            refs.setdefault(row_id, {})[handle["action"]] = result["reference"]
            if row_id in {
                    r["id"] for r in project_model.active_rows(project)}:
                destination = "design" if handle["action"] in ("nozzle", "cad") else "mesh"
                if visible is None or row_id in visible:
                    selected = row_id
            try:
                next_handle = calculations.advance(handle, refs, log)
            except (ValueError, OSError, RuntimeError) as exc:
                status = {**status, "state": "failed", "message": f"Queue stopped: {exc}", "fraction": None}
                log += f"\n{status['message']}\n"
        queue = handle.get("queue", {})
        elapsed = time.time() - queue.get("started", handle["started"])
        count = len(queue.get("rows", [handle.get("row_id")]))
        index = queue.get("index", 0)
        label = row_label(project_model.get_row(queue.get("project", project), handle.get("row_id", "stator_1")))
        message = f"{label}: {status['message']} · row {index + 1} of {count} · {elapsed:.1f} s"
        fraction = status.get("fraction")
        fraction = None if fraction is None else (index + fraction) / count
        log = (queue.get("log", "") + log)[-60000:]
        if next_handle:
            next_row = project_model.get_row(next_handle["queue"]["project"], next_handle["row_id"])
            message = f"{row_label(next_row)}: starting {calculations.STAGE_LABELS[handle['action']]} · row {next_handle['queue']['index'] + 1} of {count}"
        elif done and status["state"] == "complete" and calculations.cancellation_requested(handle):
            message += " · Remaining calculations cancelled."
        elif done and status["state"] == "complete":
            fraction = 1  # Remaining rows may have had already-current meshes.
        return (refs, handle["job"] if done else no_update, message, fraction, log,
                "spinner running" if not done or next_handle else "", destination, selected,
                next_handle if next_handle else no_update)

    @app.callback(Output("job-status", "title"), Input("job-status", "children"))
    def progress_title(message):
        return message or ""

    @app.callback(*[Output(action, "disabled") for action in ACTIONS], Output("cad", "disabled"),
        Output("volume-status", "children"), Output("calculation-controls", "title"),
        Input("config", "data"), Input("row-references", "data"), Input("job", "data"),
        Input("job-completion", "data"), Input("launch-busy", "data"),
        Input("calculation-scope", "value"))
    def readiness(state, registry, handle, completed, launching, scope=None):
        busy = bool(launching or (handle and completed != handle["job"]))
        available, reasons = [], []
        for action in ("nozzle", "reference", "volume", "cad"):
            try:
                if not state or not state.get("valid"):
                    raise ValueError("Correct the invalid settings before calculating.")
                calculations.check_prerequisites(state["project"], scope if scope is not None else state["row_id"], action, registry)
                available.append(True)
                reasons.append("")
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                available.append(False)
                reasons.append(str(exc))
        disabled = (*(busy or not value for value in available[:3]), not busy, busy or not available[3])
        message = ""
        if state and state.get("valid"):
            outdated = [row_label(row) for row in project_model.active_rows(state["project"])
                if (registry or {}).get(row["id"], {}).get("volume")
                and mesh_reporting.mesh_status(state["project"], row, registry) != "Current"]
            if outdated:
                message = f"Outdated volume meshes: {', '.join(outdated)}. Regenerate for the current settings."
        scope_note = "Select a valid calculation scope."
        if state and state.get("valid"):
            try:
                selected_rows = calculations.scoped_rows(state["project"], scope if scope is not None else state["row_id"])
                scope_note = f"Calculations: {' + '.join(row_label(row) for row in selected_rows)}."
            except ValueError:
                scope_note = "Select an active row to calculate."
            scope_note += " Rows run in machine order; plot visibility is independent."
            reason = next((item for item in reasons[:3] if item), "")
            if reason:
                scope_note += " " + reason
        return (*disabled, message, scope_note)

    @app.callback(Output("nozzle", "children"), Input("config", "data"))
    def design_action_label(state):
        if state and state.get("project") and nozzle_mode(state["project"]):
            return "Compute nozzle"
        return "Compute blade design"

    @app.callback(Output("interspace-note", "title"), Input("config", "data"))
    def interspace_note(state):
        if not state or not state.get("valid"):
            return ""
        messages = []
        try:
            interfaces = {(item["upstream_id"], item["downstream_id"]): item
                for item in project_model.shared_interfaces(state["project"])}
        except ValueError as exc:
            return f"Shared interface unavailable: {exc}"
        for spacing in project_model.assembly_interspaces(state["project"]):
            upstream = row_label(project_model.get_row(state["project"], spacing["upstream_id"]))
            downstream = row_label(project_model.get_row(state["project"], spacing["downstream_id"]))
            gap = spacing["gap_mm"]
            if spacing["geometry_type"] == "axial":
                description = f"Minimum axial interspace: {gap:g} mm; prescribed on {downstream}."
            else:
                description = f"Radial interspace along the flow: {gap:g} mm; derived from outlet/inlet radii."
            interface = interfaces.get((spacing["upstream_id"], spacing["downstream_id"]))
            if interface:
                coordinate_name = "z" if spacing["geometry_type"] == "axial" else "r"
                description += f" Shared interface: {coordinate_name} = {interface['coordinate_mm']:g} mm (mid-gap)."
            messages.append(f"{upstream} → {downstream} · {description}")
        return "\n".join(messages)

    @app.callback(Output("download-project", "disabled"), Output("download-project", "title"),
        Input("config", "data"), Input("row-references", "data"), Input("job", "data"),
        Input("job-completion", "data"), Input("launch-busy", "data"))
    def project_download_ready(state, registry, handle, completed, launching):
        if launching or (handle and completed != handle["job"]):
            return True, "Wait for the active job to finish before downloading the project."
        if not state or not state.get("valid"):
            return True, "Correct the configuration before downloading the project."
        if nozzle_mode(state["project"]):
            try:
                project_export.nozzle_ready(state["project"], registry)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return True, str(exc)
            return False, "Download the nozzle case, coordinates, MoC result, meshes, mesh report and activity log."
        try:
            meshes, omitted, stages = project_export.stage_meshes(state["project"], registry)
        except (ValueError, TypeError, KeyError) as exc:
            return True, str(exc)
        title = "Download the complete stage CGNS, individual row files, final YAML, mesh report and activity log."
        if omitted:
            title += " Unavailable or outdated meshes are listed in the README."
        return False, title

    @app.callback(Output("project-download", "data"), Output("project-download-message", "children"),
        Input("download-project", "n_clicks"), State("config", "data"), State("row-references", "data"),
        State("job", "data"), State("job-completion", "data"), State("launch-busy", "data"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "value"),
        State({"type": "parameter", "row": ALL, "path": ALL}, "id"), prevent_initial_call=True)
    def download_project(clicks, state, registry, handle, completed, launching, values, identifiers):
        try:
            if launching or (handle and completed != handle["job"]):
                raise ValueError("Wait for the active job to finish before downloading the project.")
            if not state or not state.get("valid"):
                raise ValueError("Correct the invalid settings before downloading the project.")
            project = project_model.validate_project(flow.edited_project(state["project"], values, identifiers))
            build = project_export.nozzle_archive if nozzle_mode(project) else project_export.build_archive
            data, filename = build(project, registry)
            return dcc.send_bytes(data, filename, type="application/zip"), ""
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return no_update, f"Cannot download project: {exc}"

    app.clientside_callback("function(data, settings, rows) { return window.dash_clientside.studio.render(data, settings, 'geometry', rows, 1); }",
        Output("geometry-scene", "children"), Output("geometry-note", "children"), Input("geometry-data", "data"),
        Input("geometry-display", "data"), Input("plotted-rows", "value"))
    app.clientside_callback("""function(data, settings, width, rows, result, request) {
        return window.dash_clientside.studio.render(data, settings, 'mesh', rows, width,
            window.turbomocInspection.active(result, request));
    }""",
        Output("mesh-scene", "children"), Output("mesh-note", "children"), Input("volume-data", "data"),
        Input("mesh-display", "data"), Input("mesh-width", "data"), Input("plotted-rows", "value"),
        Input("mesh-inspection-result", "data"), Input("mesh-inspection-request", "data"))
    return app


