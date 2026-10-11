"""Internal-section contours and periodic display copies for 3D Fluent cases.

All coordinates are in metres. Creating display copies does not change the
single-passage computational mesh. PyFluent is loaded only when graphics are
enabled and a live solver is supplied.
"""

from pathlib import Path

import numpy as np

from .common import contour_definitions
from turbo_moc.reporting import display_path, get_logger

logger = get_logger("fluent.runner")
SECTION_NAME = "workflow-midspan"


def _fluid_zone(solver):
    """Fluent 24.2 keys periodic display settings by an existing cell zone."""
    names = solver.settings.setup.cell_zone_conditions.fluid.get_object_names()
    if len(names) != 1:
        raise RuntimeError("3D periodic display requires exactly one fluid cell zone.")
    return names[0]


def _vector(value, name, *, normalize=False):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite components.")
    if normalize:
        length = np.linalg.norm(vector)
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"{name} must be a nonzero vector.")
        vector = vector / length
    return vector


def _section_settings(geometry):
    """Return the settings type and its state for the requested physical cut."""
    section = geometry["section"]
    if section["type"] == "cylinder":
        radius = float(section["radius_m"])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("The contour cylinder radius must be positive and finite.")
        origin = _vector(geometry["axis_origin"], "Cylinder axis origin")
        axis = _vector(
            geometry["axis_direction"], "Cylinder axis direction", normalize=True
        )
        # |(I - aa.T)(x - origin)|^2 = radius^2. Fluent's ten quadric
        # coefficients are x², y², z², xy, yz, zx, x, y, z, constant.
        projection = np.eye(3) - np.outer(axis, axis)
        linear = -2 * projection @ origin
        coefficients = [
            *np.diag(projection),
            2 * projection[0, 1],
            2 * projection[1, 2],
            2 * projection[2, 0],
            *linear,
            origin @ projection @ origin,
        ]
        return "quadric_surface", {
            "attribute": [float(value) for value in coefficients],
            "value": radius**2,
        }
    if section["type"] == "plane":
        normal = _vector(section["normal"], "Contour plane normal", normalize=True)
        point = _vector(section["point_m"], "Contour plane point")
        return "plane_slice", {
            "normal": normal.tolist(),
            "distance_from_origin": float(normal @ point),
        }
    raise ValueError("The 3D contour section must be a cylinder or a plane.")


def _instance_settings(options, geometry):
    settings = {
        "surfaces": [SECTION_NAME],
        "repeats": options["periodic_instancing"]["repeats"],
    }
    if geometry["kind"] == "linear":
        translation = _vector(geometry["translation_m"], "Periodic translation")
        if np.linalg.norm(translation) <= 0:
            raise ValueError("Periodic display translation must be nonzero.")
        settings.update(periodic_type="translational", translation=translation.tolist())
    elif geometry["kind"] in ("axial", "radial"):
        angle = float(geometry["angle_deg"])
        if not np.isfinite(angle) or angle == 0:
            raise ValueError("Periodic display rotation must be finite and nonzero.")
        settings.update(
            periodic_type="rotational",
            rotation=_vector(
                geometry["axis_direction"], "Periodic axis direction", normalize=True
            ).tolist(),
            axis_origin=_vector(
                geometry["axis_origin"], "Periodic axis origin"
            ).tolist(),
            # Fluent settings use SI angle units (radians), unlike the
            # degree-valued geometry metadata and mesh-pairing command.
            rotation_angle=float(np.deg2rad(angle)),
        )
    else:
        raise ValueError(
            "3D contour graphics require axial, radial or linear geometry."
        )
    return settings


def _validate_created_section(solver):
    """Reject an empty cut before recording contours that could produce blank PNGs."""
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    data = solver.fields.field_data.get_field_data(
        SurfaceFieldDataRequest(
            surfaces=[SECTION_NAME],
            data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity],
        )
    )[SECTION_NAME]
    vertices = np.asarray(data.vertices, dtype=float)
    faces = data.connectivity
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
        or faces is None
        or len(faces) == 0
    ):
        raise RuntimeError(
            "The requested 3D contour section has no valid surface faces in the mesh."
        )
    return {"surface": SECTION_NAME, "vertices": len(vertices), "faces": len(faces)}


def prepare_contour_graphics(solver, options, geometry):
    """Create an internal cylinder/plane and pressure/Mach contour definitions."""
    definitions = [item for item in contour_definitions(options) if item[2]["enabled"]]
    if not definitions:
        return None
    # Validate all geometric inputs before modifying the live results tree.
    surface_type, surface_settings = _section_settings(geometry)
    instances = (
        _instance_settings(options, geometry)
        if options["periodic_instancing"]["enabled"]
        else None
    )
    fluid_zone = _fluid_zone(solver) if instances is not None else None
    results = solver.settings.results
    getattr(results.surfaces, surface_type)[SECTION_NAME] = surface_settings
    summary = _validate_created_section(solver)
    if instances is not None:
        results.graphics.periodic_instancing[fluid_zone] = instances
    for name, field, _ in definitions:
        results.graphics.contour[name] = {
            "field": field,
            "filled": True,
            "node_values": True,
            "surfaces_list": [SECTION_NAME],
        }
    return summary


def export_contours(solver, options, output_dir):
    """Render the configured physical section and save every requested PNG."""
    definitions = [item for item in contour_definitions(options) if item[2]["enabled"]]
    if not definitions:
        return []
    fluid_zone = (
        _fluid_zone(solver) if options["periodic_instancing"]["enabled"] else None
    )
    graphics = solver.settings.results.graphics
    picture = graphics.picture
    picture.driver_options.hardcopy_format = "png"
    picture.use_window_resolution = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = []
    for name, _, settings in definitions:
        graphics.contour[name].display()
        if options["periodic_instancing"]["enabled"]:
            graphics.periodic_instancing[fluid_zone].display()
        graphics.views.restore_view(view_name=options.get("view", "isometric"))
        graphics.views.auto_scale()
        picture.x_resolution = settings["width"]
        picture.y_resolution = settings["height"]
        filename = output_dir / settings["filename"]
        picture.save_picture(file_name=str(filename))
        if not filename.is_file() or filename.stat().st_size == 0:
            raise RuntimeError(f"Fluent did not produce the {name} contour image.")
        logger.info(
            "  %s contour saved: %s",
            name.replace("-", " ").capitalize(),
            display_path(filename),
        )
        filenames.append(filename)
    return filenames
