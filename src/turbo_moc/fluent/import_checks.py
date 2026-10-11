"""Check a Fluent volume import against the exported mesh before solving."""

import numpy as np

BOUNDARY_ROLES = (
    "inlet",
    "outlet",
    "blade",
    "periodic_left",
    "periodic_right",
    "hub",
    "shroud",
)


def _positive_count(value, label):
    """Require a nonempty discrete entity count from Fluent's reduction API."""
    value = float(value)
    if not np.isfinite(value) or value < 1 or not value.is_integer():
        raise RuntimeError(f"Fluent imported an invalid {label} count: {value}.")
    return int(value)


def validate_import(solver, zones, expected_mesh=None):
    """Return JSON-ready mesh counts and bounds, rejecting an incomplete import.

    ``zones`` maps the seven boundary roles to the actual imported names.
    ``expected_mesh``, when supplied, contains ``cell_count`` and ``bounds_m``;
    the latter holds minimum and maximum XYZ coordinates in metres. The
    boundary vertices determine the imported volume's coordinate envelope.
    This routine inspects geometry only and does not initialize the solution.
    """
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    fluids = solver.settings.setup.cell_zone_conditions.fluid.get_object_names()
    if len(fluids) != 1:
        raise RuntimeError(f"Expected one imported fluid cell zone; found {fluids}.")
    missing = set(BOUNDARY_ROLES) - set(zones)
    if missing:
        raise RuntimeError(f"Missing imported boundary roles: {sorted(missing)}.")
    boundary_names = [zones[role] for role in BOUNDARY_ROLES]
    if len(set(boundary_names)) != len(boundary_names):
        raise RuntimeError(
            "The seven imported boundary roles must identify distinct zones."
        )

    reduction = solver.fields.reduction
    # Passing the cell-zone object avoids PyFluent 0.42's string-location
    # validation, which recognizes boundary surfaces but rejects fluid names.
    fluid = solver.settings.setup.cell_zone_conditions.fluid[fluids[0]]
    cell_count = _positive_count(reduction.count(locations=[fluid]), "fluid cell")
    boundary_counts = {
        role: _positive_count(reduction.count(locations=[zones[role]]), f"{role} face")
        for role in BOUNDARY_ROLES
    }
    data = solver.fields.field_data.get_field_data(
        SurfaceFieldDataRequest(
            surfaces=boundary_names,
            data_types=[SurfaceDataType.Vertices],
        )
    )
    boundary_vertices = {}
    for role in BOUNDARY_ROLES:
        vertices = np.asarray(data[zones[role]].vertices, dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 3
            or not np.isfinite(vertices).all()
        ):
            raise RuntimeError(
                f"The imported {role} boundary has invalid XYZ vertices."
            )
        boundary_vertices[role] = vertices
    all_vertices = np.vstack(list(boundary_vertices.values()))
    bounds = np.asarray([all_vertices.min(axis=0), all_vertices.max(axis=0)])
    if np.any(bounds[1] <= bounds[0]):
        raise RuntimeError("The imported mesh does not span three spatial dimensions.")
    report = {
        "fluid_zone": fluids[0],
        "cell_count": cell_count,
        "bounds_m": bounds.tolist(),
        "boundaries": {
            role: {
                "zone": zones[role],
                "face_count": boundary_counts[role],
                "vertex_count": len(boundary_vertices[role]),
            }
            for role in BOUNDARY_ROLES
        },
    }
    if expected_mesh is not None:
        expected_count = _positive_count(
            expected_mesh["cell_count"], "exported fluid cell"
        )
        expected_bounds = np.asarray(expected_mesh["bounds_m"], dtype=float)
        if (
            expected_bounds.shape != (2, 3)
            or not np.isfinite(expected_bounds).all()
            or np.any(expected_bounds[1] <= expected_bounds[0])
        ):
            raise ValueError(
                "Exported bounds_m must contain increasing finite minimum/maximum XYZ."
            )
        if cell_count != expected_count:
            raise RuntimeError(
                f"Fluent imported {cell_count} fluid cells; the exported mesh has {expected_count}."
            )
        scale = max(float((expected_bounds[1] - expected_bounds[0]).max()), 1e-6)
        tolerance = max(1e-9, scale * 1e-6)
        mismatch = float(np.abs(bounds - expected_bounds).max())
        if mismatch > tolerance:
            raise RuntimeError(
                "Fluent's imported coordinate bounds differ from the metre-scaled export "
                f"by {mismatch:.6g} m (tolerance {tolerance:.6g} m)."
            )
        report.update(
            expected_cell_count=expected_count,
            expected_bounds_m=expected_bounds.tolist(),
            max_bounds_mismatch_m=mismatch,
            bounds_tolerance_m=tolerance,
        )
    return report
