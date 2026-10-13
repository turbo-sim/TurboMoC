"""Smooth, closed geometry previews, independent of CFD mesh discretization."""
import numpy as np

from turbo_moc.geometry import passage_2d as planar
from turbo_moc.geometry.blade_row import section_data
from turbo_moc.geometry.flow_interfaces import interface_prime


def _polydata(points, triangles):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
    cells = vtkCellArray()
    offsets = 3 * np.arange(len(triangles) + 1, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.asarray(triangles, dtype=np.int64).ravel(), deep=True))
    result = vtkPolyData()
    result.SetPoints(vertices)
    result.SetPolys(cells)
    return result


def _triangulate(loops):
    """Triangulate a conformal polygon, including an optional blade-shaped hole."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    points = np.vstack(loops)
    offset = 0
    lines = []
    offsets = [0]
    for loop in loops:
        ids = np.arange(len(loop), dtype=np.int64) + offset
        lines.append(np.r_[ids, ids[0]])
        offsets.append(offsets[-1] + len(ids) + 1)
        offset += len(loop)
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(np.column_stack((points, np.zeros(len(points)))), deep=True))
    contours = vtkCellArray()
    contours.SetData(numpy_to_vtkIdTypeArray(np.asarray(offsets, dtype=np.int64), deep=True),
                     numpy_to_vtkIdTypeArray(np.concatenate(lines), deep=True))
    polygon = vtkPolyData()
    polygon.SetPoints(vertices)
    polygon.SetLines(contours)
    triangulator = vtkContourTriangulator()
    triangulator.SetInputData(polygon)
    triangulator.Update()
    if triangulator.GetTriangulationError():
        raise ValueError("Geometry preview contour triangulation failed.")
    output = triangulator.GetOutput()
    return (vtk_to_numpy(output.GetPoints().GetData())[:, :2],
            vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3))


def _display_surface(design, parameters, triangles, subdivisions, *, boundary_name=None):
    """Refine in (q_x, q_y, η), then map and compute smooth physical normals.

    Refining after mapping would interpolate chords through curved stream sheets.
    Refining shared parameter edges instead keeps blade sides and caps aligned.
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkPolyDataNormals
    from vtkmodules.vtkFiltersModeling import vtkLinearSubdivisionFilter

    clean = vtkCleanPolyData()
    clean.SetInputData(_polydata(parameters, triangles))
    clean.SetTolerance(0)
    refine = vtkLinearSubdivisionFilter()
    refine.SetInputConnection(clean.GetOutputPort())
    refine.SetNumberOfSubdivisions(subdivisions)
    refine.Update()
    output = refine.GetOutput()
    if not output.GetNumberOfPoints():
        raise ValueError("Geometry preview refinement produced no surface.")
    raw_stations = vtk_to_numpy(output.GetPoints().GetData())
    # VTK subdivision may return float32 even for double input. Use that
    # precision for boundary identification, then map in double precision.
    parameter_tolerance = 8 * np.finfo(raw_stations.dtype).eps * max(1., np.abs(parameters).max())
    stations = np.asarray(raw_stations, dtype=float).copy()
    # Linear interpolation in (q, eta) need not remain on an axial plane after
    # the conformal mapping, particularly for taper. Keep refined cap vertices
    # and their shared periodic-side edges on the same physical interface.
    interfaces = design.assembly.get("interfaces", {})
    for side, interface in interfaces.items():
        if boundary_name == side:
            cap_vertices = np.ones(len(stations), dtype=bool)
        elif boundary_name in ("periodic_left", "periodic_right"):
            original_eta = np.unique(parameters[:, 2])
            extrema = np.max if side == "inlet" else np.min
            cap_y = [extrema(parameters[parameters[:, 2] == value, 1]) for value in original_eta]
            interpolated = np.interp(stations[:, 2], original_eta, cap_y)
            cap_vertices = np.isclose(stations[:, 1], interpolated, rtol=0, atol=parameter_tolerance)
        else:
            continue
        for eta in np.unique(stations[cap_vertices, 2]):
            mask = cap_vertices & (stations[:, 2] == eta)
            stations[mask, 1] = -design.meridional_channel.reference_radius * interface_prime(
                design.meridional_channel, interface, design.assembly.get("placement", {}), float(eta))
    xyz = np.empty_like(stations)
    for eta in np.unique(stations[:, 2]):
        mask = stations[:, 2] == eta
        xyz[mask] = design.meridional_channel.map_points(stations[mask, :2], float(eta))
    output.GetPoints().SetData(numpy_to_vtk(xyz, deep=True))
    output.Modified()

    normals = vtkPolyDataNormals()
    normals.SetInputData(output)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    # Split cap/side and other genuine sharp edges; smooth curved blade faces.
    normals.SplittingOn()
    normals.SetFeatureAngle(50)
    normals.Update()
    surface = normals.GetOutput()
    faces = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return {
        "points": vtk_to_numpy(surface.GetPoints().GetData()).ravel().tolist(),
        "polys": np.column_stack((np.full(len(faces), 3), faces)).ravel().tolist(),
        "normals": vtk_to_numpy(surface.GetPointData().GetNormals()).ravel().tolist(),
    }


def preview_geometry(design, *, span_sections=9, curve_points=180, subdivisions=1):
    """Create smoothly shaded, capped display surfaces without running Gmsh.

    The default refinement gives 17 span stations and 360 contour segments.
    It is a rendering approximation, unrelated to the numerical mesh controls.
    Both blade ends are capped so hiding hub/shroud never reveals a hollow blade.
    """
    def sample(points, count, closed=False):
        return planar.resample_curve(points, count + int(closed))[:count]

    def loops(blade, curves):
        result = {"blade": sample(planar.blade_contour(blade), curve_points, True)}
        for name in ("periodic_left", "periodic_right", "inlet", "outlet"):
            result[name] = sample(curves[name], curve_points if "periodic" in name else 24)
        result["outer"] = np.vstack((result["periodic_left"], result["outlet"][1:],
                                    result["periodic_right"][-2::-1], result["inlet"][-2:0:-1]))
        return result

    eta = np.linspace(0, 1, span_sections)
    sections = [loops(*design.section(float(value))) for value in eta]
    surfaces = {}
    for name in ("blade", "periodic_left", "periodic_right", "inlet", "outlet"):
        q = np.array([row[name] for row in sections])
        width = q.shape[1]
        parameters = np.concatenate((q, np.broadcast_to(eta[:, None, None],
                                    (*q.shape[:2], 1))), axis=2).reshape(-1, 3)
        edge = np.arange(width if name == "blade" else width - 1)
        nxt = (edge + 1) % width
        offset = np.arange(span_sections - 1)[:, None] * width
        quads = np.stack((edge + offset, nxt + offset, nxt + offset + width,
                          edge + offset + width), axis=-1).reshape(-1, 4)
        triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
        if name == "blade":
            # Cap triangulation uses the same contour as the sides. Welding
            # before subdivision makes the blade a single closed display shell.
            for row, value in ((sections[0], 0.0), (sections[-1], 1.0)):
                cap_q, cap_cells = _triangulate([row["blade"]])
                cap = np.column_stack((cap_q, np.full(len(cap_q), value)))
                cap_cells = cap_cells[:, ::-1] if value == 0 else cap_cells
                triangles = np.vstack((triangles, cap_cells + len(parameters)))
                parameters = np.vstack((parameters, cap))
        surfaces[name] = _display_surface(design, parameters, triangles, subdivisions, boundary_name=name)
    # Both span supports follow their mapped periodic edges. In an annular row
    # these are revolution sectors; in a linear cascade they are planar strips.
    # These are continuous display surfaces, with no blade-shaped cutouts. This
    # leaves the numerical CFD boundary and the closed blade shell unchanged.
    mapping = design.meridional_channel
    angular_samples = (3 if mapping.is_linear else
                       max(3, int(np.ceil(mapping.pitch_angle / np.deg2rad(4))) + 1))
    fraction = np.linspace(0, 1, angular_samples)
    for name, row, value in (("hub", sections[0], 0.0), ("shroud", sections[-1], 1.0)):
        left, right = row["periodic_left"], row["periodic_right"]
        q = left[:, None, :] + fraction[None, :, None] * (right - left)[:, None, :]
        parameters = np.column_stack((q.reshape(-1, 2), np.full(q.shape[0] * q.shape[1], value)))
        start = (np.arange(len(left) - 1)[:, None] * angular_samples
                 + np.arange(angular_samples - 1))
        quads = np.stack((start, start + 1, start + angular_samples + 1,
                          start + angular_samples), axis=-1).reshape(-1, 4)
        triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
        surfaces[name] = _display_surface(design, parameters, triangles, subdivisions)
    result = {"section": section_data(design), "surfaces": surfaces,
              "is_linear": mapping.is_linear, "pitch_mm": mapping.pitch,
              "shift_mm": design.curves["shift_mm"]}
    if not mapping.is_linear:
        result.update(num_blades=mapping.num_blades, reference_radius_mm=mapping.reference_radius)
    return result
