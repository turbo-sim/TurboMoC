"""Headless Fluent extraction of the solved stage: surfaces, integrals, mesh quality and residuals.

Everything the report needs is written to ``data_dir`` so that figures and
tables can be rebuilt later without Fluent.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

ROWS = ("stator_1", "rotor_1")
WALLS = ("blade", "hub", "shroud")
FIELDS = ("absolute-pressure", "density", "x-velocity", "y-velocity", "z-velocity",
          "relative-x-velocity", "relative-y-velocity", "relative-z-velocity")


def boundary(setup, row, name):
    return setup["boundaries"][f"{row}_fluid_{name}"]


def midspan_surface(row):
    return f"{row}_midspan"


def channel_radii(mesh_report):
    """Hub and shroud radii in metres from the app mesh report (constant-radius axial channel)."""
    channel = mesh_report["rows"]["stator_1"]["meshes"]["3d"]["summary"]["geometry"]
    return channel["hub"]["control_points_rz_mm"][0][0] / 1e3, channel["shroud"]["control_points_rz_mm"][0][0] / 1e3


def surface_arrays(field_data, surface, wall):
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest
    mesh = field_data.get_field_data(SurfaceFieldDataRequest(
        surfaces=[surface], data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity,
                                        SurfaceDataType.FacesNormal, SurfaceDataType.FacesCentroid]))[surface]
    arrays = {
        f"{surface}|vertices": np.asarray(mesh.vertices, dtype=float).reshape(-1, 3),
        f"{surface}|faces": np.concatenate([[len(face), *face] for face in mesh.connectivity]).astype(np.int64),
        f"{surface}|normals": np.asarray(mesh.face_normals, dtype=float).reshape(-1, 3),
        f"{surface}|centroids": np.asarray(mesh.face_centroids, dtype=float).reshape(-1, 3),
    }
    for location, node_value, fields in (("node", True, FIELDS),
                                         ("face", False, FIELDS + (("y-plus",) if wall else ()))):
        for field in fields:
            arrays[f"{surface}|{location}|{field}"] = np.asarray(field_data.get_field_data(ScalarFieldDataRequest(
                field_name=field, surfaces=[surface], node_value=node_value))[surface], dtype=float)
    return arrays


def fluent_reports(settings, setup):
    definitions = settings.solution.report_definitions
    flow_surfaces = [boundary(setup, row, name) for row in ROWS for name in ("inlet", "outlet")]
    definitions.surface.create(name="report_mass_flow")
    mass_flow = definitions.surface["report_mass_flow"]
    mass_flow.report_type = "surface-massflowrate"
    mass_flow.surface_names = flow_surfaces
    mass_flow.per_surface = True
    moments = {}
    for row in ROWS:
        name = f"report_moment_{row}"
        definitions.moment.create(name=name)
        moment = definitions.moment[name]
        moment.report_output_type = "Moment"
        moment.mom_center = [0.0, 0.0, 0.0]
        moment.mom_axis = [0.0, 0.0, 1.0]
        moment.zones = [boundary(setup, row, wall) for wall in WALLS]
        moment.per_zone = True
        moments[row] = name
    values = definitions.compute(report_defs=["report_mass_flow", *moments.values()])
    values = {key: value[0] for item in values for key, value in item.items()}
    return {"mass_flow": {surface: values[f"report_mass_flow({surface})"] for surface in flow_surfaces},
            "moment": {row: {wall: values[f"{name}({boundary(setup, row, wall)})"] for wall in WALLS}
                       for row, name in moments.items()}}


def mesh_quality(solver):
    lines = []
    solver.transcript.register_callback(lines.append)
    solver.transcript.start(write_to_stdout=False)
    solver.settings.mesh.quality()
    time.sleep(2)
    text = "".join(lines)
    return {key: float(re.search(pattern, text).group(1)) for key, pattern in (
        ("min_orthogonal_quality", r"Minimum Orthogonal Quality =\s*([0-9.eE+-]+)"),
        ("max_aspect_ratio", r"Maximum Aspect Ratio =\s*([0-9.eE+-]+)"))}


def extract(case_file, setup, data_dir, options, radii):
    """Read case and data headless; write ``surfaces.npz`` and ``fluent_reports.json`` to ``data_dir``."""
    import ansys.fluent.core as pyfluent

    data_dir.mkdir(parents=True, exist_ok=True)
    solver = pyfluent.launch_fluent(product_version=options["product_version"], mode="solver", dimension=3,
                                    precision="double", processor_count=options["processor_count"],
                                    ui_mode="no_gui", start_timeout=options["start_timeout"], cwd=str(data_dir))
    try:
        settings = solver.settings
        settings.file.read_case_data(file_name=str(case_file))
        iso = settings.results.surfaces.iso_surface
        for row in ROWS:
            iso.create(name=midspan_surface(row))
            surface = iso[midspan_surface(row)]
            surface.field = "radial-coordinate"
            surface.zones = [setup["cell_zones"][f"{row}_fluid"]]
            surface.iso_values = [0.5 * sum(radii)]

        field_data = solver.fields.field_data
        arrays = {}
        for row in ROWS:
            for name in ("inlet", "outlet", *WALLS):
                arrays.update(surface_arrays(field_data, boundary(setup, row, name), wall=name in WALLS))
            arrays.update(surface_arrays(field_data, midspan_surface(row), wall=False))
        np.savez_compressed(data_dir / "surfaces.npz", **arrays)

        reports = fluent_reports(settings, setup)
        reports["mesh_quality"] = mesh_quality(solver)
        reports["methods"] = settings.solution.methods.get_state()
        reports["viscous"] = settings.setup.models.viscous.model()
        (data_dir / "fluent_reports.json").write_text(json.dumps(reports, indent=2, default=float), encoding="utf-8")
    finally:
        solver.exit()


def residual_table(transcript_dir, data_dir):
    """Residual history of the longest Fluent transcript, written to ``residuals.csv``."""
    import barotropy
    transcripts = sorted(Path(transcript_dir).glob("*.trn"), key=lambda path: path.stat().st_size)
    if not transcripts:
        return None
    parsed = barotropy.read_residual_file(str(transcripts[-1]))
    history = parsed[0] if isinstance(parsed, tuple) else parsed
    history = history.drop_duplicates("iter", keep="last").sort_values("iter")
    history.to_csv(data_dir / "residuals.csv", index=False)
    return data_dir / "residuals.csv"
