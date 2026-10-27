"""Headless extraction of the solved stage (machine axis x) for 05_make_report.py.

python 04_extract_results.py [processors]
Reads stage_fluent.cas.h5/.dat.h5 from the run folder in its own Fluent session (the GUI is left alone) and
writes surfaces.npz and fluent_reports.json to <run>/report/data.
"""
import json
import sys

import numpy as np

from settings import CASE_NAME, RUN_DIR as RUN

DATA = RUN / "report" / "data"
ROWS = ("stator_1", "rotor_1")
WALLS = ("blade", "hub", "shroud")
PARTS = ("inlet", "outlet") + WALLS
SPANS = (0.1, 0.5, 0.9)
FIELDS = ("absolute-pressure", "density", "x-velocity", "y-velocity", "z-velocity",
          "relative-x-velocity", "relative-y-velocity", "relative-z-velocity", "turb-kinetic-energy")
R_HUB, R_SHROUD = 0.4361714416288, 0.4361714416288 + 0.1090428604088


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


def main(processors=8):
    import ansys.fluent.core as pyfluent
    DATA.mkdir(parents=True, exist_ok=True)
    solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                    processor_count=int(processors), ui_mode="no_gui", start_timeout=600,
                                    cwd=str(DATA))
    try:
        settings = solver.settings
        settings.file.read_case_data(file_name=str(RUN / CASE_NAME))
        field_data = solver.fields.field_data
        arrays = {}
        for row in ROWS:
            for part in PARTS:
                arrays.update(surface_arrays(field_data, f"{row}_{part}", wall=part in WALLS))

        # Blade-to-blade surfaces at constant radius about x, and axial cuts through the rotor passage
        iso = settings.results.surfaces.iso_surface
        radius_expression = "sqrt(Position.y^2 + Position.z^2)"
        settings.setup.named_expressions["radius_x"] = {"definition": radius_expression}
        rotor_x = arrays["rotor_1_blade|vertices"][:, 0]
        cuts = {"le": rotor_x.min(), "mid": 0.5 * (rotor_x.min() + rotor_x.max()), "te": rotor_x.max()}
        cuts = {"rotor_1_x_" + key: value + offset for (key, value), offset in
                zip(cuts.items(), (-0.003, 0.0, 0.003))}
        for row in ROWS:
            for span in SPANS:
                name = f"{row}_span{int(round(100 * span)):02d}"
                iso.create(name=name)
                iso[name].field = "expr:radius_x"
                iso[name].zones = [f"{row}_fluid"]
                iso[name].iso_values = [R_HUB + span * (R_SHROUD - R_HUB)]
                arrays.update(surface_arrays(field_data, name, wall=False))
        for name, x in cuts.items():
            iso.create(name=name)
            iso[name].field = "x-coordinate"
            iso[name].zones = ["rotor_1_fluid"]
            iso[name].iso_values = [x]
            arrays.update(surface_arrays(field_data, name, wall=False))
        np.savez_compressed(DATA / "surfaces.npz", **arrays)

        definitions = settings.solution.report_definitions
        flow_surfaces = [f"{row}_{part}" for row in ROWS for part in ("inlet", "outlet")]
        definitions.surface.create(name="report_mass_flow")
        definitions.surface["report_mass_flow"].report_type = "surface-massflowrate"
        definitions.surface["report_mass_flow"].surface_names = flow_surfaces
        definitions.surface["report_mass_flow"].per_surface = True
        names = []
        for row in ROWS:
            name = f"report_moment_{row}"
            definitions.moment.create(name=name)
            moment = definitions.moment[name]
            moment.report_output_type = "Moment"
            moment.mom_center = [0.0, 0.0, 0.0]
            moment.mom_axis = [1.0, 0.0, 0.0]
            moment.zones = [f"{row}_{wall}" for wall in WALLS]
            moment.per_zone = True
            names.append(name)
        raw = definitions.compute(report_defs=["report_mass_flow", *names])
        values = {key: value[0] for item in raw for key, value in item.items()}
        reports = {"values": values, "cuts_x_m": cuts, "spans": SPANS, "radii_m": [R_HUB, R_SHROUD]}
        (DATA / "fluent_reports.json").write_text(json.dumps(reports, indent=2, default=float), encoding="utf-8")
        print("written", DATA, sorted(values))
    finally:
        solver.exit()


if __name__ == "__main__":
    main(*sys.argv[1:])
