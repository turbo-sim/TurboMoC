"""Quasi-2D stage: the medium midspan mesh as a thin annulus in Fluent 3D, with the stage's rotation, mixing
plane, inlet and outlet conditions and barotropic model (Fluent has no working mixing plane in its 2D solver).

python 06_midspan_annulus_stage.py setup [processors]    writes the case
python 06_midspan_annulus_stage.py solve [processors]    solves it with the staged procedure of run_night_recipe.py
python 06_midspan_annulus_stage.py report [processors]   prints mass flows, moments and interface averages

Mesh: each node (x, r_m theta) of meshes/midspan_2D is swept from hub to shroud, SPAN apart at the same angle, with
the machine axis along z (old axes (x, y, z) -> (Z, X, Y), a cyclic permutation, so +omega about +x becomes +omega about
+z). Hub and shroud are slip cylinders, so the blade-to-blade flow at r_m has no end-wall boundary layers or radial
equilibrium. Mass flows and moments are per passage of the SPAN-high annulus.
"""
import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from turbo_moc.fluent.barotropic_model import generate_barotropic_model
from turbo_moc.fluent.common import load_dependencies
from turbo_moc.fluent.stage import case
from turbo_moc.fluent.stage.case import ROTOR, STATOR
from turbo_moc.io.cgns_adf import AdfNode, new_root, read_adf, write_adf

from settings import STATOR_CONFIG

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fluent_2D_stage"))
stage_3d = importlib.import_module("01_setup_stage_case")   # Same speed, outlet pressure, options and barotropic model
recipe = importlib.import_module("run_night_recipe")

MESH_DIR = HERE / "meshes" / "midspan_2D"
LAYERS = int(os.environ.get("ANNULUS_LAYERS", 4))   # One cell cannot hold the radial pressure gradient: spurious V_r
RUN_DIR = HERE / f"run_annulus_{LAYERS}layers"
ROWS = {"stator_1": ("stator", 44), "rotor_1": ("rotor", 120)}
EDGE_NAMES = {"INFLOW": "inlet", "OUTFLOW": "outlet", "BLADE": "blade", "PER1": "periodic_left",
              "PER2": "periodic_right"}
BOUNDARY_TYPES = {"inlet": "BCInflowSubsonic", "outlet": "BCOutflowSubsonic", "blade": "BCWallViscous",
                  "hub": "BCWallViscous", "shroud": "BCWallViscous", "periodic_left": "BCGeneral",
                  "periodic_right": "BCGeneral"}
SPAN = 2.0e-3                   # Annulus height [m], as in the thin stage cases
CASE_NAME = "stage_annulus.cas.h5"


def read_row(row):
    """Coordinates [m], radius and mean angle of the unwrapping, quad cells and named edges of a midspan row."""
    root = read_adf(str(MESH_DIR / f"{ROWS[row][0]}_medium_midspan_2D.cgns"))
    zone = root.find_all("CGNSBase_t")[0].find_all("Zone_t")[0]
    grid = zone.get(label="GridCoordinates_t")
    xy = np.column_stack([np.asarray(grid.get(f"Coordinate{axis}").data, dtype=float) for axis in "XY"])
    note = zone.get("SurfaceExtraction").text
    radius = float(re.search(r"cylinder of radius ([\d.eE+-]+)", note).group(1))
    theta0 = float(re.search(r"theta0 = ([\d.eE+-]+) rad", note).group(1))
    sections = {section.name: np.asarray(section.get("ElementConnectivity").data).ravel().astype(np.int64) - 1
                for section in zone.find_all("Elements_t")}
    return {"xy": xy, "radius": radius, "theta0": theta0, "cells": sections["MIDSPAN"].reshape(-1, 4),
            "edges": {name: sections[label].reshape(-1, 2) for label, name in EDGE_NAMES.items()}}


def annulus_sections(row):
    """Nodes (X, Y, Z), hexahedra and boundary quads of a row, swept from hub to shroud in LAYERS equal cells."""
    count = len(row["xy"])
    theta = row["xy"][:, 1] / row["radius"] + row["theta0"]
    radii = row["radius"] + SPAN * (np.arange(LAYERS + 1) / LAYERS - 0.5)
    points = np.vstack([np.column_stack([r * np.cos(theta), r * np.sin(theta), row["xy"][:, 0]]) for r in radii])
    cells = row["cells"]
    a, b, d = (points[cells[:, k]] for k in (0, 1, 3))
    e = points[cells[:, 0] + count]
    flipped = np.einsum("ij,ij->i", np.cross(b - a, d - a), e - a) < 0   # HEXA_8: 1-2-3-4 turns towards 5-6-7-8
    cells = np.where(flipped[:, None], cells[:, ::-1], cells)
    a, b, d = (points[cells[:, k]] for k in (0, 1, 3))
    if (np.einsum("ij,ij->i", np.cross(b - a, d - a), points[cells[:, 0] + count] - a) <= 0).any():
        raise ValueError("Inverted hexahedra in the annulus.")
    hexa = np.vstack([np.column_stack([cells + k * count, cells + (k + 1) * count]) for k in range(LAYERS)])
    quads = {name: np.vstack([np.column_stack([edge + k * count, edge[:, ::-1] + (k + 1) * count])
                              for k in range(LAYERS)]) for name, edge in row["edges"].items()}
    quads["hub"], quads["shroud"] = cells, cells + LAYERS * count
    return points, hexa, quads


def write_mesh(rows, filename):
    """One 3D CGNS base with a zone per row, zones and boundaries named as case.configure_stage expects."""
    root = new_root()
    root.add(AdfNode("CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([3.4], dtype=np.float32)))
    base = root.add(AdfNode("Base", "CGNSBase_t", np.array([3, 3], dtype=np.int32)))
    for row_id, row in rows.items():
        points, hexa, quads = annulus_sections(row)
        name = f"{row_id}_fluid"
        zone = base.add(AdfNode(name, "Zone_t", np.array([[len(points), len(hexa), 0]], dtype=np.int32)))
        zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
        grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
        for axis, values in zip("XYZ", points.T):
            grid.add(AdfNode(f"Coordinate{axis}", "DataArray_t", np.ascontiguousarray(values, dtype=np.float64)))

        def add_section(label, kind, start, nodes):
            section = zone.add(AdfNode(label, "Elements_t", np.array([kind, 0], dtype=np.int32)))
            section.add(AdfNode("ElementRange", "IndexRange_t", np.array([start, start + len(nodes) - 1], dtype=np.int32)))
            section.add(AdfNode("ElementConnectivity", "DataArray_t", (nodes + 1).ravel().astype(np.int32)))

        add_section(f"{name}_domain", 17, 1, hexa)
        zone_bc = zone.add(AdfNode("ZoneBC", "ZoneBC_t"))
        start = len(hexa) + 1
        for boundary, nodes in quads.items():
            label = f"{row_id}_{boundary}"
            add_section(label, 7, start, nodes)
            bc = zone_bc.add(AdfNode(label, "BC_t", BOUNDARY_TYPES[boundary]))
            bc.add(AdfNode("GridLocation", "GridLocation_t", "FaceCenter"))
            bc.add(AdfNode("PointRange", "IndexRange_t", np.array([[start, start + len(nodes) - 1]], dtype=np.int32)))
            start += len(nodes)
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_adf(str(filename), root)
    return filename


def launch(processors, pyfluent):
    return pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                  processor_count=processors, ui_mode="no_gui", start_timeout=600, cwd=str(RUN_DIR))


def setup(processors):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    barotropy, pyfluent = load_dependencies(False)
    rows = {row: read_row(row) for row in ROWS}
    mesh_file = write_mesh(rows, RUN_DIR / "mesh" / "stage_annulus.cgns")
    config = yaml.safe_load(STATOR_CONFIG.read_text(encoding="utf-8"))
    parameters = dict(config["rows"][0]["geometry"]["method_of_characteristics"], fluid_name="Cyclopentane")
    expressions, state = generate_barotropic_model(barotropy, {"nozzle": parameters}, {}, stage_3d.BAROTROPIC,
                                                   RUN_DIR / "barotropy")
    options = dict(stage_3d.OPTIONS, case_filename=CASE_NAME, interface="mixing_plane", slip_endwalls=True,
                   slip_blades=False)
    point = {"rpm": stage_3d.RPM, "swirl_sign": 1.0, "rotor_exit_pressure": stage_3d.P_OUT, "design": {}}
    solver = launch(processors, pyfluent)
    try:
        settings = solver.settings
        configured = case.configure_stage(solver, mesh_file, state, expressions, point, options)
        omega = configured["rotor_omega_rad_s"]
        settings.setup.named_expressions["relative_mach"] = {"definition": (
            f"sqrt(Velocity.z^2 + (Velocity.x + {omega:.6f} [s^-1]*Position.y)^2 + (Velocity.y - {omega:.6f} [s^-1]"
            "*Position.x)^2) / barotropic_speed_of_sound")}
        settings.file.batch_options.confirm_overwrite = False
        settings.file.write_case(file_name=str(RUN_DIR / CASE_NAME))
    finally:
        solver.exit()
    (RUN_DIR / "setup_summary.json").write_text(json.dumps({
        "inlet_state": state, "rpm": stage_3d.RPM, "omega_rad_s": omega, "outlet_pressure": stage_3d.P_OUT, "span_m": SPAN,
        "blade_speed_m_s": omega * rows["rotor_1"]["radius"], "radius_m": {row: rows[row]["radius"] for row in rows},
        "blades": {row: count for row, (_, count) in ROWS.items()}, "span_layers": LAYERS,
        "cells": {row: LAYERS * len(rows[row]["cells"]) for row in rows}, **configured}, indent=2, default=float), encoding="utf-8")
    print("saved", RUN_DIR / CASE_NAME, flush=True)


def solve(processors):
    _, pyfluent = load_dependencies(False)
    (RUN_DIR / "autosave").mkdir(exist_ok=True)
    solver = launch(processors, pyfluent)
    monitor = recipe.Monitor()
    solver.transcript.register_callback(monitor)
    settings = solver.settings
    try:
        settings.file.read_case(file_name=str(RUN_DIR / CASE_NAME))
        settings.file.batch_options.confirm_overwrite = False
        relax = settings.solution.controls.pseudo_time_explicit_relaxation_factor.global_dt_pseudo_relax
        for equation, value in recipe.DEFAULT_RELAXATION.items():
            if equation in set(relax.get_state()):
                relax[equation] = value
        auto = settings.file.auto_save
        auto.root_name = str(RUN_DIR / "autosave" / "stage")
        auto.data_frequency = 250
        auto.retain_most_recent_files = True
        auto.max_files = 2
        final = recipe.FINAL_SCALE
        for attempt in range(1, recipe.ATTEMPTS + 1):
            if recipe.procedure(settings, monitor, RUN_DIR, attempt, final):
                recipe.log(f"finished at iteration {monitor.iteration}, residuals {monitor.residuals}")
                break
            recipe.log(f"attempt {attempt} crashed at iteration {monitor.iteration}: hybrid re-initialization")
            if monitor.iteration > 1:  # As run_night_recipe: a crash at iteration 1 is normal, a later one halves the scale
                final /= 2
        else:
            recipe.log("all attempts crashed")
        settings.file.write_case_data(file_name=str(RUN_DIR / "stage_annulus_solved.cas.h5"))
        recipe.log("saved", RUN_DIR / "stage_annulus_solved.cas.h5")
    finally:
        solver.exit()


def report(processors):
    from ansys.fluent.core import ScalarFieldDataRequest, SurfaceDataType, SurfaceFieldDataRequest
    _, pyfluent = load_dependencies(False)
    summary = json.loads((RUN_DIR / "setup_summary.json").read_text(encoding="utf-8"))
    zones = summary["boundaries"]
    solver = launch(processors, pyfluent)
    settings = solver.settings
    try:
        settings.file.read_case_data(file_name=str(RUN_DIR / "stage_annulus_solved.cas.h5"))
        definitions = settings.solution.report_definitions
        flows = {key: zones[f"{key}"] for key in (f"{STATOR}_inlet", f"{STATOR}_outlet", f"{ROTOR}_inlet", f"{ROTOR}_outlet")}
        names = []
        for key, surface in flows.items():
            name = f"mass_{key}"
            definitions.surface.create(name=name)
            definitions.surface[name].report_type = "surface-massflowrate"
            definitions.surface[name].surface_names = [surface]
            names.append(name)
        for row in (STATOR, ROTOR):
            name = f"moment_{row}"
            definitions.moment.create(name=name)
            definitions.moment[name].report_output_type = "Moment"
            definitions.moment[name].mom_center = [0.0, 0.0, 0.0]
            definitions.moment[name].mom_axis = [0.0, 0.0, 1.0]
            definitions.moment[name].zones = [zones[f"{row}_blade"]]
            names.append(name)
        values = {key: value[0] for item in definitions.compute(report_defs=names) for key, value in item.items()}
        for key, value in values.items():
            print(f"{key} {value:.8g}", flush=True)
        omega, data = summary["omega_rad_s"], solver.fields.field_data
        for key, surface in flows.items():
            centroid = np.asarray(data.get_field_data(SurfaceFieldDataRequest(
                surfaces=[surface], data_types=[SurfaceDataType.FacesCentroid]))[surface].face_centroids,
                dtype=float).reshape(-1, 3)
            area = np.asarray(data.get_field_data(ScalarFieldDataRequest(
                field_name="density", surfaces=[surface], node_value=False))[surface], dtype=float)
            vx, vy, vz, pressure = (np.asarray(data.get_field_data(ScalarFieldDataRequest(
                field_name=field, surfaces=[surface], node_value=False))[surface], dtype=float)
                for field in ("x-velocity", "y-velocity", "z-velocity", "absolute-pressure"))
            theta = np.arctan2(centroid[:, 1], centroid[:, 0])
            radius = np.hypot(centroid[:, 0], centroid[:, 1])
            v_r = vx * np.cos(theta) + vy * np.sin(theta)
            v_t = vy * np.cos(theta) - vx * np.sin(theta)
            print(f"{key} (z {centroid[:, 2].min() * 1e3:.1f}-{centroid[:, 2].max() * 1e3:.1f} mm): p {pressure.mean():.0f} Pa, rho {area.mean():.3f}, V_r mean {v_r.mean():.2f} "
                  f"max|.| {np.abs(v_r).max():.2f}, V_theta {v_t.mean():.2f}, V_ax {vz.mean():.2f}, "
                  f"alpha {np.degrees(np.arctan2(v_t.mean(), vz.mean())):.2f} deg, "
                  f"W_theta {(v_t - omega * radius).mean():.2f} m/s", flush=True)
    finally:
        solver.exit()


if __name__ == "__main__":
    mode = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    {"setup": setup, "solve": solve, "report": report}[mode](count)
    sys.stdout.flush()
    os._exit(0)  # pyfluent's transcript-streaming thread would keep the interpreter alive
