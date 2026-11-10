"""Midspan blade-to-blade mesh of a TurboGrid passage as a 2-D CGNS/ADF file (stator and rotor of the medium mesh).

The hub/shroud tool cgns_adf_3Dto2D.py (copied unchanged next to this script) extracts boundary surfaces. Midspan is
an interior TurboGrid layer instead: this script finds the layer of mesh faces closest to midspan, adds it to the zone
as an extra QUAD_4 section named MIDSPAN, and hands the zone to the tool's own extraction. The layer is then unwrapped
as a cylinder (CoordinateX = axial x, CoordinateY = R (theta - theta0)), oriented counter-clockwise, and its boundary
edges become BAR_2 sections + BCs named after the neighbouring 3-D boundaries (blade, periodics, inlet, outlet).

Two sources:
  python midspan_3Dto2D.py turbogrid <mesh.cgns> [-o out.cgns]   TurboGrid 3-D CGNS/ADF export (layer from HEXA_8)
  python midspan_3Dto2D.py fluent [case.cas.h5]                  Fluent stage case (layer from its interior faces);
                                                                 default: the medium-mesh stage case, both rows
The Fluent route keeps the stage coordinates: the rotor sits where the stage case put it (interfaces at the same x).
"""
import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np

import cgns_adf_3Dto2D as tool
from cgns_adf_3Dto2D import AdfNode

HERE = Path(__file__).resolve().parent
MEDIUM_CASE = HERE.parent / "Simulations" / "medium_sim" / "stage_fluent_medium_2order_tsn.cas.h5"
FLUENT_ROWS = {"stator_1": ("stator", "default-interior"), "rotor_1": ("rotor", "default-interior.1")}
FLUENT_BCS = {"blade": "BLADE", "periodic_left": "PER1", "periodic_left-shadow": "PER2", "inlet": "INFLOW",
              "outlet": "OUTFLOW"}        # Fluent zone suffix -> BC name in the 2-D file (TurboGrid style)
HEXA_8 = 17                                                      # CGNS element type
HEXA_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
LAYER = "MIDSPAN"
CYLINDER_TOL = 1e-3     # relative radius spread tolerated on the layer (TurboGrid layers are near-cylindrical)
log = logging.getLogger("midspan_3Dto2D")


def midspan_layer(points, quads, axis, r_hub, r_shroud):
    """Faces of the TurboGrid layer closest to midspan: faces lying at (nearly) one radius about the machine axis,
    grouped by radius, the group nearest the mean of hub and shroud radii. Duplicates (shared by two cells) are
    removed."""
    radius = np.hypot(points[:, (axis + 1) % 3], points[:, (axis + 2) % 3])
    r = radius[quads]
    span = abs(r_shroud - r_hub)
    flat = np.ptp(r, axis=1) < 1e-3 * span              # layer faces; spanwise faces span a whole cell height
    mean = r[flat].mean(axis=1)
    levels = np.sort(mean)
    groups = np.split(levels, np.nonzero(np.diff(levels) > 1e-2 * span)[0] + 1)
    target = 0.5 * (r_hub + r_shroud)
    centre = min((g.mean() for g in groups), key=lambda value: abs(value - target))
    layer = quads[flat][np.abs(mean - centre) < 1e-2 * span]
    _, first = np.unique(np.sort(layer, axis=1), axis=0, return_index=True)
    layer = layer[np.sort(first)]
    log.info("midspan layer at r = %.6g about %s (hub %.6g, shroud %.6g): %d faces", centre, "XYZ"[axis], r_hub,
             r_shroud, len(layer))
    return layer


def endwalls(zone, points):
    """Machine axis and hub/shroud radii, from the zone's HUB and SHROUD face sections (TurboGrid names)."""
    fits = {}
    for wall in ("HUB", "SHROUD"):
        sections = [s for s in zone.find_all("Elements_t") if wall in s.name.upper().split()]
        if not sections:
            raise tool.AdfError(f"zone {zone.name!r}: no {wall} section to locate midspan")
        nodes = np.unique(np.concatenate([tool._int_array(s.get("ElementConnectivity")) for s in sections])) - 1
        fits[wall] = tool._cylinder_fit(points[nodes])
    (axis, r_hub, spread_hub), (axis_shroud, r_shroud, spread_shroud) = fits["HUB"], fits["SHROUD"]
    if axis != axis_shroud or max(spread_hub, spread_shroud) > CYLINDER_TOL:
        raise tool.AdfError(f"zone {zone.name!r}: hub and shroud are not cylinders about one axis "
                            f"({fits}); a constant-radius midspan layer does not exist")
    return axis, r_hub, r_shroud


def add_layer_section(zone, layer):
    """Append the layer as a QUAD_4 section (1-based connectivity) after the zone's last element."""
    sections = zone.find_all("Elements_t")
    last = max(int(tool._int_array(s.get("ElementRange"))[1]) for s in sections)
    template = sections[0]
    conn_dtype = np.asarray(template.get("ElementConnectivity").data).dtype
    range_node = template.get("ElementRange")
    section = zone.add(AdfNode(LAYER, "Elements_t", np.array([tool.QUAD_4, 0],
                                                            dtype=np.asarray(template.data).dtype)))
    section.add(AdfNode("ElementRange", "IndexRange_t", np.array([last + 1, last + len(layer)],
                        dtype=np.asarray(range_node.data).dtype).reshape(np.asarray(range_node.data).shape)))
    section.add(AdfNode("ElementConnectivity", "DataArray_t", (layer + 1).ravel().astype(conn_dtype)))


def write_2d(zones, base_name, output, source):
    """Run the tool's extraction of the MIDSPAN section on each zone and write one 2-D CGNS/ADF file."""
    root = tool.new_root()
    root.add(AdfNode("CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([4.0], np.float32)))
    base = root.add(AdfNode(base_name, "CGNSBase_t", np.array([2, 2], np.int32)))
    for zone in zones:
        zone2, info = tool._extract_zone(zone, LAYER, CYLINDER_TOL, "cylinder", 4.0, source)
        base.add(zone2)
        log.info("zone %r: %d nodes, %d cells; %s; edges: %s", info["zone"], info["nodes"], info["cells"],
                 info["plane"], ", ".join(f"{n} ({k})" for n, k in info["edges"]))
    tool.write_adf(str(output), root)
    log.info("wrote %s", output)
    return output


# ------------------------------------------------------------------------------------- TurboGrid 3-D CGNS/ADF input
def from_turbogrid(filename, output=None):
    root = tool.read_adf(str(filename))
    zones = []
    for base in root.find_all("CGNSBase_t"):
        for zone in base.find_all("Zone_t"):
            grid = zone.get(label="GridCoordinates_t")
            points = np.column_stack([np.asarray(grid.get(f"Coordinate{a}").data).ravel(order="F") for a in "XYZ"])
            cells = [tool._int_array(s.get("ElementConnectivity")).reshape(-1, 8) - 1
                     for s in zone.find_all("Elements_t") if int(tool._int_array(s)[0]) == HEXA_8]
            if not cells:
                raise tool.AdfError(f"zone {zone.name!r}: no HEXA_8 cells to build the midspan layer from")
            hexa = np.concatenate(cells)
            quads = np.concatenate([hexa[:, list(face)] for face in HEXA_FACES])
            add_layer_section(zone, midspan_layer(points, quads, *endwalls(zone, points)))
            zones.append(zone)
    output = output or tool.default_output_filename(str(filename)).replace("_2D", "_midspan_2D")
    return write_2d(zones, root.find_all("CGNSBase_t")[0].name, output, os.path.basename(filename))


# ------------------------------------------------------------------------------------------------ Fluent case input
def fluent_zone(case, row, interior):
    """A TurboGrid-like unstructured zone of one row: coordinates, the boundary faces next to midspan as named
    QUAD_4 sections with face-centre BCs, and the MIDSPAN section."""
    mesh = case["meshes/1"]
    xyz = mesh["nodes/coords/12"][:]
    offsets = np.concatenate([[0], np.cumsum(mesh["faces/nodes/1/nnodes"][:], dtype=np.int64)])
    nodes = mesh["faces/nodes/1/nodes"][:].astype(np.int64) - 1
    table = mesh["faces/zoneTopology"]
    ranges = {name: (int(lo), int(hi)) for name, lo, hi in
              zip(table["name"][0].decode().split(";"), table["minId"][:], table["maxId"][:])}
    quads = lambda name: nodes[offsets[ranges[name][0] - 1]:offsets[ranges[name][1]]].reshape(-1, 4)

    interior_quads = quads(interior)
    boundary = {bc: quads(f"{row}_{suffix}") for suffix, bc in FLUENT_BCS.items()}
    used = np.unique(np.concatenate([interior_quads.ravel(), *[q.ravel() for q in boundary.values()]]))
    local = np.full(len(xyz), -1, np.int64)
    local[used] = np.arange(len(used))
    points = xyz[used]
    (axis, r_hub, _), (_, r_shroud, _) = (tool._cylinder_fit(xyz[np.unique(quads(f"{row}_{wall}"))])
                                          for wall in ("hub", "shroud"))
    layer = midspan_layer(points, local[interior_quads], axis, r_hub, r_shroud)

    zone = AdfNode(row, "Zone_t", np.array([[len(points), 0, 0]], np.int32))
    zone.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
    grid = zone.add(AdfNode("GridCoordinates", "GridCoordinates_t"))
    for axis, column in zip("XYZ", points.T):
        grid.add(AdfNode(f"Coordinate{axis}", "DataArray_t", np.ascontiguousarray(column, np.float64)))
    zone_bc = AdfNode("ZoneBC", "ZoneBC_t")
    first = 1
    for bc, faces in boundary.items():
        last = first + len(faces) - 1
        section = zone.add(AdfNode(bc, "Elements_t", np.array([tool.QUAD_4, 0], np.int32)))
        section.add(AdfNode("ElementRange", "IndexRange_t", np.array([first, last], np.int32)))
        section.add(AdfNode("ElementConnectivity", "DataArray_t", (local[faces] + 1).ravel().astype(np.int32)))
        node = zone_bc.add(AdfNode(bc, "BC_t", "Null"))
        node.add(AdfNode("PointRange", "IndexRange_t", np.array([[first, last]], np.int32)))
        node.add(AdfNode("GridLocation", "GridLocation_t", "FaceCenter"))
        first = last + 1
    add_layer_section(zone, layer)
    zone.add(zone_bc)
    return zone


def from_fluent(case_file=MEDIUM_CASE, output_dir=HERE, tag="medium"):
    outputs = []
    with h5py.File(case_file, "r") as case:
        for row, (name, interior) in FLUENT_ROWS.items():
            zone = fluent_zone(case, row, interior)
            output = Path(output_dir) / f"{name}_{tag}_midspan_2D.cgns"
            outputs.append(write_2d([zone], "Base", output, os.path.basename(case_file)))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="source", required=True)
    turbogrid = sub.add_parser("turbogrid", help="TurboGrid 3-D CGNS/ADF export")
    turbogrid.add_argument("mesh")
    turbogrid.add_argument("-o", "--output")
    fluent = sub.add_parser("fluent", help="Fluent stage case (.cas.h5)")
    fluent.add_argument("case", nargs="?", default=str(MEDIUM_CASE))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.source == "turbogrid":
        from_turbogrid(args.mesh, args.output)
    else:
        from_fluent(args.case)


if __name__ == "__main__":
    main()
