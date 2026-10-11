"""Stage packaging preserves local topology and independent interface sides."""
import numpy as np
import pytest

from turbo_moc.io.cgns_adf import AdfNode as Node, new_root, read_adf, read_adf_bytes, write_adf
from turbo_moc.io.cgns_stage import combine_rows


def row_file(path, dimension=3):
    """Small valid cube/square with outward faces and reciprocal periodic maps."""
    root = new_root()
    root.add(Node("CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([3.4], np.float32)))
    base = root.add(Node("Base", "CGNSBase_t", np.array([dimension, dimension], np.int32)))
    base.add(Node("DataClass", "DataClass_t", "Dimensional"))
    units = np.full((32, 5), b" ", dtype="S1")
    for column, unit in enumerate(("Kilogram", "Meter", "Second", "Kelvin", "Radian")):
        units[:len(unit), column] = np.frombuffer(unit.encode(), dtype="S1")
    base.add(Node("DimensionalUnits", "DimensionalUnits_t", units))
    base.add(Node("fluid_family", "Family_t"))
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    points = np.column_stack((np.tile(xy, (2, 1)), np.repeat([0., 1.], 4))) if dimension == 3 else xy
    zone = base.add(Node("fluid_domain", "Zone_t", np.array([[len(points), 1, 0]], np.int32)))
    zone.add(Node("ZoneType", "ZoneType_t", "Unstructured"))
    zone.add(Node("FamilyName", "FamilyName_t", "fluid_family"))
    grid = zone.add(Node("GridCoordinates", "GridCoordinates_t"))
    for axis, values in zip("XYZ", points.T):
        grid.add(Node(f"Coordinate{axis}", "DataArray_t", values))

    def section(name, kind, tag, connectivity):
        node = zone.add(Node(name, "Elements_t", np.array([kind, 0], np.int32)))
        node.add(Node("ElementRange", "IndexRange_t", np.array([tag, tag], np.int32)))
        node.add(Node("ElementConnectivity", "DataArray_t", np.array(connectivity, np.int32)))
        return node

    section("fluid_domain", 20, 1, [17 if dimension == 3 else 7, *range(1, len(points)+1)])
    faces = {"inlet": [1, 2, 6, 5], "outlet": [4, 8, 7, 3],
             "periodic_left": [1, 5, 8, 4], "periodic_right": [2, 3, 7, 6],
             "hub": [1, 4, 3, 2], "shroud": [5, 6, 7, 8]} if dimension == 3 else {
                 "inlet": [1, 2], "outlet": [3, 4], "periodic_left": [4, 1], "periodic_right": [2, 3]}
    bcs = zone.add(Node("ZoneBC", "ZoneBC_t"))
    for tag, (name, face) in enumerate(faces.items(), 2):
        section(name, 7 if dimension == 3 else 3, tag, face).add(
            Node("ParentElements", "DataArray_t", np.array([[1, 0]], np.int32)))
        bc = bcs.add(Node(name, "BC_t", "BCWallViscous"))
        bc.add(Node("GridLocation", "GridLocation_t", "FaceCenter" if dimension == 3 else "EdgeCenter"))
        bc.add(Node("PointRange", "IndexRange_t", np.array([[tag, tag]], np.int32)))
    connections = zone.add(Node("ZoneGridConnectivity", "ZoneGridConnectivity_t"))
    right, left = ([2, 3, 6, 7], [1, 4, 5, 8]) if dimension == 3 else ([2, 3], [1, 4])
    for name, current, donor, offset in (("right_to_left", right, left, -1), ("left_to_right", left, right, 1)):
        connection = connections.add(Node(name, "GridConnectivity_t", zone.name))
        connection.add(Node("GridConnectivityType", "GridConnectivityType_t", "Abutting1to1"))
        connection.add(Node("GridLocation", "GridLocation_t", "Vertex"))
        connection.add(Node("PointList", "IndexArray_t", np.array([current], np.int32)))
        connection.add(Node("PointListDonor", "IndexArray_t", np.array([donor], np.int32)))
        periodic = connection.add(Node("GridConnectivityProperty", "GridConnectivityProperty_t")).add(
            Node("Periodic", "Periodic_t"))
        for key, values in (("RotationCenter", [0]*dimension), ("RotationAngle", [0]*dimension),
                            ("Translation", [offset]+[0]*(dimension-1))):
            periodic.add(Node(key, "DataArray_t", np.array(values, np.float32)))
    write_adf(path, root)
    return root


def assert_preserved(source, target):
    """Names may change; numerical arrays, BC types and periodic arrays may not."""
    for original, copied in zip(source.children, target.children):
        assert original.label == copied.label
        if isinstance(original.data, np.ndarray):
            assert np.array_equal(original.data, copied.data)
        elif original.label not in ("FamilyName_t", "GridConnectivity_t"):
            assert original.data == copied.data
        assert_preserved(original, copied)


def test_combination_keeps_row_local_arrays_and_updates_names(tmp_path):
    sources = [(name, tmp_path / f"{name}.cgns") for name in ("stator_1", "rotor_1")]
    originals = [row_file(path) for _, path in sources]
    data, manifest = combine_rows(sources)
    base = read_adf_bytes(data).get(label="CGNSBase_t")
    zones = base.find_all("Zone_t")
    assert len(zones) == 2
    # Identical coordinates deliberately stay distinct, as do the interface faces.
    assert sum(zone.data.ravel()[0] for zone in zones) == 16
    for (row_id, path), original, zone in zip(sources, originals, zones):
        assert zone.name == f"{row_id}_fluid"
        assert_preserved(original.get(label="CGNSBase_t").get(label="Zone_t"), zone)
        assert zone.get(label="FamilyName_t").text == f"{row_id}_fluid_family"
        assert all(node.text == zone.name for node in zone.get(label="ZoneGridConnectivity_t").children)
        assert manifest["rows"][row_id]["boundaries"]["inlet"] == f"{row_id}_inlet"
        assert read_adf(path).get(label="CGNSBase_t").get(label="Zone_t").name == "fluid_domain"


def test_long_row_names_are_stable_unique_and_bounded(tmp_path):
    row_file(tmp_path / "row.cgns")
    names = ["a"*50 + "stator", "a"*50 + "rotor"]
    data, manifest = combine_rows([(name, tmp_path / "row.cgns") for name in names])
    base = read_adf_bytes(data).get(label="CGNSBase_t")
    assert len({row["zone"] for row in manifest["rows"].values()}) == 2
    assert all(len(node.name) <= 32 for _, node in base.walk())


def test_different_dimensions_are_not_combined(tmp_path):
    row_file(tmp_path / "two.cgns", 2)
    row_file(tmp_path / "three.cgns", 3)
    with pytest.raises(ValueError, match="same CGNS dimensions"):
        combine_rows([("stator", tmp_path / "two.cgns"), ("rotor", tmp_path / "three.cgns")])
