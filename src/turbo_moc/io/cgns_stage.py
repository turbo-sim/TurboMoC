"""Combine canonical row exports without welding their interface meshes.

CGNS connectivity is local to each zone. Keeping separate zones preserves all
node/element IDs, boundary parents and periodic maps; only names and donor-zone
references change. The CFD solver must configure the inter-row flow interface.
"""
from hashlib import sha256
import json

import numpy as np

from .cgns_adf import AdfNode, new_root, read_adf, serialize_adf


def _name(row_id, original):
    """Use stable row-qualified names within ADF's 32-character limit."""
    value = f"{row_id}_{original}"
    if len(value) > 32:
        value = value[:23] + "_" + sha256(value.encode("ascii")).hexdigest()[:8]
    return value


def _audit_zone(zone):
    """Check the references affected by assembly and retain local connectivity."""
    if zone.get(label="ZoneType_t").text != "Unstructured":
        raise ValueError("Stage export requires unstructured row meshes.")
    nodes, cells, _ = np.asarray(zone.data).ravel()
    if nodes <= 0 or cells <= 0:
        raise ValueError("A stage row contains an empty mesh.")
    grid = zone.get(label="GridCoordinates_t")
    if grid is None or any(array.data.size != nodes or not np.all(np.isfinite(array.data))
                           for array in grid.find_all("DataArray_t")):
        raise ValueError("Invalid row coordinate arrays.")
    ranges = [section.get("ElementRange").data.ravel()
              for section in zone.find_all("Elements_t")]
    last = max(end for _, end in ranges)
    for _, node in zone.walk():
        if node.label == "BC_t":
            point_range = node.get("PointRange")
            if point_range is None or not any(
                    np.array_equal(point_range.data.ravel(), extent) for extent in ranges):
                raise ValueError("A boundary does not refer to its row's element section.")
        elif node.label == "GridConnectivity_t":
            if node.text != zone.name:
                raise ValueError("Row periodic connectivity must refer to its own zone.")
            for name in ("PointList", "PointListDonor"):
                points = node.get(name)
                if points is None or np.any(points.data < 1) or np.any(points.data > nodes):
                    raise ValueError("Invalid periodic node indices.")
    if last < cells:
        raise ValueError("The row's element sections do not cover its fluid cells.")
    return int(nodes), int(cells)


def combine_rows(sources):
    """Return one ADF image and manifest for ordered ``(row_id, path)`` sources.

    Sources must be canonical one-zone exports of the same dimension and units.
    Coordinates already contain assembly placement; applying it again here
    would move the rotor twice. Coincident inter-row vertices remain separate.
    """
    if not sources or len({row_id for row_id, _ in sources}) != len(sources):
        raise ValueError("Stage export needs distinct, nonempty row sources.")
    root, base, version, units = new_root(), None, None, None
    manifest = {"format": "CGNS 3.4 / ADF", "rows": {}}
    for row_id, path in sources:
        source = read_adf(path)
        bases = source.find_all("CGNSBase_t")
        versions = source.find_all("CGNSLibraryVersion_t")
        if len(bases) != 1 or len(versions) != 1 or len(bases[0].find_all("Zone_t")) != 1:
            raise ValueError("Stage export requires one base and one zone per row file.")
        previous = bases[0]
        source_units = previous.get(label="DimensionalUnits_t")
        if source_units is None:
            raise ValueError("Row export is missing CGNS dimensional units.")
        if base is None:
            version = versions[0].data
            units = source_units.data
            root.add(versions[0].copy())
            base = root.add(AdfNode("TurboMoC", "CGNSBase_t", previous.data))
            for label in ("DataClass_t", "DimensionalUnits_t"):
                node = previous.get(label=label)
                if node is not None:
                    base.add(node.copy())
            manifest["cell_dimension"], manifest["physical_dimension"] = map(int, previous.data)
        elif (not np.array_equal(previous.data, base.data) or
              not np.array_equal(versions[0].data, version) or
              not np.array_equal(source_units.data, units)):
            raise ValueError("Combined rows must have the same CGNS dimensions, version and units.")

        family_names = {}
        for family in previous.find_all("Family_t"):
            copied = family.copy()
            copied.name = family_names[family.name] = _name(row_id, family.name)
            base.add(copied)
        original = previous.get(label="Zone_t")
        zone = original.copy()
        zone.name = _name(row_id, "fluid")
        boundaries, sections = {}, {}
        for _, node in zone.walk():
            if node.label in ("Elements_t", "BC_t", "GridConnectivity_t"):
                old_name = node.name
                node.name = _name(row_id, old_name)
                if node.label == "BC_t":
                    boundaries[old_name] = node.name
                elif node.label == "Elements_t":
                    sections[old_name] = node.name
                else:
                    if node.text != original.name:
                        raise ValueError("Only row-local periodic donors are supported in row exports.")
                    node.set_data(zone.name)
            elif node.label in ("FamilyName_t", "AdditionalFamilyName_t"):
                if node.text not in family_names:
                    raise ValueError("A row refers to a missing CGNS family.")
                node.set_data(family_names[node.text])
        # Keep per-row provenance next to its zone rather than retaining only
        # the first row's base-level Assembly descriptor.
        assembly = previous.get("Assembly", "Descriptor_t")
        if assembly is not None:
            zone.add(assembly.copy())
        nodes, cells = _audit_zone(zone)
        base.add(zone)
        manifest["rows"][row_id] = {"zone": zone.name, "nodes": nodes, "cells": cells,
            "boundaries": boundaries, "element_sections": sections}
    # ADF truncates names on disk; catch collisions before writing instead of
    # silently aliasing distinct boundary/zone names.
    for _, parent in root.walk():
        if len({child.name for child in parent.children}) != len(parent.children):
            raise ValueError("Combined CGNS node names collide.")
    base.add(AdfNode("StageExport", "Descriptor_t", json.dumps(manifest, sort_keys=True)))
    return serialize_adf(root), manifest
