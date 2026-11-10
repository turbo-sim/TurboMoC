#!/usr/bin/env python3
"""
cgns_adf_3Dto2D
===============

Extract a planar boundary surface (by default the hub) from a 3-D CGNS file
written in the ADF dialect -- as exported by ANSYS TurboGrid -- and write it as
a 2-D unstructured CGNS/ADF file.

The module carries its own small reader/writer for the ADF container format
(the CGNS "ADF" dialect, database version ``A02011``), so the only runtime
dependency is numpy.  No CGNS / HDF5 library and no ANSYS installation is
needed, which keeps the tool portable between machines.

Library use::

    from cgns_adf_3Dto2D import extract_planar_surface_2d
    extract_planar_surface_2d("mesh.cgns")                       # -> mesh_2D.cgns, surface "HUB"
    extract_planar_surface_2d("mesh.cgns", "hub.cgns", "SHROUD")

Command line::

    python cgns_adf_3Dto2D.py mesh.cgns [-o mesh_2D.cgns] [-s HUB] [--mapping plane|cylinder]
    python cgns_adf_3Dto2D.py mesh.cgns --list        # show zones / surfaces

What the conversion does
------------------------
* Reads the whole ADF node tree of the 3-D file.
* For every unstructured zone, collects all 2-D boundary elements
  (``TRI_3`` / ``QUAD_4``, also inside ``MIXED`` sections) and all
  ``BC_t`` boundary conditions.
* Selects the boundary conditions whose name matches ``surface_name``
  (exact name, or a whitespace-separated token such as ``HUB`` in
  ``"INBlock HUB"``, ``"Passage HUB"``, ``"OUTBlock HUB"``).
* Fits a plane through the selected faces (warns if they are not planar),
  drops the plane-normal coordinate axis (or projects onto an orthonormal
  in-plane basis if the plane is not axis aligned) and renumbers the nodes.
  With ``mapping="cylinder"`` a surface of constant radius about a
  coordinate axis is unwrapped into (axial coordinate, radius * angle)
  instead.
* Re-orients the faces counter-clockwise in 2-D.
* Turns the boundary of the extracted surface into ``BAR_2`` element
  sections, one per neighbouring 3-D boundary condition, and writes a
  matching ``BC_t`` (``EdgeCenter``) for each of them.  The names, BC types
  and family names of the 3-D boundary conditions are preserved.
* Writes a new ADF file with ``CellDimension = PhysicalDimension = 2``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "AdfNode",
    "read_adf",
    "write_adf",
    "extract_planar_surface_2d",
    "list_surfaces",
    "default_output_filename",
    "DEFAULT_SURFACE_NAME",
]

log = logging.getLogger("cgns_adf_3Dto2D")

#: Default surface to extract.  TurboGrid names the hub patches e.g.
#: ``"INBlock HUB"``, ``"Passage HUB"`` and ``"OUTBlock HUB"``; the token
#: ``HUB`` selects all of them.
DEFAULT_SURFACE_NAME = "HUB"

# =============================================================================
#  ADF container format (CGNS "ADF" dialect)
# =============================================================================
#
# Layout (all sizes in bytes, all integers/pointers ASCII upper-case hex):
#
#   file header      186   @0      "AdF0".."AdF5" tags, format chars, pointers
#   free chunk table  80   @186    "fCbt" ... "Fcte"   (all blank here)
#   root node header 246   @266    "NoDe" ... "TaiL"
#
#   node header (246): "NoDe" name[32] label[32] num_sub_nodes[8]
#       entries_for_sub_nodes[8] sub_node_table_ptr[12] data_type[32]
#       num_dims[2] dims[12*8] num_data_chunks[4] data_ptr[12] "TaiL"
#   sub-node table:   "SNTb" end_ptr[12] {name[32] ptr[12]}* "snTE"
#   data chunk:       "DaTa" end_ptr[12] payload "dEnD"
#   data chunk table: "DCtb" end_ptr[12] {start[12] end[12]}* "dcTE"
#   disk pointer: block[8 hex] offset[4 hex], byte = block*4096 + offset;
#       the blank pointer is block 0, offset 4096.
#
# Reference: src/adf/ADF_internals.c of the CGNS library.

DISK_BLOCK_SIZE = 4096
FILE_HEADER_SIZE = 186
FREE_CHUNK_TABLE_SIZE = 80
NODE_HEADER_SIZE = 246
ROOT_NODE_OFFSET = FILE_HEADER_SIZE + FREE_CHUNK_TABLE_SIZE
TAG_SIZE = 4
PTR_SIZE = 12
NAME_LENGTH = 32
MAX_DIMENSIONS = 12
BLANK_POINTER = -1  # in-memory marker for "block 0 / offset 4096"

ADF_WHAT_STRING = b"\xc0\xa8\xa3\xa9ADF Database Version A02011>"
ROOT_NODE_NAME = "ADF MotherNode"
ROOT_NODE_LABEL = "Root Node of ADF File"

_ADF_TO_NUMPY = {
    "I4": "i4", "I8": "i8", "U4": "u4", "U8": "u8",
    "R4": "f4", "R8": "f8", "X4": "c8", "X8": "c16",
    "C1": "S1", "B1": "u1", "LK": "S1",
}
_NUMPY_TO_ADF = {
    np.dtype("i4"): "I4", np.dtype("i8"): "I8",
    np.dtype("u4"): "U4", np.dtype("u8"): "U8",
    np.dtype("f4"): "R4", np.dtype("f8"): "R8",
    np.dtype("c8"): "X4", np.dtype("c16"): "X8",
    np.dtype("S1"): "C1", np.dtype("u1"): "B1",
}


class AdfError(ValueError):
    """Malformed or unsupported ADF/CGNS content."""


class AdfNode:
    """One node of an ADF tree.

    ``label`` is the CGNS node type (``Zone_t`` ...).  ``data`` is ``None``
    (data type ``MT``), a ``str`` (data type ``C1``) or a numpy array whose
    dtype maps to an ADF data type.  Multi-dimensional arrays keep their
    Fortran (column-major) ordering on disk.
    """

    __slots__ = ("name", "label", "dtype", "data", "children")

    def __init__(self, name: str, label: str = "", data=None, dtype: str | None = None,
                 children: Iterable["AdfNode"] | None = None):
        self.name = name
        self.label = label
        self.children: list[AdfNode] = list(children) if children else []
        self.set_data(data, dtype)

    # -- data ---------------------------------------------------------------
    def set_data(self, data, dtype: str | None = None) -> None:
        if data is None:
            self.dtype, self.data = "MT", None
            return
        if isinstance(data, str):
            self.dtype, self.data = "C1", data
            return
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if dtype is None:
            code = _NUMPY_TO_ADF.get(arr.dtype)
            if code is None:
                raise TypeError(f"no ADF data type for numpy dtype {arr.dtype}")
        else:
            code = dtype
            if code not in _ADF_TO_NUMPY:
                raise TypeError(f"unknown ADF data type {code!r}")
            arr = arr.astype(_ADF_TO_NUMPY[code], copy=False)
        self.dtype, self.data = code, arr

    @property
    def text(self) -> str:
        """String content of a ``C1`` node (trailing NULs/blanks stripped)."""
        if isinstance(self.data, str):
            return self.data.rstrip("\x00 ")
        if self.data is None:
            return ""
        return self.data.tobytes(order="F").decode("latin-1").rstrip("\x00 ")

    # -- tree ---------------------------------------------------------------
    def get(self, name: str | None = None, label: str | None = None) -> "AdfNode | None":
        for c in self.children:
            if (name is None or c.name == name) and (label is None or c.label == label):
                return c
        return None

    def find_all(self, label: str) -> list["AdfNode"]:
        return [c for c in self.children if c.label == label]

    def add(self, node: "AdfNode") -> "AdfNode":
        self.children.append(node)
        return node

    def copy(self) -> "AdfNode":
        """Deep copy of the sub-tree (array data is shared, never mutated)."""
        n = AdfNode.__new__(AdfNode)
        n.name, n.label, n.dtype, n.data = self.name, self.label, self.dtype, self.data
        n.children = [c.copy() for c in self.children]
        return n

    def walk(self, path: str = ""):
        p = f"{path}/{self.name}" if path or self.name != ROOT_NODE_NAME else ""
        yield p or "/", self
        for c in self.children:
            yield from c.walk(p)

    def __repr__(self) -> str:
        shape = "" if self.data is None else (
            f" str[{len(self.data)}]" if isinstance(self.data, str) else f" {self.dtype}{list(self.data.shape)}")
        return f"<AdfNode {self.name!r} {self.label}{shape} children={len(self.children)}>"


def new_root() -> AdfNode:
    """Create an empty ADF root node."""
    return AdfNode(ROOT_NODE_NAME, ROOT_NODE_LABEL)


# ----------------------------------------------------------------------------
#  reader
# ----------------------------------------------------------------------------
class _AdfReader:
    def __init__(self, buf: bytes, filename: str):
        self.buf = buf
        if buf[:8] == b"\x89HDF\r\n\x1a\n":
            raise AdfError(f"{filename}: this is an HDF5-based CGNS file, only the ADF dialect is supported")
        if len(buf) < ROOT_NODE_OFFSET + NODE_HEADER_SIZE or buf[4:24] != b"ADF Database Version":
            raise AdfError(f"{filename}: not an ADF (CGNS) file")
        for i, off in enumerate((32, 64, 96, 102, 130, 182)):
            if buf[off:off + 4] != b"AdF%d" % i:
                raise AdfError(f"{filename}: corrupt ADF file header (tag {i})")
        # 'A....' versions store dimensions and pointers as ASCII hex, 'B....'
        # versions store them as binary integers.
        self.old_version = buf[25:26] != b"B"
        fmt, os_size = chr(buf[100]), chr(buf[101])
        if fmt in ("L", "N"):
            self.endian = "<"
        elif fmt == "B":
            self.endian = ">"
        else:
            raise AdfError(f"{filename}: unsupported ADF numeric format {fmt!r}")
        self.numeric_format, self.os_size = fmt, os_size
        self.root_pos = self.ptr(134)

    def ptr(self, off: int) -> int:
        """Absolute byte position of the disk pointer stored at ``off``."""
        if self.old_version:
            block = int(self.buf[off:off + 8], 16)
            offset = int(self.buf[off + 8:off + 12], 16)
        else:
            block = int(np.frombuffer(self.buf[off:off + 8], self.endian + "i8")[0])
            offset = int(np.frombuffer(self.buf[off + 8:off + 12], self.endian + "i4")[0])
        if block == 0 and offset == DISK_BLOCK_SIZE:
            return BLANK_POINTER
        return block * DISK_BLOCK_SIZE + offset

    def read_root(self) -> AdfNode:
        return self.read_node(self.root_pos)

    def read_node(self, pos: int) -> AdfNode:
        buf = self.buf
        hdr = buf[pos:pos + NODE_HEADER_SIZE]
        if hdr[:4] != b"NoDe" or hdr[242:246] != b"TaiL":
            raise AdfError(f"corrupt node header at byte {pos}")
        name = hdr[4:36].decode("latin-1").rstrip()
        label = hdr[36:68].decode("latin-1").rstrip()
        num_sub = int(hdr[68:76], 16)
        sub_pos = self.ptr(pos + 84)
        dtype = hdr[96:128].decode("latin-1").strip()
        ndims = int(hdr[128:130], 16)
        if self.old_version:
            dims = [int(hdr[130 + 8 * i:138 + 8 * i], 16) for i in range(ndims)]
        else:
            dims = [int(d) for d in np.frombuffer(hdr[130:226], self.endian + "i8")[:ndims]]
        nchunks = int(hdr[226:230], 16)
        data_pos = self.ptr(pos + 230)

        node = AdfNode(name, label)
        if dtype != "MT" and ndims > 0 and nchunks > 0:
            node.dtype, node.data = self._read_data(dtype, dims, nchunks, data_pos, name)
        if num_sub > 0:
            if buf[sub_pos:sub_pos + 4] != b"SNTb":
                raise AdfError(f"corrupt sub-node table of node {name!r}")
            p = sub_pos + TAG_SIZE + PTR_SIZE
            for _ in range(num_sub):
                node.children.append(self.read_node(self.ptr(p + NAME_LENGTH)))
                p += NAME_LENGTH + PTR_SIZE
        return node

    def _read_data(self, dtype, dims, nchunks, data_pos, name):
        code = _ADF_TO_NUMPY.get(dtype)
        if code is None:
            raise AdfError(f"node {name!r}: unsupported ADF data type {dtype!r}")
        itemsize = np.dtype(code).itemsize
        nbytes = itemsize * int(np.prod(dims, dtype=np.int64))
        if nchunks == 1:
            raw = self._chunk_bytes(data_pos, nbytes, name)
        else:  # chunk table
            if self.buf[data_pos:data_pos + 4] != b"DCtb":
                raise AdfError(f"corrupt data chunk table of node {name!r}")
            parts, got, p = [], 0, data_pos + TAG_SIZE + PTR_SIZE
            for _ in range(nchunks):
                start, end = self.ptr(p), self.ptr(p + PTR_SIZE)
                p += 2 * PTR_SIZE
                avail = end - start - TAG_SIZE - PTR_SIZE
                take = min(avail, nbytes - got)
                if take <= 0:
                    break
                parts.append(self._chunk_bytes(start, take, name))
                got += take
            raw = b"".join(parts)
        if len(raw) < nbytes:
            log.warning("node %r: data incomplete (%d of %d bytes), padding with zeros", name, len(raw), nbytes)
            raw = raw + b"\x00" * (nbytes - len(raw))
        if dtype in ("C1", "LK") and len(dims) == 1:
            return dtype, raw.decode("latin-1")
        arr = np.frombuffer(raw, self.endian + code).reshape(dims, order="F")
        if arr.dtype.byteorder not in ("=", "|") and arr.dtype != arr.dtype.newbyteorder("="):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        return dtype, np.array(arr, copy=True)

    def _chunk_bytes(self, pos, nbytes, name):
        if self.buf[pos:pos + 4] != b"DaTa":
            raise AdfError(f"corrupt data chunk of node {name!r} at byte {pos}")
        end = self.ptr(pos + TAG_SIZE)
        start = pos + TAG_SIZE + PTR_SIZE
        return self.buf[start:start + min(nbytes, end - start)]


def read_adf(filename: str) -> AdfNode:
    """Read a complete ADF (CGNS) file into an :class:`AdfNode` tree."""
    with open(filename, "rb") as fh:
        buf = fh.read()
    return _AdfReader(buf, filename).read_root()


# ----------------------------------------------------------------------------
#  writer
# ----------------------------------------------------------------------------
def _hex_ptr(pos: int) -> bytes:
    if pos == BLANK_POINTER:
        return b"00000000" + b"%04X" % DISK_BLOCK_SIZE
    block, offset = divmod(pos, DISK_BLOCK_SIZE)
    if block > 0xFFFFFFFF:
        raise AdfError("file too large for the ADF pointer format")
    return b"%08X%04X" % (block, offset)


def _fixed(text: str, length: int, what: str) -> bytes:
    raw = text.encode("latin-1")
    if len(raw) > length:
        raise AdfError(f"{what} {text!r} longer than {length} characters")
    return raw.ljust(length)


class _AdfWriter:
    """Serialise an :class:`AdfNode` tree into a fresh ADF file image.

    Space is handed out at the end of the file exactly like
    ``ADFI_file_malloc`` in the reference implementation (small chunks never
    straddle a 4096-byte block boundary; skipped bytes are filled with ``z``).
    """

    def __init__(self):
        self.buf = bytearray(ROOT_NODE_OFFSET)
        self.eof = ROOT_NODE_OFFSET - 1  # last byte in use

    def alloc(self, size: int) -> int:
        eof_block, eof_off = divmod(self.eof, DISK_BLOCK_SIZE)
        if eof_off != DISK_BLOCK_SIZE - 1 and eof_off + size >= DISK_BLOCK_SIZE and size <= DISK_BLOCK_SIZE:
            pos = (eof_block + 1) * DISK_BLOCK_SIZE
        else:
            pos = self.eof + 1
        gap = pos - len(self.buf)
        if gap > 0:
            self.buf.extend(b"z" * gap)
        self.buf.extend(bytes(size))
        self.eof = pos + size - 1
        return pos

    def put(self, pos: int, data: bytes) -> None:
        self.buf[pos:pos + len(data)] = data

    @staticmethod
    def payload(node: AdfNode) -> tuple[bytes, list[int]]:
        if isinstance(node.data, str):
            raw = node.data.encode("latin-1")
            return raw, [len(raw)]
        arr = node.data
        if node.dtype in ("C1", "LK"):
            arr = np.asarray(arr, dtype="S1")
        else:
            arr = np.asarray(arr).astype("<" + _ADF_TO_NUMPY[node.dtype], copy=False)
        dims = list(arr.shape)
        if len(dims) > MAX_DIMENSIONS or any(d > 0xFFFFFFFF for d in dims):
            raise AdfError(f"node {node.name!r}: array shape {dims} not representable in ADF")
        return arr.tobytes(order="F"), dims

    def emit(self, node: AdfNode) -> int:
        pos = self.alloc(NODE_HEADER_SIZE)
        # ---- data
        nchunks, data_pos, dims = 0, BLANK_POINTER, []
        if node.data is not None and node.dtype != "MT":
            raw, dims = self.payload(node)
            if raw:
                data_pos = self.alloc(TAG_SIZE + PTR_SIZE + len(raw) + TAG_SIZE)
                end = data_pos + TAG_SIZE + PTR_SIZE + len(raw)
                self.put(data_pos, b"DaTa" + _hex_ptr(end) + raw + b"dEnD")
                nchunks = 1
        # ---- children
        child_pos = [self.emit(c) for c in node.children]
        nsub, sub_pos = len(child_pos), BLANK_POINTER
        if nsub:
            size = TAG_SIZE + PTR_SIZE + nsub * (NAME_LENGTH + PTR_SIZE) + TAG_SIZE
            sub_pos = self.alloc(size)
            end = sub_pos + size - TAG_SIZE
            entries = b"".join(_fixed(c.name, NAME_LENGTH, "node name") + _hex_ptr(p)
                               for c, p in zip(node.children, child_pos))
            self.put(sub_pos, b"SNTb" + _hex_ptr(end) + entries + b"snTE")
        # ---- header
        dim_field = b"".join(b"%08X" % d for d in dims) + b"00000000" * (MAX_DIMENSIONS - len(dims))
        hdr = (b"NoDe" + _fixed(node.name, NAME_LENGTH, "node name")
               + _fixed(node.label, NAME_LENGTH, "node label")
               + b"%08X%08X" % (nsub, nsub) + _hex_ptr(sub_pos)
               + _fixed(node.dtype if node.data is not None else "MT", NAME_LENGTH, "data type")
               + b"%02X" % len(dims) + dim_field + b"%04X" % nchunks + _hex_ptr(data_pos) + b"TaiL")
        assert len(hdr) == NODE_HEADER_SIZE
        self.put(pos, hdr)
        return pos

    def finish(self, root: AdfNode) -> bytes:
        root_pos = self.emit(root)
        assert root_pos == ROOT_NODE_OFFSET
        date = _fixed(time.strftime("%a %b %d %H:%M:%S %Y"), 28, "date")
        sizes = b"01" + b"02" + b"04" + b"04" + b"04" + b"08" + b"08" * 6  # char short int long float double + 6 pointers
        header = (ADF_WHAT_STRING + b"AdF0" + date + b"AdF1" + date + b"AdF2" + b"LL" + b"AdF3" + sizes
                  + b"AdF4" + _hex_ptr(ROOT_NODE_OFFSET) + _hex_ptr(self.eof) + _hex_ptr(FILE_HEADER_SIZE)
                  + _hex_ptr(BLANK_POINTER) + b"AdF5")
        assert len(header) == FILE_HEADER_SIZE
        free_table = b"fCbt" + _hex_ptr(BLANK_POINTER) * 6 + b"Fcte"
        assert len(free_table) == FREE_CHUNK_TABLE_SIZE
        self.put(0, header + free_table)
        pad = (-len(self.buf)) % DISK_BLOCK_SIZE
        return bytes(self.buf) + bytes(pad)


def write_adf(filename: str, root: AdfNode) -> None:
    """Write an :class:`AdfNode` tree as a new little-endian ADF (CGNS) file."""
    image = _AdfWriter().finish(root)
    with open(filename, "wb") as fh:
        fh.write(image)


# =============================================================================
#  CGNS helpers
# =============================================================================
BAR_2, TRI_3, QUAD_4, MIXED = 3, 5, 7, 20
NODES_PER_ELEMENT = {2: 1, 3: 2, 4: 3, 5: 3, 6: 6, 7: 4, 8: 8, 9: 9, 10: 4, 11: 10, 12: 5, 13: 14,
                     14: 6, 15: 15, 16: 18, 17: 8, 18: 20, 19: 27, 21: 13, 24: 4, 25: 9, 26: 10,
                     27: 12, 28: 16, 29: 20, 30: 35, 31: 21, 32: 30, 33: 40, 34: 55, 35: 64, 36: 24,
                     37: 21, 38: 31, 39: 44, 40: 50}
SURFACE_ELEMENT_TYPES = {TRI_3: 3, QUAD_4: 4}


def _require(node: AdfNode | None, what: str) -> AdfNode:
    if node is None:
        raise AdfError(f"missing {what}")
    return node


def _int_array(node: AdfNode) -> np.ndarray:
    return np.asarray(node.data).ravel(order="F").astype(np.int64)


def _match_surface_names(query: str, names: Sequence[str]) -> tuple[list[str], str | None]:
    """Return the names matching ``query`` and the label to use for the result.

    Matching order: exact, case-insensitive exact, whitespace/underscore
    separated token (``hub`` matches ``"INBlock HUB"``), sub-string.
    """
    q = query.strip()
    if not q:
        return [], None
    hit = [n for n in names if n == q]
    if hit:
        return hit, q
    hit = [n for n in names if n.strip().lower() == q.lower()]
    if hit:
        return hit, hit[0].strip()
    label = None
    for n in names:
        for tok in re.split(r"[\s_\-]+", n.strip()):
            if tok.lower() == q.lower():
                hit.append(n)
                label = label or tok
                break
    if hit:
        return hit, label
    hit = [n for n in names if q.lower() in n.lower()]
    return hit, (q if hit else None)


class _FaceTable:
    """All linear surface elements (TRI_3 / QUAD_4) of one zone.

    ``ids`` are the global CGNS element numbers (sorted), ``nodes`` the
    0-based vertex indices padded with -1 for triangles.
    """

    def __init__(self, ids: np.ndarray, nodes: np.ndarray, section: np.ndarray, section_names: list[str]):
        order = np.argsort(ids, kind="stable")
        self.ids, self.nodes, self.section = ids[order], nodes[order], section[order]
        self.section_names = section_names

    def __len__(self) -> int:
        return len(self.ids)

    def rows_for_ids(self, ids: np.ndarray, what: str) -> np.ndarray:
        rows = np.searchsorted(self.ids, ids)
        bad = (rows >= len(self.ids)) | (self.ids[np.minimum(rows, len(self.ids) - 1)] != ids)
        if bad.any():
            raise AdfError(f"{what}: element {int(ids[bad][0])} is not a TRI_3/QUAD_4 boundary face")
        return rows

    def rows_for_vertices(self, mask: np.ndarray) -> np.ndarray:
        inside = np.where(self.nodes >= 0, mask[np.maximum(self.nodes, 0)], True)
        return np.nonzero(inside.all(axis=1))[0]


def _collect_faces(zone: AdfNode) -> _FaceTable:
    ids, nodes, section, names = [], [], [], []
    for sec in zone.find_all("Elements_t"):
        etype = int(_int_array(sec)[0])
        lo, hi = (int(v) for v in _int_array(_require(sec.get("ElementRange"), f"ElementRange of {sec.name!r}")))
        conn_node = sec.get("ElementConnectivity")
        if conn_node is None or conn_node.data is None:
            continue
        conn = _int_array(conn_node)
        sec_index = len(names)
        names.append(sec.name)
        if etype in SURFACE_ELEMENT_TYPES:
            npe = SURFACE_ELEMENT_TYPES[etype]
            elem = conn.reshape(-1, npe) - 1
            if len(elem) != hi - lo + 1:
                raise AdfError(f"section {sec.name!r}: connectivity size does not match ElementRange")
            if npe == 3:
                elem = np.column_stack([elem, np.full(len(elem), -1, np.int64)])
            ids.append(np.arange(lo, hi + 1, dtype=np.int64))
            nodes.append(elem)
            section.append(np.full(len(elem), sec_index, np.int64))
        elif etype == MIXED:
            off_node = sec.get("ElementStartOffset")
            if off_node is not None and off_node.data is not None:
                starts = _int_array(off_node)[:-1]
            else:  # CGNS < 4.0: walk the connectivity
                starts, p = [], 0
                while p < len(conn):
                    starts.append(p)
                    p += 1 + NODES_PER_ELEMENT[int(conn[p])]
                starts = np.asarray(starts, np.int64)
            types = conn[starts]
            for t, npe in SURFACE_ELEMENT_TYPES.items():
                sel = np.nonzero(types == t)[0]
                if not len(sel):
                    continue
                elem = conn[starts[sel, None] + 1 + np.arange(npe)] - 1
                if npe == 3:
                    elem = np.column_stack([elem, np.full(len(elem), -1, np.int64)])
                ids.append(lo + sel)
                nodes.append(elem)
                section.append(np.full(len(elem), sec_index, np.int64))
    if not ids:
        raise AdfError(f"zone {zone.name!r}: no TRI_3/QUAD_4 boundary face sections found")
    return _FaceTable(np.concatenate(ids), np.concatenate(nodes), np.concatenate(section), names)


def _bc_faces(bc: AdfNode, faces: _FaceTable, nverts: int) -> np.ndarray:
    """Rows of ``faces`` that belong to boundary condition ``bc``."""
    loc_node = bc.get(label="GridLocation_t")
    location = loc_node.text if loc_node is not None else "Vertex"
    rng = bc.get("PointRange") or bc.get("ElementRange")
    lst = bc.get("PointList") or bc.get("ElementList")
    if rng is not None:
        lo, hi = (int(v) for v in _int_array(rng)[:2])
        ids = np.arange(lo, hi + 1, dtype=np.int64)
    elif lst is not None:
        ids = _int_array(lst)
    else:
        raise AdfError(f"BC {bc.name!r}: no PointRange/PointList")
    element_based = "Center" in location or (rng or lst).name.startswith("Element")
    if element_based:
        return faces.rows_for_ids(ids, f"BC {bc.name!r}")
    mask = np.zeros(nverts, bool)
    mask[ids - 1] = True
    return faces.rows_for_vertices(mask)


# ----------------------------------------------------------------------------
#  geometry
# ----------------------------------------------------------------------------
MAPPINGS = ("plane", "cylinder")
_AXES = "XYZ"


def _cylinder_fit(P: np.ndarray) -> tuple[int, float, float]:
    """Best coordinate axis for a surface of constant radius: (axis, mean radius, relative radius spread)."""
    best = None
    for k in range(3):
        r = np.hypot(P[:, (k + 1) % 3], P[:, (k + 2) % 3])
        mean = float(r.mean())
        rel = float(np.ptp(r) / mean) if mean > 0 else np.inf
        if best is None or rel < best[2]:
            best = (k, mean, rel)
    return best


def _map_to_2d(P: np.ndarray, tol: float, mapping: str, what: str) -> dict:
    """Map the surface points ``P`` (m, 3) to 2-D coordinates.

    ``plane``: least-squares plane.  If its normal is a coordinate axis that
    axis is dropped and the two other coordinates are kept unchanged;
    otherwise the points are projected onto an in-plane orthonormal basis
    (u, v) that follows the two other axes in cyclic order (Y, Z for a
    normal close to X, ...).

    ``cylinder``: surface of constant radius about a coordinate axis,
    developed into (axial coordinate, radius * angle), the angle being
    measured from the mean angular position of the surface.
    """
    extent = float(np.ptp(P, axis=0).max()) or 1.0
    if mapping == "cylinder":
        k, radius, rel = _cylinder_fit(P)
        if rel > tol:
            log.warning("%s: surface is not a cylinder about a coordinate axis "
                        "(radius about %s varies by %.3g relative to %.6g)", what, _AXES[k], rel, radius)
        c, s = P[:, (k + 1) % 3], P[:, (k + 2) % 3]
        cm, sm = c.mean(), s.mean()
        norm = np.hypot(cm, sm)
        cm, sm = cm / norm, sm / norm
        theta = np.arctan2(s * cm - c * sm, c * cm + s * sm)
        theta0 = float(np.arctan2(sm, cm))
        xy = np.column_stack([P[:, k], radius * theta])
        desc = (f"cylinder of radius {radius:.9g} about the {_AXES[k]} axis; "
                f"CoordinateX = {_AXES[k]}, CoordinateY = R*(theta - theta0) with theta = atan2("
                f"{_AXES[(k + 2) % 3]}, {_AXES[(k + 1) % 3]}), theta0 = {theta0:.9g} rad")
        return {"xy": xy, "deviation": float(np.ptp(np.hypot(c, s))), "extent": extent, "description": desc,
                "normal": None}
    if mapping != "plane":
        raise ValueError(f"unknown mapping {mapping!r}; choose one of {MAPPINGS}")

    centre = P.mean(axis=0)
    Q = P - centre
    _, evecs = np.linalg.eigh(Q.T @ Q)
    normal = evecs[:, 0]
    k = int(np.argmax(np.abs(normal)))
    if normal[k] < 0:
        normal = -normal
    deviation = float(np.abs(Q @ normal).max())
    if deviation > tol * extent:
        hint = ""
        ck, cr, crel = _cylinder_fit(P)
        if crel <= tol:
            hint = (f"; it is a cylinder of radius {cr:.6g} about the {_AXES[ck]} axis, "
                    f"consider mapping='cylinder' (--mapping cylinder)")
        log.warning("%s: surface is not planar (max out-of-plane deviation %.3g, extent %.3g)%s",
                    what, deviation, extent, hint)
    if abs(normal[k]) > 1.0 - 1e-6:
        eye = np.eye(3)
        u, v = eye[(k + 1) % 3], eye[(k + 2) % 3]
        desc = (f"axis-aligned plane, {_AXES[k]} = {centre[k]:.9g} dropped; "
                f"CoordinateX = {_AXES[(k + 1) % 3]}, CoordinateY = {_AXES[(k + 2) % 3]}")
    else:
        a = np.eye(3)[(k + 1) % 3]
        u = a - np.dot(a, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        desc = ("general plane; CoordinateX = P.u, CoordinateY = P.v with "
                f"u = ({u[0]:.9g}, {u[1]:.9g}, {u[2]:.9g}), v = ({v[0]:.9g}, {v[1]:.9g}, {v[2]:.9g})")
    desc += f"; normal = ({normal[0]:.9g}, {normal[1]:.9g}, {normal[2]:.9g})"
    return {"xy": P @ np.column_stack([u, v]), "deviation": deviation, "extent": extent,
            "description": desc, "normal": normal}


def _signed_area(faces: np.ndarray, xy: np.ndarray) -> np.ndarray:
    idx = np.where(faces >= 0, faces, faces[:, [0]])  # repeat first vertex for triangles
    p = xy[idx]
    q = np.roll(p, -1, axis=1)
    return 0.5 * np.sum(p[:, :, 0] * q[:, :, 1] - q[:, :, 0] * p[:, :, 1], axis=1)


def _reverse_winding(faces: np.ndarray) -> np.ndarray:
    out = faces.copy()
    tri = faces[:, 3] < 0
    out[~tri, 1], out[~tri, 3] = faces[~tri, 3], faces[~tri, 1]
    out[tri, 1], out[tri, 2] = faces[tri, 2], faces[tri, 1]
    return out


def _face_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges (m, 4, 2) in winding order and a validity mask (m, 4)."""
    nxt = np.roll(faces, -1, axis=1)
    tri = faces[:, 3] < 0
    nxt[tri, 2] = faces[tri, 0]
    valid = np.ones(faces.shape, bool)
    valid[tri, 3] = False
    return np.stack([faces, nxt], axis=-1), valid


# ----------------------------------------------------------------------------
#  extraction of one zone
# ----------------------------------------------------------------------------
_ZONE_CHILDREN_REBUILT = {"ZoneType_t", "GridCoordinates_t", "Elements_t", "ZoneBC_t", "FlowSolution_t",
                          "ZoneGridConnectivity_t", "ZoneIterativeData_t", "DiscreteData_t", "ZoneSubRegion_t"}
_BC_CHILDREN_COPIED = {"FamilyName_t", "AdditionalFamilyName_t", "Descriptor_t", "Ordinal_t"}


def _extract_zone(zone: AdfNode, surface_name: str, planar_tol: float, mapping: str, lib_version: float,
                  source_note: str) -> tuple[AdfNode, dict] | None:
    zone_type = zone.get(label="ZoneType_t")
    if zone_type is None or zone_type.text != "Unstructured":
        raise NotImplementedError(f"zone {zone.name!r}: only Unstructured zones are supported "
                                  f"(found {zone_type.text if zone_type else 'no ZoneType'})")
    what = f"zone {zone.name!r}"
    grid = _require(zone.get(label="GridCoordinates_t"), f"GridCoordinates in {what}")
    coord_nodes = [_require(grid.get(f"Coordinate{a}"), f"Coordinate{a} in {what}") for a in "XYZ"]
    P_all = np.column_stack([np.asarray(c.data).ravel(order="F") for c in coord_nodes]).astype(np.float64)
    nverts = len(P_all)

    faces = _collect_faces(zone)
    zone_bc = zone.get(label="ZoneBC_t")
    bcs = zone_bc.find_all("BC_t") if zone_bc is not None else []
    bc_rows = {bc.name: _bc_faces(bc, faces, nverts) for bc in bcs}

    # ---- select the surface -------------------------------------------------
    matched, label = _match_surface_names(surface_name, [bc.name for bc in bcs])
    if matched:
        sel_rows = np.unique(np.concatenate([bc_rows[n] for n in matched]))
    else:
        matched, label = _match_surface_names(surface_name, faces.section_names)
        if not matched:
            return None
        sel_rows = np.nonzero(np.isin(faces.section, [faces.section_names.index(n) for n in matched]))[0]
    if not len(sel_rows):
        raise AdfError(f"{what}: surface {matched} contains no faces")
    selected = np.zeros(len(faces), bool)
    selected[sel_rows] = True
    face_nodes = faces.nodes[sel_rows]

    # ---- nodes, plane and 2-D coordinates -----------------------------------
    used = np.unique(face_nodes[face_nodes >= 0])
    new_index = np.full(nverts, -1, np.int64)
    new_index[used] = np.arange(len(used))
    plane = _map_to_2d(P_all[used], planar_tol, mapping, f"{what} surface {label!r}")
    xy = plane["xy"]
    local = np.where(face_nodes >= 0, new_index[np.maximum(face_nodes, 0)], -1)

    area = _signed_area(local, xy)
    reversed_winding = int((area < 0).sum()) > int((area > 0).sum())
    if reversed_winding:
        local, face_nodes, area = _reverse_winding(local), _reverse_winding(face_nodes), -area
    if (area <= 0).any():
        log.warning("%s: %d of %d faces are degenerate or inconsistently oriented in 2-D",
                    what, int((area <= 0).sum()), len(area))

    # ---- boundary edges and their neighbouring boundary conditions ----------
    edges, valid = _face_edges(face_nodes)
    e_flat = edges[valid]                                       # (ne, 2) original ids, winding order
    key = np.minimum(e_flat[:, 0], e_flat[:, 1]) * nverts + np.maximum(e_flat[:, 0], e_flat[:, 1])
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    boundary = first[counts == 1]                               # rows in e_flat, in face order
    boundary.sort()

    all_edges, all_valid = _face_edges(faces.nodes)
    a_flat = all_edges[all_valid]
    a_key = np.minimum(a_flat[:, 0], a_flat[:, 1]) * nverts + np.maximum(a_flat[:, 0], a_flat[:, 1])
    a_face = np.repeat(np.arange(len(faces)), all_valid.sum(axis=1))
    order = np.argsort(a_key, kind="stable")
    a_key, a_face = a_key[order], a_face[order]

    bc_of_face = np.full(len(faces), -1, np.int64)
    for i, bc in enumerate(bcs):
        rows = bc_rows[bc.name]
        rows = rows[bc_of_face[rows] < 0]
        bc_of_face[rows] = i
    selected_bc = {i for i, bc in enumerate(bcs) if bc.name in matched}
    vertex_masks: dict[int, np.ndarray] = {}

    def vertex_mask(i: int) -> np.ndarray:
        if i not in vertex_masks:
            m = np.zeros(nverts, bool)
            nodes = faces.nodes[bc_rows[bcs[i].name]]
            m[nodes[nodes >= 0]] = True
            vertex_masks[i] = m
        return vertex_masks[i]

    groups: dict[int, list[int]] = {}      # bc index (-1 = unassigned) -> rows of e_flat
    ambiguous: dict[tuple[int, ...], int] = {}
    for row in boundary:
        k = key[row]
        lo, hi = np.searchsorted(a_key, k, "left"), np.searchsorted(a_key, k, "right")
        neighbours = [f for f in a_face[lo:hi] if not selected[f]]
        cands = sorted({int(bc_of_face[f]) for f in neighbours if bc_of_face[f] >= 0})
        if not cands:  # no adjacent face: fall back to vertex membership
            n0, n1 = e_flat[row]
            cands = [i for i in range(len(bcs)) if i not in selected_bc
                     and vertex_mask(i)[n0] and vertex_mask(i)[n1]]
        if len(cands) > 1:
            # TurboGrid writes overlapping regions (e.g. BLADE and BLD LOW) as
            # duplicated faces: prefer the boundary condition with most faces.
            ambiguous[tuple(cands)] = ambiguous.get(tuple(cands), 0) + 1
            cands.sort(key=lambda i: (-len(bc_rows[bcs[i].name]), i))
        groups.setdefault(cands[0] if cands else -1, []).append(int(row))
    for cands, count in ambiguous.items():
        log.info("%s: %d boundary edges touch %s; assigned to the largest, %r",
                 what, count, [bcs[i].name for i in cands],
                 bcs[max(cands, key=lambda i: (len(bc_rows[bcs[i].name]), -i))].name)
    if -1 in groups:
        log.warning("%s: %d boundary edges are not adjacent to any boundary condition (-> 'Unassigned')",
                    what, len(groups[-1]))

    # ---- build the 2-D zone ---------------------------------------------------
    conn_dtype = np.asarray(_require(zone.find_all("Elements_t")[0].get("ElementConnectivity"),
                                     "ElementConnectivity").data).dtype
    range_node = zone.find_all("Elements_t")[0].get("ElementRange")
    range_dtype, range_shape = np.asarray(range_node.data).dtype, np.asarray(range_node.data).shape
    elem_dtype = np.asarray(zone.find_all("Elements_t")[0].data).dtype

    def index_range(lo: int, hi: int, shape=range_shape) -> np.ndarray:
        return np.array([lo, hi], dtype=range_dtype).reshape(shape, order="F")

    ncells = len(local)
    zone_data = np.array([len(used), ncells, 0], dtype=np.asarray(zone.data).dtype).reshape(
        np.asarray(zone.data).shape, order="F")
    zone2 = AdfNode(zone.name, "Zone_t", zone_data)
    zone2.add(AdfNode("ZoneType", "ZoneType_t", "Unstructured"))
    for c in zone.children:
        if c.label not in _ZONE_CHILDREN_REBUILT:
            zone2.add(c.copy())
        elif c.label in ("ZoneGridConnectivity_t", "ZoneIterativeData_t", "DiscreteData_t", "ZoneSubRegion_t"):
            log.warning("%s: %s node %r is not carried over", what, c.label, c.name)

    grid2 = zone2.add(AdfNode(grid.name, "GridCoordinates_t"))
    for c in grid.children:
        if c.label != "DataArray_t":
            grid2.add(c.copy())
    coord_dtype = np.asarray(coord_nodes[0].data).dtype
    for j, name in enumerate(("CoordinateX", "CoordinateY")):
        cn = grid2.add(AdfNode(name, "DataArray_t", xy[:, j].astype(coord_dtype)))
        for c in coord_nodes[j].children:
            cn.add(c.copy())

    # surface cells
    tri = local[:, 3] < 0
    if not tri.any():
        etype, conn = QUAD_4, (local + 1).ravel()
    elif tri.all():
        etype, conn = TRI_3, (local[:, :3] + 1).ravel()
    else:
        etype = MIXED
        parts, offsets, pos = [], [0], 0
        for f in local:
            n = 3 if f[3] < 0 else 4
            parts.append(np.concatenate([[TRI_3 if n == 3 else QUAD_4], f[:n] + 1]))
            pos += n + 1
            offsets.append(pos)
        conn = np.concatenate(parts)
    sec = zone2.add(AdfNode(label, "Elements_t", np.array([etype, 0], dtype=elem_dtype)))
    sec.add(AdfNode("ElementRange", "IndexRange_t", index_range(1, ncells)))
    sec.add(AdfNode("ElementConnectivity", "DataArray_t", conn.astype(conn_dtype)))
    if etype == MIXED and lib_version >= 4.0:
        sec.add(AdfNode("ElementStartOffset", "DataArray_t", np.asarray(offsets, dtype=conn_dtype)))

    # boundary edges: one BAR_2 section + BC per neighbouring boundary condition
    zbc2 = AdfNode("ZoneBC", "ZoneBC_t")
    next_id = ncells + 1
    edge_summary = []
    for i in sorted(groups, key=lambda i: (i < 0, i)):
        rows = np.asarray(groups[i])
        e_local = new_index[e_flat[rows]] + 1                   # (k, 2), 1-based, domain on the left
        name = bcs[i].name if i >= 0 else "Unassigned"
        lo, hi = next_id, next_id + len(rows) - 1
        next_id = hi + 1
        es = zone2.add(AdfNode(name, "Elements_t", np.array([BAR_2, 0], dtype=elem_dtype)))
        es.add(AdfNode("ElementRange", "IndexRange_t", index_range(lo, hi)))
        es.add(AdfNode("ElementConnectivity", "DataArray_t", e_local.ravel().astype(conn_dtype)))
        bc2 = zbc2.add(AdfNode(name, "BC_t", bcs[i].text if i >= 0 else "BCTypeNull"))
        pr_shape = range_shape
        if i >= 0 and (bcs[i].get("PointRange") is not None):
            pr_shape = np.asarray(bcs[i].get("PointRange").data).shape
        bc2.add(AdfNode("PointRange", "IndexRange_t", index_range(lo, hi, (1, 2) if len(pr_shape) != 2 else pr_shape)))
        bc2.add(AdfNode("GridLocation", "GridLocation_t", "EdgeCenter"))
        if i >= 0:
            for c in bcs[i].children:
                if c.label in _BC_CHILDREN_COPIED:
                    bc2.add(c.copy())
        edge_summary.append((name, len(rows)))
    zone2.add(zbc2)

    # vertex-based flow solutions can be carried over, cell-based ones cannot
    for fs in zone.find_all("FlowSolution_t"):
        loc_node = fs.get(label="GridLocation_t")
        if loc_node is not None and loc_node.text != "Vertex":
            log.warning("%s: FlowSolution %r (%s) is not carried over", what, fs.name, loc_node.text)
            continue
        fs2 = zone2.add(AdfNode(fs.name, "FlowSolution_t"))
        for c in fs.children:
            if c.label == "DataArray_t" and c.data is not None and np.asarray(c.data).size == nverts:
                fs2.add(AdfNode(c.name, "DataArray_t", np.asarray(c.data).ravel(order="F")[used],
                                children=[g.copy() for g in c.children]))
            elif c.label != "DataArray_t":
                fs2.add(c.copy())

    n_quad, n_tri = int((~tri).sum()), int(tri.sum())
    info = {
        "zone": zone.name, "surface": label, "matched": matched, "nodes": len(used), "cells": ncells,
        "quads": n_quad, "tris": n_tri, "plane": plane["description"], "deviation": plane["deviation"],
        "extent": plane["extent"], "reversed_winding": reversed_winding, "edges": edge_summary,
    }
    text = "\n".join([
        f"2-D surface extraction by cgns_adf_3Dto2D from {source_note}",
        f"zone: {zone.name}",
        f"surface: {label} (boundary conditions: {', '.join(matched)})",
        f"faces: {ncells} ({n_quad} QUAD_4, {n_tri} TRI_3), nodes: {len(used)}",
        f"mapping: {plane['description']}",
        f"max deviation from the fitted surface: {plane['deviation']:.3g} (extent {plane['extent']:.6g})",
        f"element winding reversed for counter-clockwise orientation: {'yes' if reversed_winding else 'no'}",
        "boundary edges: " + ", ".join(f"{n}: {k}" for n, k in edge_summary),
    ])
    zone2.add(AdfNode("SurfaceExtraction", "Descriptor_t", text))
    return zone2, info


# =============================================================================
#  public API
# =============================================================================
def default_output_filename(input_filename: str) -> str:
    """``mesh.cgns`` -> ``mesh_2D.cgns`` (``_2D`` is inserted before the extension)."""
    stem, ext = os.path.splitext(input_filename)
    return f"{stem}_2D{ext}"


def _library_version(root: AdfNode) -> float:
    node = root.get(label="CGNSLibraryVersion_t")
    try:
        return float(np.asarray(node.data).ravel()[0]) if node is not None else 4.0
    except (TypeError, ValueError, IndexError):
        return 4.0


def list_surfaces(input_filename: str) -> dict[str, dict[str, list[str]]]:
    """Return ``{base: {zone: [BC names...]}}`` and the element sections of an ADF/CGNS file."""
    root = read_adf(input_filename)
    out: dict[str, dict[str, list[str]]] = {}
    for base in root.find_all("CGNSBase_t"):
        out[base.name] = {}
        for zone in base.find_all("Zone_t"):
            zbc = zone.get(label="ZoneBC_t")
            out[base.name][zone.name] = [bc.name for bc in (zbc.find_all("BC_t") if zbc else [])]
    return out


def extract_planar_surface_2d(input_filename: str, output_filename: str | None = None,
                              surface_name: str = DEFAULT_SURFACE_NAME, planar_tol: float = 1e-6,
                              mapping: str = "plane") -> str:
    """Extract a planar boundary surface of a 3-D CGNS/ADF file into a 2-D CGNS/ADF file.

    Parameters
    ----------
    input_filename
        3-D CGNS file in the ADF dialect (e.g. exported by ANSYS TurboGrid).
    output_filename
        Output file; defaults to the input name with ``_2D`` inserted before
        the extension.
    surface_name
        Boundary condition(s) to extract.  Exact ``BC_t`` name, or a token
        of it: the default ``"HUB"`` selects ``"INBlock HUB"``,
        ``"Passage HUB"`` and ``"OUTBlock HUB"`` in a TurboGrid file.
    planar_tol
        Relative tolerance (fraction of the surface extent) above which a
        non-planar surface triggers a warning.
    mapping
        ``"plane"`` (default): project onto the fitted plane (an axis-aligned
        plane keeps the two in-plane coordinates unchanged).
        ``"cylinder"``: unwrap a surface of constant radius about a
        coordinate axis into (axial coordinate, radius * angle).

    Returns
    -------
    str
        The output filename.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown mapping {mapping!r}; choose one of {MAPPINGS}")
    if output_filename is None:
        output_filename = default_output_filename(input_filename)
    if os.path.abspath(output_filename) == os.path.abspath(input_filename):
        raise ValueError("output_filename must differ from input_filename")

    root = read_adf(input_filename)
    lib_version = _library_version(root)
    bases = root.find_all("CGNSBase_t")
    if not bases:
        raise AdfError(f"{input_filename}: no CGNSBase_t node found")

    root2 = new_root()
    lib = root.get(label="CGNSLibraryVersion_t")
    root2.add(lib.copy() if lib is not None else AdfNode("CGNSLibraryVersion", "CGNSLibraryVersion_t",
                                                          np.array([4.0], np.float32)))
    n_zones = 0
    source_note = os.path.basename(input_filename)
    for base in bases:
        base2 = AdfNode(base.name, "CGNSBase_t", np.array([2, 2], dtype=np.asarray(base.data).dtype))
        for c in base.children:
            if c.label != "Zone_t":
                base2.add(c.copy())
        for zone in base.find_all("Zone_t"):
            result = _extract_zone(zone, surface_name, planar_tol, mapping, lib_version, source_note)
            if result is None:
                log.info("zone %r: no boundary condition matches %r, skipped", zone.name, surface_name)
                continue
            zone2, info = result
            base2.add(zone2)
            n_zones += 1
            log.info("zone %r: surface %r -> %d nodes, %d cells (%d quads, %d tris); %s; "
                     "max deviation %.3g; winding reversed: %s; edges: %s",
                     info["zone"], info["surface"], info["nodes"], info["cells"], info["quads"], info["tris"],
                     info["plane"], info["deviation"], info["reversed_winding"],
                     ", ".join(f"{n} ({k})" for n, k in info["edges"]))
        if base2.find_all("Zone_t"):
            root2.add(base2)
    if n_zones == 0:
        available = sorted({bc for zones in list_surfaces(input_filename).values() for bcs in zones.values() for bc in bcs})
        raise AdfError(f"no boundary condition matches {surface_name!r}; available: {available}")

    write_adf(output_filename, root2)
    log.info("wrote %s", output_filename)
    return output_filename


# =============================================================================
#  command line
# =============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cgns_adf_3Dto2D",
        description="Extract a planar boundary surface (default: the hub) from a 3-D CGNS/ADF file "
                    "written by ANSYS TurboGrid into a 2-D CGNS/ADF file.")
    parser.add_argument("input_filename", help="3-D CGNS file (ADF dialect)")
    parser.add_argument("-o", "--output", dest="output_filename", default=None,
                        help="2-D output file (default: input name with _2D inserted before the extension)")
    parser.add_argument("-s", "--surface", dest="surface_name", default=DEFAULT_SURFACE_NAME,
                        help=f"boundary condition name or name token to extract (default: {DEFAULT_SURFACE_NAME})")
    parser.add_argument("--tol", dest="planar_tol", type=float, default=1e-6,
                        help="relative planarity tolerance used for the non-planar warning (default: 1e-6)")
    parser.add_argument("--mapping", choices=MAPPINGS, default="plane",
                        help="how to flatten the surface: 'plane' (default) projects onto the fitted plane, "
                             "'cylinder' unwraps a constant-radius surface into (axial, radius*angle)")
    parser.add_argument("--list", action="store_true", help="list bases, zones and boundary conditions and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.list:
            for base, zones in list_surfaces(args.input_filename).items():
                print(f"Base {base!r}")
                for zone, names in zones.items():
                    print(f"  Zone {zone!r}")
                    for n in names:
                        print(f"    {n}")
            return 0
        extract_planar_surface_2d(args.input_filename, args.output_filename, args.surface_name, args.planar_tol,
                                  args.mapping)
        return 0
    except (AdfError, NotImplementedError, OSError, ValueError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
