"""Small NumPy-only CGNS/ADF container reader and writer.

Extracted from ParaBlade projects/mesh_examples/3D/cgns_adf_3Dto2D.py.
This module handles the container; callers construct the CGNS node tree.
It supports ADF files, not HDF5 files.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

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


