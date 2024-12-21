"""Minimal GGUF reader: enough of the header to size a model.

Layout, from the spec: magic `GGUF`, a u32 version, a u64 tensor count,
a u64 metadata-KV count, then that many key/value pairs, then one
tensor-info record per tensor, then alignment padding, then the data.

Only the header is read. A 5 GiB file is sized from its first few
hundred kilobytes, which is also what makes this usable on a half
downloaded file.
"""
import struct

MAGIC = b"GGUF"

# metadata value types
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64,
 F64) = range(13)

FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<B", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}


class GGUFError(Exception):
    pass


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise GGUFError(f"short read: wanted {n}, got {len(b)}")
        return b

    def fixed(self, vtype):
        fmt, size = FIXED[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self):
        n = self.fixed(U64)
        return self.raw(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in FIXED:
            return self.fixed(vtype)
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = self.fixed(U32)
            count = self.fixed(U64)
            return [self.value(etype) for _ in range(count)]
        raise GGUFError(f"unknown value type {vtype}")


def read_header(path):
    """Return (version, metadata dict, tensor list)."""
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        version = r.fixed(U32)
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors = r.fixed(U64)
        n_kv = r.fixed(U64)

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.fixed(U32))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.fixed(U32)
            dims = [r.fixed(U64) for _ in range(n_dims)]
            gtype = r.fixed(U32)
            offset = r.fixed(U64)
            tensors.append({"name": name, "dims": dims,
                            "type": gtype, "offset": offset})
    return version, meta, tensors
