"""Read a safetensors header.

Format: a little-endian u64 giving the length of a JSON header, then the
JSON, then the tensor data. The JSON maps tensor name to `dtype`,
`shape` and `data_offsets`, with an optional `__metadata__` key.

Same deal as the GGUF reader: the header is all that is needed to size
the model, so a 15 GiB shard costs a few hundred kilobytes to inspect.
Sharded repos have an `index.json` listing which shard holds what, and
for sizing purposes the union of the shard headers is the model.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

# bytes per element
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}

MAX_HEADER = 100 * 1024 * 1024


class SafetensorsError(Exception):
    pass


def read_header(path) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{path}: too short for a header")
        (n,) = struct.unpack("<Q", raw)
        if not 0 < n <= MAX_HEADER:
            raise SafetensorsError(
                f"{path}: implausible header length {n}")
        blob = fh.read(n)
        if len(blob) != n:
            raise SafetensorsError(f"{path}: truncated header")
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"{path}: header is not JSON: {exc}") from None


def tensors(header: dict) -> list:
    """(name, dtype, shape, nbytes) per tensor, metadata excluded."""
    out = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES:
            raise SafetensorsError(f"{name}: unknown dtype {dtype}")
        shape = entry.get("shape") or []
        n = 1
        for d in shape:
            n *= d
        out.append((name, dtype, shape, n * DTYPE_BYTES[dtype]))
    return out


def shard_paths(directory) -> list:
    """Every safetensors file in a repo directory, index-aware."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        names = sorted(set(data.get("weight_map", {}).values()))
        return [directory / n for n in names]
    return sorted(directory.glob("*.safetensors"))
