import json
import struct

import pytest

from vramfit import safetensors


def write_st(path, header):
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 8)
    return path


HEADER = {
    "__metadata__": {"format": "pt"},
    "model.embed_tokens.weight": {"dtype": "BF16",
                                  "shape": [128256, 4096],
                                  "data_offsets": [0, 1050673152]},
    "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "BF16", "shape": [4096, 4096],
        "data_offsets": [1050673152, 1084227584]},
}


def test_reads_header_and_skips_metadata(tmp_path):
    p = write_st(tmp_path / "model.safetensors", HEADER)
    header = safetensors.read_header(p)
    assert header["__metadata__"]["format"] == "pt"
    names = [t[0] for t in safetensors.tensors(header)]
    assert "__metadata__" not in names
    assert len(names) == 2


def test_nbytes_is_elements_times_dtype(tmp_path):
    p = write_st(tmp_path / "model.safetensors", HEADER)
    by_name = {t[0]: t[3]
               for t in safetensors.tensors(
                   safetensors.read_header(p))}
    assert by_name["model.layers.0.self_attn.q_proj.weight"] == \
        4096 * 4096 * 2


def test_implausible_header_length_is_rejected(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", 1 << 60) + b"{}")
    with pytest.raises(safetensors.SafetensorsError,
                       match="implausible"):
        safetensors.read_header(p)


def test_shard_index_is_used_when_present(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert len(safetensors.shard_paths(tmp_path)) == 2
