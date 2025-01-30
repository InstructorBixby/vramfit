import pytest

from vramfit import gguf

from .conftest import FIXTURES


def test_reads_metadata_and_tensors():
    version, meta, tensors = gguf.read_header(
        FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")
    assert version == 3
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert len(tensors) == 32 * 9 + 3


def test_tokenizer_arrays_are_counted_not_materialised():
    _, meta, _ = gguf.read_header(
        FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")
    toks = meta["tokenizer.ggml.tokens"]
    assert isinstance(toks, gguf.Skipped)
    assert len(toks) > 0


def test_rejects_a_non_gguf_file(tmp_path):
    p = tmp_path / "nope.gguf"
    p.write_bytes(b"ORFF" + b"\x00" * 64)
    with pytest.raises(gguf.GGUFError, match="not a GGUF"):
        gguf.read_header(p)


def test_truncated_file_is_a_clean_error(tmp_path):
    src = (FIXTURES / "tiny-mqa-f16.header.gguf").read_bytes()
    p = tmp_path / "half.gguf"
    p.write_bytes(src[:len(src) // 3])
    with pytest.raises(gguf.GGUFError, match="short read"):
        gguf.read_header(p)
