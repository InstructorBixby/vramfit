import json

import pytest

from vramfit import load

from .conftest import FIXTURES

CFG = {
    "model_type": "qwen2",
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "max_position_embeddings": 32768,
}


def test_gguf_file():
    sh, how = load.load(
        FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")
    assert how == "gguf header"
    assert sh.n_head_kv == 8


def test_directory_of_gguf(tmp_path):
    src = FIXTURES / "tiny-mqa-f16.header.gguf"
    (tmp_path / "model.gguf").write_bytes(src.read_bytes())
    sh, how = load.load(tmp_path)
    assert how == "gguf header"
    assert sh.n_head_kv == 1


def test_config_only_says_so(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CFG))
    sh, how = load.load(tmp_path)
    assert "no weights" in how
    assert sh.params == 0


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(load.LoadError, match="no config.json"):
        load.load(tmp_path)


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(load.LoadError, match="no such file"):
        load.load(tmp_path / "ghost.gguf")
