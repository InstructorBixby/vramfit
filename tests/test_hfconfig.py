import json

from vramfit import hfconfig

GQA = {
    "model_type": "llama",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}

PRE_GQA = {
    "model_type": "llama",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
}


def test_gqa_heads_are_read():
    sh = hfconfig.from_config(GQA)
    assert sh.n_head_kv == 8
    assert sh.head_dim == 128


def test_missing_kv_heads_defaults_to_attention_heads():
    # Not to 1. Defaulting the other way under-predicts the cache by
    # the head count.
    assert hfconfig.from_config(PRE_GQA).n_head_kv == 32


def test_explicit_head_dim_wins_over_the_ratio():
    cfg = dict(GQA, head_dim=96)
    assert hfconfig.from_config(cfg).head_dim == 96


def test_rope_scaling_reports_both_numbers():
    cfg = dict(GQA, rope_scaling={
        "factor": 8.0, "original_max_position_embeddings": 8192})
    assert hfconfig.rope_scaling(cfg) == (8192, 131072)


def test_no_rope_scaling_is_none():
    assert hfconfig.rope_scaling(GQA) is None


def test_directory_without_weights_has_no_params(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(GQA))
    sh = hfconfig.from_directory(tmp_path)
    assert sh.n_layers == 32
    assert sh.tensors == []
