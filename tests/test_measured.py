from vramfit import measured

ENTRY = {
    "model": "Meta-Llama-3.1-8B-Instruct",
    "quant": "q4_k_m",
    "bits_per_weight": 4.83,
    "context": 8192,
    "backend": "llama.cpp",
    "backend_version": "b4341",
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "560.35.03",
    "cache_dtype": "f16",
    "decode_tok_s": 162.8,
    "nvidia_smi": "name, memory.total, driver_version\nx, 24564, 1.0",
}


def test_published_shapes_load():
    shapes = measured.known_shapes()
    assert "meta-llama-3.1-8b-instruct" in shapes
    assert shapes["qwen2.5-7b-instruct"].n_head_kv == 4


def test_param_count_from_config_is_close_to_the_published_one():
    sh = measured.known_shapes()["meta-llama-3.1-8b-instruct"]
    assert 7.7e9 < sh.params_from_config() < 8.3e9


def test_fraction_is_computed_for_a_known_card_and_model():
    row = measured.row_for(ENTRY)
    assert row.fraction is not None
    assert 0.2 < row.fraction < 1.2
    assert row.achieved_gbs > 100


def test_unknown_model_gets_no_fraction_and_says_why():
    row = measured.row_for(dict(ENTRY, model="SomeFinetune-9B"))
    assert row.fraction is None
    assert "shapes.csv" in row.why_no_fraction


def test_unknown_card_gets_no_fraction():
    row = measured.row_for(dict(ENTRY, gpu="NVIDIA GeForce RTX 9999"))
    assert row.fraction is None
    assert "spec row" in row.why_no_fraction


def test_markdown_footnotes_the_missing_ones():
    md = measured.markdown([ENTRY, dict(ENTRY, model="Unknown-3B")])
    assert "% of peak" in md
    assert "Missing fractions" in md
