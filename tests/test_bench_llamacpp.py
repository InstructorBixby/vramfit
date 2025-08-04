import json

import pytest

from vramfit.bench import llamacpp

# Trimmed from a real `llama-bench -o json` run: two rows, one pp and
# one tg, with the fields this parser reads.
ROWS = [
    {"model_filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
     "build_commit": "a1b2c3d", "n_prompt": 512, "n_gen": 0,
     "n_gpu_layers": 99, "avg_ts": 2914.33, "stddev_ts": 18.2},
    {"model_filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
     "build_commit": "a1b2c3d", "n_prompt": 0, "n_gen": 128,
     "n_gpu_layers": 99, "avg_ts": 118.47, "stddev_ts": 0.41},
]


def test_parse_accepts_a_bare_list():
    assert len(llamacpp.parse(json.dumps(ROWS))) == 2


def test_parse_accepts_a_wrapped_object():
    assert len(llamacpp.parse(json.dumps({"results": ROWS}))) == 2


def test_split_rows_separates_the_two_measurements():
    prefill, decode = llamacpp.split_rows(ROWS)
    assert prefill == 2914.33
    assert decode == 118.47


def test_missing_one_half_is_an_error():
    with pytest.raises(llamacpp.BenchError, match="both a pp and a tg"):
        llamacpp.split_rows(ROWS[:1])


def test_garbage_output_is_a_clean_error():
    with pytest.raises(llamacpp.BenchError, match="could not parse"):
        llamacpp.parse("llama_model_load: error loading model")


def test_result_carries_ratios(llama8b, monkeypatch):
    monkeypatch.setattr(llamacpp, "find_binary", lambda b="": "stub")
    monkeypatch.setattr(llamacpp, "run_llama_bench",
                        lambda *a, **k: ROWS)
    res = llamacpp.bench(llama8b, model="m.gguf", gpu="RTX 3090",
                         context=4096, quant="q4_k_m")
    assert res.backend == "llama.cpp"
    assert res.backend_version == "a1b2c3d"
    assert 0 < res.decode_frac_peak < 1
    assert res.achieved_bandwidth_gbs > 0


def test_unknown_gpu_leaves_ratios_empty_and_says_why(llama8b,
                                                     monkeypatch):
    monkeypatch.setattr(llamacpp, "find_binary", lambda b="": "stub")
    monkeypatch.setattr(llamacpp, "run_llama_bench",
                        lambda *a, **k: ROWS)
    res = llamacpp.bench(llama8b, model="m.gguf",
                         gpu="RTX 9999", context=4096)
    assert res.decode_frac_peak == 0
    assert any("no spec row" in n for n in res.notes)
