import json

import pytest

from vramfit.bench import exl2

LOG = """\
Loading model ...
Prompt processed: 512 tokens, 3104.88 tokens/s
Response generated: 128 tokens, 142.70 tokens/s
"""


def test_scrapes_the_log():
    out = exl2.parse(LOG)
    assert out["prefill"] == 3104.88
    assert out["decode"] == 142.70


def test_prefers_json_when_offered():
    out = exl2.parse(json.dumps({"prompt_tokens_s": 3000.0,
                                 "gen_tokens_s": 140.0,
                                 "version": "0.2.3"}))
    assert out["version"] == "0.2.3"


def test_one_sided_output_is_an_error():
    with pytest.raises(exl2.BenchError, match="both a prompt"):
        exl2.parse("Prompt processed: 512 tokens, 3104.88 tokens/s")


def test_unknown_cache_mode_is_rejected(llama8b):
    with pytest.raises(exl2.BenchError, match="cache mode"):
        exl2.bench(llama8b, model="m", gpu="RTX 3090",
                   cache_type="q8_0")


def test_cache_cost_uses_its_own_table(llama8b, monkeypatch):
    monkeypatch.setattr(exl2, "find_script", lambda b="": "stub")
    monkeypatch.setattr(exl2, "run", lambda *a, **k: {
        "prefill": 3000.0, "decode": 140.0, "version": "0.2.3"})
    a = exl2.bench(llama8b, model="m", gpu="RTX 3090", context=32768,
                   cache_type="fp16")
    b = exl2.bench(llama8b, model="m", gpu="RTX 3090", context=32768,
                   cache_type="q4")
    assert b.decode_frac_peak < a.decode_frac_peak
    assert any("not ggml block sizes" in n for n in b.notes)
