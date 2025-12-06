import pytest

from vramfit.bench import vllm


def fake_stream(gaps):
    """Chunks arriving at the given cumulative offsets."""
    def gen(url, payload, timeout=600.0):
        for t in gaps:
            yield t, {"choices": [{"text": "x"}]}
    return gen


def test_prefill_and_decode_are_separated(monkeypatch):
    monkeypatch.setattr(vllm, "post_stream",
                        fake_stream([1.0, 1.1, 1.2, 1.3]))
    monkeypatch.setattr(vllm.time, "perf_counter", lambda: 0.5)
    ttft, decode_s, n = vllm.timings("u", "m", "p", 4)
    assert ttft == pytest.approx(0.5)
    assert decode_s == pytest.approx(0.3)
    assert n == 4


def test_one_token_cannot_be_split(monkeypatch):
    monkeypatch.setattr(vllm, "post_stream", fake_stream([1.0]))
    monkeypatch.setattr(vllm.time, "perf_counter", lambda: 0.5)
    with pytest.raises(vllm.BenchError, match="fewer than two"):
        vllm.timings("u", "m", "p", 1)


def test_result_says_where_the_split_came_from(llama8b, monkeypatch):
    monkeypatch.setattr(vllm, "timings",
                        lambda *a, **k: (0.4, 1.0, 101))
    monkeypatch.setattr(vllm, "server_version", lambda base: "0.6.2")
    res = vllm.bench(llama8b, model="m", gpu="RTX 3090", context=4096)
    assert res.backend == "vllm"
    assert res.decode_tok_s == pytest.approx(100.0)
    assert res.prefill_tok_s == pytest.approx(512 / 0.4)
    assert any("time-to-first-token" in n for n in res.notes)
