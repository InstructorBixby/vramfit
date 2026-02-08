from vramfit import explain, specs


def test_names_the_source_of_every_term(llama8b):
    out = explain.explain(llama8b, 8192, source="gguf header")
    for marker in ("[tensor table]", "block ratios", "policy",
                   "backend-specific", "gguf header"):
        assert marker in out


def test_shows_the_kv_head_arithmetic(llama8b):
    out = explain.explain(llama8b, 8192)
    assert "8 kv heads" in out
    assert "not the 32 attention heads" in out


def test_says_when_the_cuda_context_is_assumed(llama8b):
    assumed = explain.explain(llama8b, 8192)
    measured = explain.explain(llama8b, 8192,
                               cuda_context=304 * 1024 ** 2)
    assert "[assumed]" in assumed
    assert "measured on this machine" in measured


def test_ceilings_appear_only_with_a_spec(llama8b):
    without = explain.explain(llama8b, 8192)
    with_spec = explain.explain(llama8b, 8192,
                                spec=specs.lookup("RTX 3090"))
    assert "ceilings" not in without
    assert "decode" in with_spec
    assert "936.2 GB/s" in with_spec
