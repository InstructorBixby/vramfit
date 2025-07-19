from vramfit import roofline

GIB = 1024 ** 3


def test_decode_bytes_include_weights_and_cache(llama8b):
    empty = roofline.decode_bytes_per_token(llama8b, 0)
    full = roofline.decode_bytes_per_token(llama8b, 32768)
    assert empty == llama8b.weight_bytes
    # 128 KiB per token of cache, 32k positions: 4 GiB.
    assert full - empty == 4 * GIB


def test_decode_ceiling_falls_as_context_grows(llama8b):
    bw = 936.2e9
    a = roofline.decode_ceiling(llama8b, 0, bw).tok_s
    b = roofline.decode_ceiling(llama8b, 32768, bw).tok_s
    assert b < a
    # Cache is comparable to the weights at this context, so the
    # ceiling is close to halved. Worth asserting because it is the
    # fact that surprises people.
    assert 0.45 < b / a < 0.6


def test_q8_cache_restores_some_of_it(llama8b):
    bw = 936.2e9
    full = roofline.decode_ceiling(llama8b, 32768, bw).tok_s
    half = roofline.decode_ceiling(llama8b, 32768, bw,
                                   cache_dtype_bytes=1).tok_s
    assert half > full


def test_prefill_ceiling_is_flops_over_two_params(llama8b):
    c = roofline.prefill_ceiling(llama8b, 71.0e12)
    assert c.basis == "flops"
    assert abs(c.tok_s - 71.0e12 / (2 * llama8b.params)) < 1e-6


def test_fraction_and_achieved_agree(llama8b):
    bw = 936.2e9
    c = roofline.decode_ceiling(llama8b, 4096, bw)
    measured = c.tok_s * 0.8
    assert abs(roofline.fraction_of_peak(measured, c) - 0.8) < 1e-9
    achieved = roofline.achieved_bandwidth(measured, c.bytes_per_token)
    assert abs(achieved / bw - 0.8) < 1e-9


def test_verdict_thresholds():
    assert "bandwidth bound" in roofline.verdict(0.81)
    assert "headroom" in roofline.verdict(0.61)
    assert "table" in roofline.verdict(0.3)
    assert "wrong" in roofline.verdict(0.1)
