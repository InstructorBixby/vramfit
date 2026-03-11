from vramfit import api

from .conftest import FIXTURES

GIB = 1024 ** 3
MODEL = FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf"


def test_load_shape_returns_a_shape():
    sh = api.load_shape(MODEL)
    assert sh.n_head_kv == 8


def test_fit_returns_separable_terms():
    b = api.fit(MODEL, 8192, cuda_context=304 * 1024 ** 2)
    assert b.total == sum(n for _, n in b.rows())
    assert b.kv_cache > 0


def test_fit_accepts_a_hypothetical_quant():
    heavy = api.fit(MODEL, 8192, quant="q8_0")
    light = api.fit(MODEL, 8192, quant="q4_k_m")
    assert heavy.weights > light.weights


def test_max_context_and_quant_for_context():
    assert api.max_context(MODEL, 16 * GIB) > 0
    assert api.quant_for_context(MODEL, 32768, 16 * GIB) is not None


def test_bits_per_weight_is_not_the_name():
    assert api.bits_per_weight("q4_k_m") > 4.7
    assert api.bits_per_weight("nonsense") is None


def test_ceilings():
    dec = api.decode_ceiling(MODEL, 4096, 936.2e9)
    pre = api.prefill_ceiling(MODEL, 71.0e12)
    assert dec.basis == "bandwidth"
    assert pre.basis == "flops"


def test_everything_in_all_exists():
    for name in api.__all__:
        assert hasattr(api, name), name
