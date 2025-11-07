import pytest

from vramfit import budget, cache


def test_f16_is_two_bytes():
    assert cache.element_bytes("f16") == 2.0


def test_q8_0_is_not_one_byte():
    # 34 bytes per 32 elements: 8.5 bits, same as the weights.
    assert cache.element_bytes("q8_0") == 34 / 32


def test_q4_0_is_four_and_a_half_bits():
    assert cache.lookup("q4_0").bits_per_element == 4.5


def test_quantised_v_cache_declares_the_flash_attention_dependency():
    assert cache.lookup("q8_0").needs_flash_attn
    assert not cache.lookup("f16").needs_flash_attn


def test_unknown_type_lists_what_there_is():
    with pytest.raises(KeyError, match="unknown cache type"):
        cache.element_bytes("q3_k_m")


def test_cache_cost_at_32k(llama8b):
    gib = 1024 ** 3
    f16 = budget.kv_bytes(llama8b, 32768, cache.element_bytes("f16"))
    q8 = budget.kv_bytes(llama8b, 32768, cache.element_bytes("q8_0"))
    assert abs(f16 / gib - 4.0) < 0.01
    assert 2.0 < q8 / gib < 2.2
