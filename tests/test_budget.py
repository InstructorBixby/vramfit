from vramfit import budget

MIB = 1024 ** 2
GIB = 1024 ** 3


def test_kv_uses_kv_heads_not_attention_heads(llama8b):
    # 32 layers, 8 kv heads, head_dim 128, f16 -> 128 KiB per token.
    assert budget.kv_bytes_per_token(llama8b) == 2 * 32 * 8 * 128 * 2
    wrong = 2 * 32 * llama8b.n_head * 128 * 2
    assert budget.kv_bytes_per_token(llama8b) * 4 == wrong


def test_mqa_cache_is_tiny(mqa):
    assert budget.kv_bytes_per_token(mqa) == 2 * 24 * 1 * 64 * 2


def test_kv_is_linear_in_context(llama8b):
    a = budget.kv_bytes(llama8b, 4096)
    assert budget.kv_bytes(llama8b, 8192) == 2 * a


def test_q8_cache_halves_it(llama8b):
    full = budget.kv_bytes(llama8b, 8192, dtype_bytes=2)
    assert budget.kv_bytes(llama8b, 8192, dtype_bytes=1) == full // 2


def test_total_is_the_sum_of_the_terms(llama8b):
    b = budget.build(llama8b, 8192)
    assert b.total == sum(n for _, n in b.rows())


def test_unmeasured_context_is_flagged(llama8b):
    b = budget.build(llama8b, 8192)
    assert any("assumed" in note for note in b.assumptions)


def test_measured_context_is_not_flagged(llama8b):
    b = budget.build(llama8b, 8192, cuda_context=304 * MIB)
    assert b.assumptions == []
    assert b.cuda_context == 304 * MIB
