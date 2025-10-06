from vramfit import budget

GIB = 1024 ** 3
MIB = 1024 ** 2


def test_full_offload_is_the_old_answer(llama8b):
    a = budget.build(llama8b, 8192, cuda_context=304 * MIB)
    b = budget.build(llama8b, 8192, cuda_context=304 * MIB,
                     n_gpu_layers=llama8b.n_layers)
    assert a.total == b.total
    assert b.offloaded_layers is None


def test_cache_scales_with_the_offloaded_layers(llama8b):
    full = budget.kv_bytes(llama8b, 8192)
    half = budget.kv_bytes(llama8b, 8192, layers=16)
    assert half * 2 == full


def test_half_the_layers_is_not_half_the_weights(llama8b):
    # The embedding and output tensors stay off the card, so the
    # resident weights are less than half, not exactly half.
    full = llama8b.weight_bytes
    half = budget.offloaded_weight_bytes(llama8b, 16)
    assert half < full / 2


def test_block_split_accounts_for_everything(llama8b):
    block, other = budget.block_weight_bytes(llama8b)
    assert block + other == llama8b.weight_bytes
    # token_embd at q4_k plus output at q6_k, 4096 x 128256 each:
    # 295 MB + 431 MB, so a little over 690 MiB. NOT "about a
    # gigabyte", which is what I had written here from memory.
    # The number matters because this term is resident whatever
    # --ngl says, so it is what decides whether a split fits.
    assert other > 0.6 * GIB
    assert other / llama8b.weight_bytes > 0.1


def test_zero_layers_puts_no_weights_on_the_card(llama8b):
    b = budget.build(llama8b, 8192, cuda_context=304 * MIB,
                     n_gpu_layers=0)
    assert b.weights == 0
    assert b.kv_cache == 0
    assert b.compute > 0


def test_partial_offload_says_it_is_approximate(llama8b):
    b = budget.build(llama8b, 8192, cuda_context=304 * MIB,
                     n_gpu_layers=20)
    assert b.offloaded_layers == 20
    assert any("build detail" in n for n in b.assumptions)
