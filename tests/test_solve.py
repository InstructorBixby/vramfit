from vramfit import budget, solve

GIB = 1024 ** 3
MIB = 1024 ** 2


def test_max_context_fits_and_one_step_more_does_not(llama8b):
    vram = 16 * GIB
    ctx = solve.max_context(llama8b, vram, cuda_context=304 * MIB)
    assert ctx > 0
    assert budget.build(llama8b, ctx,
                        cuda_context=304 * MIB).total <= vram
    assert budget.build(llama8b, ctx + solve.ALIGN,
                        cuda_context=304 * MIB).total > vram


def test_result_is_aligned(llama8b):
    ctx = solve.max_context(llama8b, 16 * GIB, cuda_context=304 * MIB)
    assert ctx % solve.ALIGN == 0


def test_does_not_exceed_the_trained_context(llama8b):
    # 131072 trained. A 48 GiB card could hold more cache than that.
    ctx = solve.max_context(llama8b, 48 * GIB, cuda_context=304 * MIB)
    assert ctx == llama8b.train_context


def test_zero_when_the_weights_alone_do_not_fit(llama8b):
    assert solve.max_context(llama8b, 2 * GIB,
                             cuda_context=304 * MIB) == 0


def test_q8_cache_buys_context(llama8b):
    full = solve.max_context(llama8b, 16 * GIB, cuda_context=304 * MIB)
    half = solve.max_context(llama8b, 16 * GIB, cuda_context=304 * MIB,
                             cache_dtype_bytes=1)
    assert half > full
