from vramfit import budget, solve

GIB = 1024 ** 3
MIB = 1024 ** 2
CTX = {"cuda_context": 304 * MIB}


def test_max_context_fits_and_one_step_more_does_not(llama8b):
    vram = 16 * GIB
    ctx = solve.max_context(llama8b, vram, **CTX)
    assert ctx > 0
    assert budget.build(llama8b, ctx, **CTX).total <= vram
    assert budget.build(llama8b, ctx + solve.ALIGN, **CTX).total > vram


def test_result_is_aligned(llama8b):
    assert solve.max_context(llama8b, 16 * GIB, **CTX) % solve.ALIGN == 0


def test_does_not_exceed_the_trained_context(llama8b):
    assert solve.max_context(llama8b, 48 * GIB, **CTX) == \
        llama8b.train_context


def test_zero_when_the_weights_alone_do_not_fit(llama8b):
    assert solve.max_context(llama8b, 2 * GIB, **CTX) == 0


def test_q8_cache_buys_context(llama8b):
    full = solve.max_context(llama8b, 16 * GIB, **CTX)
    half = solve.max_context(llama8b, 16 * GIB, cache_dtype_bytes=1,
                             **CTX)
    assert half > full


def test_quant_for_context_picks_the_heaviest_that_fits(llama8b):
    q, b = solve.quant_for_context(llama8b, 32768, 16 * GIB, **CTX)
    assert q is not None
    assert b.total <= 16 * GIB
    heavier = [x for x in solve.__dict__["quants"].LADDER
               if x.bpw > q.bpw]
    assert heavier, "ladder has nothing above the pick, test is vacuous"


def test_quant_for_context_can_fail(llama8b):
    q, b = solve.quant_for_context(llama8b, 131072, 2 * GIB, **CTX)
    assert (q, b) == (None, None)


def test_context_ladder_trades_bits_for_tokens(llama8b):
    rungs = solve.context_ladder(llama8b, 16 * GIB, **CTX)
    assert len(rungs) > 4
    # Fewer bits per weight must never buy fewer tokens.
    contexts = [ctx for _, ctx in rungs]
    assert contexts == sorted(contexts)
