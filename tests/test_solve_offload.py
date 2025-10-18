from vramfit import budget, solve

GIB = 1024 ** 3
MIB = 1024 ** 2
CTX = {"cuda_context": 304 * MIB}


def test_model_that_fits_gets_every_layer(llama8b):
    assert solve.max_gpu_layers(llama8b, 24 * GIB, 8192, **CTX) == \
        llama8b.n_layers


def test_tight_card_gets_some_layers(llama8b):
    n = solve.max_gpu_layers(llama8b, 4 * GIB, 8192, **CTX)
    assert 0 < n < llama8b.n_layers


def test_the_answer_is_the_largest_that_fits(llama8b):
    vram = 4 * GIB
    n = solve.max_gpu_layers(llama8b, vram, 8192, **CTX)
    assert budget.build(llama8b, 8192, n_gpu_layers=n,
                        **CTX).total <= vram
    assert budget.build(llama8b, 8192, n_gpu_layers=n + 1,
                        **CTX).total > vram


def test_no_room_at_all_is_zero(llama8b):
    assert solve.max_gpu_layers(llama8b, 128 * 1024 ** 2, 8192,
                                **CTX) == 0


def test_more_context_costs_layers(llama8b):
    short = solve.max_gpu_layers(llama8b, 6 * GIB, 4096, **CTX)
    long = solve.max_gpu_layers(llama8b, 6 * GIB, 65536, **CTX)
    assert long <= short
