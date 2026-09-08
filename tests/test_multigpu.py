from vramfit import devices, multigpu

GIB = 1024 ** 3
MIB = 1024 ** 2


def card(idx, name, gib, uuid):
    return devices.Device(idx, name, int(gib * GIB), 0,
                          driver="550.78", uuid=uuid)


PAIR = [card(0, "NVIDIA GeForce RTX 3090", 24, "GPU-aaaa1111"),
        card(1, "NVIDIA GeForce RTX 3060", 12, "GPU-bbbb2222")]


def test_layer_counts_sum_to_the_layer_count(llama8b):
    split = multigpu.by_memory(PAIR)
    counts = split.layer_counts(llama8b.n_layers)
    assert sum(counts) == llama8b.n_layers
    assert counts[0] > counts[1]


def test_remainder_goes_to_the_largest_share(llama8b):
    split = multigpu.Split(PAIR, [0.34, 0.66], primary=0)
    counts = split.layer_counts(31)
    assert sum(counts) == 31
    assert counts[1] > counts[0]


def test_primary_card_carries_the_embedding_and_output(llama8b):
    split = multigpu.Split(PAIR, [0.5, 0.5], primary=0)
    a, b = split.budgets(llama8b, 8192, cuda_context=304 * MIB)
    assert a.weights > b.weights


def test_every_card_pays_its_own_cuda_context(llama8b):
    split = multigpu.even(PAIR)
    for b in split.budgets(llama8b, 8192, cuda_context=304 * MIB):
        assert b.cuda_context == 304 * MIB


def test_bandwidth_is_the_slowest_card_not_the_sum():
    bw, why = multigpu.even(PAIR).decode_bandwidth()
    assert bw == 360.0 * 1e9      # the 3060's, not 936 + 360
    assert "slowest" in why and "not the sum" in why


def test_unknown_cards_are_called_out():
    pair = [card(0, "NVIDIA GeForce RTX 3090", 24, "GPU-a"),
            card(1, "NVIDIA GeForce RTX 9999", 16, "GPU-b")]
    _, why = multigpu.Split(pair, [0.5, 0.5]).decode_bandwidth()
    assert "may be optimistic" in why


def test_fits_reports_slack_per_card(llama8b):
    rows = multigpu.by_memory(PAIR).fits(llama8b, 8192,
                                         cuda_context=304 * MIB)
    assert len(rows) == 2
    assert all(slack > 0 for _, _, slack in rows)
