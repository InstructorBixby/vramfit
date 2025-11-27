from vramfit import vllm_plan

GIB = 1024 ** 3
MIB = 1024 ** 2


def test_usable_is_the_utilisation_fraction(llama8b):
    p = vllm_plan.plan(llama8b, 24 * GIB, cuda_context=304 * MIB)
    assert p.usable == int(24 * GIB * 0.9)


def test_everything_left_becomes_blocks(llama8b):
    p = vllm_plan.plan(llama8b, 24 * GIB, cuda_context=304 * MIB)
    assert p.blocks > 0
    assert p.cache == p.usable - p.weights - p.activations


def test_concurrency_falls_as_context_grows(llama8b):
    p = vllm_plan.plan(llama8b, 24 * GIB, cuda_context=304 * MIB)
    assert p.sequences_at(4096) > p.sequences_at(32768)


def test_short_sequences_still_cost_a_whole_block(llama8b):
    p = vllm_plan.plan(llama8b, 24 * GIB, cuda_context=304 * MIB)
    # 1 token and 16 tokens both cost one block.
    assert p.sequences_at(1) == p.sequences_at(16)


def test_a_card_too_small_says_it_will_accept_nothing(llama8b):
    p = vllm_plan.plan(llama8b, 5 * GIB, cuda_context=304 * MIB)
    assert p.cache == 0
    assert any("accept nothing" in n for n in p.notes)


def test_prefix_sharing_is_called_out_as_pessimism(llama8b):
    p = vllm_plan.plan(llama8b, 24 * GIB, cuda_context=304 * MIB)
    assert any("prefix caching" in n for n in p.notes)
