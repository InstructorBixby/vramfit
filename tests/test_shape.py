def test_param_count_matches_the_model_card(llama8b):
    # 8.03B is the published number for this one.
    assert 7.9e9 < llama8b.params < 8.1e9


def test_effective_bpw_is_not_the_name(llama8b):
    # Q4_K_M. If this ever returns 4.5 the tensor table is being
    # ignored and the mix has been replaced by a guess.
    assert 4.7 < llama8b.bpw < 4.95


def test_flat_quant_bpw_is_the_block_ratio(qwen7b):
    # q8_0 is 34 bytes per 32 weights, plus f32 norms.
    assert 8.5 <= qwen7b.bpw < 8.6


def test_quant_mix_shows_the_bumped_tensors(llama8b):
    mix = llama8b.quant_mix()
    assert set(mix) == {"q4_k", "q6_k", "f32"}
    assert mix["q6_k"] / sum(mix.values()) > 0.05


def test_gqa_ratio(llama8b, qwen7b, mqa):
    assert llama8b.gqa_ratio == 4
    assert qwen7b.gqa_ratio == 7
    assert mqa.gqa_ratio == 32
