from vramfit import speculative
from vramfit.shape import Shape, Tensor

MIB = 1024 ** 2


def tiny(name, layers, embd):
    return Shape(name=name, arch="llama", n_layers=layers,
                 n_embd=embd, n_head=32, n_head_kv=8, head_dim=128,
                 vocab=32000, train_context=8192,
                 tensors=[Tensor("blk.0.attn_q.weight",
                                 [embd, embd * layers], "q4_k")])


def test_both_models_are_budgeted():
    spec = speculative.Speculative(tiny("target", 32, 4096),
                                   tiny("draft", 22, 2048))
    t, d = spec.budget(8192, cuda_context=304 * MIB)
    assert t.total > d.total > 0


def test_combined_total_is_printed():
    spec = speculative.Speculative(tiny("target", 32, 4096),
                                   tiny("draft", 22, 2048))
    out = spec.table(8192, cuda_context=304 * MIB)
    assert "combined" in out


def test_no_bandwidth_fraction_is_claimed():
    spec = speculative.Speculative(tiny("target", 32, 4096),
                                   tiny("draft", 22, 2048))
    out = spec.table(8192, cuda_context=304 * MIB)
    assert "does not apply with speculation" in out
    assert "%" not in out.split("! the decode")[1]


def test_draft_overhead_fraction():
    spec = speculative.Speculative(tiny("target", 32, 4096),
                                   tiny("draft", 4, 1024))
    assert 0 < spec.draft_overhead_frac < 0.2


def test_draft_context_can_be_smaller():
    spec = speculative.Speculative(tiny("target", 32, 4096),
                                   tiny("draft", 22, 2048))
    _, small = spec.budget(32768, draft_context=4096,
                           cuda_context=304 * MIB)
    _, same = spec.budget(32768, cuda_context=304 * MIB)
    assert small.total < same.total
