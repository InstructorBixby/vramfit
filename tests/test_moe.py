from vramfit import moe
from vramfit.shape import Shape, Tensor

GIB = 1024 ** 3


def fake_moe_shape(n_layers=48, n_experts=128):
    tensors = [Tensor("token_embd.weight", [2048, 151936], "q4_k")]
    for i in range(n_layers):
        p = f"blk.{i}."
        tensors += [
            Tensor(p + "attn_q.weight", [2048, 4096], "q4_k"),
            Tensor(p + "attn_k.weight", [2048, 512], "q4_k"),
            Tensor(p + "attn_v.weight", [2048, 512], "q4_k"),
            Tensor(p + "attn_output.weight", [4096, 2048], "q4_k"),
            Tensor(p + "ffn_gate_exps.weight",
                   [2048, 768, n_experts], "q4_k"),
            Tensor(p + "ffn_up_exps.weight",
                   [2048, 768, n_experts], "q4_k"),
            Tensor(p + "ffn_down_exps.weight",
                   [768, 2048, n_experts], "q4_k"),
        ]
    return Shape(name="moe-test", arch="qwen3moe", n_layers=n_layers,
                 n_embd=2048, n_head=32, n_head_kv=4, head_dim=128,
                 vocab=151936, train_context=32768, tensors=tensors)


def test_dense_model_is_not_an_moe(llama8b):
    assert moe.from_shape(llama8b) is None
    assert moe.from_shape(llama8b, n_experts=1, n_used=1) is None


def test_residency_is_every_expert():
    sh = fake_moe_shape()
    m = moe.from_shape(sh, n_experts=128, n_used=8)
    assert m.resident_bytes == sh.weight_bytes


def test_bytes_per_token_is_far_smaller_than_residency():
    m = moe.from_shape(fake_moe_shape(), n_experts=128, n_used=8)
    assert m.bytes_per_token < m.resident_bytes / 5
    assert m.speed_ratio > 5


def test_more_experts_used_narrows_the_gap():
    few = moe.from_shape(fake_moe_shape(), n_experts=128, n_used=8)
    many = moe.from_shape(fake_moe_shape(), n_experts=128, n_used=64)
    assert many.bytes_per_token > few.bytes_per_token
    assert many.speed_ratio < few.speed_ratio


def test_shared_expert_is_always_read():
    sh = fake_moe_shape(n_layers=2)
    sh.tensors.append(
        Tensor("blk.0.ffn_gate_shexp.weight", [2048, 4096], "q4_k"))
    m = moe.from_shape(sh, n_experts=128, n_used=8)
    assert m.shared_bytes > 0
    assert m.bytes_per_token >= m.dense_bytes + m.shared_bytes


def test_summary_says_both_numbers():
    m = moe.from_shape(fake_moe_shape(), n_experts=128, n_used=8)
    out = m.summary()
    assert "resident" in out and "per token" in out
    assert "fewer bytes" in out
