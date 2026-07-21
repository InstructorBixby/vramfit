"""The shape of a model, as far as sizing it is concerned.

Deliberately small. Every number downstream -- weights, cache, buffers,
the largest context that fits -- is a function of this and nothing else,
so a reader for a new metadata format only has to produce one of these.

K and V are sized separately. On most architectures they are the same
and equal to `head_dim`, but GGUF has `attention.key_length` and
`attention.value_length` precisely because they need not be, and a
report of a 40% cache over-prediction turned out to be an architecture
where `rope.dimension_count` is smaller than the key length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vramfit import ggml_types


@dataclass
class Tensor:
    name: str
    dims: list
    type_name: str

    @property
    def elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return ggml_types.tensor_bytes(self.elements, self.type_name)


@dataclass
class Shape:
    name: str
    arch: str
    n_layers: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    vocab: int
    train_context: int
    n_ff: int = 0
    # Per-head K and V widths. Default to head_dim, which is right for
    # every architecture that does not say otherwise.
    key_dim: int = 0
    value_dim: int = 0
    # 0 for a dense model.
    n_experts: int = 0
    n_experts_used: int = 0
    tensors: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.key_dim = self.key_dim or self.head_dim
        self.value_dim = self.value_dim or self.head_dim

    @property
    def params(self) -> int:
        return sum(t.elements for t in self.tensors)

    @property
    def weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def bpw(self) -> float:
        return self.weight_bytes * 8 / self.params

    @property
    def gqa_ratio(self) -> float:
        return self.n_head / self.n_head_kv

    @property
    def kv_dim_total(self) -> int:
        """K plus V width per kv head. The 2 in the old formula."""
        return self.key_dim + self.value_dim

    @property
    def asymmetric_kv(self) -> bool:
        return self.key_dim != self.value_dim

    @property
    def moe(self):
        """MoE accounting, or None for a dense model."""
        from vramfit import moe as moe_mod

        return moe_mod.from_shape(self, self.n_experts,
                                  self.n_experts_used)

    def params_from_config(self) -> int:
        """Parameter count from the dimensions, with no tensor table.

        For a shape that came from a published-architecture row rather
        than from a file. Accurate to within the bias and norm terms,
        which are a fraction of a percent of the total and are left out
        rather than guessed at.
        """
        if self.params:
            return self.params
        n_ff = self.n_ff or int(self.n_embd * 3.5)
        per_layer = (
            self.n_embd * self.n_head * self.head_dim
            + self.n_embd * self.n_head_kv * self.key_dim
            + self.n_embd * self.n_head_kv * self.value_dim
            + self.n_head * self.head_dim * self.n_embd
            + 3 * self.n_embd * n_ff * max(1, self.n_experts)
        )
        return (2 * self.n_embd * self.vocab
                + self.n_layers * per_layer)

    def quant_mix(self) -> dict:
        """How many bytes each ggml type accounts for.

        Reading this is how you find out that your Q4_K_M file is 15%
        Q6_K by volume.
        """
        out: dict = {}
        for t in self.tensors:
            out[t.type_name] = out.get(t.type_name, 0) + t.nbytes
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def from_gguf(meta: dict, tensors: list, path: str = "") -> Shape:
    arch = meta.get("general.architecture", "unknown")

    def m(suffix, default=None):
        return meta.get(f"{arch}.{suffix}", default)

    n_embd = m("embedding_length")
    n_head = m("attention.head_count")
    head_dim = m("rope.dimension_count") or (n_embd // n_head)
    vocab = m("vocab_size")
    if vocab is None:
        toks = meta.get("tokenizer.ggml.tokens")
        vocab = len(toks) if toks is not None else 0
    return Shape(
        name=meta.get("general.name") or path,
        arch=arch,
        n_layers=m("block_count"),
        n_embd=n_embd,
        n_head=n_head,
        # head_count_kv is absent on pre-GQA models, where it is equal
        # to head_count.
        n_head_kv=m("attention.head_count_kv", n_head),
        head_dim=head_dim,
        vocab=vocab,
        train_context=m("context_length", 0),
        n_ff=m("feed_forward_length", 0) or 0,
        # Explicit when the architecture sets them, head_dim otherwise.
        key_dim=m("attention.key_length", 0) or 0,
        value_dim=m("attention.value_length", 0) or 0,
        n_experts=m("expert_count", 0) or 0,
        n_experts_used=m("expert_used_count", 0) or 0,
        tensors=[Tensor(t["name"], t["dims"],
                        ggml_types.name(t["type"])) for t in tensors],
    )
