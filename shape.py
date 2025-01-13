"""The shape of a model, as far as sizing it is concerned.

Deliberately small. Every number downstream -- weights, cache, buffers,
the largest context that fits -- is a function of this and nothing else,
so a reader for a new metadata format only has to produce one of these.
"""
from dataclasses import dataclass, field

import ggml_types


@dataclass
class Tensor:
    name: str
    dims: list
    type_name: str

    @property
    def elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self):
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
    tensors: list = field(default_factory=list)

    @property
    def params(self):
        return sum(t.elements for t in self.tensors)

    @property
    def weight_bytes(self):
        return sum(t.nbytes for t in self.tensors)

    @property
    def bpw(self):
        return self.weight_bytes * 8 / self.params

    @property
    def gqa_ratio(self):
        return self.n_head / self.n_head_kv


def from_gguf(meta, tensors, path=""):
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
        # head_count_kv is absent on pre-GQA models, where it equals
        # head_count.
        n_head_kv=m("attention.head_count_kv", n_head),
        head_dim=head_dim,
        vocab=vocab,
        train_context=m("context_length", 0),
        tensors=[Tensor(t["name"], t["dims"],
                        ggml_types.name(t["type"])) for t in tensors],
    )
