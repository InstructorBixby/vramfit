"""The VRAM budget, as five terms that are each separately wrong.

Keeping them separate is the point: when a prediction misses, the useful
question is which term missed, and a single number cannot answer it.

  weights        exact, from the tensor table
  kv cache       exact arithmetic, given the cache dtype
  compute buffer backend-specific, scales with the batch
  cuda context   a per-process constant, measured not guessed
  headroom       fragmentation slack, a policy and not a measurement
"""
from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass
class Budget:
    weights: int
    kv_cache: int
    compute: int
    cuda_context: int
    headroom: int

    @property
    def total(self) -> int:
        return (self.weights + self.kv_cache + self.compute
                + self.cuda_context + self.headroom)

    def rows(self):
        return [
            ("weights", self.weights),
            ("kv cache", self.kv_cache),
            ("compute buffer", self.compute),
            ("cuda context", self.cuda_context),
            ("headroom", self.headroom),
        ]

    def table(self) -> str:
        out = [f"{label:16} {n / GIB:8.3f} GiB"
               for label, n in self.rows()]
        out.append(f"{'total':16} {self.total / GIB:8.3f} GiB")
        return "\n".join(out)


def kv_bytes(shape, context: int, dtype_bytes: int = 2) -> int:
    """K and V, per layer, for `context` positions.

    n_head_kv, not n_head. See NOTES.md.
    """
    return (2 * shape.n_layers * shape.n_head_kv * shape.head_dim
            * context * dtype_bytes)


def compute_bytes(shape, context: int, batch: int = 512) -> int:
    """Scratch for one forward pass, and the roughest term in here.

    llama.cpp sizes its compute buffer from the largest intermediate it
    will materialise, which for a logits-producing pass is the output
    projection: batch x vocab in f32. The hidden-state terms are
    smaller but not negligible.
    """
    logits = batch * shape.vocab * 4
    hidden = batch * shape.n_embd * 4 * 6
    return logits + hidden


def build(shape, context: int, *, cache_dtype_bytes: int = 2,
          batch: int = 512, cuda_context: int = 350 * MIB,
          headroom_frac: float = 0.03) -> Budget:
    weights = shape.weight_bytes
    kv = kv_bytes(shape, context, cache_dtype_bytes)
    compute = compute_bytes(shape, context, batch)
    base = weights + kv + compute + cuda_context
    return Budget(weights=weights, kv_cache=kv, compute=compute,
                  cuda_context=cuda_context,
                  headroom=int(base * headroom_frac))
