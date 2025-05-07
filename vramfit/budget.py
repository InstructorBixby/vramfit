"""The VRAM budget, as five terms that are each separately wrong.

Keeping them separate is the point: when a prediction misses, the useful
question is which term missed, and a single number cannot answer it.

  weights        exact, from the tensor table
  kv cache       exact arithmetic, given the cache dtype
  compute buffer backend-specific, scales with the batch
  cuda context   measured per (device, driver), cached
  headroom       fragmentation slack, a policy and not a measurement
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIB = 1024 ** 2
GIB = 1024 ** 3

# Fallback when nothing has been measured on this machine. High enough
# to be safe on the cards seen so far and explicitly flagged, because a
# prediction that admits it is guessing is useful and one that guesses
# silently is not.
ASSUMED_CUDA_CONTEXT = 350 * MIB


@dataclass
class Budget:
    weights: int
    kv_cache: int
    compute: int
    cuda_context: int
    headroom: int
    assumptions: list = field(default_factory=list)

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
        for note in self.assumptions:
            out.append(f"  ! {note}")
        return "\n".join(out)


def kv_bytes(shape, context: int, dtype_bytes: int = 2) -> int:
    """K and V for `context` positions, across every layer.

    The head count here is `n_head_kv` and not `n_head`. Under GQA the
    K and V projections are shared across a group of query heads, so the
    cache holds `n_head_kv x head_dim` per position per tensor. Under
    MQA `n_head_kv` is 1.

    Getting this wrong over-predicts the cache by the GQA ratio, which
    is 4x on an 8B of this family and 32x on a multi-query model.
    """
    return (2 * shape.n_layers * shape.n_head_kv * shape.head_dim
            * context * dtype_bytes)


def kv_bytes_per_token(shape, dtype_bytes: int = 2) -> int:
    """The slope of the cache against context."""
    return 2 * shape.n_layers * shape.n_head_kv * shape.head_dim * dtype_bytes


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
          batch: int = 512, cuda_context: int | None = None,
          headroom_frac: float = 0.03,
          weights_override: int | None = None) -> Budget:
    """Build a budget.

    `weights_override` is for asking hypotheticals -- "this model at
    some other bits per weight" -- without inventing a fake shape. The
    weight term is the only one that moves with the quant.
    """
    notes = []
    if cuda_context is None:
        cuda_context = ASSUMED_CUDA_CONTEXT
        notes.append("cuda context is assumed, not measured: run "
                     "`vramfit calibrate`")
    weights = (shape.weight_bytes if weights_override is None
               else weights_override)
    if weights_override is not None:
        notes.append("weights are a hypothetical, not a file on disk")
    kv = kv_bytes(shape, context, cache_dtype_bytes)
    compute = compute_bytes(shape, context, batch)
    base = weights + kv + compute + cuda_context
    return Budget(weights=weights, kv_cache=kv, compute=compute,
                  cuda_context=cuda_context,
                  headroom=int(base * headroom_frac),
                  assumptions=notes)
