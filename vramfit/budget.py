"""The VRAM budget, as terms that are each separately wrong.

Keeping them separate is the point: when a prediction misses, the useful
question is which term missed, and a single number cannot answer it.

  weights        exact, from the tensor table
  kv cache       exact arithmetic, given the cache dtype
  compute buffer backend-specific, scales with the batch
  cuda context   measured per (device, driver), cached
  headroom       fragmentation slack, a policy and not a measurement

Partial offload is not a sixth term, it is a scale factor on the first
two -- and on both of them, which is the part that is easy to get wrong.
See docs/offload.md.
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

# Tensors that are not part of a transformer block. Under partial
# offload these are the ones whose placement is a build detail.
NON_BLOCK = ("token_embd", "output.weight", "output_norm")


@dataclass
class Budget:
    weights: int
    kv_cache: int
    compute: int
    cuda_context: int
    headroom: int
    assumptions: list = field(default_factory=list)
    offloaded_layers: int | None = None

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
        if self.offloaded_layers is not None:
            out.append(f"{'offloaded':16} "
                       f"{self.offloaded_layers} layers")
        for note in self.assumptions:
            out.append(f"  ! {note}")
        return "\n".join(out)


def kv_bytes(shape, context: int, dtype_bytes: float = 2,
             layers: int | None = None) -> int:
    """K and V for `context` positions, across `layers` layers.

    The head count here is `n_head_kv` and not `n_head`. Under GQA the
    K and V projections are shared across a group of query heads, so the
    cache holds `n_head_kv x head_dim` per position per tensor. Under
    MQA `n_head_kv` is 1.

    `dtype_bytes` is a float because a quantised cache is not a whole
    number of bytes per element: q8_0 is 34 bytes per 32, which is 8.5
    bits and not 8.

    `layers` defaults to all of them. It is a parameter because the
    cache for a layer lives wherever that layer lives, so a partially
    offloaded model has a partially offloaded cache.
    """
    n = shape.n_layers if layers is None else layers
    return int(2 * n * shape.n_head_kv * shape.head_dim
               * context * dtype_bytes)


def kv_bytes_per_token(shape, dtype_bytes: float = 2) -> int:
    """The slope of the cache against context."""
    return int(2 * shape.n_layers * shape.n_head_kv * shape.head_dim
               * dtype_bytes)


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


def is_block_tensor(name: str) -> bool:
    return not any(name.startswith(p) or name == p for p in NON_BLOCK)


def block_weight_bytes(shape) -> tuple:
    """(bytes in transformer blocks, bytes outside them)."""
    block = sum(t.nbytes for t in shape.tensors
                if is_block_tensor(t.name))
    return block, shape.weight_bytes - block


def offloaded_weight_bytes(shape, n_gpu_layers: int) -> int:
    """Weight bytes resident on the card at `-ngl n_gpu_layers`.

    The block tensors scale with the layer count. The embedding and
    output tensors do not move until everything does, which is an
    approximation of a build detail -- see docs/offload.md.
    """
    if n_gpu_layers >= shape.n_layers:
        return shape.weight_bytes
    if n_gpu_layers <= 0:
        return 0
    block, _ = block_weight_bytes(shape)
    return int(block * n_gpu_layers / shape.n_layers)


def build(shape, context: int, *, cache_dtype: str = "f16",
          cache_dtype_bytes: float | None = None,
          batch: int = 512, cuda_context: int | None = None,
          headroom_frac: float = 0.03,
          weights_override: int | None = None,
          n_gpu_layers: int | None = None,
          flash_attn: bool = False) -> Budget:
    """Build a budget.

    `weights_override` is for asking hypotheticals -- the same model at
    some other bits per weight -- without inventing a fake shape.
    `n_gpu_layers` is `-ngl`: None or >= n_layers means fully resident.
    `cache_dtype_bytes` overrides the named type, for callers that have
    a number rather than a name.
    """
    from vramfit import cache as cache_mod

    notes = []
    if cuda_context is None:
        cuda_context = ASSUMED_CUDA_CONTEXT
        notes.append("cuda context is assumed, not measured: run "
                     "`vramfit calibrate`")

    if cache_dtype_bytes is None:
        cache_dtype_bytes = cache_mod.element_bytes(cache_dtype)
        ctype = cache_mod.lookup(cache_dtype)
        if ctype is not None and ctype.needs_flash_attn \
                and not flash_attn:
            notes.append(
                f"{cache_dtype} cache needs flash attention in "
                f"llama.cpp; without it V stays f16 and the real cache "
                f"is larger than this")

    full = n_gpu_layers is None or n_gpu_layers >= shape.n_layers
    resident_layers = shape.n_layers if full else max(0, n_gpu_layers)

    if weights_override is not None:
        weights = weights_override
        if not full:
            weights = int(weights_override
                          * resident_layers / shape.n_layers)
        notes.append("weights are a hypothetical, not a file on disk")
    else:
        weights = (shape.weight_bytes if full
                   else offloaded_weight_bytes(shape, resident_layers))

    kv = kv_bytes(shape, context, cache_dtype_bytes, resident_layers)
    compute = compute_bytes(shape, context, batch)
    if not full:
        notes.append(
            f"partial offload: {resident_layers}/{shape.n_layers} "
            f"layers and the same fraction of the cache are on the "
            f"card; placement of the embedding and output tensors is a "
            f"build detail (docs/offload.md)")
    base = weights + kv + compute + cuda_context
    return Budget(weights=weights, kv_cache=kv, compute=compute,
                  cuda_context=cuda_context,
                  headroom=int(base * headroom_frac),
                  assumptions=notes,
                  offloaded_layers=None if full else resident_layers)
