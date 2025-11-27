"""What to ask about vLLM instead of "will it fit".

It always fits. The question is how much cache is left after the weights
and the profiled activations, how many blocks that is, and therefore how
many concurrent sequences of a given length the server will actually
accept before it starts queueing.

Everything here is derived from the same model shape as the llama.cpp
budget. The difference is entirely in what gets reported, which is the
argument for having profiles rather than one budget with flags.
"""
from __future__ import annotations

from dataclasses import dataclass

from vramfit import budget, profiles

GIB = 1024 ** 3


@dataclass
class Plan:
    total: int
    usable: int
    weights: int
    activations: int
    cache: int
    block_bytes: int
    blocks: int
    notes: list

    def sequences_at(self, context: int) -> int:
        """Concurrent sequences of `context` tokens, no prefix sharing.

        Deliberately the pessimistic number: prefix caching shares
        blocks between sequences, so the real figure for a workload with
        a common system prompt is higher. Over-promising concurrency is
        the worse error.
        """
        p = profiles.VLLM
        per_seq_tokens = p.cache_tokens(context)
        per_seq_blocks = per_seq_tokens // p.block_tokens
        return self.blocks // max(1, per_seq_blocks)

    def table(self) -> str:
        out = [
            f"{'card':20} {self.total / GIB:8.3f} GiB",
            f"{'usable (util)':20} {self.usable / GIB:8.3f} GiB",
            f"{'weights':20} {self.weights / GIB:8.3f} GiB",
            f"{'activations':20} {self.activations / GIB:8.3f} GiB",
            f"{'kv cache':20} {self.cache / GIB:8.3f} GiB",
            f"{'blocks':20} {self.blocks:8}  "
            f"({self.block_bytes / 1024:.0f} KiB each)",
        ]
        for n in self.notes:
            out.append(f"  ! {n}")
        return "\n".join(out)


def plan(shape, total_vram: int, *, utilisation: float | None = None,
         cache_dtype_bytes: float = 2, block_tokens: int | None = None,
         cuda_context: int | None = None, batch: int = 512,
         weights_override: int | None = None) -> Plan:
    p = profiles.VLLM
    util = p.utilisation if utilisation is None else utilisation
    block = p.block_tokens if block_tokens is None else block_tokens

    weights = (shape.weight_bytes if weights_override is None
               else weights_override)
    ctx = (budget.ASSUMED_CUDA_CONTEXT if cuda_context is None
           else cuda_context)
    activations = budget.compute_bytes(shape, 0, batch) + ctx

    usable = int(total_vram * util)
    cache = max(0, usable - weights - activations)
    per_token = budget.kv_bytes_per_token(shape, cache_dtype_bytes) \
        // shape.n_layers * shape.n_layers
    block_bytes = int(budget.kv_bytes(shape, block, cache_dtype_bytes))

    notes = list(p.notes)
    if cuda_context is None:
        notes.insert(0, "cuda context is assumed, not measured")
    if cache == 0:
        notes.insert(0, "no cache left after weights and activations: "
                        "the server will start and accept nothing")
    return Plan(total=total_vram, usable=usable, weights=weights,
                activations=activations, cache=cache,
                block_bytes=block_bytes,
                blocks=cache // block_bytes if block_bytes else 0,
                notes=notes)
