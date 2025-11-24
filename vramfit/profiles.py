"""How each backend spends VRAM, as the parts that differ from llama.cpp.

A profile is three things: how the cache is sized, how the compute
buffer is sized, and what the headroom policy is. Everything else --
weights, CUDA context -- is the same for everybody because it is a
property of the model and the driver, not of the runtime.

See docs/backends.md for what each of these is approximating.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

MIB = 1024 ** 2


@dataclass
class Profile:
    name: str
    # vLLM and exllamav2 page the cache; llama.cpp does not.
    block_tokens: int = 1
    # Fraction of the card the runtime will use at all.
    utilisation: float = 1.0
    headroom_frac: float = 0.03
    notes: tuple = ()

    def cache_tokens(self, context: int, sequences: int = 1) -> int:
        """Tokens of cache that must be allocated.

        A paged cache rounds each sequence up to a block, so 17 tokens
        in blocks of 16 costs 32 and ten sequences of 17 cost 320. The
        rounding is per sequence, which is why this takes both numbers
        rather than their product.
        """
        if self.block_tokens <= 1:
            return context * sequences
        blocks = ceil(context / self.block_tokens)
        return blocks * self.block_tokens * sequences


LLAMA_CPP = Profile(
    "llama.cpp",
    notes=("cache is preallocated contiguously for the full -c",))

VLLM = Profile(
    "vllm", block_tokens=16, utilisation=0.9, headroom_frac=0.0,
    notes=("gpu_memory_utilization is a fraction of the card, not a "
           "byte figure: the runtime takes that much and divides what "
           "is left after weights into KV blocks",
           "it will not fail to load for want of cache, it will load "
           "with fewer blocks and refuse long requests later",
           "prefix caching shares blocks, so concurrent cache use is "
           "not the sum of the per-sequence costs"))

EXLLAMAV2 = Profile(
    "exllamav2", block_tokens=1, headroom_frac=0.05,
    notes=("cache is preallocated for the declared max_seq_len",
           "its cache quantisation modes are not the ggml block sizes",
           "paged attention when batching, which makes the cache a "
           "block count rather than a product"))

PROFILES = {p.name: p for p in (LLAMA_CPP, VLLM, EXLLAMAV2)}


def lookup(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"no profile for backend {name!r}; have "
                       f"{', '.join(sorted(PROFILES))}") from None
