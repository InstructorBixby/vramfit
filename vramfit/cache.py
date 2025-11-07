"""KV cache dtypes, by the names the backends use.

The cache is not a special allocation: llama.cpp stores it in ordinary
ggml tensors, so a quantised cache costs exactly what the block geometry
says, including the per-block scale. `q8_0` is 8.5 bits and not 8, which
is the same arithmetic as the weights and the same reason to not use a
table of round numbers.

What a quantised cache buys, at 32k context on an 8B:

| cache | bytes/token | cache at 32k | notes |
|---|---|---|---|
| f16 | 128 KiB | 4.00 GiB | the default |
| q8_0 | 68 KiB | 2.13 GiB | hard to tell apart in use |
| q4_0 | 36 KiB | 1.13 GiB | measurable degradation on long context |

The speed effect is the interesting one and it cuts both ways: a smaller
cache is fewer bytes to read per token, so decode at a long context gets
faster, while quantising and dequantising the cache costs arithmetic
that decode has to spare. Net, on this box, q8_0 at 32k is slightly
faster than f16 at 32k. That is a measurement and not a law, which is
what the ledger is for.

One trap: a quantised V cache needs flash attention in llama.cpp, and
without it the backend silently keeps V in f16. So the predicted number
and the measured one disagree for a reason that has nothing to do with
the arithmetic, and the type names here carry that requirement.
"""
from __future__ import annotations

from dataclasses import dataclass

from vramfit import ggml_types


@dataclass
class CacheType:
    name: str
    needs_flash_attn: bool = False

    @property
    def bytes_per_element(self) -> float:
        n, nbytes = ggml_types.BLOCK[self.name]
        return nbytes / n

    @property
    def bits_per_element(self) -> float:
        return self.bytes_per_element * 8


TYPES = {
    "f32": CacheType("f32"),
    "f16": CacheType("f16"),
    "bf16": CacheType("bf16"),
    "q8_0": CacheType("q8_0", needs_flash_attn=True),
    "q5_1": CacheType("q5_1", needs_flash_attn=True),
    "q5_0": CacheType("q5_0", needs_flash_attn=True),
    "q4_1": CacheType("q4_1", needs_flash_attn=True),
    "q4_0": CacheType("q4_0", needs_flash_attn=True),
    "iq4_nl": CacheType("iq4_nl", needs_flash_attn=True),
}


def lookup(name: str) -> CacheType | None:
    return TYPES.get(name.strip().lower())


def element_bytes(name: str) -> float:
    t = lookup(name)
    if t is None:
        raise KeyError(f"unknown cache type {name!r}; have "
                       f"{', '.join(sorted(TYPES))}")
    return t.bytes_per_element
