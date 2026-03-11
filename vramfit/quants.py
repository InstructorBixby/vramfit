"""Bits per weight for the quantisation schemes, and where each number
comes from.

For a GGUF file on disk none of this is needed: the tensor table gives
the exact answer. This is for the other direction -- "what would this
model cost at Q4_K_M" -- where there is no file to read yet, and for
the formats whose overhead is a documented constant rather than a block
table.

| scheme | bpw | why |
|---|---|---|
| exl2 | as named | the target bpw is the parameter you quantise with |
| gptq 4-bit g128 | 4.16 | 4 bits + one fp16 scale and one 4-bit zero per 128 |
| awq 4-bit g128 | 4.16 | same accounting |
| gguf K-mixes | measured | the `_M` mixes are architecture-dependent |

The GGUF numbers below are *observed* effective bpw averaged over the
llama-family files on disk here, and they are a fallback for planning
only. Anything with a file reads the file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantEstimate:
    name: str
    bpw: float
    source: str


def _group_bpw(bits: int, group: int, scale_bits: int = 16,
               zero_bits: int | None = None) -> float:
    """Grouped integer quantisation: payload plus per-group metadata."""
    zero_bits = bits if zero_bits is None else zero_bits
    return bits + (scale_bits + zero_bits) / group


LADDER = [
    QuantEstimate("iq1_s", 1.5625, "block"),
    QuantEstimate("iq2_xxs", 2.0625, "block"),
    QuantEstimate("iq2_m", 2.7, "observed"),
    QuantEstimate("q2_k", 2.625, "block"),
    QuantEstimate("q2_k_m", 2.96, "observed"),
    QuantEstimate("iq3_xxs", 3.0625, "block"),
    QuantEstimate("q3_k_s", 3.44, "observed"),
    QuantEstimate("q3_k_m", 3.91, "observed"),
    QuantEstimate("q3_k_l", 4.27, "observed"),
    QuantEstimate("iq4_xs", 4.25, "block"),
    QuantEstimate("q4_0", 4.5, "block"),
    QuantEstimate("gptq-4bit-128g", _group_bpw(4, 128, 16, 4), "arithmetic"),
    QuantEstimate("awq-4bit-128g", _group_bpw(4, 128, 16, 4), "arithmetic"),
    QuantEstimate("q4_k_s", 4.58, "observed"),
    QuantEstimate("q4_k_m", 4.83, "observed"),
    QuantEstimate("exl2-5.0", 5.0, "as-named"),
    QuantEstimate("q5_k_s", 5.52, "observed"),
    QuantEstimate("q5_k_m", 5.69, "observed"),
    QuantEstimate("q6_k", 6.5625, "block"),
    QuantEstimate("exl2-6.5", 6.5, "as-named"),
    QuantEstimate("q8_0", 8.5, "block"),
    QuantEstimate("f16", 16.0, "exact"),
]

BY_NAME = {q.name: q for q in LADDER}
# Public alias: api.py reads this, and renaming BY_NAME later must not
# break an import.
LADDER_BY_NAME = BY_NAME


def exl2(bpw: float) -> QuantEstimate:
    """exl2 takes a target bpw, so the name is the number."""
    return QuantEstimate(f"exl2-{bpw:g}", bpw, "as-named")


def lookup(name: str) -> QuantEstimate | None:
    n = name.strip().lower()
    if n.startswith("exl2-"):
        try:
            return exl2(float(n.split("-", 1)[1]))
        except ValueError:
            return None
    return BY_NAME.get(n)


def ladder_at_most(bpw: float) -> list:
    """Everything at or below a bits-per-weight, largest first."""
    return sorted((q for q in LADDER if q.bpw <= bpw),
                  key=lambda q: -q.bpw)
