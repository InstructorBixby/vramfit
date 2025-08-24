"""Compare measurements under one key.

This is the payoff for keeping history. After a driver upgrade, a
backend rebuild or a quant change, the question is not "how fast is it"
but "is it different, and by how much, and what else changed at the same
time" -- and the last clause is the one that makes the first two mean
anything.

A diff that reports a speed change without reporting that the driver
also moved is an invitation to the wrong conclusion, so provenance
changes are printed next to the numbers and not in a footnote.
"""
from __future__ import annotations

from dataclasses import dataclass

MEASURED = ("prefill_tok_s", "decode_tok_s", "prefill_frac_peak",
            "decode_frac_peak", "achieved_bandwidth_gbs", "vram_peak")
PROVENANCE = ("driver", "backend_version", "cache_dtype",
              "n_gpu_layers", "batch")

# Below this, a difference is run-to-run noise on this hardware and
# calling it a change is worse than saying nothing.
NOISE = 0.02


@dataclass
class Change:
    field: str
    before: object
    after: object
    ratio: float | None = None

    @property
    def significant(self) -> bool:
        if self.ratio is None:
            return self.before != self.after
        return abs(self.ratio - 1.0) >= NOISE

    def line(self) -> str:
        if self.ratio is None:
            return f"{self.field:24} {self.before} -> {self.after}"
        pct = (self.ratio - 1.0) * 100
        return (f"{self.field:24} {self.before:>10.2f} -> "
                f"{self.after:>10.2f}  ({pct:+.1f}%)")


def diff(before: dict, after: dict) -> tuple:
    """(measured changes, provenance changes)."""
    measured, prov = [], []
    for f in MEASURED:
        a, b = before.get(f) or 0.0, after.get(f) or 0.0
        if not a and not b:
            continue
        ratio = (b / a) if a else None
        measured.append(Change(f, a, b, ratio))
    for f in PROVENANCE:
        a, b = before.get(f), after.get(f)
        if a != b:
            prov.append(Change(f, a, b))
    return measured, prov


def render(before: dict, after: dict, *, all_fields: bool = False) -> str:
    measured, prov = diff(before, after)
    out = [f"{before.get('taken_at', '?')} -> "
           f"{after.get('taken_at', '?')}", ""]
    shown = [c for c in measured if all_fields or c.significant]
    if not shown:
        out.append(f"  no measured change above {NOISE:.0%}")
    for c in shown:
        out.append("  " + c.line())
    if prov:
        out += ["", "  changed at the same time:"]
        for c in prov:
            out.append("    " + c.line())
        out.append("    -- attribute the change to these before "
                   "attributing it to anything else")
    return "\n".join(out)
