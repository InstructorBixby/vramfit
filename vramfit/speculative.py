"""Speculative decoding: one more thing in VRAM, one fewer thing the
decode ceiling can describe.

A draft model proposes several tokens and the target verifies them in
one batched forward pass. Two consequences, in increasing order of how
much trouble they cause:

1. **Capacity.** The draft model's weights and its own KV cache are both
   resident. That is arithmetic, it is the same arithmetic as the target,
   and it is handled by budgeting the draft as a second model.

2. **The decode ceiling stops being meaningful.** The whole ceiling is
   `bandwidth / bytes_per_token`, derived from one forward pass per
   token. Speculation exists precisely to break that: a verify pass
   reads the target's weights once and emits several tokens, so measured
   tok/s can legitimately exceed the "ceiling" by the acceptance rate.

The second point is why this module reports a ratio of *nothing*. An
honest tool says the measurement is not comparable rather than printing
160% of peak and letting somebody conclude the card is magic.
"""
from __future__ import annotations

from dataclasses import dataclass

from vramfit import budget

GIB = 1024 ** 3


@dataclass
class Speculative:
    target: object
    draft: object
    draft_tokens: int = 5

    def budget(self, context: int, *, draft_context: int | None = None,
               **kw: object) -> tuple:
        """(target budget, draft budget).

        The draft's context need not match the target's -- it only has
        to hold the window it drafts from -- but defaulting it to the
        target's is the safe assumption, since a backend that
        preallocates will do exactly that.
        """
        t = budget.build(self.target, context, **kw)  # type: ignore[arg-type]
        d = budget.build(self.draft, draft_context or context, **kw)  # type: ignore[arg-type]
        return t, d

    def table(self, context: int, **kw: object) -> str:
        t, d = self.budget(context, **kw)
        return "\n".join([
            f"target {self.target.name}",
            t.table(),
            "",
            f"draft {self.draft.name}",
            d.table(),
            "",
            f"{'combined':16} {(t.total + d.total) / GIB:8.3f} GiB",
            "",
            "  ! the decode bandwidth ceiling does not apply with "
            "speculation on: it assumes one forward pass per token, "
            "which is the assumption a draft model removes. Measured "
            "tok/s can exceed it legitimately, so no fraction is "
            "reported.",
        ])

    @property
    def draft_overhead_frac(self) -> float:
        """Draft bytes per token as a fraction of the target's.

        The useful sanity check on a draft choice: a draft that is 20%
        of the target cannot pay for itself unless its acceptance rate
        is high, because every drafted token costs a draft forward pass
        whether it is accepted or not.
        """
        tb = getattr(self.target, "weight_bytes", 0)
        db = getattr(self.draft, "weight_bytes", 0)
        return (db / tb) if tb else 0.0
