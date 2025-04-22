"""Invert the budget.

`budget.build` answers "how much does this configuration need". The
questions people actually ask are the other way round:

  - I have 16 GiB. What is the longest context I can run?
  - I need 32k context. What quant do I have to drop to?
  - I have 10 GiB free. How many layers can I offload?

All three are monotonic in one variable, so bisection is enough and is
preferable to an algebraic inverse: the budget has terms that are not
linear in context and the formula would have to be kept in sync with
them. Bisecting calls the real function.
"""
from __future__ import annotations

from vramfit import budget

# Contexts are reported rounded down to this, because a backend that
# rounds its cache up to a block boundary will otherwise quietly spend
# the last few hundred tokens of slack.
ALIGN = 256


def fits(shape, context: int, vram: int, **kw) -> bool:
    return budget.build(shape, context, **kw).total <= vram


def max_context(shape, vram: int, *, align: int = ALIGN,
                ceiling: int | None = None, **kw) -> int:
    """Longest context that fits in `vram`, or 0.

    `ceiling` defaults to the trained context length: a longer one is
    arithmetically fine and is not a thing to recommend.
    """
    hi = ceiling or shape.train_context or 1 << 20
    if not fits(shape, align, vram, **kw):
        return 0
    if fits(shape, hi, vram, **kw):
        return hi
    lo = align
    while hi - lo > align:
        mid = (lo + hi) // 2
        if fits(shape, mid, vram, **kw):
            lo = mid
        else:
            hi = mid
    return (lo // align) * align
