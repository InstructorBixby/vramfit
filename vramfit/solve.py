"""Invert the budget.

`budget.build` answers "how much does this configuration need". The
questions people actually ask are the other way round:

  - I have 16 GiB. What is the longest context I can run?
  - I need 32k context. What quant do I have to drop to?
  - I have 10 GiB free. How many layers can I offload?

All of these are monotonic in one variable, so bisection is enough and
is preferable to an algebraic inverse: the budget has terms that are not
linear in context and a formula would have to be kept in sync with them.
Bisecting calls the real function.
"""
from __future__ import annotations

from vramfit import budget, quants

# Contexts are reported rounded down to this, because a backend that
# rounds its cache up to a block boundary will otherwise quietly spend
# the last few hundred tokens of slack.
ALIGN = 256


def fits(shape, context: int, vram: int, **kw) -> bool:
    return budget.build(shape, context, **kw).total <= vram


def max_context(shape, vram: int, *, align: int = ALIGN,
                ceiling: int | None = None, **kw) -> int:
    """Longest context that fits in `vram`, or 0.

    `ceiling` defaults to the trained context length: going beyond it is
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


def quant_for_context(shape, context: int, vram: int, **kw):
    """Heaviest quant on the ladder that fits `context` in `vram`.

    Returns (QuantEstimate, Budget), or (None, None) when even the
    bottom rung does not fit.
    """
    for q in sorted(quants.LADDER, key=lambda q: -q.bpw):
        b = budget.build(
            shape, context,
            weights_override=int(shape.params * q.bpw / 8), **kw)
        if b.total <= vram:
            return q, b
    return None, None


def context_ladder(shape, vram: int, **kw) -> list:
    """(quant, max context) for every rung that fits at all.

    This is the table I actually want when deciding what to download.
    """
    out = []
    for q in sorted(quants.LADDER, key=lambda q: -q.bpw):
        ctx = max_context(
            shape, vram,
            weights_override=int(shape.params * q.bpw / 8), **kw)
        if ctx:
            out.append((q, ctx))
    return out


def max_gpu_layers(shape, vram: int, context: int, **kw) -> int:
    """Largest `-ngl` that fits, 0 .. n_layers.

    Monotonic in the layer count, so the same bisection applies. The
    answer being `n_layers` means the model fits outright and partial
    offload is not the question.
    """
    kw.pop("n_gpu_layers", None)
    if budget.build(shape, context, n_gpu_layers=shape.n_layers,
                    **kw).total <= vram:
        return shape.n_layers
    lo, hi = 0, shape.n_layers
    if budget.build(shape, context, n_gpu_layers=0, **kw).total > vram:
        return 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if budget.build(shape, context, n_gpu_layers=mid,
                        **kw).total <= vram:
            lo = mid
        else:
            hi = mid
    return lo
