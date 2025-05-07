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

from dataclasses import replace

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


def at_bpw(shape, bpw: float):
    """The same model with its weights scaled to a different bpw.

    Only the weight term moves: the cache, the buffers and the context
    do not care how the weights are stored. Returns a shape-alike with
    `weight_bytes` overridden, which is all the budget reads.
    """
    scaled = int(shape.params * bpw / 8)

    class _Scaled:
        pass

    # Cheaper and clearer than a second dataclass: copy the real shape
    # and shadow the one derived property the budget uses.
    out = replace(shape, tensors=[])
    out.__dict__["_weight_bytes"] = scaled
    object.__setattr__(out, "tensors", shape.tensors)
    return out, scaled


def quant_for_context(shape, context: int, vram: int, **kw):
    """Heaviest quant on the ladder that fits `context` in `vram`.

    Returns (QuantEstimate, Budget) or (None, None).
    """
    for q in sorted(quants.LADDER, key=lambda q: -q.bpw):
        b = budget.build(shape, context,
                         weights_override=int(shape.params * q.bpw / 8),
                         **kw)
        if b.total <= vram:
            return q, b
    return None, None


def max_layers_offloaded(shape, vram: int, context: int, **kw) -> int:
    """Placeholder: partial offload needs its own budget. See NOTES."""
    raise NotImplementedError
