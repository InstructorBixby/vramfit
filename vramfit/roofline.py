"""Ceilings for prefill and decode, and what fraction of one was reached.

No measurement in here, just the denominators. See docs/speed.md for the
argument; the short version is that prefill is bound by FLOP/s and
decode by memory bandwidth, so one number cannot describe both.
"""
from __future__ import annotations

from dataclasses import dataclass

from vramfit import budget
from vramfit.shape import Shape


@dataclass
class Ceiling:
    tok_s: float
    basis: str          # "bandwidth" or "flops"
    bytes_per_token: int = 0
    flops_per_token: int = 0
    notes: tuple = ()


def decode_bytes_per_token(shape: Shape, context: int,
                           cache_dtype_bytes: float = 2,
                           weights: int | None = None) -> int:
    """Bytes that must cross the memory bus to emit one token.

    Every weight that the token routes through, read once, plus the
    whole cache up to the current position.

    On a mixture-of-experts model "every weight" is not the model: a
    token reads the dense layers plus the experts it routes to, so this
    is far smaller than residency. Using residency here under-predicts
    MoE decode speed by the expert ratio, which on a 128-expert model
    is most of an order of magnitude.
    """
    if weights is None:
        mix = shape.moe
        weights = (mix.bytes_per_token if mix is not None
                   else shape.weight_bytes)
    return weights + budget.kv_bytes(shape, context, cache_dtype_bytes)


def decode_ceiling(shape: Shape, context: int, bandwidth_bps: float,
                   cache_dtype_bytes: float = 2,
                   weights: int | None = None) -> Ceiling:
    per = decode_bytes_per_token(shape, context, cache_dtype_bytes,
                                 weights)
    notes: tuple = ()
    mix = shape.moe
    if mix is not None and weights is None:
        notes = (f"moe: {mix.n_used} of {mix.n_experts} experts read "
                 f"per token, so decode moves "
                 f"{mix.speed_ratio:.1f}x fewer bytes than the model's "
                 f"size -- residency is unaffected",)
    return Ceiling(tok_s=bandwidth_bps / per, basis="bandwidth",
                   bytes_per_token=per, notes=notes)


def prefill_flops_per_token(shape: Shape) -> int:
    """Two FLOPs per active parameter, attention term ignored.

    Active, not total: on an MoE model a token only does the arithmetic
    of the experts it routes to. The byte and the FLOP accounting agree
    about that, which they do not about residency.
    """
    mix = shape.moe
    if mix is not None and shape.weight_bytes:
        active = shape.params * (mix.bytes_per_token
                                 / shape.weight_bytes)
        return int(2 * active)
    return 2 * shape.params


def prefill_ceiling(shape: Shape, flops: float) -> Ceiling:
    per = prefill_flops_per_token(shape)
    return Ceiling(tok_s=flops / per, basis="flops",
                   flops_per_token=per)


def achieved_bandwidth(measured_tok_s: float,
                       bytes_per_token: int) -> float:
    return measured_tok_s * bytes_per_token


def fraction_of_peak(measured_tok_s: float, ceiling: Ceiling) -> float:
    return measured_tok_s / ceiling.tok_s if ceiling.tok_s else 0.0


def verdict(fraction: float) -> str:
    """Plain words for the ratio, with the thresholds in one place.

    Deliberately coarse. The point is to stop someone reading 0.73 as
    a grade; it is a statement about where the next improvement is.
    """
    if fraction >= 0.75:
        return "bandwidth bound: the lever is fewer bytes per token"
    if fraction >= 0.5:
        return "near the roof: some headroom in the backend"
    if fraction >= 0.25:
        return ("leaving performance on the table: check offload, "
                "build flags, batch size")
    return "far off the roof: something is wrong, not slow"
