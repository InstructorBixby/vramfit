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


def decode_bytes_per_token(shape: Shape, context: int,
                           cache_dtype_bytes: float = 2,
                           weights: int | None = None) -> int:
    """Bytes that must cross the memory bus to emit one token.

    Every weight, read once, plus the whole cache up to the current
    position. Averaging the cache over the generation would be more
    accurate for a long run and less useful for a ceiling, so this is
    the cost at `context` and the caller decides what context means.
    """
    w = shape.weight_bytes if weights is None else weights
    return w + budget.kv_bytes(shape, context, cache_dtype_bytes)


def decode_ceiling(shape: Shape, context: int, bandwidth_bps: float,
                   cache_dtype_bytes: float = 2,
                   weights: int | None = None) -> Ceiling:
    per = decode_bytes_per_token(shape, context, cache_dtype_bytes,
                                 weights)
    return Ceiling(tok_s=bandwidth_bps / per, basis="bandwidth",
                   bytes_per_token=per)


def prefill_flops_per_token(shape: Shape) -> int:
    """Two FLOPs per parameter, attention term ignored. See docs."""
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
