"""Splitting a model across cards.

The tempting model is "two 24 GiB cards are a 48 GiB card". They are
not, in three ways that all matter:

1. **The split is per layer, and layers are not equal.** llama.cpp's
   `--tensor-split` divides by proportion and the embedding and output
   tensors land on one specific card, so the two halves are not the same
   size. The card that gets them needs more free memory than the
   arithmetic suggests.

2. **Each card pays its own CUDA context.** Two cards is two contexts,
   roughly 300 MiB each, and a compute buffer each.

3. **The decode ceiling is the *slowest* card's bandwidth, not the
   sum.** Layer-wise split is sequential: card 0 computes its layers,
   hands the activations over, card 1 computes the rest. Nothing
   overlaps, so adding a slow card to a fast one makes generation
   slower than the fast card alone while making more context possible.
   That trade is the real decision and summing the bandwidths hides it.

Tensor-parallel splits in other backends do overlap and are a different
calculation; this models the layer-wise case, which is what llama.cpp
and exllamav2 do by default.
"""
from __future__ import annotations

from dataclasses import dataclass

from vramfit import budget, specs

GIB = 1024 ** 3


@dataclass
class Split:
    devices: list          # Device objects
    proportions: list      # one float per device, summing to 1
    primary: int = 0       # which card holds embedding and output

    def layer_counts(self, n_layers: int) -> list:
        raw = [p * n_layers for p in self.proportions]
        out = [int(x) for x in raw]
        # Give the remainder to the largest share, not to the first
        # card: rounding everything down and dumping the slack on card
        # 0 is how the primary card ends up over-committed.
        while sum(out) < n_layers:
            big = max(range(len(out)), key=lambda i: raw[i] - out[i])
            out[big] += 1
        return out

    def budgets(self, shape, context: int, **kw: object) -> list:
        counts = self.layer_counts(shape.n_layers)
        block, other = budget.block_weight_bytes(shape)
        out = []
        for i, n in enumerate(counts):
            weights = int(block * n / shape.n_layers)
            if i == self.primary:
                weights += other
            b = budget.build(
                shape, context,
                weights_override=weights,
                n_gpu_layers=None, **kw)  # type: ignore[arg-type]
            # The cache follows the layers, like any other offload.
            b.kv_cache = budget.kv_bytes(
                shape, context,
                kw.get("cache_dtype_bytes") or 2, n)  # type: ignore[arg-type]
            out.append(b)
        return out

    def fits(self, shape, context: int, **kw: object) -> list:
        """(device, budget, slack) per card."""
        return [(d, b, d.total - b.total)
                for d, b in zip(self.devices,
                                self.budgets(shape, context, **kw))]

    def decode_bandwidth(self) -> tuple:
        """(bandwidth to use, why).

        The slowest card's, because a layer-wise split is sequential.
        """
        rows = [(d, specs.lookup(d.tag)) for d in self.devices]
        known = [(d, s) for d, s in rows if s is not None]
        if not known:
            return 0.0, "no spec row for any card in the split"
        slowest = min(known, key=lambda ds: ds[1].bandwidth_gbs)
        if len(known) < len(rows):
            return (slowest[1].bandwidth_bps,
                    "some cards have no spec row; using the slowest of "
                    "those that do, which may be optimistic")
        return (slowest[1].bandwidth_bps,
                f"layer-wise split is sequential, so the bound is the "
                f"slowest card ({slowest[0].name}, "
                f"{slowest[1].bandwidth_gbs:.0f} GB/s) and not the sum")


def even(devices_: list) -> Split:
    n = len(devices_)
    return Split(devices_, [1.0 / n] * n)


def by_memory(devices_: list) -> Split:
    """Proportional to each card's total memory.

    The obvious default and still not right, because the primary card
    carries the embedding and output tensors on top of its share.
    """
    total = sum(d.total for d in devices_) or 1
    props = [d.total / total for d in devices_]
    primary = max(range(len(devices_)),
                  key=lambda i: devices_[i].total)
    return Split(devices_, props, primary=primary)
