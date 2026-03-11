"""The public API.

Everything below is covered by the compatibility promise in
docs/stability.md. Everything else in the package is internal and will
move without notice -- a distinction worth drawing explicitly, because
the CLI was the whole interface for a long time and people have started
importing the internals.

The shape is deliberately boring: functions that take a path and return
a dataclass. No client object, no configuration singleton, nothing that
has to be constructed before a question can be asked.
"""
from __future__ import annotations

from pathlib import Path

from vramfit import budget as _budget
from vramfit import cache as _cache
from vramfit import load as _load
from vramfit import quants as _quants
from vramfit import roofline as _roofline
from vramfit import solve as _solve
from vramfit import specs as _specs
from vramfit.budget import Budget
from vramfit.devices import Device, discover
from vramfit.quants import QuantEstimate
from vramfit.roofline import Ceiling
from vramfit.shape import Shape, Tensor
from vramfit.specs import Spec

__all__ = [
    "Budget", "Ceiling", "Device", "QuantEstimate", "Shape", "Spec",
    "Tensor", "bits_per_weight", "decode_ceiling", "devices", "fit",
    "load_shape", "max_context", "prefill_ceiling",
    "quant_for_context", "spec_for",
]


def load_shape(target: str | Path) -> Shape:
    """Read a GGUF file, safetensors repo or config.json directory."""
    shape, _ = _load.load(target)
    return shape


def fit(target: str | Path, context: int, *, cache_dtype: str = "f16",
        cuda_context: int | None = None, batch: int = 512,
        n_gpu_layers: int | None = None,
        quant: str | None = None) -> Budget:
    """VRAM a model needs at a context. Terms are on the Budget."""
    shape = load_shape(target)
    override = (int(shape.params * _quants.LADDER_BY_NAME[quant].bpw / 8)
                if quant else None)
    return _budget.build(shape, context, cache_dtype=cache_dtype,
                         cuda_context=cuda_context, batch=batch,
                         n_gpu_layers=n_gpu_layers,
                         weights_override=override)


def max_context(target: str | Path, vram: int, **kw: object) -> int:
    """Longest context that fits in `vram` bytes, or 0."""
    return _solve.max_context(load_shape(target), vram, **kw)  # type: ignore[arg-type]


def quant_for_context(target: str | Path, context: int,
                      vram: int) -> QuantEstimate | None:
    """Heaviest quant that fits, or None."""
    quant, _ = _solve.quant_for_context(load_shape(target), context,
                                        vram)
    return quant


def bits_per_weight(name: str) -> float | None:
    """Effective bpw for a quantisation scheme name."""
    q = _quants.lookup(name)
    return None if q is None else q.bpw


def decode_ceiling(target: str | Path, context: int,
                   bandwidth_bps: float,
                   cache_dtype: str = "f16") -> Ceiling:
    """Memory-bandwidth-bound tokens per second."""
    return _roofline.decode_ceiling(
        load_shape(target), context, bandwidth_bps,
        _cache.element_bytes(cache_dtype))


def prefill_ceiling(target: str | Path, flops: float) -> Ceiling:
    """Compute-bound prompt-processing tokens per second."""
    return _roofline.prefill_ceiling(load_shape(target), flops)


def spec_for(device_name: str) -> Spec | None:
    """Vendor specification row for a device, or None."""
    return _specs.lookup(device_name)


def devices() -> list[Device]:
    """Visible CUDA devices. Empty list when there are none."""
    return discover()
