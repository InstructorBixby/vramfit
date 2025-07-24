"""One measured result.

The key is (model, quant, context, backend, gpu). Everything else is
either measurement or provenance, and the provenance fields exist
because a number without the driver and backend version it was taken
under cannot be compared to anything later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Result:
    # key
    model: str
    quant: str
    context: int
    backend: str
    gpu: str

    # measurement
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    prefill_frac_peak: float = 0.0
    decode_frac_peak: float = 0.0
    achieved_bandwidth_gbs: float = 0.0
    vram_peak: int = 0

    # provenance
    driver: str = ""
    backend_version: str = ""
    n_gpu_layers: int | None = None
    cache_dtype: str = "f16"
    batch: int = 512
    notes: list = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.model, self.quant, self.context, self.backend,
                self.gpu)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.model} {self.quant} ctx {self.context} "
            f"{self.backend} on {self.gpu}\n"
            f"  prefill {self.prefill_tok_s:8.1f} tok/s  "
            f"({self.prefill_frac_peak:.0%} of flops peak)\n"
            f"  decode  {self.decode_tok_s:8.1f} tok/s  "
            f"({self.decode_frac_peak:.0%} of bandwidth peak, "
            f"{self.achieved_bandwidth_gbs:.0f} GB/s)"
        )
