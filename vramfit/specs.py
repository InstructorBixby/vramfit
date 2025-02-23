"""Load the GPU specification table.

Kept as CSV and not as a dict literal so that adding a card is a data
change, and so that `source` travels with the number. Anything that
consumes a row has to cope with the row being absent.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

TABLE = Path(__file__).parent / "data" / "gpus.csv"
MIB = 1024 ** 2


@dataclass
class Spec:
    key: str
    vram_mib: int
    bandwidth_gbs: float
    fp16_tflops: float
    source: str

    @property
    def bandwidth_bps(self) -> float:
        """GB/s as advertised is 10^9 bytes, not GiB."""
        return self.bandwidth_gbs * 1e9


def load(path=TABLE) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh
        pass
    return out
