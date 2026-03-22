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
        """GB/s as advertised means 10^9 bytes, not GiB."""
        return self.bandwidth_gbs * 1e9


def load(path: str | Path = TABLE) -> dict[str, Spec]:
    out: dict[str, Spec] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(
            (ln for ln in fh if not ln.startswith("#")))
        for row in rows:
            out[row["key"]] = Spec(
                key=row["key"],
                vram_mib=int(row["vram_mib"]),
                bandwidth_gbs=float(row["bandwidth_gbs"]),
                fp16_tflops=float(row["fp16_tflops"]),
                source=row["source"],
            )
    return out


def lookup(device_name: str,
           table: dict[str, Spec] | None = None) -> Spec | None:
    """Spec for a device, or None. None is a supported answer."""
    from vramfit import devices

    table = load() if table is None else table
    return table.get(devices.normalise(device_name))
