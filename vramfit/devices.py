"""What card is in the box, and what is it theoretically capable of.

Two separate questions and two separate sources. The amount of memory
and how much of it is already in use come from the driver. Peak memory
bandwidth does not -- NVML will not tell you, so it comes from a table
of vendor specifications keyed on the device name.

A table of vendor numbers is a liability, so: it is only ever used as a
denominator, every row is marked with where it came from, and an unknown
card degrades to "no ratio available" rather than to a plausible lie.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

MIB = 1024 ** 2


@dataclass
class Device:
    index: int
    name: str
    total: int           # bytes
    used: int            # bytes
    driver: str = ""

    @property
    def free(self) -> int:
        return self.total - self.used

    def __str__(self) -> str:
        return (f"[{self.index}] {self.name}  "
                f"{self.free / MIB:.0f} / {self.total / MIB:.0f} MiB free")


def _from_pynvml():
    import pynvml

    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(Device(i, name, mem.total, mem.used, driver))
        return out
    finally:
        pynvml.nvmlShutdown()


def _from_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    proc = subprocess.run(
        [exe, "--query-gpu=index,name,memory.total,memory.used,"
              "driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, total, used, driver = parts[:5]
        out.append(Device(int(idx), name, int(float(total)) * MIB,
                          int(float(used)) * MIB, driver))
    return out


def discover() -> list:
    """Every visible CUDA device. NVML first, nvidia-smi as a fallback.

    Both can be absent -- a laptop, a CI runner -- and that is not an
    error here. It only means the ratio columns stay empty.
    """
    try:
        return _from_pynvml()
    except Exception:
        return _from_smi()


def normalise(name: str) -> str:
    """Device names are not stable enough to key a table on directly."""
    n = name.lower()
    n = re.sub(r"\bnvidia\b|\bgeforce\b|\blaptop gpu\b", " ", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return " ".join(n.split())
