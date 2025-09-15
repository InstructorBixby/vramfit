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
    uuid: str = ""       # NVML GPU-UUID, stable across reboots
    bus_id: str = ""

    @property
    def free(self) -> int:
        return self.total - self.used

    @property
    def tag(self) -> str:
        """Name plus enough to tell two identical cards apart.

        A box with two of the same card produced two different
        measurements under one ledger key, which is worse than having
        no measurement: the diff reported a 30% regression that was
        really the slower slot. The UUID is the only identifier that is
        stable across reboots -- the index is not, and the bus id is
        stable but changes if the card moves.
        """
        if self.uuid:
            return f"{self.name} [{self.uuid[-8:]}]"
        if self.bus_id:
            return f"{self.name} [{self.bus_id}]"
        return self.name

    def __str__(self) -> str:
        return (f"[{self.index}] {self.tag}  "
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

            def text(fn):
                v = fn(h)
                return v.decode() if isinstance(v, bytes) else v

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(Device(
                i, text(pynvml.nvmlDeviceGetName),
                mem.total, mem.used, driver,
                uuid=text(pynvml.nvmlDeviceGetUUID),
                bus_id=text(lambda hh: pynvml.nvmlDeviceGetPciInfo(hh)
                            .busId)))
        return out
    finally:
        pynvml.nvmlShutdown()


def _from_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    proc = subprocess.run(
        [exe, "--query-gpu=index,name,memory.total,memory.used,"
              "driver_version,uuid,pci.bus_id",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, total, used, driver = parts[:5]
        uuid = parts[5] if len(parts) > 5 else ""
        bus = parts[6] if len(parts) > 6 else ""
        out.append(Device(int(idx), name, int(float(total)) * MIB,
                          int(float(used)) * MIB, driver,
                          uuid=uuid, bus_id=bus))
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
    """Device names are not stable enough to key a table on directly.

    Also strips the disambiguating suffix that `Device.tag` adds, so a
    ledger row written with a tag still finds its spec.
    """
    n = name.lower()
    n = re.sub(r"\[[^\]]*\]", " ", n)
    n = re.sub(r"\bnvidia\b|\bgeforce\b|\blaptop gpu\b", " ", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return " ".join(n.split())
