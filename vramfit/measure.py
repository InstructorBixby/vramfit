"""Measure the two terms that cannot be derived.

The CUDA context is a per-process allocation that exists before a single
weight is loaded. It is not in any header and it is not constant across
drivers or cards, so the only honest thing to do is measure it on the
machine the prediction is for and cache the answer.

Method: read used memory, start a process that initialises a context and
allocates nothing, read used memory again, difference. The child is a
separate process because a context in *this* process would be counted in
the baseline.

The cache is keyed on (device, driver) because a driver upgrade moved
the figure from 304 MiB to 322 MiB on one card here. A cache keyed on
the device alone would have kept serving the old number, and a
prediction that is wrong in a cached way is worse than one that admits
it has not measured anything.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024 ** 2

CHILD = (
    "import ctypes, sys, time\n"
    "lib = ctypes.CDLL('libcuda.so.1')\n"
    "assert lib.cuInit(0) == 0\n"
    "ctx = ctypes.c_void_p()\n"
    "dev = ctypes.c_int()\n"
    "assert lib.cuDeviceGet(ctypes.byref(dev), int(sys.argv[1])) == 0\n"
    "assert lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0\n"
    "print('ready', flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
)


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "vramfit" / "context.json"


def measure_cuda_context(index: int = 0, settle: float = 1.5) -> int:
    """Bytes of VRAM a bare CUDA context costs on this device."""
    from vramfit import devices

    def used() -> int:
        for d in devices.discover():
            if d.index == index:
                return d.used
        raise RuntimeError(f"device {index} not visible")

    before = used()
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(index), str(settle + 2)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline() if proc.stdout else ""
        if "ready" not in line:
            tail = proc.stderr.read()[:200] if proc.stderr else ""
            raise RuntimeError("child did not create a context: "
                               + tail)
        time.sleep(settle)
        after = used()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return max(0, after - before)


def _key(device_name: str, driver: str) -> str:
    return f"{device_name}|{driver}"


def cached_cuda_context(index: int = 0, device_name: str = "",
                        driver: str = "") -> int | None:
    """Measured value for this (device, driver), if one was recorded."""
    path = cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    value = data.get(_key(device_name, driver))
    return int(value) if value else None


def stale_entries(device_name: str = "", driver: str = "") -> list[str]:
    """Entries for this device under a different driver.

    Surfaced rather than deleted: a measurement from an older driver is
    still a true fact about that driver, and the only wrong thing to do
    with it is use it for this one.
    """
    path = cache_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return sorted(k for k in data
                  if k.startswith(f"{device_name}|")
                  and k != _key(device_name, driver))


def record_cuda_context(value: int, device_name: str = "",
                        driver: str = "") -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[_key(device_name, driver)] = int(value)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
