"""Measure the two terms that cannot be derived.

The CUDA context is a per-process allocation that exists before a single
weight is loaded. It is not in any header and it is not constant across
drivers or cards, so the only honest thing to do is measure it on the
machine the prediction is for and cache the answer.

Method: read used memory, start a process that initialises a context and
allocates nothing, read used memory again, difference. The child is a
separate process because a context in *this* process would be counted in
the baseline.
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

    def used():
        for d in devices.discover():
            if d.index == index:
                return d.used
        raise RuntimeError(f"device {index} not visible")

    before = used()
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(index), str(settle + 2)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        if "ready" not in line:
            raise RuntimeError("child did not create a context: "
                               + proc.stderr.read()[:200])
        time.sleep(settle)
        after = used()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return max(0, after - before)


def cached_cuda_context(index: int = 0, device_name: str = "",
                        driver: str = "") -> int | None:
    """Measured value for this (device, driver), if one was recorded.

    Keyed on the driver version because it moves with the driver: see
    NOTES.md.
    """
    path = cache_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get(f"{device_name}|{driver}")


def record_cuda_context(value: int, device_name: str = "",
                        driver: str = "") -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[f"{device_name}|{driver}"] = int(value)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
