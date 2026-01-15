"""Sample used VRAM while something else is running.

The point is to have a measured peak to put next to the predicted total.
Without it, "the prediction was close" is a recollection.

A thread and a poll rather than anything clever: the driver's used-memory
figure is a counter, there is no high-water mark to read, and a 50 ms
poll is enough to catch an allocation that is going to be held for the
length of a benchmark. It will miss a transient spike, and that is worth
writing down because a transient spike is exactly what kills a tight
fit.
"""
from __future__ import annotations

import threading
import time


class PeakSampler:
    def __init__(self, index: int = 0, interval: float = 0.05):
        self.index = index
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        from vramfit import devices

        while not self._stop.is_set():
            for d in devices.discover():
                if d.index == self.index:
                    self.peak = max(self.peak, d.used)
                    self.samples += 1
                    break
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def note(self) -> str:
        if not self.samples:
            return "no memory samples taken: peak is unknown, not zero"
        return (f"peak sampled every {self.interval * 1000:.0f} ms "
                f"over {self.samples} samples; a shorter spike than "
                f"that is not captured")


def sample_peak(fn, index: int = 0, interval: float = 0.05):
    """Run `fn`, return (result, peak bytes, note)."""
    with PeakSampler(index, interval) as s:
        out = fn()
    return out, s.peak, s.note
