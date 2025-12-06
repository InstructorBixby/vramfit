"""Measuring, as opposed to predicting.

A backend here is anything that can be asked two questions separately:
how fast does it process a prompt, and how fast does it generate. Any
harness that only reports one combined number is not usable for this,
which rules out most of the convenience wrappers and is why the backends
drive the underlying tools directly.

Nothing in here imports a model library. Each backend shells out to the
binary, or talks to the server, that it is measuring -- that is the
thing whose performance is in question, and a Python binding would
measure a different build with different flags.
"""
from __future__ import annotations

from typing import Callable

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def available() -> list:
    return sorted(REGISTRY)


def load_builtins() -> None:
    from vramfit.bench import llamacpp, vllm  # noqa: F401
