"""Work out what kind of thing was pointed at, and get a Shape from it.

Three formats, one entry point, because the question "will this fit" is
asked about whatever is on disk and nobody wants to pick the right
subcommand for it.

A config.json with no weight files produces a Shape with no tensors.
That is a legitimate thing to have -- the cache arithmetic is complete
without it -- but the weight term then has to come from a quant estimate
and the caller has to know that, so this returns the reason too.
"""
from __future__ import annotations

from pathlib import Path

from vramfit import gguf, hfconfig, safetensors
from vramfit import shape as shape_mod
from vramfit.shape import Tensor


class LoadError(Exception):
    pass


def load(target) -> tuple:
    """Return (Shape, how) for a file or directory."""
    p = Path(target)
    if p.is_dir():
        if (p / "config.json").exists():
            sh = hfconfig.from_directory(p)
            how = ("config.json + safetensors" if sh.tensors
                   else "config.json only, no weights found")
            return sh, how
        ggufs = sorted(p.glob("*.gguf"))
        if ggufs:
            return load(ggufs[0])
        raise LoadError(f"{p}: no config.json and no .gguf inside")

    if not p.exists():
        raise LoadError(f"{p}: no such file")

    if p.suffix == ".gguf":
        _, meta, tensors = gguf.read_header(p)
        return shape_mod.from_gguf(meta, tensors, p.name), "gguf header"

    if p.suffix == ".safetensors":
        header = safetensors.read_header(p)
        tensors = [
            Tensor(n, list(shape),
                   hfconfig.ST_TO_GGML.get(dtype, "f16"))
            for n, dtype, shape, _ in safetensors.tensors(header)]
        cfg = p.parent / "config.json"
        if not cfg.exists():
            raise LoadError(
                f"{p}: safetensors on its own has no architecture "
                f"metadata; need config.json beside it")
        return (hfconfig.from_config(
                    hfconfig._read(cfg), p.parent.name, tensors),
                "safetensors + config.json")

    if p.name == "config.json":
        return (hfconfig.from_config(hfconfig._read(p), p.parent.name),
                "config.json only, no weights found")

    raise LoadError(f"{p}: do not know how to read this")
