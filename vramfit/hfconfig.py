"""Read a Hugging Face `config.json` into a Shape.

Different vocabulary for the same facts, and a few traps:

  - `num_key_value_heads` is absent on pre-GQA models, where it equals
    `num_attention_heads`. Defaulting it to 1 would under-predict the
    cache by the head count, which is the mirror image of the mistake
    the GGUF reader used to make.
  - `head_dim` is usually `hidden_size / num_attention_heads` but some
    architectures set it explicitly and it is not that ratio, so read it
    when it is there.
  - `max_position_embeddings` is the trained context, except where a
    rope-scaling block extends it, in which case both numbers are
    interesting and only the base one is a fact.
  - the parameter count is not in the config at all. It comes from the
    weight files, so a config alone gives a shape with no tensors and an
    estimate has to say so.
"""
from __future__ import annotations

import json
from pathlib import Path

from vramfit import safetensors
from vramfit.shape import Shape, Tensor

ST_TO_GGML = {"F32": "f32", "F16": "f16", "BF16": "bf16",
              "F8_E4M3": "i8", "F8_E5M2": "i8", "I8": "i8"}


def _read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def from_config(cfg: dict, name: str = "", tensors=None) -> Shape:
    n_head = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    return Shape(
        name=name or cfg.get("_name_or_path", ""),
        arch=(cfg.get("model_type")
              or (cfg.get("architectures") or ["unknown"])[0]),
        n_layers=cfg["num_hidden_layers"],
        n_embd=hidden,
        n_head=n_head,
        n_head_kv=cfg.get("num_key_value_heads", n_head),
        head_dim=cfg.get("head_dim") or hidden // n_head,
        vocab=cfg["vocab_size"],
        train_context=cfg.get("max_position_embeddings", 0),
        tensors=list(tensors or []),
    )


def from_directory(directory) -> Shape:
    """A repo checkout: config.json plus whatever weight files exist."""
    directory = Path(directory)
    cfg = _read(directory / "config.json")
    tensors = []
    for shard in safetensors.shard_paths(directory):
        if shard.stat().st_size == 0:
            continue
        header = safetensors.read_header(shard)
        for tname, dtype, shape, _ in safetensors.tensors(header):
            tensors.append(Tensor(tname, list(shape),
                                  ST_TO_GGML.get(dtype, "f16")))
    return from_config(cfg, directory.name, tensors)


def rope_scaling(cfg: dict):
    """(base context, scaled context) or None.

    A model whose config claims 128k via a rope factor is a different
    planning problem from one trained at 128k, and the cache cost is
    identical either way -- which is the point worth making.
    """
    rs = cfg.get("rope_scaling")
    base = cfg.get("max_position_embeddings")
    if not rs or not base:
        return None
    factor = rs.get("factor")
    original = rs.get("original_max_position_embeddings")
    if original:
        return original, base
    if factor:
        return base, int(base * float(factor))
    return None
