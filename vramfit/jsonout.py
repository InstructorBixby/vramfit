"""`--json` on the prediction subcommands.

Added because the first thing anybody does with a CLI that prints a
table is write a fragile parser for the table. The schema is flat and
every number is in bytes, so no unit guessing: `GiB` belongs in a
display layer and nowhere else.

`assumptions` is part of the schema and not a footnote. A consumer that
drops it is making a decision on a number whose provenance it threw
away.
"""
from __future__ import annotations

import json


def budget_dict(shape, context, b, *, vram=None, source="") -> dict:
    out = {
        "model": shape.name,
        "arch": shape.arch,
        "source": source,
        "params": shape.params,
        "bits_per_weight": round(shape.bpw, 4),
        "layers": shape.n_layers,
        "attention_heads": shape.n_head,
        "kv_heads": shape.n_head_kv,
        "head_dim": shape.head_dim,
        "vocab": shape.vocab,
        "trained_context": shape.train_context,
        "context": context,
        "bytes": {
            "weights": b.weights,
            "kv_cache": b.kv_cache,
            "compute": b.compute,
            "cuda_context": b.cuda_context,
            "headroom": b.headroom,
            "total": b.total,
        },
        "offloaded_layers": b.offloaded_layers,
        "assumptions": list(b.assumptions),
    }
    if vram is not None:
        out["vram"] = vram
        out["slack"] = vram - b.total
        out["fits"] = vram >= b.total
    return out


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=False)
