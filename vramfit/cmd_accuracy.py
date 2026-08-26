"""`vramfit accuracy`: the predictions against the measurements."""
from __future__ import annotations

import argparse

from vramfit import accuracy, budget, load, measured
from vramfit.ledger import Ledger


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("accuracy",
                      help="prediction error across the ledger")
    p.add_argument("--db")
    p.add_argument("--models", nargs="*", default=[],
                   help="paths to the model files, to read their "
                        "metadata; published shapes are used otherwise")
    p.set_defaults(fn=run)
    return p


def shape_index(paths: list) -> dict:
    """Model name -> Shape, from files given and from shapes.csv."""
    index = dict(measured.known_shapes())
    for path in paths:
        try:
            shape, _ = load.load(path)
        except Exception:
            continue
        index[shape.name.lower()] = shape
    return index


def run(args: argparse.Namespace) -> int:
    index = shape_index(args.models)

    def predict(row: dict) -> int | None:
        shape = index.get(str(row["model"]).lower())
        if shape is None:
            return None
        ngl = row.get("n_gpu_layers")
        return budget.build(
            shape, int(row["context"]),
            cache_dtype=str(row.get("cache_dtype") or "f16"),
            cache_dtype_bytes=None,
            batch=int(row.get("batch") or 512),
            n_gpu_layers=(int(ngl) if ngl is not None
                          and int(ngl) < shape.n_layers else None),
            flash_attn=True,
        ).total

    with Ledger(args.db) as led:
        rows = []
        for key in led.keys():
            k = {c: key[c] for c in ("model", "quant", "context",
                                     "backend", "gpu")}
            latest = led.latest(**k)
            if latest:
                rows.append(latest)
    print(accuracy.residuals(rows, predict).table())
    return 0
