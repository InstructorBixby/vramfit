"""`vramfit contrib`: validate and import contributed measurements."""
from __future__ import annotations

import argparse
from pathlib import Path

from vramfit import contrib
from vramfit.bench.result import Result
from vramfit.ledger import Ledger


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("contrib",
                      help="validate or import contributed results")
    p.add_argument("action", choices=("check", "import"))
    p.add_argument("paths", nargs="+")
    p.add_argument("--db")
    p.set_defaults(fn=run)
    return p


def to_result(entry: dict) -> Result:
    return Result(
        model=entry["model"],
        quant=entry["quant"],
        context=int(entry["context"]),
        backend=entry["backend"],
        gpu=entry["gpu"],
        prefill_tok_s=float(entry.get("prefill_tok_s") or 0.0),
        decode_tok_s=float(entry["decode_tok_s"]),
        vram_peak=int(entry.get("vram_peak_mib") or 0) * 1024 ** 2,
        driver=entry["driver"],
        backend_version=entry["backend_version"],
        n_gpu_layers=entry.get("n_gpu_layers"),
        cache_dtype=entry["cache_dtype"],
        batch=int(entry.get("batch") or 512),
        notes=[n for n in (entry.get("notes"),
                           f"contributed by {entry['reported_by']}"
                           if entry.get("reported_by") else None) if n],
    )


def run(args: argparse.Namespace) -> int:
    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]

    bad = 0
    entries: list[dict] = []
    for path in files:
        found, problems = contrib.load_file(path)
        if problems:
            bad += 1
            print(f"{path}: {len(problems)} problem(s)")
            for prob in problems:
                print(f"  - {prob}")
            continue
        print(f"{path}: {len(found)} entr(y|ies) ok")
        entries += found

    if bad:
        print(f"\n{bad} file(s) rejected")
        return 1
    if args.action == "check":
        return 0

    with Ledger(args.db) as led:
        for entry in entries:
            led.record(to_result(entry))
    print(f"imported {len(entries)} entr(y|ies)")
    return 0
