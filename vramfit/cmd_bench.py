"""The bench subcommand, split out because it has its own flag soup."""
from __future__ import annotations

from vramfit import bench, devices, load
from vramfit.ledger import Ledger

MIB = 1024 ** 2


def add_parser(sub):
    p = sub.add_parser("bench", help="measure prefill and decode")
    p.add_argument("model")
    p.add_argument("--backend", default="llama.cpp")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--quant", default="")
    p.add_argument("--bin", default="",
                   help="path to the backend binary")
    p.add_argument("--cache-type", default="f16")
    p.add_argument("--n-gpu-layers", type=int, default=999)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--record", action="store_true",
                   help="append the result to the ledger")
    p.add_argument("--db")
    p.set_defaults(fn=run)
    return p


def run(args) -> int:
    bench.load_builtins()
    fn = bench.REGISTRY.get(args.backend)
    if fn is None:
        print(f"unknown backend {args.backend!r}; have "
              f"{', '.join(bench.available())}")
        return 2

    sh, _ = load.load(args.model)
    dev = next((d for d in devices.discover()
                if d.index == args.device), None)
    if dev is None:
        print("no CUDA device visible; a measurement needs one")
        return 1

    # dev.tag and not dev.name: two identical cards in one box have to
    # be distinguishable in the ledger.
    res = fn(sh, model=args.model, gpu=dev.tag, context=args.context,
             quant=args.quant, binary=args.bin,
             cache_type=args.cache_type,
             n_gpu_layers=args.n_gpu_layers, batch=args.batch,
             reps=args.reps, driver=dev.driver)
    print(res.summary())
    for note in res.notes:
        print(f"  ! {note}")

    if args.record:
        with Ledger(args.db) as led:
            led.record(res)
        print("recorded")
    return 0
