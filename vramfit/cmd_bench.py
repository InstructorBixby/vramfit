"""The bench subcommand, split out because it has its own flag soup."""
from __future__ import annotations

from vramfit import bench, budget, cmd_fit, devices, load, sampler
from vramfit.ledger import Ledger

GIB = 1024 ** 3
MIB = 1024 ** 2


def add_parser(sub):
    p = sub.add_parser("bench", help="measure prefill and decode")
    p.add_argument("model")
    p.add_argument("--backend", default="llama.cpp")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--quant", default="")
    p.add_argument("--bin", default="",
                   help="path to the backend binary")
    p.add_argument("--base-url", default="http://127.0.0.1:8000",
                   help="for server backends")
    p.add_argument("--cache-type", default="f16")
    p.add_argument("--n-gpu-layers", type=int, default=999)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-peak", action="store_true",
                   help="skip memory sampling")
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

    def measure_it():
        # dev.tag and not dev.name: two identical cards in one box have
        # to be distinguishable in the ledger.
        return fn(sh, model=args.model, gpu=dev.tag,
                  context=args.context, quant=args.quant,
                  binary=args.bin, base_url=args.base_url,
                  cache_type=args.cache_type,
                  n_gpu_layers=args.n_gpu_layers, batch=args.batch,
                  reps=args.reps, driver=dev.driver)

    if args.no_peak:
        res, peak, note = measure_it(), 0, ""
    else:
        res, peak, note = sampler.sample_peak(measure_it, dev.index)
    res.vram_peak = peak
    if note:
        res.notes.append(note)

    print(res.summary())

    predicted = budget.build(
        sh, args.context, cache_dtype=args.cache_type,
        cuda_context=cmd_fit.cuda_context_for(dev),
        batch=args.batch,
        n_gpu_layers=(args.n_gpu_layers
                      if args.n_gpu_layers < sh.n_layers else None))
    if peak:
        err = (peak - predicted.total) / predicted.total
        print(f"  vram    predicted {predicted.total / GIB:6.3f} GiB  "
              f"measured peak {peak / GIB:6.3f} GiB  "
              f"({err:+.1%})")
    for n in res.notes:
        print(f"  ! {n}")

    if args.record:
        with Ledger(args.db) as led:
            led.record(res)
        print("recorded")
    return 0
