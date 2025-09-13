"""vramfit entry point."""
from __future__ import annotations

import argparse

from vramfit import (budget, cmd_bench, cmd_ledger, devices, load,
                     measure, quants, solve, specs)

GIB = 1024 ** 3
MIB = 1024 ** 2


def parse_size(text: str) -> int:
    t = text.strip().lower().replace(" ", "")
    for suffix, mult in (("gib", GIB), ("mib", MIB),
                         ("gb", 10 ** 9), ("mb", 10 ** 6),
                         ("g", GIB), ("m", MIB)):
        if t.endswith(suffix):
            return int(float(t[:-len(suffix)]) * mult)
    return int(float(t))


def human_ctx(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def pick_device(index: int = 0):
    for d in devices.discover():
        if d.index == index:
            return d
    return None


def resolve_vram(args, dev) -> int | None:
    if getattr(args, "vram", None):
        return parse_size(args.vram)
    if getattr(args, "free", False) and dev is not None:
        return dev.free
    return dev.total if dev is not None else None


def cuda_context_for(dev) -> int | None:
    if dev is None:
        return None
    return measure.cached_cuda_context(dev.index, dev.name, dev.driver)


def weights_override(sh, args) -> int | None:
    """None when the file told us, a number when it could not."""
    if getattr(args, "quant", None):
        q = quants.lookup(args.quant)
        if q is None:
            raise SystemExit(f"unknown quant {args.quant!r}; see "
                             f"`vramfit quants`")
        return int(sh.params * q.bpw / 8)
    if sh.weight_bytes:
        return None
    raise SystemExit(
        "no weight files behind this metadata, so the weight term is "
        "unknown: pass --quant to estimate it")


def cmd_shape(args) -> int:
    sh, how = load.load(args.model)
    print(f"name          {sh.name}")
    print(f"source        {how}")
    print(f"arch          {sh.arch}")
    if sh.params:
        print(f"params        {sh.params / 1e9:.3f}B")
        print(f"bpw           {sh.bpw:.3f}")
    print(f"layers        {sh.n_layers}")
    print(f"heads         {sh.n_head} attention / {sh.n_head_kv} kv "
          f"(gqa {sh.gqa_ratio:.0f}x)")
    print(f"head_dim      {sh.head_dim}")
    print(f"vocab         {sh.vocab}")
    print(f"train context {sh.train_context}")
    print(f"kv per token  "
          f"{budget.kv_bytes_per_token(sh) / 1024:.0f} KiB (f16)")
    if sh.params:
        print()
        for name, nbytes in sh.quant_mix().items():
            print(f"  {name:8} {nbytes / GIB:7.3f} GiB")
    return 0


def cmd_fit(args) -> int:
    sh, how = load.load(args.model)
    dev = pick_device(args.device)
    vram = resolve_vram(args, dev)
    kw = {"cuda_context": cuda_context_for(dev),
          "cache_dtype_bytes": args.cache_bytes,
          "weights_override": weights_override(sh, args)}

    label = f"{sh.name}  {sh.params / 1e9:.2f}B" if sh.params \
        else f"{sh.name}"
    print(f"{label}  ({how})")
    if dev is not None and not args.vram:
        print(f"against {dev.name}")
    print()

    if args.max_context:
        if vram is None:
            print("need --vram or a visible device")
            return 2
        ctx = solve.max_context(sh, vram, **kw)
        if not ctx:
            print(f"does not fit at any context in "
                  f"{vram / GIB:.1f} GiB")
            return 1
        print(f"largest context that fits: {ctx} ({human_ctx(ctx)})")
        print()
        print(budget.build(sh, ctx, **kw).table())
        return 0

    if args.for_context:
        if vram is None:
            print("need --vram or a visible device")
            return 2
        kw.pop("weights_override")
        q, b = solve.quant_for_context(sh, args.for_context, vram, **kw)
        if q is None:
            print(f"{human_ctx(args.for_context)} does not fit in "
                  f"{vram / GIB:.1f} GiB at any quant on the ladder")
            return 1
        print(f"{human_ctx(args.for_context)} context needs "
              f"{q.name} or lighter ({q.bpw:.2f} bpw, {q.source})")
        print()
        print(b.table())
        return 0

    b = budget.build(sh, args.context, **kw)
    print(f"ctx {args.context}")
    print(b.table())
    if vram is None:
        return 0
    slack = vram - b.total
    print()
    print(f"{'fits' if slack >= 0 else 'does not fit'}: "
          f"{slack / GIB:+.2f} GiB against {vram / GIB:.1f} GiB")
    return 0 if slack >= 0 else 1


def cmd_ladder(args) -> int:
    sh, _ = load.load(args.model)
    dev = pick_device(args.device)
    vram = resolve_vram(args, dev)
    if vram is None:
        print("need --vram or a visible device")
        return 2
    kw = {"cuda_context": cuda_context_for(dev),
          "cache_dtype_bytes": args.cache_bytes}
    print(f"{sh.name}  in {vram / GIB:.1f} GiB")
    print()
    print(f"{'quant':16} {'bpw':>5}  {'weights':>9}  {'max ctx':>9}")
    for q, ctx in solve.context_ladder(sh, vram, **kw):
        w = sh.params * q.bpw / 8
        print(f"{q.name:16} {q.bpw:5.2f}  {w / GIB:7.2f} GiB  "
              f"{human_ctx(ctx):>9}")
    return 0


def cmd_quants(args) -> int:
    print(f"{'name':18} {'bpw':>5}  source")
    for q in sorted(quants.LADDER, key=lambda q: q.bpw):
        print(f"{q.name:18} {q.bpw:5.2f}  {q.source}")
    return 0


def cmd_devices(args) -> int:
    found = devices.discover()
    if not found:
        print("no CUDA devices visible")
        return 1
    table = specs.load()
    for d in found:
        print(d)
        spec = table.get(devices.normalise(d.name))
        if spec is None:
            print("    no spec row: bandwidth ratios unavailable")
        else:
            print(f"    peak {spec.bandwidth_gbs:.1f} GB/s, "
                  f"{spec.fp16_tflops:.1f} TFLOP/s fp16 "
                  f"({spec.source})")
        if d.driver:
            print(f"    driver {d.driver}")
        ctx = measure.cached_cuda_context(d.index, d.name, d.driver)
        if ctx:
            print(f"    cuda context {ctx / MIB:.0f} MiB (measured)")
    return 0


def cmd_calibrate(args) -> int:
    dev = pick_device(args.device)
    if dev is None:
        print(f"device {args.device} not visible")
        return 1
    value = measure.measure_cuda_context(dev.index)
    measure.record_cuda_context(value, dev.name, dev.driver)
    print(f"{dev.name} driver {dev.driver}: "
          f"cuda context {value / MIB:.0f} MiB")
    print(f"recorded in {measure.cache_path()}")
    return 0


def add_target_args(p) -> None:
    p.add_argument("--vram", help="e.g. 16GiB; default is the device")
    p.add_argument("--free", action="store_true",
                   help="budget against free memory, not total")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cache-bytes", type=int, default=2,
                   choices=(1, 2), help="kv cache dtype size")
    p.add_argument("--quant",
                   help="estimate weights at this scheme instead of "
                        "reading them")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vramfit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("shape", help="what the metadata says")
    p.add_argument("model")
    p.set_defaults(fn=cmd_shape)

    p = sub.add_parser("fit", help="will it fit")
    p.add_argument("model")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--max-context", action="store_true",
                   help="largest context that fits")
    p.add_argument("--for-context", type=int,
                   help="which quant this context needs")
    add_target_args(p)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("ladder", help="quant against max context")
    p.add_argument("model")
    add_target_args(p)
    p.set_defaults(fn=cmd_ladder)

    cmd_bench.add_parser(sub)
    cmd_ledger.add_parser(sub)

    p = sub.add_parser("quants", help="the bpw ladder")
    p.set_defaults(fn=cmd_quants)

    p = sub.add_parser("devices", help="what is in the box")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("calibrate",
                       help="measure this machine's cuda context cost")
    p.add_argument("--device", type=int, default=0)
    p.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
