"""vramfit entry point.

Thin: argument wiring and the small subcommands. Anything with more than
a couple of flags lives in its own module.
"""
from __future__ import annotations

import argparse

from vramfit import (__version__, budget, cmd_accuracy, cmd_bench,
                     cmd_contrib, cmd_fit, cmd_ledger, cmd_plan,
                     devices, explain, load, measure, quants, solve,
                     specs)

GIB = 1024 ** 3
MIB = 1024 ** 2

parse_size = cmd_fit.parse_size
human_ctx = cmd_fit.human_ctx
pick_device = cmd_fit.pick_device
resolve_vram = cmd_fit.resolve_vram
cuda_context_for = cmd_fit.cuda_context_for


def cmd_shape(args: argparse.Namespace) -> int:
    sh, how = load.load(args.model)
    mix = sh.moe
    if getattr(args, "json", False):
        from vramfit import jsonout
        print(jsonout.dump({
            "model": sh.name, "arch": sh.arch, "source": how,
            "params": sh.params,
            "bits_per_weight": round(sh.bpw, 4) if sh.params else None,
            "layers": sh.n_layers, "attention_heads": sh.n_head,
            "kv_heads": sh.n_head_kv, "head_dim": sh.head_dim,
            "key_dim": sh.key_dim, "value_dim": sh.value_dim,
            "vocab": sh.vocab, "trained_context": sh.train_context,
            "kv_bytes_per_token": budget.kv_bytes_per_token(sh),
            "experts": sh.n_experts or None,
            "experts_used": sh.n_experts_used or None,
            "bytes_per_token": (mix.bytes_per_token if mix
                                else sh.weight_bytes),
            "quant_mix": sh.quant_mix() if sh.params else {},
        }))
        return 0
    print(f"name          {sh.name}")
    print(f"source        {how}")
    print(f"arch          {sh.arch}")
    if sh.params:
        print(f"params        {sh.params / 1e9:.3f}B")
        print(f"bpw           {sh.bpw:.3f}")
    print(f"layers        {sh.n_layers}")
    print(f"heads         {sh.n_head} attention / {sh.n_head_kv} kv "
          f"(gqa {sh.gqa_ratio:.0f}x)")
    if sh.asymmetric_kv:
        print(f"head_dim      {sh.head_dim}  "
              f"(key {sh.key_dim} / value {sh.value_dim})")
    else:
        print(f"head_dim      {sh.head_dim}")
    print(f"vocab         {sh.vocab}")
    print(f"train context {sh.train_context}")
    print(f"kv per token  "
          f"{budget.kv_bytes_per_token(sh) / 1024:.0f} KiB (f16)")
    if mix is not None:
        print()
        print(mix.summary())
    if sh.params:
        print()
        for name, nbytes in sh.quant_mix().items():
            print(f"  {name:8} {nbytes / GIB:7.3f} GiB")
    return 0


def cmd_ladder(args: argparse.Namespace) -> int:
    sh, _ = load.load(args.model)
    dev = pick_device(args.device)
    vram = resolve_vram(args, dev)
    if vram is None:
        print("need --vram or a visible device")
        return 2
    kw = cmd_fit.budget_kwargs(args, dev)
    rows = solve.context_ladder(sh, vram, **kw)
    if args.json:
        from vramfit import jsonout
        print(jsonout.dump([
            {"quant": q.name, "bits_per_weight": q.bpw,
             "source": q.source,
             "weight_bytes": int(sh.params * q.bpw / 8),
             "max_context": ctx} for q, ctx in rows]))
        return 0
    print(f"{sh.name}  in {vram / GIB:.1f} GiB  "
          f"({args.cache_type} cache)")
    print()
    print(f"{'quant':16} {'bpw':>5}  {'weights':>9}  {'max ctx':>9}")
    for q, ctx in rows:
        w = sh.params * q.bpw / 8
        print(f"{q.name:16} {q.bpw:5.2f}  {w / GIB:7.2f} GiB  "
              f"{human_ctx(ctx):>9}")
    return 0


def cmd_quants(args: argparse.Namespace) -> int:
    print(f"{'name':18} {'bpw':>5}  source")
    for q in sorted(quants.LADDER, key=lambda q: q.bpw):
        print(f"{q.name:18} {q.bpw:5.2f}  {q.source}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
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
            print("    a pull request adding it is welcome; see "
                  "docs/cards.md")
        else:
            print(f"    peak {spec.bandwidth_gbs:.1f} GB/s, "
                  f"{spec.fp16_tflops:.1f} TFLOP/s fp16 "
                  f"({spec.source})")
        if d.driver:
            print(f"    driver {d.driver}")
        ctx = measure.cached_cuda_context(d.index, d.name, d.driver)
        if ctx:
            print(f"    cuda context {ctx / MIB:.0f} MiB (measured)")
        else:
            stale = measure.stale_entries(d.name, d.driver)
            if stale:
                print(f"    cuda context not measured on this driver; "
                      f"have {len(stale)} entr(y|ies) from another")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vramfit")
    ap.add_argument("--version", action="version",
                    version=f"vramfit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("shape", help="what the metadata says")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_shape)

    cmd_fit.add_parser(sub)
    cmd_plan.add_parser(sub)
    explain.add_parser(sub)

    p = sub.add_parser("ladder", help="quant against max context")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    cmd_fit.add_target_args(p)
    p.set_defaults(fn=cmd_ladder)

    cmd_bench.add_parser(sub)
    cmd_ledger.add_parser(sub)
    cmd_contrib.add_parser(sub)
    cmd_accuracy.add_parser(sub)

    p = sub.add_parser("quants", help="the bpw ladder")
    p.set_defaults(fn=cmd_quants)

    p = sub.add_parser("devices", help="what is in the box")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("calibrate",
                       help="measure this machine's cuda context cost")
    p.add_argument("--device", type=int, default=0)
    p.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.fn(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
