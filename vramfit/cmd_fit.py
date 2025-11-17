"""The fit subcommand.

Lifted out of cli.py once it had four mutually exclusive modes and the
dispatch was longer than the work.
"""
from __future__ import annotations

from vramfit import budget, cache, devices, load, measure, quants, solve

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


def budget_kwargs(args, dev) -> dict:
    if cache.lookup(args.cache_type) is None:
        raise SystemExit(f"unknown cache type {args.cache_type!r}")
    return {"cuda_context": cuda_context_for(dev),
            "cache_dtype": args.cache_type,
            "flash_attn": args.flash_attn}


def add_target_args(p) -> None:
    p.add_argument("--vram", help="e.g. 16GiB; default is the device")
    p.add_argument("--free", action="store_true",
                   help="budget against free memory, not total")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cache-type", default="f16",
                   help="kv cache dtype: f16, q8_0, q4_0 ...")
    p.add_argument("--flash-attn", action="store_true",
                   help="required for a quantised V cache")
    p.add_argument("--quant",
                   help="estimate weights at this scheme instead of "
                        "reading them")


def add_parser(sub):
    p = sub.add_parser("fit", help="will it fit")
    p.add_argument("model")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--max-context", action="store_true",
                   help="largest context that fits")
    p.add_argument("--for-context", type=int,
                   help="which quant this context needs")
    p.add_argument("--n-gpu-layers", type=int,
                   help="partial offload, as llama.cpp's -ngl")
    p.add_argument("--max-layers", action="store_true",
                   help="largest -ngl that fits")
    add_target_args(p)
    p.set_defaults(fn=run)
    return p


def run(args) -> int:
    sh, how = load.load(args.model)
    dev = pick_device(args.device)
    vram = resolve_vram(args, dev)
    kw = budget_kwargs(args, dev)
    kw["weights_override"] = weights_override(sh, args)
    if args.n_gpu_layers is not None:
        kw["n_gpu_layers"] = args.n_gpu_layers

    label = (f"{sh.name}  {sh.params / 1e9:.2f}B" if sh.params
             else sh.name)
    print(f"{label}  ({how})")
    if dev is not None and not args.vram:
        print(f"against {dev.tag}")
    print()

    if args.max_layers:
        if vram is None:
            print("need --vram or a visible device")
            return 2
        n = solve.max_gpu_layers(sh, vram, args.context, **kw)
        if n == sh.n_layers:
            print(f"fits outright at ctx {args.context}: all "
                  f"{n} layers")
        elif n == 0:
            print(f"no layers fit at ctx {args.context}")
            return 1
        else:
            print(f"largest -ngl that fits at ctx {args.context}: "
                  f"{n} of {sh.n_layers}")
        print()
        print(budget.build(sh, args.context, n_gpu_layers=n,
                           **kw).table())
        return 0

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
