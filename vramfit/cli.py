"""vramfit entry point."""
from __future__ import annotations

import argparse

from vramfit import budget, devices, gguf, specs
from vramfit import shape as shape_mod

GIB = 1024 ** 3


def parse_size(text: str) -> int:
    t = text.strip().lower().replace(" ", "")
    for suffix, mult in (("gib", GIB), ("mib", 1024 ** 2),
                         ("gb", 10 ** 9), ("mb", 10 ** 6),
                         ("g", GIB), ("m", 1024 ** 2)):
        if t.endswith(suffix):
            return int(float(t[:-len(suffix)]) * mult)
    return int(float(t))


def load_shape(path: str):
    _, meta, tensors = gguf.read_header(path)
    return shape_mod.from_gguf(meta, tensors, path)


def cmd_shape(args) -> int:
    sh = load_shape(args.model)
    print(f"name          {sh.name}")
    print(f"arch          {sh.arch}")
    print(f"params        {sh.params / 1e9:.3f}B")
    print(f"bpw           {sh.bpw:.3f}")
    print(f"layers        {sh.n_layers}")
    print(f"heads         {sh.n_head} attention / {sh.n_head_kv} kv "
          f"(gqa {sh.gqa_ratio:.0f}x)")
    print(f"head_dim      {sh.head_dim}")
    print(f"vocab         {sh.vocab}")
    print(f"train context {sh.train_context}")
    print()
    for name, nbytes in sh.quant_mix().items():
        print(f"  {name:8} {nbytes / GIB:7.3f} GiB")
    return 0


def cmd_fit(args) -> int:
    sh = load_shape(args.model)
    b = budget.build(sh, args.context)
    print(f"{sh.name}  {sh.params / 1e9:.2f}B  {sh.bpw:.2f} bpw  "
          f"ctx {args.context}")
    print()
    print(b.table())

    vram = parse_size(args.vram) if args.vram else None
    if vram is None:
        found = devices.discover()
        if found:
            vram = found[0].total
            print(f"\nagainst {found[0].name}")
    if vram is None:
        return 0
    slack = vram - b.total
    print()
    print(f"{'fits' if slack >= 0 else 'does not fit'}: "
          f"{slack / GIB:+.2f} GiB against {vram / GIB:.1f} GiB")
    return 0 if slack >= 0 else 1


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
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vramfit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("shape", help="what the metadata says")
    p.add_argument("model")
    p.set_defaults(fn=cmd_shape)

    p = sub.add_parser("fit", help="will it fit")
    p.add_argument("model")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--vram", help="e.g. 16GiB; default is device 0")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("devices", help="what is in the box")
    p.set_defaults(fn=cmd_devices)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
