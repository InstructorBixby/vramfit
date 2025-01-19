"""Will it fit: read a GGUF, build the budget, print the terms."""
import argparse

import budget
import gguf
import shape as shape_mod

GIB = 1024 ** 3


def parse_size(text):
    t = text.strip().lower()
    for suffix, mult in (("gib", GIB), ("gb", 10 ** 9),
                         ("mib", 1024 ** 2), ("g", GIB)):
        if t.endswith(suffix):
            return int(float(t[:-len(suffix)]) * mult)
    return int(float(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--vram")
    args = ap.parse_args()

    _, meta, tensors = gguf.read_header(args.model)
    sh = shape_mod.from_gguf(meta, tensors, args.model)
    b = budget.build(sh, args.context)

    print(f"{sh.name}  {sh.arch}  {sh.params / 1e9:.2f}B params  "
          f"{sh.bpw:.2f} bpw")
    print(f"{sh.n_layers} layers  {sh.n_head} heads / "
          f"{sh.n_head_kv} kv heads  head_dim {sh.head_dim}")
    print()
    print(b.table())

    if args.vram:
        vram = parse_size(args.vram)
        slack = vram - b.total
        verdict = "fits" if slack >= 0 else "does not fit"
        print()
        print(f"{verdict}: {slack / GIB:+.2f} GiB against "
              f"{vram / GIB:.1f} GiB")


if __name__ == "__main__":
    main()
