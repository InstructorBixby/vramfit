"""Print what the GGUF header says. Mostly a debugging aid for gguf.py."""
import sys

import gguf


def main():
    path = sys.argv[1]
    version, meta, tensors = gguf.read_header(path)
    print(f"gguf v{version}  {len(meta)} kv  {len(tensors)} tensors")
    for k in sorted(meta):
        v = meta[k]
        if isinstance(v, list):
            v = f"[{len(v)} items]"
        print(f"  {k:48} {v}")
    print()
    for t in tensors[:6]:
        print(f"  {t['name']:36} {t['dims']} type={t['type']}")
    if len(tensors) > 6:
        print(f"  ... {len(tensors) - 6} more")


if __name__ == "__main__":
    main()
