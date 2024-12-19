"""How much VRAM a quantised model wants: weights plus KV cache."""
import argparse

GIB = 1024 ** 3

# (weights per block, bytes per block) from the ggml type definitions.
# Exact, unlike a bits-per-weight table.
BLOCK = {
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "q8_k": (256, 292),
}


def bpw(gtype):
    n, nbytes = BLOCK[gtype]
    return nbytes * 8 / n


def weights_bytes(params, gtype):
    return params * bpw(gtype) / 8


def kv_bytes(n_layers, n_heads, head_dim, context, dtype_bytes=2):
    # 2 for K and V.
    return 2 * n_layers * n_heads * head_dim * context * dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True)
    ap.add_argument("--quant", default="q4_k")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    args = ap.parse_args()

    w = weights_bytes(args.params, args.quant)
    kv = kv_bytes(args.layers, args.heads, args.head_dim, args.context)
    print(f"bpw         {bpw(args.quant):7.4f}  (nominal, flat mix)")
    print(f"weights     {w / GIB:7.2f} GiB")
    print(f"kv cache    {kv / GIB:7.2f} GiB  ({args.context} ctx)")
    print(f"total       {(w + kv) / GIB:7.2f} GiB")


if __name__ == "__main__":
    main()
