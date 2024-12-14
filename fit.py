"""How much VRAM a quantised model wants: weights plus KV cache."""
import argparse

GIB = 1024 ** 3

BPW = {
    "f32": 32.0,
    "f16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.5625,
    "q5_k_m": 5.5,
    "q4_k_m": 4.5,
    "q4_0": 4.5,
    "q3_k_m": 3.4375,
    "q2_k": 2.625,
}


def weights_bytes(params, quant):
    return params * BPW[quant] / 8


def kv_bytes(n_layers, n_heads, head_dim, context, dtype_bytes=2):
    # 2 for K and V.
    return 2 * n_layers * n_heads * head_dim * context * dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True)
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    args = ap.parse_args()

    w = weights_bytes(args.params, args.quant)
    kv = kv_bytes(args.layers, args.heads, args.head_dim, args.context)
    print(f"weights     {w / GIB:7.2f} GiB")
    print(f"kv cache    {kv / GIB:7.2f} GiB  ({args.context} ctx)")
    print(f"total       {(w + kv) / GIB:7.2f} GiB")
    print(f"+20% slop   {(w + kv) * 1.2 / GIB:7.2f} GiB")


if __name__ == "__main__":
    main()
