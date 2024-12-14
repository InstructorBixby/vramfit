"""How much VRAM do the weights of a quantised model want."""
import argparse

GIB = 1024 ** 3

# Bits per weight. These are the nominal numbers off the quant name;
# see README for why that is not good enough for the K-quants.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True)
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()

    w = weights_bytes(args.params, args.quant)
    print(f"weights     {w / GIB:7.2f} GiB")
    print(f"+20% slop   {w * 1.2 / GIB:7.2f} GiB")


if __name__ == "__main__":
    main()
