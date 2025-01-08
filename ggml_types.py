"""ggml type ids, block geometry, and the bpw that falls out of it.

The ids are the on-disk enum values. The geometry is the struct
definition for each block type, which is where the real bits-per-weight
comes from -- there is nothing to estimate.
"""

# id -> name
TYPES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
    18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
    22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32",
    27: "i64", 28: "f64", 29: "iq1_m", 30: "bf16",
}

# name -> (weights per block, bytes per block)
BLOCK = {
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "f64": (1, 8),
    "i8": (1, 1),
    "i16": (1, 2),
    "i32": (1, 4),
    "i64": (1, 8),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q8_1": (32, 36),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "q8_k": (256, 292),
    "iq1_s": (256, 50),
    "iq1_m": (256, 56),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq2_s": (256, 82),
    "iq3_xxs": (256, 98),
    "iq3_s": (256, 110),
    "iq4_xs": (256, 136),
    "iq4_nl": (32, 18),
}


def name(type_id):
    try:
        return TYPES[type_id]
    except KeyError:
        raise KeyError(f"unknown ggml type id {type_id}") from None


def bpw(type_name):
    n, nbytes = BLOCK[type_name]
    return nbytes * 8 / n


def tensor_bytes(elements, type_name):
    n, nbytes = BLOCK[type_name]
    if elements % n:
        # ggml pads the last block. Round up rather than lie low.
        return ((elements + n - 1) // n) * nbytes
    return elements // n * nbytes
