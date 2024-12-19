# notes

## the quant name does not tell you the bits per weight

Downloaded an 8B at Q4_K_M, 4.58 GiB on disk. The table said 4.5 bpw,
which predicts 4.20 GiB for 8.03e9 parameters. That is 9% low, and 9%
is the difference between fitting and not fitting on a 16 GiB card at a
useful context.

The reason is that the `_S` / `_M` / `_L` suffixes are not quant types.
They are *mixes*. llama.cpp quantises most tensors to the named type and
bumps a few to something larger -- the output tensor, and `attn_v` /
`ffn_down` in the outer layers. So the effective bits per weight for a
file depends on the architecture it was applied to.

Block sizes, which *are* exact, from the ggml type definitions:

| type | weights per block | bytes per block | bpw |
|---|---|---|---|
| q4_0 | 32 | 18 | 4.5 |
| q5_0 | 32 | 22 | 5.5 |
| q8_0 | 32 | 34 | 8.5 |
| q2_K | 256 | 84 | 2.625 |
| q3_K | 256 | 110 | 3.4375 |
| q4_K | 256 | 144 | 4.5 |
| q5_K | 256 | 176 | 5.5 |
| q6_K | 256 | 210 | 6.5625 |

So: take the per-tensor types out of the GGUF tensor table, multiply
element counts by the block ratio, and the number comes out exact. No
table of guesses. That is the next thing to build.

## where the KV cache number comes from

Two tensors per layer (K and V), each `context x n_kv_heads x head_dim`
elements, at the cache dtype's size. The head count is the one that
people get wrong, including me -- see below.
