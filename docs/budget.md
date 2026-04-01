# the VRAM budget

Five terms, kept separate on purpose. When a prediction misses, the only
useful question is which term missed, and a single number cannot answer
it.

## weights

    sum over the tensor table of ceil(elements / block_weights) * block_bytes

Exact. The block geometry is in the ggml type definitions and does not
depend on the file:

| type | weights/block | bytes/block | bpw |
|---|---|---|---|
| q4_0 | 32 | 18 | 4.5 |
| q8_0 | 32 | 34 | 8.5 |
| q2_K | 256 | 84 | 2.625 |
| q3_K | 256 | 110 | 3.4375 |
| q4_K | 256 | 144 | 4.5 |
| q5_K | 256 | 176 | 5.5 |
| q6_K | 256 | 210 | 6.5625 |
| iq4_xs | 256 | 136 | 4.25 |

The `_S` / `_M` / `_L` suffixes on a GGUF filename are mixes of these,
not types, which is why the effective figure for a Q4_K_M file is nearer
4.83 than 4.5.

## kv cache

    2 * layers * kv_heads * head_dim * context * dtype_bytes

The 2 is K and V. `kv_heads` is `attention.head_count_kv`, which is
absent on pre-GQA models and equal to the attention head count there.
`dtype_bytes` is not an integer for a quantised cache: `q8_0` is 34
bytes per 32 elements, 8.5 bits.

Under partial offload the cache for a layer lives wherever the layer
lives, so this is multiplied by the resident fraction.

## compute buffer

    batch * vocab * 4  +  batch * n_embd * 4 * 6

Rough, backend-specific, and the term to suspect first.

## cuda context

A per-process allocation that exists before a weight is loaded, and not
a constant: measured here at 304 MiB on one card, 268 on another, and it
moved to 322 after a driver upgrade. `vramfit calibrate` measures it and
caches it per (device, driver). Without a measurement the figure is 350
MiB and the output says it is assumed.

## headroom

3% of the rest, for fragmentation. A policy. See
[where this is wrong](wrong.md).
