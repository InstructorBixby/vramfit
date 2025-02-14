# notes

## the quant name does not tell you the bits per weight

Downloaded an 8B at Q4_K_M, 4.58 GiB on disk. The table said 4.5 bpw,
which predicts 4.20 GiB for 8.03e9 parameters. That is 9% low, and 9%
is the difference between fitting and not fitting on a 16 GiB card at a
useful context.

The `_S` / `_M` / `_L` suffixes are not quant types. They are *mixes*.
llama.cpp quantises most tensors to the named type and bumps a few to
something larger -- the output tensor, and `attn_v` / `ffn_down` in the
outer layers. The effective bits per weight therefore depends on the
architecture the mix was applied to.

Fix: do not have a bpw table at all. Walk the tensor-info records,
multiply each tensor's element count by its own type's block ratio, add
up. Block geometry is exact and does not depend on the file.
`ggml_types.BLOCK` is that table.

On the 8B above this gives 4.83 bpw and 4.52 GiB of weights against
4.58 GiB of file -- the remainder is the header and the alignment
padding, which is the right size for being ignored.

## kv cache: kv heads, not attention heads

Got this wrong first time and the wrongness was invisible, because an
over-estimate still produces the sentence "it does not fit" and you
believe it.

    kv_bytes = 2 * n_layers * n_head_kv * head_dim * context * dtype

The 2 is K and V. The head count is `n_head_kv`. Under grouped-query
attention several query heads share one K/V projection, so the cache
holds `n_head_kv x head_dim` per position, not `n_head x head_dim`.
Multi-query attention is the extreme case: `n_head_kv` is 1.

Size of the error, with f16 cache:

| model | n_head | n_head_kv | per token, right | per token, wrong |
|---|---|---|---|---|
| 8B, 32 layers | 32 | 8 | 128 KiB | 512 KiB |
| 7B, 28 layers | 28 | 4 | 56 KiB | 392 KiB |
| 1.1B MQA, 24 layers | 32 | 1 | 6 KiB | 192 KiB |

At 32k context the middle row is 1.75 GiB against 12.25 GiB. That is
not a rounding error, it is the whole answer.

Pre-GQA models have no `head_count_kv` key at all, and for those
`n_head_kv == n_head`, which is why the reader defaults it that way
rather than to 1.

## where the five budget terms come from

| term | how it is obtained | how wrong it is |
|---|---|---|
| weights | sum over the tensor table | exact |
| kv cache | arithmetic above | exact for a contiguous cache |
| compute buffer | batch x vocab x 4, plus hidden terms | rough, backend-specific |
| cuda context | per-process constant | currently a guess, needs measuring |
| headroom | 3% of the rest | a policy |

Two of those five are honest numbers. The compute buffer and the CUDA
context are the ones to attack next, in that order, because they are
the ones that make a 16 GiB card fail on a prediction that said it had
half a gigabyte spare.
