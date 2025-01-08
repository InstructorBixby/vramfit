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

Fix: do not have a bpw table at all. Walk the tensor-info records,
multiply each tensor's element count by its own type's block ratio, add
up. Block geometry is exact, it is in the ggml type definitions, and it
does not depend on the file. `ggml_types.BLOCK` is that table.

On the 8B above this gives 4.83 bpw and 4.52 GiB of weights, against
4.58 GiB of file -- the remainder is the header and the alignment
padding, which is the right size for being ignored.

## where the KV cache number comes from

Two tensors per layer (K and V), each `context x n_kv_heads x head_dim`
elements, at the cache dtype's size. The head count is the one that
people get wrong, including me -- see below.
