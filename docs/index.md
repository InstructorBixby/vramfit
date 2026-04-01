# vramfit

Will this model fit in this card, and roughly how fast will it be.

Two questions that normally get answered by downloading five gigabytes
and watching what happens. Both are arithmetic over metadata the model
file already contains, so they can be answered in a tenth of a second,
and then measured so the arithmetic can be checked.

    uv pip install -e .

    vramfit fit ./llama-3.1-8b-instruct-q4_k_m.gguf --vram 16GiB
    vramfit fit ./llama-3.1-8b-instruct-q4_k_m.gguf --max-context
    vramfit bench ./llama-3.1-8b-instruct-q4_k_m.gguf --record

## the short version of why this is not trivial

Two numbers that almost everybody gets wrong by enough to matter.

**Bits per weight is not the quant name.** `Q4_K_M` is a *mix*: most
tensors at Q4_K, the output tensor and some `attn_v` / `ffn_down`
layers at Q6_K. The effective figure is around 4.83 bits, not 4.5, and
it depends on the architecture the mix was applied to. vramfit reads the
per-tensor types out of the GGUF tensor table and sums them, so there is
nothing to estimate.

**The KV cache is sized by KV heads.** Under grouped-query attention
several query heads share one K/V projection, so the cache holds
`n_head_kv x head_dim` per position -- a quarter of the attention head
count on a typical 8B, and one under multi-query attention. Using
`n_head` over-predicts the cache by the GQA ratio, which is enough to
report that a model does not fit when it fits four times over.

## what it will not do

See [where this is wrong](wrong.md), which is the page to read before
trusting a number from here in anything that matters.
