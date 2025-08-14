# vramfit

Will this model fit in this card, and roughly how fast will it be.

Both halves with arithmetic over the model's own metadata, then a
measurement to find out how wrong the arithmetic was.

## install

    pip install -e ".[dev]"
    pip install -e ".[nvml]"   # optional, nvidia-smi works too

## use

    vramfit devices
    vramfit calibrate

    # will it fit
    vramfit fit models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --context 8192 --vram 16GiB

    # what is the most context i can have
    vramfit fit models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --max-context --vram 16GiB

    # i need 32k, what do i have to download
    vramfit fit models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --for-context 32768 --vram 16GiB

    # quant against max context, all rungs
    vramfit ladder ./Qwen2.5-32B-Instruct --vram 24GiB

    # and now measure it
    vramfit bench models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --context 8192

`fit` exits 1 when it does not fit, so it works in a script.

## what it computes

Five VRAM terms, kept separate on purpose, because when a prediction
misses the only useful question is which term missed.

| term | source |
|---|---|
| weights | sum over the tensor table, exact |
| kv cache | `2 x layers x kv_heads x head_dim x context x dtype` |
| compute buffer | batch x vocab x 4 plus hidden terms, rough |
| cuda context | measured by `vramfit calibrate`, cached per driver |
| headroom | 3% for fragmentation, a policy |

And two speeds, which are different problems:

| measurement | bound by | ceiling |
|---|---|---|
| prompt processing | arithmetic | `peak_flops / (2 x params)` |
| token generation | memory bandwidth | `peak_bw / bytes_per_token` |

The fraction of the ceiling is the number worth reading. 80% of peak
bandwidth on decode means the only remaining lever is reading fewer
bytes per token; 35% means the backend or the build is leaving half the
card idle. Absolute tok/s cannot distinguish those.

## two details that make the numbers trustworthy

**Bits per weight comes from the tensor table.** A Q4_K_M file is about
4.83 bpw, not 4.5: the `_M` suffix is a *mix*, where the output tensor
and some `attn_v` / `ffn_down` layers are quantised heavier. Using 4.5
under-predicts an 8B by nearly half a gigabyte.

**The cache is sized with `n_head_kv`.** Under grouped-query attention
that is a quarter of the attention head count, and under multi-query
attention it is 1. Using `n_head` over-predicts the cache by the GQA
ratio, which is enough to tell you a model does not fit when it fits
four times over.

## where this is wrong

- **Fragmentation is a policy, not a prediction.** The 3% headroom is a
  number that has worked here. An allocator that has been churning for
  an hour behaves differently from a fresh process.
- **The compute buffer is backend-specific** and only llama.cpp's shape
  of it is modelled. A backend that preallocates its cache in blocks,
  or batches more aggressively, will not match.
- **It predicts load, not life.** The cache term is for a full context.
  A process that loads happily can still die later, because the cache
  grows as the conversation does.
- **Speculative decoding breaks the decode ceiling** -- it is derived
  from one forward pass per token, which is exactly the assumption a
  draft model removes. The ratio becomes meaningless, not merely
  inaccurate.
- **Attention FLOPs are not in the prefill ceiling**, so long-context
  prefill ceilings are optimistic.
- **`data/gpus.csv` is vendor specifications.** Used only as
  denominators; an unknown card gets no ratio rather than a plausible
  one.
