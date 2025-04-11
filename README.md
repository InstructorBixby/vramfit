# vramfit

Will this model fit in this card, and roughly how fast will it be.

Answers the first half with arithmetic over the model's own metadata
instead of loading it to find out.

## install

    pip install -e ".[dev]"
    pip install -e ".[nvml]"   # optional, nvidia-smi works too

## use

    vramfit devices
    vramfit calibrate
    vramfit shape models/llama-3.1-8b-instruct-q4_k_m.gguf
    vramfit fit models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --context 8192 --vram 16GiB

`fit` exits 1 when it does not fit, so it works in a script.

## what it computes

Five terms, kept separate on purpose, because when a prediction misses
the only useful question is which term missed.

| term | source |
|---|---|
| weights | sum over the GGUF tensor table, exact |
| kv cache | `2 x layers x kv_heads x head_dim x context x dtype` |
| compute buffer | batch x vocab x 4 plus hidden terms, rough |
| cuda context | measured by `vramfit calibrate`, cached per driver |
| headroom | 3% for fragmentation, a policy |

Two things it does that the arithmetic people do in their heads does
not. One, the bits per weight come from the tensor table, so a Q4_K_M
file is 4.83 bpw and not 4.5 -- the `_M` suffix is a mix, not a type.
Two, the cache is sized with `n_head_kv`, which under grouped-query
attention is a quarter of the attention head count and under
multi-query attention is one. NOTES.md has both in full.

## where this is wrong

- **Fragmentation is a policy, not a prediction.** The 3% headroom is
  a number that has worked on this machine. An allocator that has been
  churning for an hour behaves differently from a fresh process.
- **The compute buffer is backend-specific** and this only models
  llama.cpp's shape of it. A backend that preallocates its cache in
  blocks, or batches more aggressively, will not match.
- **It predicts load, not life.** The cache term is for a full context.
  A process that loads happily can still die later, because the cache
  grows as the conversation does.
- **The bandwidth and FLOP numbers in `data/gpus.csv` are vendor
  specifications.** They are only ever used as denominators, and an
  unknown card gets no ratio rather than a plausible one.
