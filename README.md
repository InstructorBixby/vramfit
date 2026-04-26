# vramfit

**Will this model fit in this card, and roughly how fast will it be.**

Answered with arithmetic over metadata the model file already contains,
instead of by downloading five gigabytes and watching what happens.
Then measured, so the arithmetic can be checked and the error kept.

## how I run it

    uv pip install -e .
    uv pip install -e ".[nvml]"   # optional; nvidia-smi works too

Editable, from the checkout. On the GPU box and the work laptop it is
the same, which is the only reason a ledger row from one is comparable
to a row from the other.

## the commands

    vramfit devices
    vramfit calibrate                  # once per machine

    vramfit fit  ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --vram 16GiB
    vramfit fit  ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --max-context
    vramfit fit  ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --for-context 32768
    vramfit ladder ./Qwen2.5-32B-Instruct --vram 24GiB
    vramfit explain ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --context 8192
    vramfit bench ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --record

`fit` exits 1 when it does not fit, so it drops straight into a
script, which is how most of my use of it actually happens. `--json`
on the prediction subcommands gives a flat schema with every number in
bytes.

## worked example

An 8B at Q4_K_M, on a 16 GiB card, and the question is how much context
is actually available.

    $ vramfit fit ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --max-context --vram 16GiB
    Meta-Llama-3.1-8B-Instruct  8.03B  4.83 bpw  (gguf header)

    largest context that fits: 55296 (54k)

    weights             4.520 GiB
    kv cache            6.750 GiB
    compute buffer      0.296 GiB
    cuda context        0.297 GiB
    headroom            0.356 GiB
    total              12.219 GiB

Two things to notice. The cache is bigger than the weights -- past about
35k context on this model it is the dominant term, and nothing about the
quant choice changes that. And the quant is 4.83 bits per weight, not
4.5, because `_M` is a mix.

Cutting the cache to q8_0 buys most of the rest of the window:

    $ vramfit fit ./Llama-3.1-8B-Instruct-Q4_K_M.gguf \
          --max-context --vram 16GiB --cache-type q8_0 --flash-attn
    largest context that fits: 101888 (99k)

Then check it:

    $ vramfit bench ./Llama-3.1-8B-Instruct-Q4_K_M.gguf --context 8192
    Meta-Llama-3.1-8B-Instruct q4_k_m ctx 8192 llama.cpp on NVIDIA GeForce RTX 3090
      prefill   2914.3 tok/s  (25% of flops peak)
      decode     118.5 tok/s  (82% of bandwidth peak, 768 GB/s)
      vram    predicted  5.414 GiB  measured peak  5.502 GiB  (+1.6%)

82% of peak memory bandwidth on decode means the backend is doing about
as well as the hardware allows and the only remaining lever is reading
fewer bytes per token. That is the output worth having: absolute tok/s
cannot tell "the card is saturated" from "half the card is idle".

## why not just do it by hand

I did, for months. The back-of-envelope is `params x bits / 8`, plus a
fudge factor, plus a guess at the cache. Where that went wrong:

| | by hand | vramfit |
|---|---|---|
| bits per weight | 4.0 or 4.5 from the quant name | 4.83, summed over the tensor table |
| kv cache heads | attention heads | `head_count_kv` -- 4x smaller here, 32x on MQA |
| cache dtype | assumed f16 | q8_0 costs 8.5 bits, not 8 |
| cuda context | in the fudge factor | measured per (device, driver) |
| partial offload | weights scaled by layer fraction | cache scaled too, embeddings not |
| speed | one tok/s number | prefill and decode, each against its own ceiling |

On the example above, the by-hand version lands at about 4.2 GiB of
weights against 4.52 real, and over-predicts the cache by 4x. Those two
errors point in opposite directions, which is why the fudge factor
appeared to work for months and then did not, on the one afternoon I
had the rented card for.

## where this is wrong

The honest list, in full, is [docs/wrong.md](docs/wrong.md). The short
version:

- **Fragmentation is a policy, not a prediction.** The 3% headroom is a
  number that has worked here; free memory is a total, not the largest
  contiguous block.
- **The compute buffer is backend-specific** and only llama.cpp's shape
  of it is modelled.
- **It predicts load, not life.** The cache grows with the
  conversation, so a model that loads can still die an hour later.
- **Speculative decoding invalidates the decode ceiling** -- it removes
  the one-forward-pass-per-token assumption the ceiling rests on.
- **Attention FLOPs are not in the prefill ceiling**, so long-context
  prefill ceilings are optimistic.
- **`data/gpus.csv` is vendor numbers**, used only as denominators. An
  unknown card gets no ratio rather than a plausible one.
- **The Windows path is only ever checked by hand.** There is no
  Windows runner on the instance, so the `nvidia-smi` fallback and the
  ledger's default path there are as good as whenever I last looked.

## layout

| path | what |
|---|---|
| `vramfit/api.py` | the public API; everything else is internal |
| `vramfit/gguf.py`, `safetensors.py`, `hfconfig.py` | metadata readers |
| `vramfit/ggml_types.py` | block geometry, and therefore exact bpw |
| `vramfit/budget.py` | the five VRAM terms |
| `vramfit/solve.py` | invert it: max context, min quant, max `-ngl` |
| `vramfit/roofline.py` | the two ceilings |
| `vramfit/bench/` | one module per backend |
| `vramfit/ledger.py` | measurements, kept and diffable |

## licence

MIT. See [LICENSE](LICENSE).
