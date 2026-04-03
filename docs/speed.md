# the two speeds

Every "tokens per second" figure quoted for a local model is one of two
completely different measurements averaged together, and averaging them
is what makes the number useless for deciding anything.

## prefill: compute bound

Processing the prompt runs the whole context through the model in large
batches. Every weight gets loaded once and used for hundreds of
positions, so arithmetic dominates and the ceiling is FLOP/s:

    flops_per_token ~ 2 * params
    prefill_ceiling_tok_s = peak_fp16_flops / (2 * params)

The 2 is one multiply and one add per parameter. Attention adds a term
that is quadratic in context and is small below about 8k, so it is
ignored here and the limitation is written down rather than hidden.

## decode: memory bandwidth bound

Generating a token does one forward pass for one position. Every weight
is read from VRAM and used exactly once, so arithmetic is irrelevant and
the ceiling is bandwidth:

    bytes_per_token = weight_bytes + kv_bytes(context)
    decode_ceiling_tok_s = peak_bandwidth / bytes_per_token

This is why a 4-bit quant generates roughly twice as fast as an 8-bit
one of the same model: it is half as many bytes to read, not half as
much maths.

It is also why the KV cache term matters for speed and not only for
capacity. At 32k context on an 8B the cache is 4 GiB against 4.5 GiB of
weights, so decode at a full context is near half the speed of decode at
an empty one, with nothing having gone wrong.

## the ratio is the number worth having

    achieved = measured_tok_s * bytes_per_token
    fraction = achieved / peak_bandwidth

A decode loop reaching 75-85% of peak bandwidth is doing about as well
as the hardware allows, and the only remaining lever is reading fewer
bytes. One at 40% is leaving half the card unused and the lever is the
backend, the build flags or the batch.

Absolute tok/s cannot tell those apart, which is the whole argument for
reporting the fraction.

## what this does not model

- Attention FLOPs, so long-context prefill ceilings are optimistic.
- Prompt caching, which makes the prefill number conditional on what
  was in the cache.
- Speculative decoding, which breaks the "one forward pass per token"
  assumption that the decode ceiling rests on entirely.
- Batched serving. All of the above is batch size 1, which is the
  interactive case and not the throughput case.

## see also

- [the VRAM budget](budget.md), since the cache term drives both the
  capacity answer and the decode ceiling.
- [the ledger](ledger.md), which is where a measured fraction goes so
  that it can be compared to the next one.
- [where this is wrong](wrong.md).
