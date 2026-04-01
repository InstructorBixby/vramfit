# where this is wrong

Every term in the budget has a different claim to accuracy, and the
totals are only as good as the weakest one in play. In rough order of
how much trouble each has caused:

## fragmentation is a policy, not a prediction

The headroom term is 3% of everything else. That is a number that has
worked on the machines here. The driver's free-memory figure is a
*total*, not the largest contiguous block, so a 600 MiB allocation can
fail with 900 MiB free, and an allocator that has been churning for an
hour behaves differently from a fresh process.

## the compute buffer is backend-specific and modelled roughly

`batch x vocab x 4` plus hidden-state terms is llama.cpp's shape of it.
A backend that batches more aggressively, or preallocates its cache in
blocks, will not match. This is the term to suspect first when a
prediction misses by a few hundred megabytes.

## it predicts load, not life

The cache is the only term that grows during a run. A process that
loads happily can die an hour later because the conversation outgrew the
context the prediction was made for. `--reserve` and the thin-margin
warning make the gap visible; they do not close it.

## speculative decoding invalidates the decode ceiling

The bandwidth ceiling is derived from one forward pass per token, which
is exactly the assumption a draft model removes. With speculation on,
the achieved fraction is not inaccurate, it is meaningless.

## attention FLOPs are not in the prefill ceiling

Ignored because the term is small below roughly 8k context. Above that,
the prefill ceiling is optimistic by a growing margin.

## the vendor table is vendor numbers

`data/gpus.csv` holds advertised peak bandwidth and dense FP16 FLOP/s.
They are used only as denominators, every row carries its source, and a
card with no row gets no ratio rather than a plausible one. A ratio
against an advertised number is not a ratio against a number anybody
has ever achieved.

## partial offload is an approximation of a build detail

Which of the embedding and output tensors end up on the card at a given
`-ngl` depends on the backend build. The model here is "block tensors
scale with the layer count, the rest stays off until everything is on",
and it is marked as an approximation wherever it is used.

## quantised caches need the right flags

A quantised V cache needs flash attention in llama.cpp. Without it the
backend keeps V in f16, so the prediction and the measurement disagree
for a reason that has nothing to do with the arithmetic. vramfit warns;
it cannot check the backend's build flags from here.
