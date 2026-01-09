# OOM at generation time

How it goes: `vramfit fit` said it fits with 0.6 GiB spare, the model
loaded, the first few replies were fine, and forty minutes into a
conversation the backend died with an allocation failure. Twice on the
3090 before I worked out it was not a bug in the prediction.

Nothing was wrong with the prediction. It answered the question it was
asked, which was about a 4096-token context, and by then the
conversation was longer than that.

## why this is the default experience

The cache is the only term that grows during a run, and it grows
linearly with the conversation. Three ways that turns into a crash:

1. **`-c` larger than the budgeted context.** llama.cpp preallocates
   for `-c`, so this one actually fails at load -- which is the good
   case, because it fails immediately and visibly.
2. **A server that grows the cache per slot.** With `--parallel N` the
   cache is `N` times the context, allocated up front, and the flag is
   easy to forget when reading a fit result.
3. **Fragmentation after hours of churn.** The free figure is a total,
   not the largest contiguous block. A 600 MiB allocation can fail with
   900 MiB free.

The third is why the headroom term exists and why it is a policy rather
than a prediction. 3% has been enough here; it is not a law.

## what the tool does about it

- `fit` defaults to the model's trained context, not 4096, when no
  `--context` is given. The old default made the optimistic answer the
  easy one to get.
- `--reserve` takes a figure to hold back, for the allocator and for
  anything else sharing the card (a display, another process).
- A fit result whose slack is under 5% of the card says so. Slack that
  small is inside the error bars of the compute-buffer term and should
  not be read as "it fits".
- `bench` samples peak used memory during the run and records it, so the
  prediction can be checked against a measurement rather than against a
  recollection.

None of that prevents the failure. It makes the margin visible, which
is the most a prediction can honestly do.
