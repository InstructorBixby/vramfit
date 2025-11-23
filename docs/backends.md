# backends, and why the budget is not portable

The five-term budget describes llama.cpp. Other backends allocate
differently enough that applying it unchanged produces a number that
looks right and is not.

## llama.cpp

Allocates the cache for the full `-c` up front, in one contiguous
tensor per layer, and the compute buffer from the batch. So "does it
fit" is a question about load time and the answer is stable for the
whole run. This is the easy case and it is the one the budget models.

## vLLM

Does not take a VRAM figure, it takes a *fraction*:
`gpu_memory_utilization`, default 0.9. On startup it profiles a forward
pass, subtracts the weights and the profiled activations from that
fraction of the card, and turns everything left into KV blocks. So:

- It will not fail to load for want of cache. It will load with fewer
  blocks, and then refuse or queue requests whose context does not fit
  in a block budget it never told you about.
- The cache is paged, in blocks of 16 tokens by default, so a sequence
  costs a whole number of blocks and the per-request arithmetic needs
  a ceiling, not a product.
- Prefix caching makes blocks shared between sequences, so total cache
  use is not the sum of the per-sequence costs.

The useful prediction for vLLM is therefore not "will it fit" but "how
many concurrent sequences of length N will the block budget hold", which
is a different function with a different answer. It is implemented as
its own profile rather than by bending the llama.cpp one.

## exllamav2

Closer to llama.cpp -- the cache is preallocated for a declared max
sequence length -- with two differences that matter:

- Its own cache quantisation (`Q4`, `Q6`, `Q8` cache modes) with its own
  byte costs, which are not the ggml block sizes.
- Paged attention when batching, which changes the cache from a product
  into a block count the way vLLM's does.

## what this means for the ledger

A result row carries the backend, so numbers from two backends never get
compared by accident. It is not a fair comparison in either direction:
they are not doing the same amount of work, do not use the same kernels,
and in the vLLM case are not even answering the same question.
