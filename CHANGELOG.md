# Changelog

Format follows Keep a Changelog, versions follow semver.

## [0.7.0]

### Added
- `vramfit explain`: the whole derivation, with every line naming
  its source -- file, vendor table, measurement on this machine,
  or policy. The last two are the ones worth arguing with.
- `--json` on the prediction subcommands. Flat schema, every
  number in bytes, `assumptions` included rather than footnoted.
- `--reserve`, and `--parallel` for server slots, where the cache
  is per slot and the flag is easy to forget.
- `bench` samples peak used memory during the run and prints it
  next to the prediction, so the error is a measurement.
- `docs/oom.md`: why a model that loads can still die later.

### Changed
- `fit --context` defaults to the model's trained context instead
  of 4096. The old default made the optimistic answer the easy one
  to get, and a conversation outgrows 4096 in an afternoon.
- A fit whose slack is under 5% of the card is reported as a
  maybe. That much slack is inside the error bars of the
  compute-buffer term.

## [0.6.0]

### Added
- Backend profiles. The five-term budget describes llama.cpp; the
  others allocate differently enough that applying it unchanged
  gives a number that looks right and is not. `docs/backends.md`.
- vLLM: `vramfit plan`, which answers the question vLLM actually
  raises -- how many concurrent sequences of length N does the
  block budget hold -- rather than a yes/no it will never give.
  Paged cache, so sequences round up to a block, per sequence.
- vLLM bench backend. Prefill comes from time-to-first-token and
  decode from the inter-token intervals, because a non-streamed
  response cannot be separated into the two.
- exllamav2 bench backend, with its own cache-mode byte costs,
  which are not the ggml block sizes.

## [0.5.1]

### Added
- `--cache-type` by name (f16, q8_0, q4_0 ...) with the real block
  geometry behind it: a q8_0 cache is 8.5 bits per element, not 8,
  for the same reason the weights are not 4.
- `--flash-attn`. A quantised V cache needs it in llama.cpp, and
  without it the backend keeps V in f16 -- so the prediction and
  the measurement disagree for a reason unrelated to the
  arithmetic. Now it says so.

### Changed
- `--cache-bytes` is gone, replaced by `--cache-type`.

## [0.5.0]

### Added
- Partial offload. `--n-gpu-layers` to budget a given `-ngl`, and
  `--max-layers` for the largest one that fits.
- `docs/offload.md` on what actually lives where.

### Fixed
- The KV cache under partial offload. The cache for a layer lives
  wherever the layer lives, so offloading 20 of 32 layers puts
  62.5% of the cache on the card -- not all of it, and not none.
  The all-of-it version over-predicted; the none-of-it version
  produced a confident 'it fits' followed by an OOM.
- Embedding and output tensors are no longer scaled with the layer
  count. They do not move until everything does, which is an
  approximation of a build detail and is marked as one.

### Changed
- A partial-offload bench reports tok/s and no bandwidth
  fraction. The bound is the host path, so a fraction against GPU
  bandwidth would be meaningless rather than merely inaccurate.
- `fit` moved into `cmd_fit.py`; `cli.py` is wiring now.

## [0.4.0]

### Added
- A ledger: every measurement kept in sqlite, keyed by (model,
  quant, context, backend, gpu), with driver version and backend
  build as provenance.
- `vramfit bench --record`.
- `vramfit ledger list | show | diff`. The diff reports measured
  changes above 2% and prints the provenance that moved at the
  same time next to them, because a speed change reported without
  the driver change beside it invites the wrong conclusion.
- Markdown and CSV export of the latest measurement per key.

### Fixed
- A box with two identical cards collided on one ledger key and
  the diff reported a 30% regression that was the other slot.
  Device identity now includes the NVML UUID, and `normalise`
  strips the suffix again so spec lookups still work.

## [0.3.0]

### Added
- `vramfit bench`: prompt processing and token generation measured
  separately, because they are bound by different things.
- llama.cpp backend, driven through `llama-bench -o json`.
- Roofline ceilings: FLOP/s for prefill, memory bandwidth for
  decode, and the achieved fraction of each.
- Achieved bandwidth in GB/s alongside tok/s. The fraction is the
  output worth having -- it says whether the next improvement is
  available at all.
- `docs/speed.md` on why one tok/s number describes nothing.

### Changed
- A result carries driver version, backend build and cache dtype,
  because a number without them cannot be compared later.
- An unknown GPU produces no ratio and a note saying why, rather
  than a ratio against an invented peak.

## [0.2.0]

### Added
- `vramfit fit --max-context`: longest context that fits, by
  bisecting the real budget rather than inverting it on paper.
- `vramfit fit --for-context N`: heaviest quant that still fits.
- `vramfit ladder`: quant against max context, which is the table
  I actually want before downloading anything.
- safetensors header reader, sharded-repo aware.
- Hugging Face `config.json` reader, including rope-scaling, so a
  model can be sized before any weights are downloaded.
- One loader entry point: GGUF file, repo directory, safetensors
  or config.json.
- `vramfit quants`: the bits-per-weight ladder and where each
  number came from.
- `--free` to budget against free memory instead of total, and
  `--cache-bytes` for a quantised KV cache.

### Changed
- `budget.build` takes `weights_override` so hypotheticals do not
  need a fabricated shape.
- A metadata-only load refuses to guess the weight term and asks
  for `--quant`.

## [0.1.0]

### Added
- VRAM estimate from GGUF metadata: weights, KV cache, compute
  buffer, CUDA context, fragmentation headroom.
- GGUF metadata and tensor-info reader.
- Effective bits-per-weight computed from the tensor table rather
  than from the quant name.
- `vramfit fit` and `vramfit shape`.
- Device query via nvidia-smi, pynvml when it is importable.
