# Changelog

Format follows Keep a Changelog, versions follow semver.

## [1.3.0]

### Added
- Speculative decoding: the draft model budgeted as a second
  model, and an explicit refusal to report a bandwidth fraction
  while speculation is on. The ceiling assumes one forward pass
  per token, which is the assumption a draft model removes, so
  measured tok/s can exceed it legitimately.
- Multi-GPU layer-wise splits. Three things two cards are not:
  one card with more memory, an even split (the primary card
  carries the embedding and output tensors), or the sum of their
  bandwidths (a layer-wise split is sequential, so decode is bound
  by the *slowest* card -- adding a slow card buys context and
  costs speed, and that trade is the actual decision).

## [1.2.0]

### Added
- Mixture-of-experts support. Residency needs every expert
  because any token may route anywhere; decode reads only the
  routed ones. Predicting speed from the total size under-predicts
  by most of an order of magnitude, and predicting VRAM from the
  active size is how you get an OOM. `docs/moe.md`.
- `vramfit accuracy`: prediction error across the ledger, signed,
  because a systematically low prediction is a missing term and a
  systematically high one is wasted capacity.
- `docs/cards.md` and twelve more cards, including the current
  generation, reported with citations.
- `docs/findings.md`: the things keeping every measurement has
  caught that would otherwise have been a vague feeling.
- `measure.stale_entries`, and `devices` says when the cached CUDA
  context figure is from a different driver.

### Fixed
- `attention.key_length` / `attention.value_length` are read when
  the architecture sets them. A contributed report of a 40% cache
  over-prediction was an architecture where the key width is not
  `rope.dimension_count`.
- The prefill ceiling uses active parameters on an MoE model.

## [1.1.0]

### Added
- A schema for contributed measurements, a validator, `vramfit
  contrib check | import`, and a CI job that rejects a report
  nobody could reproduce. Every required field is required because
  a report without it has already gone nowhere.
- `contributions/` and the published table at `docs/measured.md`,
  built from it by `tools/build_measured_table.py`.
- `vramfit/data/shapes.csv`: published architectures, so a
  contributed number can be turned into a fraction of that card's
  bandwidth without anybody shipping weights.
- Issue template for a hardware report, asking for `nvidia-smi`
  output and the exact backend version.
- `Shape.params_from_config()` and `Shape.n_ff`.

### Deprecated
- `vramfit plan --utilisation`, renamed to
  `--gpu-memory-utilization` to match vLLM's own spelling. The old
  flag keeps working and warns; it will be removed in 2.0.

## [1.0.0]

### Changed
- Nothing. 1.0.0 is the same code as 0.7.1 with a promise
  attached: see `docs/stability.md`. The version number is for
  other people's benefit, and holding it at 0.x while people
  package it was the less honest option.

### Added
- `Development Status :: 5 - Production/Stable`, which is a claim
  about the interface and not about the predictions. Those are
  still wrong in the ways `docs/wrong.md` lists.

## [0.7.1]

### Added
- Typed public API in `vramfit.api`, and `py.typed`. Everything
  else in the package is internal.
- Full package metadata, so an index page says something.
- `--version`, and `shape --json`.
- Docs site, `SECURITY.md`, `CONTRIBUTING.md`, `LICENSE`, issue
  and PR templates, Dependabot.
- Release workflow publishing through PyPI trusted publishing.
- `docs/stability.md`, which is what 1.0 will be promising.

### Changed
- CI runs ruff, mypy and coverage across 3.10 to 3.13.
- `devices` points at CONTRIBUTING.md when a card has no spec row.

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
