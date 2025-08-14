# Changelog

Format follows Keep a Changelog, versions follow semver.

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
