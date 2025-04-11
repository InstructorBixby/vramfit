# Changelog

Format follows Keep a Changelog, versions follow semver.

## [0.1.0]

### Added
- VRAM estimate from GGUF metadata: weights, KV cache, compute
  buffer, CUDA context, fragmentation headroom.
- GGUF metadata and tensor-info reader.
- Effective bits-per-weight computed from the tensor table rather
  than from the quant name.
- `vramfit fit` and `vramfit shape`.
- Device query via nvidia-smi, pynvml when it is importable.
