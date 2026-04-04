# contributing

## the useful kinds of contribution

In descending order of how much they help:

1. **A measurement from hardware I do not have.** This is the most
   valuable thing anybody can send. The predictions are calibrated on a
   small number of consumer NVIDIA cards and I cannot check them
   anywhere else. See `docs/contributing-measurements.md`.
2. **A card missing from `vramfit/data/gpus.csv`.** One row, with a link
   to the vendor page the numbers came from. No link, no row -- the
   table is only defensible because every entry is traceable.
3. **A model architecture whose metadata is read wrongly.** Attach the
   output of `vramfit shape --json`, and the `config.json` or the GGUF
   header if you can share it.
4. **A prediction that missed by more than 5%.** With `vramfit explain`
   output and what actually happened, that is a bug report about a
   specific term rather than about the tool.

## code

- `pip install -e ".[dev]"`, then `pytest`, `ruff check .` and
  `mypy vramfit`. CI runs all three on 3.10 through 3.13.
- A behaviour change lands with a test. No exceptions; the whole value
  of this tool is that its numbers are checkable.
- New arithmetic needs a comment saying where the formula came from. A
  magic constant with no provenance will be asked about in review and
  the answer "it worked on my machine" is fine *if it says so in the
  code*.
- The public API is `vramfit/api.py` and nothing else. Internal modules
  move without notice.

## what will be declined

- A dependency on a model runtime. Nothing in this package imports
  torch, transformers, llama-cpp-python or vllm. Backends shell out to
  the binary or talk to the server, because that is the build whose
  performance is in question.
- A bpw table of round numbers. The whole point of reading the tensor
  table is to not have one.
- A ratio against an invented peak. An unknown card gets no ratio.
- Averaging prefill and decode into one tok/s figure.
