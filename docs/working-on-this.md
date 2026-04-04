# working on this

## the measurements worth going out of my way for

In descending order of how much they are worth, because I will not get
a second afternoon on most of these:

1. **A number from a card that is not mine.** The predictions are
   calibrated on two consumer NVIDIA cards and the box in the rack.
   They are arithmetic, so they should hold anywhere, and "should" is
   doing a lot of work in that sentence. The work laptop, a borrowed
   machine, an hour of rented GPU -- every one of those is a data point
   I cannot get back later. `docs/measuring.md` is what to record
   before the machine goes away.
2. **A card missing from `vramfit/data/gpus.csv`.** One row, with a
   link to the vendor page the numbers came from. No link, no row --
   the table is only defensible because every entry is traceable, and
   in a year I will not remember which review I read a bandwidth
   figure in.
3. **A model architecture whose metadata reads wrongly.** Keep the
   output of `vramfit shape --json` and the `config.json` or GGUF
   header next to it. That is a reproducible bug; "it looked wrong" is
   not.
4. **A prediction that missed by more than 5%.** With `vramfit explain`
   output and what actually happened, that is a bug about a specific
   term rather than a vague feeling about the tool.

## code

- `pip install -e ".[dev]"`, then `pytest`, `ruff check .` and
  `mypy vramfit`. The pipeline runs all three on 3.10 through 3.13.
- A behaviour change lands with a test. No exceptions; the whole value
  of this is that its numbers are checkable, including by me, later.
- New arithmetic needs a comment saying where the formula came from. A
  magic constant with no provenance is one I will not be able to defend
  or correct -- "it worked on my machine" is a fine answer *as long as
  it says so in the code*.
- The public surface is `vramfit/api.py` and nothing else. The internal
  modules move, and the only reason that distinction exists is that I
  import this from two other repositories.

## things I keep deciding against

Written down so I stop re-deciding them at midnight:

- **A dependency on a model runtime.** Nothing in this package imports
  torch, transformers, llama-cpp-python or vllm. Backends shell out to
  the binary or talk to the server, because that build is the thing
  whose performance is in question -- a wheel's opinion of it is not.
- **A bpw table of round numbers.** The whole point of reading the
  tensor table is to not have one.
- **A ratio against an invented peak.** An unknown card gets no ratio.
  A plausible denominator is worse than a missing one, because it
  survives into a report.
- **Averaging prefill and decode into one tok/s figure.** They are
  limited by different things. One number hides which.
