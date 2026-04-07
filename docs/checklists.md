# two checklists

Both exist because the thing I skip when I am tired is the thing that
costs the afternoon.

## before I believe a number this printed

- the exact command, including `-ngl`, `-c`, `-b`, flash attention and
  the cache types. Not "I ran the 8B"
- the full output, not the one line I thought was interesting
- `vramfit --version`, because the arithmetic changes between releases
  and a number with no version attached is a number I cannot place
- the Python version, for the one bug a year that is actually mine
- for a metadata complaint: `vramfit shape --json <model>`. That output
  has everything needed to reproduce a metadata bug without the
  weights, which are the part that cannot be kept

## before I commit a change to the arithmetic

Three questions, and the third is the one that matters:

1. **What does this change.**
2. **How was it verified** -- a test, a measurement, or both.
3. **If it changes a number: where does the new number come from.** A
   vendor page, a block-size definition, an architecture paper, a
   measurement on a machine I can name. "It matches what I see" is a
   legitimate answer *as long as the code says so.*

And then: `pytest`, `ruff check .`, `mypy vramfit`, a test for any
behaviour change, and a `CHANGELOG.md` line under Unreleased. The
changelog line is not bookkeeping -- it is the only place that records
why a prediction moved, and a prediction moving silently is the one
failure that makes every earlier ledger row unreadable.
