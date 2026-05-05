# stability

## what is covered

- **`vramfit.api`.** The functions and dataclasses re-exported there.
  Fields are added to the dataclasses, never removed or repurposed
  within a major version.
- **The CLI subcommands and their flags**, and the exit codes: 0 fits,
  1 does not fit or no rows, 2 the question was unanswerable as asked.
- **The `--json` schema.** Keys are added, never removed or
  re-typed. Every byte count stays a byte count.
- **The ledger schema.** Columns are added with defaults; an older
  database opens.

## what is not covered

- Everything in the package that is not `vramfit.api`. The internal
  modules get reorganised; people have been importing
  `vramfit.budget` directly and that is fine, but it is not a promise.
- **The numbers.** This is the important one. A term getting *more
  accurate* is a bug fix, not a breaking change, and it will move
  your output. If a prediction changing between releases would break
  something of yours, pin the version and read the changelog -- a
  tool whose whole job is to be right cannot promise to stay wrong.
- The contents of `data/gpus.csv`, which grows as cards are reported.
- The exact wording of human-readable output.

## deprecation

A flag or an API name being renamed gets:

1. The old name kept and working, emitting a `DeprecationWarning` with
   the replacement in the message.
2. A line in the changelog under `Deprecated` naming the release that
   will remove it -- always the next *major*, never a minor.
3. Removal only in that major release.

Two minor releases is the minimum lifetime for a deprecated name even
if the next major arrives sooner.

## versioning

Semver, with the caveat about numbers above. Patch releases contain
fixes and new `data/gpus.csv` rows; minor releases add backends,
subcommands and terms; major releases remove deprecated names.
