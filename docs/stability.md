# stability

The reason this exists is not an audience. It is that the engagement
scaffold shells out to `vramfit fit` and reads its exit code, two other
repositories import `vramfit.api`, and a wrapper script on the GPU box
passes flags I chose eighteen months ago. Renaming something here
breaks my own work in a place I will not connect to this change. So the
shims stay, and the list below is what I am not allowed to move.

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
  modules get reorganised; I import `vramfit.budget` directly in one
  place and that is my problem, not a promise.
- **The numbers.** This is the important one. A term getting *more
  accurate* is a bug fix, not a breaking change, and it will move the
  output. Anything of mine that would break on a prediction changing
  pins the version and gets the changelog read -- a tool whose whole
  job is to be right cannot promise to stay wrong.
- The contents of `data/gpus.csv`, which grows as I meet cards.
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
