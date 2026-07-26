# the card table

`vramfit/data/gpus.csv` holds advertised peak memory bandwidth and dense
FP16 tensor FLOP/s for every card anybody has reported. They are used
only as denominators, every row carries its provenance, and a card with
no row gets **no** ratio rather than a plausible one.

## adding a card

One row, and a link in the pull request to the vendor page the numbers
came from. `source` is `vendor-spec` for rows taken straight off a
specification page and `reported` for rows that arrived through an issue
with a citation.

## two traps in the FLOP column

**Sparsity.** Vendors quote tensor throughput with 2:4 structured
sparsity, which doubles the headline figure and does not apply to
anything run here. Every number in this table is the dense one. If a row
looks half what the marketing page says, that is why.

**FP16 accumulate.** Some consumer parts are rated at double rate for
FP16 with FP16 accumulate and half that with FP32 accumulate, which is
what the kernels in question mostly use. Where the two differ the
conservative figure is in the table.

One row is a deliberate oddity: the P40 has usable bandwidth and
essentially no FP16 throughput (its FP16 rate is 1/64 of FP32). It
generates respectably and processes a prompt appallingly, and it is in
the table because that combination is exactly what the two separate
ceilings exist to describe. A single tok/s number for that card is
meaningless in a way that is hard to argue with.

## the newest generation

The current generation's cards (compute capability `sm_120`) are in the
table from reports. Two notes for anybody comparing against an older
card:

- A backend built before that architecture existed will not have kernels
  for it and falls back, sometimes to something very slow. The build
  string in a ledger row is doing real work here.
- The achieved fraction on these has been lower than on the previous
  generation in every report so far, which is what "the backend has not
  caught up" looks like in this tool's output rather than a claim about
  the hardware.
