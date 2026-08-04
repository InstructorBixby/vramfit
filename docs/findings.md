# things the ledger has caught

The argument for keeping every measurement, in the form of the things
that would otherwise have been a vague feeling.

## a driver upgrade moved decode by 10%

Routine upgrade, no backend change, no model change. `ledger diff` on
the next benchmark:

    decode_tok_s             118.40 ->     131.00  (+10.6%)
    achieved_bandwidth_gbs   758.00 ->     838.00  (+10.6%)

    changed at the same time:
      driver                   550.54.14 -> 550.78

The achieved fraction went from 81% to 90% of advertised peak on the
same card with the same build. Nothing in the arithmetic changed; the
card simply started doing more. Without the previous row this would have
been "it feels snappier".

The same upgrade moved the measured CUDA context from 304 MiB to 322
MiB, which is why the calibration cache is keyed on (device, driver)
rather than on the device alone.

## a backend rebuild cost 30% and nobody noticed for a fortnight

A build with the CUDA backend compiled but not selected at runtime. The
tok/s drop was within the range that gets explained away as "the machine
is busy"; the achieved fraction dropped to 38%, which is not within any
such range. The verdict text for that band -- "leaving performance on
the table" -- is there because of this one.

## two identical cards were not identical

A box with two of the same card reported a 30% regression on a diff. The
second card sits behind a shorter PCIe link and runs warmer, and the
benchmark had been landing on a different one each time. This is why
device identity includes the NVML UUID and why `ledger` keys on the
device tag rather than on the card's name.

## a 12 GiB card reported as a 16 GiB one

Twice, in contributed reports, where the predictions looked wrong for a
reason that had nothing to do with the arithmetic. Hence the
`nvidia_smi` field being required in a contribution: it carries the name,
the memory size and the driver in one blob that nobody retypes.
