# three numbers I lost

Not wrong numbers. Numbers I wrote down, kept, and then could not use,
because the thing that would have made them mean something was the
thing I did not write down. All three are the reason
`vramfit/schema/ledger-entry.schema.json` requires what it requires.

## 1. a decode figure with no backend version

An unusually good tok/s on the GPU box. I wanted to know what made it
good, went back a fortnight later, and could not get within 20% of it.
The backend had been rebuilt in between. Whether the fast number was
the old build, a different `-ngl`, or a cold page cache is now
unanswerable, and "latest" is what the note said.

**So:** `backend_version` is required and `latest`, `main`, `head` and
`current` are rejected outright. A commit or a tag, or the measurement
does not go in.

## 2. a card that was not the size I had written down

Twice, on borrowed machines, the predictions looked wrong by exactly
the amount a smaller card would explain -- because it was a smaller
card. A 12 GiB variant of something I had noted as 16, which made the
arithmetic look broken when the arithmetic was fine. I spent an evening
on the second one before checking.

**So:** `nvidia_smi` is required, verbatim:

    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

One blob, not retyped, carrying the name, the memory size and the
driver together. A card name typed by hand is not a card.

## 3. a bandwidth fraction from before a driver upgrade

A 304 MiB CUDA context on one card became 322 after an upgrade, and the
measurement that depended on it had no driver recorded. Not a large
error -- large enough to turn a diff into a regression that was not one.

**So:** `driver` is required and pattern-checked, and the ledger keys on
the device tag rather than the card's name.

## the checklist, before I believe any of it

Written as a list because the moment I am most certain I will remember
the rest of it is the moment I am about to forget it:

- [ ] `nvidia-smi --query-gpu=name,memory.total,driver_version
      --format=csv`, pasted whole
- [ ] backend and **exact** version -- a build commit or a release tag
- [ ] the model as published. `8b` is not a model
- [ ] `vramfit shape --json <model>`, which is everything needed to
      reproduce a metadata question without the weights
- [ ] the exact backend command, including `-ngl`, `-c`, `-b`, flash
      attention and the cache types
- [ ] what was measured, and what vramfit predicted. `vramfit explain`
      output is the right form of this
- [ ] `measured_on`: which machine, in the words I would use out loud

Then `vramfit intake check`, and only then does it count.
