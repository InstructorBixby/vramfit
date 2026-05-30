# recording a measurement off-box

The predictions are calibrated on two cards at the desk and the one in
the rack. They are arithmetic, so they should hold anywhere, and
"should" is doing a lot of work in that sentence. Every other machine I
get near is a data point I cannot get back: the work laptop, whatever
the client has on the bench, an hour of rented GPU at two in the
morning. This page is what to write down before the machine goes away.

## the routine

    vramfit bench ./model.gguf --context 8192 --record
    vramfit ledger show --model <model> --json > intake/<machine>.json

Then add `nvidia_smi` and `measured_on` by hand -- see below -- and
`vramfit intake check intake/` before trusting any of it.

## the fields, and why each one is required

| field | why |
|---|---|
| `model` | as published. `8b` is not a model |
| `quant`, `bits_per_weight` | the effective bpw differs by architecture |
| `context` | the cache term is linear in it and dominates at length |
| `backend` | the backends are not comparable to each other |
| `backend_version` | a build commit or a tag. **not** "latest" |
| `gpu`, `nvidia_smi` | see below |
| `driver` | measured bandwidth has moved on a driver upgrade here |
| `cache_dtype`, `flash_attn` | a quantised V cache without flash attention silently stays f16 |
| `decode_tok_s` | the number |
| `vram_peak_mib` | lets the prediction error be computed, not estimated |
| `measured_on` | which machine, in the words I would use out loud |

## why the nvidia-smi output

    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Because a card name typed by hand is not a card. Twice now a number of
mine turned out to be from the 12 GiB variant of a card I had written
down as 16, which made the predictions look wrong for a reason that had
nothing to do with the arithmetic. The `nvidia-smi` line carries the
name, the memory size and the driver in one blob I do not retype, and
it settles the question six months later when I have no other way to
settle it.

## why this is enforced and not merely advised

Because the moment of writing a number down is the moment I am most
certain I will remember the rest of it, and I never do. So the pipeline
validates every file under `intake/` against
`vramfit/schema/ledger-entry.schema.json`, and a file that fails is not
a rejected measurement -- it is a measurement that cannot be
reproduced, which is the only property that makes keeping it worth
anything.

From there `vramfit intake import` puts it in the ledger and
`tools/build_measured_table.py` rebuilds [measured.md](measured.md),
which is the table I actually read.
