# contributing a measurement

The predictions here are calibrated on a handful of consumer NVIDIA
cards. They are arithmetic, so they should hold anywhere, and "should"
is doing a lot of work in that sentence. A measurement from hardware I
do not have is the single most useful thing anybody can send.

## the easy way

    vramfit bench ./model.gguf --context 8192 --record
    vramfit ledger show --model <model> --json > mine.json

Then add the `nvidia_smi` field by hand -- see below -- and open a pull
request adding the file under `contributions/`.

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

## why the nvidia-smi output

    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Because a card name typed by hand is not a card. Two reports in a row
turned out to be from the 12 GiB variant of a card I had assumed was
16, which made the predictions look wrong in a way that had nothing to
do with the arithmetic. The `nvidia-smi` line carries the name, the
memory size and the driver together, and it settles the question before
it becomes an argument.

## what happens to it

CI validates the file against
`vramfit/schema/ledger-entry.schema.json`. Once merged it goes into the
published table at [measured.md](measured.md), with attribution if
`reported_by` is set.

A rejected contribution is not a rejected measurement. It means a field
is missing, and the field is missing because without it the number
cannot be reproduced -- which is the only property that makes it worth
publishing.
