# measured

Every measurement I have kept, one row per (model, quant, context,
backend, card). Generated from `intake/` -- see
[recording a measurement off-box](measuring.md).

`% peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, which is the only column that compares across
different hardware. A card with no row in `vramfit/data/gpus.csv` gets
no fraction rather than a plausible one.


| model | quant | ctx | backend | card | decode tok/s | % peak | driver | measured on |
|---|---|---|---|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | q4_k_m | 8192 | llama.cpp b4341 | NVIDIA GeForce RTX 4090 | 138.6 | see ledger | 560.35.03 | rented 4090, one afternoon |
| Meta-Llama-3.1-8B-Instruct | q4_k_m | 32768 | llama.cpp b4341 | NVIDIA GeForce RTX 4090 | 87.9 | see ledger | 560.35.03 | rented 4090, one afternoon |
| Qwen2.5-32B-Instruct | q5_k_m | 16384 | llama.cpp b4341 | NVIDIA RTX A6000 | 21.4 | see ledger | 550.90.07 | work laptop, thunderbolt dock |
