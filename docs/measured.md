# measured

Every measurement I have kept, one row per (model, quant, context,
backend, card). Generated from `intake/` -- see
[recording a measurement off-box](measuring.md).

`% of peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, and it is the only column that compares across
different hardware: decode tok/s on six different cards is six
unrelated numbers. A card or a model with no published shape gets no
fraction rather than a plausible one.

| model | quant | ctx | backend | card | decode tok/s | % of peak | GB/s | driver | measured on |
|---|---|---|---|---|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | q4_k_m | 8192 | llama.cpp b4341 | NVIDIA GeForce RTX 4090 | 138.6 | 81% | 821 | 560.35.03 | rented 4090, one afternoon |
| Meta-Llama-3.1-8B-Instruct | q4_k_m | 32768 | llama.cpp b4341 | NVIDIA GeForce RTX 4090 | 87.9 | 80% | 804 | 560.35.03 | rented 4090, one afternoon |
| Qwen2.5-32B-Instruct | q5_k_m | 16384 | llama.cpp b4341 | NVIDIA RTX A6000 | 21.4 | 71% | 548 | 550.90.07 | work laptop, thunderbolt dock |

Carried from the entries:
- Qwen2.5-32B-Instruct on NVIDIA RTX A6000: the dock card, power limited to 275W; nothing else was on it
