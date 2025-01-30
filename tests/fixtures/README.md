# fixtures

Real GGUF headers, truncated immediately after the tensor-info table.
No tensor data, so these are kilobytes rather than gigabytes, and the
reader only ever looks at the header anyway.

Produced with:

    head -c $(python -m vramfit.tools.header_len model.gguf) model.gguf

| file | why it is here |
|---|---|
| `llama-3.1-8b-instruct-q4_k_m.header.gguf` | a K-quant mix: the bpw is not the one in the name |
| `qwen2.5-7b-instruct-q8_0.header.gguf` | a flat quant, and a 7:1 GQA ratio |
| `tiny-mqa-f16.header.gguf` | `head_count_kv` of 1, where getting the cache wrong is a 32x error |
