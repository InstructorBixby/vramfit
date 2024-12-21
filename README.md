# vramfit

Will this model fit in this card, and roughly how fast will it be.

    python gguf_dump.py models/llama-3.1-8b-instruct-q4_k_m.gguf
    python fit.py --gguf models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --context 8192

Reads the GGUF header rather than asking you for parameter counts,
because the numbers people quote for a model are usually the ones off
the model card and not the ones in the file they downloaded.

See NOTES.md for where each term comes from.
