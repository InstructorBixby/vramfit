# vramfit

Will this model fit in this card, and roughly how fast will it be.

## install

    pip install -e ".[dev]"

## use

    vramfit shape models/llama-3.1-8b-instruct-q4_k_m.gguf
    vramfit fit models/llama-3.1-8b-instruct-q4_k_m.gguf \
        --context 8192 --vram 16GiB

`fit` exits 1 when it does not fit, so it works in a script.

## layout

| path | what |
|---|---|
| `vramfit/gguf.py` | GGUF header reader |
| `vramfit/ggml_types.py` | block geometry, and therefore exact bpw |
| `vramfit/shape.py` | the model shape everything is a function of |
| `vramfit/budget.py` | the five VRAM terms |
| `vramfit/cli.py` | subcommands |

See NOTES.md for where each number comes from, and the end of this file
for where it is wrong.
