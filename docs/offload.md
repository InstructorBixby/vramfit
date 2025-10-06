# partial offload

When a model does not fit, llama.cpp will put some layers on the GPU and
leave the rest on the CPU (`-ngl N`). The budget as built so far cannot
describe that at all, and the naive patch -- scale the weights by
`N / n_layers` -- is wrong in a way that matters.

## what actually lives where

Per-layer, with `N` of `L` layers offloaded:

| thing | on the GPU | on the host |
|---|---|---|
| layer weights | N layers | L - N layers |
| **kv cache** | **N layers** | **L - N layers** |
| token embedding | optional, and large | |
| output tensor | usually, unless `-ngl` is low | |
| compute buffer | full size | a second one, host side |

The row that gets missed is the cache. The cache for a layer lives
wherever the layer lives, so offloading 20 of 32 layers moves 62.5% of
the cache onto the card, not all of it and not none of it. Treating the
cache as entirely resident over-predicts; treating it as entirely host
side under-predicts and produces the much worse failure, which is a
confident "it fits" followed by an OOM.

## the other asymmetry

Embedding and output tensors are big -- on an 8B with a 128k vocab they
are about 1.1 GiB of the 4.5 GiB total -- and llama.cpp's offload order
is not simply "the first N layers". Which of them end up on the card
depends on the build and on `-ngl`, so the model here is:

- `-ngl 0`: nothing on the card except the compute buffer.
- `0 < N < L`: N layer-blocks, N/L of the cache, no embedding, no
  output tensor.
- `N >= L`: everything, which is the case the five-term budget already
  describes.

That middle rule is an approximation of a thing that is a build detail,
and it is the term to suspect first when a partial-offload prediction
misses.

## and the speed is not proportional either

Decode with any layers on the host is bound by the slowest path, and the
host path is an order of magnitude slower per byte. One layer left on
the CPU costs far more than 1/32 of the throughput, and the decode
roofline against GPU bandwidth stops meaning anything -- so a partial
offload result reports tok/s and no fraction, rather than a fraction
against the wrong denominator.
