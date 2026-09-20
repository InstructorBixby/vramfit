# FAQ

**It says 4.83 bits per weight for a Q4 model. Is that a bug?**

No, and it is the reason this tool exists. `Q4_K_M` is a *mix*: most
tensors at Q4_K (4.5 bpw by block geometry, not 4.0), with the output
tensor and some `attn_v` / `ffn_down` layers at Q6_K. vramfit reads the
per-tensor types out of the GGUF tensor table and sums the real block
sizes, so the figure is arithmetic rather than an estimate. It also
means the number differs between two files with the same quant name and
different architectures, which is correct.

**Your prediction said it fits and it OOMed.**

Most likely one of three things, in order of frequency: the context you
actually ran was longer than the one you asked about; `--parallel` on a
server multiplies the cache per slot; or the slack was under 5% of the
card, which is inside the error bars of the compute-buffer term and is
reported as a maybe rather than a yes. [docs/oom.md](oom.md) has the
long version. `vramfit explain` will show which term is doing the work.

**Why is decode slower at 32k than at 4k? Nothing changed.**

The cache is in the bytes read per token. On an 8B at 32k it is about 4
GiB against 4.5 GiB of weights, so decode is reading nearly twice as
much per token as it was at an empty context. Halving the cache with
`--cache-type q8_0 --flash-attn` gets most of it back.

**The achieved fraction is 38%. Is that bad?**

It means the backend is reading bytes at a bit over a third of what the
card can do, so there is likely something to fix: a CUDA backend
compiled but not selected, layers quietly on the host, a batch size too
small to saturate. The fraction is reported because absolute tok/s
cannot distinguish "the card is saturated" from "half the card is
idle". 75% and above means the only remaining lever is reading fewer
bytes.

**Can it tell me whether a quant is any good?**

No, and it will not try. This measures bytes and seconds. Whether
Q3_K_M is still worth using is a question about output quality, needs
an evaluation and not a calculator, and anybody who tells you a
bits-per-weight figure answers it is selling something.

**Why is there no `--total-vram` flag that sums my two cards?**

Because two cards are not one card with more memory. The split is per
layer, the primary card carries the embedding and output tensors, each
card pays its own CUDA context, and a layer-wise split is sequential so
decode is bound by the *slowest* card. See `vramfit`'s multi-GPU
handling; summing the memory would hide every one of those.
