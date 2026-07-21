# mixture of experts

Every other page here assumes that the bytes on the card and the bytes
read per token are the same bytes. On an MoE model they are not, and the
distance between them is the entire point of the architecture.

| question | what it reads |
|---|---|
| will it fit | **every** expert. Any token may route anywhere, so nothing can be left off |
| how fast will it generate | the dense layers plus the `expert_used_count` experts this token routed to |

A 30B with 8 of 128 experts active is a 30B to fit and something nearer
a 3B to generate with. The two mistakes this invites are symmetrical and
only one of them is survivable:

- Predicting **speed** from the total size under-predicts decode by most
  of an order of magnitude. Annoying, visible, harmless.
- Predicting **VRAM** from the active size over-promises capacity, the
  model loads, and then it does not. See [running out
  later](oom.md).

## where the numbers come from

GGUF declares it: `{arch}.expert_count` and
`{arch}.expert_used_count`, with the expert weights in tensors named
`blk.N.ffn_{gate,up,down}_exps.weight`. Summing those separately from
everything else gives both figures with nothing estimated.

Some architectures also have a *shared* expert
(`ffn_*_shexp.weight`) which every token reads. It counts in both
columns.

## caveats

- The routed fraction is an average. A specific token reads whichever
  experts it routed to, and if a workload concentrates on a few of them
  the hardware's caches behave better than this model expects. The
  ceiling is therefore slightly pessimistic for decode and the error is
  in the safe direction.
- Expert-offload setups -- experts on the host, dense layers on the
  card -- are a different allocation again and are not modelled. The
  partial-offload arithmetic in `budget` scales whole layers, which is
  not what those setups do.
