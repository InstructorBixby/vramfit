"""Mixture-of-experts models, where the two questions come apart.

Everything up to here has assumed that the bytes resident on the card
and the bytes read per token are both "the weights". On an MoE model
they are not, and the gap is the whole point of the architecture:

  - **capacity** needs *every* expert resident, because any token may
    route to any of them. Nothing is saved.
  - **speed** reads only the experts a token actually routes to, so
    decode moves far fewer bytes per token than the model's size.

A 30B with 8 of 128 experts active per token is a 30B to fit and
something nearer a 3B to generate with. Predicting its speed from its
total size under-predicts by an order of magnitude, and predicting its
VRAM from its active size is the mistake that produces an OOM.

GGUF declares this: `{arch}.expert_count`, `{arch}.expert_used_count`,
and expert tensors named `blk.N.ffn_*_exps.weight`.
"""
from __future__ import annotations

from dataclasses import dataclass

EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                   "ffn_down_exps.weight")
SHARED_SUFFIXES = ("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                   "ffn_down_shexp.weight")


@dataclass
class MoE:
    n_experts: int
    n_used: int
    expert_bytes: int        # all experts, every layer
    shared_bytes: int        # always-on expert, if the arch has one
    dense_bytes: int         # everything that is not an expert

    @property
    def active_fraction(self) -> float:
        return self.n_used / self.n_experts if self.n_experts else 1.0

    @property
    def resident_bytes(self) -> int:
        """What has to be on the card: all of it."""
        return self.dense_bytes + self.shared_bytes + self.expert_bytes

    @property
    def bytes_per_token(self) -> int:
        """What decode reads: the dense part plus the routed experts."""
        return (self.dense_bytes + self.shared_bytes
                + int(self.expert_bytes * self.active_fraction))

    @property
    def speed_ratio(self) -> float:
        """How much faster decode is than the model's size suggests."""
        return (self.resident_bytes / self.bytes_per_token
                if self.bytes_per_token else 1.0)

    def summary(self) -> str:
        gib = 1024 ** 3
        return (f"moe: {self.n_used} of {self.n_experts} experts per "
                f"token\n"
                f"  resident  {self.resident_bytes / gib:7.3f} GiB  "
                f"(all experts; any token may route anywhere)\n"
                f"  per token {self.bytes_per_token / gib:7.3f} GiB  "
                f"(dense + routed experts)\n"
                f"  decode reads {self.speed_ratio:.1f}x fewer bytes "
                f"than the model's size")


def is_expert_tensor(name: str) -> bool:
    return any(name.endswith(s) for s in EXPERT_SUFFIXES)


def is_shared_expert_tensor(name: str) -> bool:
    return any(name.endswith(s) for s in SHARED_SUFFIXES)


def from_shape(shape, n_experts: int = 0, n_used: int = 0) -> MoE | None:
    """None for a dense model, which is most of them."""
    if not n_experts or n_experts < 2:
        return None
    expert = shared = dense = 0
    for t in shape.tensors:
        if is_expert_tensor(t.name):
            expert += t.nbytes
        elif is_shared_expert_tensor(t.name):
            shared += t.nbytes
        else:
            dense += t.nbytes
    return MoE(n_experts=n_experts, n_used=max(1, n_used),
               expert_bytes=expert, shared_bytes=shared,
               dense_bytes=dense)
