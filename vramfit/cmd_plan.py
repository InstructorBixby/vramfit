"""`vramfit plan`: the vLLM-shaped question.

Separate subcommand rather than a flag on `fit`, because the output is
not a yes or a no. It is a block budget and a concurrency, and
pretending that is the same answer in a different format helps nobody.
"""
from __future__ import annotations

import argparse

from vramfit import cmd_fit, load, vllm_plan
from vramfit.deprecations import DeprecatedAlias

GIB = 1024 ** 3


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("plan", help="vllm block budget and concurrency")
    p.add_argument("model")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float,
                   dest="utilisation", default=0.9,
                   help="as vLLM spells it")
    # Was the only spelling. Kept because people have it in scripts.
    p.add_argument("--utilisation", type=float, dest="utilisation",
                   action=DeprecatedAlias,
                   replacement="--gpu-memory-utilization",
                   removed_in="2.0", help=argparse.SUPPRESS)
    p.add_argument("--block-tokens", type=int, default=16)
    cmd_fit.add_target_args(p)
    p.set_defaults(fn=run)
    return p


def run(args: argparse.Namespace) -> int:
    sh, how = load.load(args.model)
    dev = cmd_fit.pick_device(args.device)
    vram = cmd_fit.resolve_vram(args, dev)
    if vram is None:
        print("need --vram or a visible device")
        return 2

    p = vllm_plan.plan(
        sh, vram, utilisation=args.utilisation,
        block_tokens=args.block_tokens,
        cuda_context=cmd_fit.cuda_context_for(dev),
        weights_override=cmd_fit.weights_override(sh, args))
    print(f"{sh.name}  ({how})  vllm")
    print()
    print(p.table())
    print()
    for ctx in (args.context, args.context * 2, args.context * 4):
        n = p.sequences_at(ctx)
        print(f"  {cmd_fit.human_ctx(ctx):>8} tokens: "
              f"{n} concurrent sequence(s)")
    return 1 if p.cache == 0 else 0
