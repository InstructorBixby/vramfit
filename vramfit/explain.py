"""Print the whole derivation.

Exists because the first question anybody sensible asks a prediction is
"where did that come from", and the right answer is the arithmetic and
its provenance rather than a confident restatement of the total.

Every line names its source: the file, the vendor table, a measurement
on this machine, or a policy. The last two categories are the ones to
argue with, and separating them is the whole point.
"""
from __future__ import annotations

from vramfit import budget, cache, roofline, specs

GIB = 1024 ** 3
MIB = 1024 ** 2


def explain(shape, context, *, vram=None, spec=None, cache_dtype="f16",
            cuda_context=None, batch=512, n_gpu_layers=None,
            source="") -> str:
    cb = cache.element_bytes(cache_dtype)
    b = budget.build(shape, context, cache_dtype=cache_dtype,
                     cuda_context=cuda_context, batch=batch,
                     n_gpu_layers=n_gpu_layers)
    out = [f"{shape.name}", ""]
    if source:
        out.append(f"metadata        {source}")
    out += [
        f"params          {shape.params:,} "
        f"({shape.params / 1e9:.3f}B)  [tensor table]",
        f"effective bpw   {shape.bpw:.4f}  [sum over tensor types, "
        f"not the quant name]",
        "",
        "weights",
        f"  {shape.weight_bytes / GIB:8.3f} GiB  = sum of "
        f"{len(shape.tensors)} tensors at their own block ratios",
    ]
    for name, nbytes in shape.quant_mix().items():
        out.append(f"      {name:8} {nbytes / GIB:7.3f} GiB")

    per_token = budget.kv_bytes_per_token(shape, cb)
    out += [
        "",
        "kv cache",
        f"  2 x {shape.n_layers} layers x {shape.n_head_kv} kv heads "
        f"x {shape.head_dim} head_dim x {cb:g} bytes "
        f"= {per_token / 1024:.0f} KiB per token",
        f"  x {context} context = {b.kv_cache / GIB:.3f} GiB",
        f"  [kv heads, not the {shape.n_head} attention heads: "
        f"gqa {shape.gqa_ratio:.0f}x]",
        "",
        "compute buffer",
        f"  {b.compute / GIB:8.3f} GiB  = batch {batch} x vocab "
        f"{shape.vocab} x 4 plus hidden terms  [rough, "
        f"backend-specific]",
        "",
        "cuda context",
        f"  {b.cuda_context / MIB:8.0f} MiB  "
        f"[{'measured on this machine' if cuda_context else 'assumed'}]",
        "",
        "headroom",
        f"  {b.headroom / GIB:8.3f} GiB  = 3% of the rest  [policy, "
        f"not a measurement]",
        "",
        f"total           {b.total / GIB:8.3f} GiB",
    ]
    if vram:
        out.append(f"against         {vram / GIB:8.3f} GiB  "
                   f"-> {(vram - b.total) / GIB:+.3f} GiB")

    if spec is not None:
        dec = roofline.decode_ceiling(shape, context,
                                      spec.bandwidth_bps, cb)
        pre = roofline.prefill_ceiling(shape, spec.fp16_tflops * 1e12)
        out += [
            "",
            f"ceilings on {spec.key}  [{spec.source}]",
            f"  prefill  {pre.tok_s:10.0f} tok/s = "
            f"{spec.fp16_tflops:.1f}e12 flops / (2 x params)",
            f"  decode   {dec.tok_s:10.1f} tok/s = "
            f"{spec.bandwidth_gbs:.1f} GB/s / "
            f"{dec.bytes_per_token / 1024 ** 2:.0f} MiB per token",
        ]
    for note in b.assumptions:
        out.append(f"  ! {note}")
    return "\n".join(out)


def add_parser(sub):
    from vramfit import cmd_fit, load

    def run(args) -> int:
        sh, how = load.load(args.model)
        dev = cmd_fit.pick_device(args.device)
        vram = cmd_fit.resolve_vram(args, dev)
        print(explain(
            sh, args.context or sh.train_context or 4096,
            vram=vram,
            spec=specs.lookup(dev.tag) if dev else None,
            cache_dtype=args.cache_type,
            cuda_context=cmd_fit.cuda_context_for(dev),
            source=how))
        return 0

    p = sub.add_parser("explain", help="show the whole derivation")
    p.add_argument("model")
    p.add_argument("--context", type=int)
    cmd_fit.add_target_args(p)
    p.set_defaults(fn=run)
    return p
