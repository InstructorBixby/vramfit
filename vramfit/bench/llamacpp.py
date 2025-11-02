"""Backend: llama.cpp, driven through `llama-bench`.

llama-bench already separates the two measurements -- `pp` is prompt
processing, `tg` is token generation -- and reports them as separate
rows, which is exactly the shape wanted here. Its `-o json` output is
parsed rather than its table, because the table's columns move between
builds.

The build matters more than anything else in this file. A CPU-only build
and a CUDA build of the same commit differ by two orders of magnitude,
so the build string goes into the result and any comparison that crosses
builds is the caller's problem.
"""
from __future__ import annotations

import json
import shutil
import subprocess

from vramfit import roofline, specs
from vramfit.bench import register
from vramfit.bench.result import Result

CACHE_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "q8_0": 1, "q8_1": 1,
               "q5_1": 1, "q5_0": 1, "q4_1": 1, "q4_0": 1, "iq4_nl": 1}


class BenchError(Exception):
    pass


def find_binary(explicit: str = "") -> str:
    for cand in (explicit, "llama-bench"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    raise BenchError("llama-bench not on PATH; pass --bin")


def run_llama_bench(binary: str, model: str, *, prompt: int = 512,
                    gen: int = 128, context: int = 4096,
                    n_gpu_layers: int = 999, cache_type: str = "f16",
                    batch: int = 512, reps: int = 3,
                    extra=()) -> list:
    cmd = [binary, "-m", model, "-p", str(prompt), "-n", str(gen),
           "-c", str(context), "-ngl", str(n_gpu_layers),
           "-b", str(batch), "-r", str(reps),
           "-ctk", cache_type, "-ctv", cache_type, "-o", "json"]
    cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        raise BenchError("llama-bench failed: " + " | ".join(tail))
    return parse(proc.stdout)


def parse(blob: str) -> list:
    """llama-bench -o json emits one object per test row."""
    try:
        rows = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise BenchError(f"could not parse llama-bench output: {exc}") \
            from None
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    return [r for r in rows if isinstance(r, dict)]


def split_rows(rows: list) -> tuple:
    """(prefill avg tok/s, decode avg tok/s) out of the row set."""
    prefill = decode = 0.0
    for r in rows:
        ts = float(r.get("avg_ts") or 0.0)
        if int(r.get("n_prompt") or 0) > 0:
            prefill = max(prefill, ts)
        elif int(r.get("n_gen") or 0) > 0:
            decode = max(decode, ts)
    if not prefill or not decode:
        raise BenchError(
            "need both a pp and a tg row; got "
            f"{len(rows)} row(s). Pass -p and -n both non-zero.")
    return prefill, decode


@register("llama.cpp")
def bench(shape, *, model: str, gpu: str, context: int = 4096,
          quant: str = "", binary: str = "", cache_type: str = "f16",
          n_gpu_layers: int = 999, batch: int = 512, reps: int = 3,
          driver: str = "") -> Result:
    rows = run_llama_bench(
        find_binary(binary), model, context=context, batch=batch,
        n_gpu_layers=n_gpu_layers, cache_type=cache_type, reps=reps)
    prefill, decode = split_rows(rows)
    build = str(rows[0].get("build_commit") or "")

    res = Result(
        model=shape.name or model, quant=quant or f"{shape.bpw:.2f}bpw",
        context=context, backend="llama.cpp", gpu=gpu,
        prefill_tok_s=prefill, decode_tok_s=decode,
        driver=driver, backend_version=build,
        n_gpu_layers=n_gpu_layers, cache_dtype=cache_type, batch=batch)

    partial = n_gpu_layers < shape.n_layers
    if partial:
        # Decode with any layer on the host is bound by the host path,
        # which is an order of magnitude slower per byte. A fraction
        # against GPU bandwidth would not be inaccurate, it would be
        # meaningless, so there is no fraction.
        res.notes.append(
            f"partial offload ({n_gpu_layers}/{shape.n_layers} "
            f"layers): no bandwidth fraction, the bound is the host "
            f"path")
        return res

    spec = specs.lookup(gpu)
    if spec is None:
        res.notes.append(f"no spec row for {gpu!r}: no peak to compare "
                         f"against, ratios left at zero")
        return res

    cache_bytes = CACHE_BYTES.get(cache_type, 2)
    dec = roofline.decode_ceiling(shape, context, spec.bandwidth_bps,
                                  cache_bytes)
    pre = roofline.prefill_ceiling(shape, spec.fp16_tflops * 1e12)
    res.decode_frac_peak = roofline.fraction_of_peak(decode, dec)
    res.prefill_frac_peak = roofline.fraction_of_peak(prefill, pre)
    res.achieved_bandwidth_gbs = roofline.achieved_bandwidth(
        decode, dec.bytes_per_token) / 1e9
    return res
