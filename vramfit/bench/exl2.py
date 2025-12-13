"""Backend: exllamav2, through its own generator example script.

Driven as a subprocess for the same reason as the others. The one
wrinkle is that its cache quantisation modes do not map onto the ggml
block sizes, so the cache byte cost is taken from its own table rather
than from `vramfit.cache`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

from vramfit import roofline, specs
from vramfit.bench import register
from vramfit.bench.result import Result

# exllamav2 cache modes, bytes per element per tensor.
CACHE_BYTES = {"fp16": 2.0, "fp8": 1.0, "q8": 1.0, "q6": 0.75,
               "q4": 0.5}

# "Prompt processed: 512 tokens, 3104.88 tokens/s". The token COUNT
# sits between the label and the rate, so the gap cannot be [^0-9]* --
# that stops dead at the 512 and matches nothing at all. Lazy, and
# scoped to the line so a prompt label cannot reach a later rate.
# "response" is in the alternation because the generate line says
# "Response generated".
ROW = re.compile(r"(?P<label>prompt|response|generate)\w*[^\n]*?"
                 r"(?P<tps>\d+\.\d+)\s*tokens/s", re.I)


class BenchError(Exception):
    pass


def find_script(explicit: str = "") -> str:
    for cand in (explicit, "exl2-bench"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    raise BenchError("no exllamav2 bench entry point on PATH; "
                     "pass --bin")


def parse(blob: str) -> dict:
    """Accept JSON if it is offered, fall back to scraping the log."""
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return {"prefill": float(data.get("prompt_tokens_s") or 0),
                    "decode": float(data.get("gen_tokens_s") or 0),
                    "version": str(data.get("version") or "")}
    except json.JSONDecodeError:
        pass
    out = {"prefill": 0.0, "decode": 0.0, "version": ""}
    for m in ROW.finditer(blob):
        key = "prefill" if m.group("label").lower() == "prompt" \
            else "decode"
        out[key] = max(out[key], float(m.group("tps")))
    if not out["prefill"] or not out["decode"]:
        raise BenchError("could not find both a prompt and a generate "
                         "rate in the output")
    return out


def run(script: str, model_dir: str, *, context: int, gen: int,
        cache_mode: str) -> dict:
    proc = subprocess.run(
        [script, "-m", model_dir, "-l", str(context), "-n", str(gen),
         "-cq", cache_mode, "--json"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        raise BenchError("exllamav2 bench failed: " + " | ".join(tail))
    return parse(proc.stdout)


@register("exllamav2")
def bench(shape, *, model: str, gpu: str, context: int = 4096,
          quant: str = "", binary: str = "", cache_type: str = "fp16",
          gen: int = 128, driver: str = "", **_) -> Result:
    if cache_type not in CACHE_BYTES:
        raise BenchError(
            f"exllamav2 cache mode {cache_type!r} is not one of "
            f"{', '.join(sorted(CACHE_BYTES))}")
    data = run(find_script(binary), model, context=context, gen=gen,
               cache_mode=cache_type)

    res = Result(
        model=shape.name or model, quant=quant or f"{shape.bpw:.2f}bpw",
        context=context, backend="exllamav2", gpu=gpu,
        prefill_tok_s=data["prefill"], decode_tok_s=data["decode"],
        driver=driver, backend_version=data["version"],
        cache_dtype=cache_type)

    spec = specs.lookup(gpu)
    if spec is None:
        res.notes.append(f"no spec row for {gpu!r}: ratios left at zero")
        return res
    dec = roofline.decode_ceiling(shape, context, spec.bandwidth_bps,
                                  CACHE_BYTES[cache_type])
    pre = roofline.prefill_ceiling(shape, spec.fp16_tflops * 1e12)
    res.decode_frac_peak = roofline.fraction_of_peak(data["decode"], dec)
    res.prefill_frac_peak = roofline.fraction_of_peak(data["prefill"],
                                                      pre)
    res.achieved_bandwidth_gbs = roofline.achieved_bandwidth(
        data["decode"], dec.bytes_per_token) / 1e9
    res.notes.append("exllamav2 cache modes are not ggml block sizes; "
                     "cache cost taken from its own table")
    return res
