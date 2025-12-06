"""Backend: vLLM, measured through its OpenAI-compatible server.

No in-process import. The server is the thing being measured, and a
`LLM(...)` in this process would measure a different configuration with
different defaults.

Separating the two speeds here takes more care than with llama-bench,
because a completion request gives one wall-clock number for both
phases. The split comes from the stream: time to first token is
prefill, and the rest divided by the remaining tokens is decode. That
is what `usage` cannot tell you and the stream can.
"""
from __future__ import annotations

import json
import time
import urllib.request

from vramfit import roofline, specs
from vramfit.bench import register
from vramfit.bench.result import Result


class BenchError(Exception):
    pass


def post_stream(url: str, payload: dict, timeout: float = 600.0):
    """Yield (arrival time, chunk) for an SSE completion stream."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield time.perf_counter(), json.loads(body)
            except json.JSONDecodeError:
                continue


def timings(url: str, model: str, prompt: str, gen: int) -> tuple:
    """(ttft seconds, decode seconds, tokens generated)."""
    start = time.perf_counter()
    first = None
    last = start
    n = 0
    for at, chunk in post_stream(url, {
            "model": model, "prompt": prompt, "max_tokens": gen,
            "temperature": 0.0, "stream": True}):
        choices = chunk.get("choices") or []
        if not choices or not choices[0].get("text"):
            continue
        if first is None:
            first = at
        last = at
        n += 1
    if first is None or n < 2:
        raise BenchError("stream produced fewer than two tokens; "
                         "cannot separate prefill from decode")
    return first - start, last - first, n


def server_version(base: str) -> str:
    try:
        with urllib.request.urlopen(f"{base}/version", timeout=10) as r:
            return json.loads(r.read()).get("version", "")
    except Exception:
        return ""


@register("vllm")
def bench(shape, *, model: str, gpu: str, context: int = 4096,
          quant: str = "", base_url: str = "http://127.0.0.1:8000",
          prompt_tokens: int = 512, gen: int = 128,
          cache_type: str = "f16", driver: str = "",
          served_name: str = "", **_) -> Result:
    # Four characters per token is close enough for a synthetic prompt
    # whose content does not matter.
    prompt = "word " * prompt_tokens
    ttft, decode_s, n = timings(f"{base_url}/v1/completions",
                                served_name or model, prompt, gen)
    prefill_tok_s = prompt_tokens / ttft if ttft > 0 else 0.0
    decode_tok_s = (n - 1) / decode_s if decode_s > 0 else 0.0

    res = Result(
        model=shape.name or model, quant=quant or f"{shape.bpw:.2f}bpw",
        context=context, backend="vllm", gpu=gpu,
        prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s,
        driver=driver, backend_version=server_version(base_url),
        cache_dtype=cache_type)
    res.notes.append(
        "prefill from time-to-first-token, decode from the inter-token "
        "intervals; the two cannot be separated from a non-streamed "
        "response")

    spec = specs.lookup(gpu)
    if spec is None:
        res.notes.append(f"no spec row for {gpu!r}: ratios left at zero")
        return res
    dec = roofline.decode_ceiling(shape, context, spec.bandwidth_bps,
                                  2 if cache_type == "f16" else 1)
    pre = roofline.prefill_ceiling(shape, spec.fp16_tflops * 1e12)
    res.decode_frac_peak = roofline.fraction_of_peak(decode_tok_s, dec)
    res.prefill_frac_peak = roofline.fraction_of_peak(prefill_tok_s, pre)
    res.achieved_bandwidth_gbs = roofline.achieved_bandwidth(
        decode_tok_s, dec.bytes_per_token) / 1e9
    return res
