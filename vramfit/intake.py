"""Validate a measurement before it joins the ledger.

The schema in `vramfit/schema/` is the documentation; this is the
enforcement, hand-written so the package keeps no runtime dependency on
a JSON Schema library. `tests/test_intake.py` asserts that the two
agree on the required fields, which is the part that drifts.

Every rule here exists because a measurement without it turned out to
be unreproducible, and it is enforced on me rather than suggested to
me, because the moment of writing a number down is the moment I am most
certain I will remember the rest. Six months later I cannot tell you
which driver a figure came from.

"Latest" is not a backend version. A GPU name typed by hand is not a
GPU; the `nvidia-smi` output is, because it also carries the memory
size and the driver, and twice now a number of mine has turned out to
be from a 12 GiB variant of a card I had written down as 16.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA = Path(__file__).parent / "schema" / "ledger-entry.schema.json"

REQUIRED = ("model", "quant", "context", "backend", "backend_version",
            "gpu", "driver", "cache_dtype", "decode_tok_s",
            "nvidia_smi")
BACKENDS = ("llama.cpp", "vllm", "exllamav2")
DRIVER_RE = re.compile(r"^[0-9]+\.[0-9]+")
VAGUE = ("latest", "master", "main", "head", "current", "newest",
         "unknown", "n/a")


@dataclass
class Problem:
    field: str
    why: str

    def __str__(self) -> str:
        return f"{self.field}: {self.why}"


def validate(entry: dict) -> list[Problem]:
    out: list[Problem] = []
    for field in REQUIRED:
        if entry.get(field) in (None, "", []):
            out.append(Problem(field, "required"))

    backend = entry.get("backend")
    if backend is not None and backend not in BACKENDS:
        out.append(Problem("backend",
                           f"not one of {', '.join(BACKENDS)}"))

    version = str(entry.get("backend_version") or "")
    if version.strip().lower() in VAGUE:
        out.append(Problem(
            "backend_version",
            f"{version!r} is not a version. A build commit or a "
            f"release tag -- the whole point of the field is being "
            f"able to build the same thing again later"))

    driver = str(entry.get("driver") or "")
    if driver and not DRIVER_RE.match(driver):
        out.append(Problem("driver",
                           f"{driver!r} does not look like a driver "
                           f"version"))

    smi = str(entry.get("nvidia_smi") or "")
    if smi and len(smi) < 20:
        out.append(Problem("nvidia_smi", "too short to be real output"))
    # Checking only for a comma let "an RTX 4090 I think, 24GB of VRAM"
    # through, which is exactly the thing this field exists to refuse.
    # The header row is the tell: it names the columns, so it is proof
    # the string came out of the command and was not typed.
    if smi and ("driver_version" not in smi or smi.count(",") < 2):
        out.append(Problem(
            "nvidia_smi",
            "expected the CSV output of `nvidia-smi --query-gpu="
            "name,memory.total,driver_version --format=csv`, header "
            "row included -- a card name typed by hand is not a card"))

    for field in ("decode_tok_s", "prefill_tok_s"):
        val = entry.get(field)
        if val is not None and (not isinstance(val, (int, float))
                                or val <= 0):
            out.append(Problem(field, "must be a positive number"))

    ctx = entry.get("context")
    if ctx is not None and (not isinstance(ctx, int) or ctx < 1):
        out.append(Problem("context", "must be a positive integer"))

    known = set(json.loads(SCHEMA.read_text())["properties"])
    for field in sorted(set(entry) - known):
        out.append(Problem(field, "not in the schema"))
    return out


def load_file(path: str | Path) -> tuple[list[dict], list[Problem]]:
    """Read one intake file: a single entry or a list of them."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else [raw]
    problems: list[Problem] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(Problem(f"[{i}]", "not an object"))
            continue
        for p in validate(entry):
            problems.append(Problem(f"[{i}].{p.field}", p.why))
    return entries, problems
