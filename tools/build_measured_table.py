#!/usr/bin/env python3
"""Render docs/measured.md from intake/.

Run by hand after an import, not in the pipeline: a generated file
committed by a job is a merge conflict waiting to happen, and this
takes a second.

The achieved-bandwidth column is the point of the table. Absolute tok/s
across six different cards says nothing; the fraction of each card's own
peak says whether the backend is doing its job on that hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vramfit import intake, specs  # noqa: E402

HEADER = """\
# measured

Every measurement I have kept, one row per (model, quant, context,
backend, card). Generated from `intake/` -- see
[recording a measurement off-box](measuring.md).

`% peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, which is the only column that compares across
different hardware. A card with no row in `vramfit/data/gpus.csv` gets
no fraction rather than a plausible one.
"""


def rows() -> list[dict]:
    out: list[dict] = []
    for path in sorted((ROOT / "intake").glob("*.json")):
        entries, problems = intake.load_file(path)
        if problems:
            raise SystemExit(f"{path}: {problems[0]}")
        out += entries
    return out


def main() -> int:
    table = specs.load()
    lines = [HEADER, "",
             "| model | quant | ctx | backend | card | decode tok/s | "
             "% peak | driver | measured on |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(rows(), key=lambda r: (r["model"], r["context"])):
        spec = table.get(
            __import__("vramfit.devices", fromlist=["normalise"])
            .normalise(e["gpu"]))
        pct = "-"
        if spec is not None and e.get("bits_per_weight"):
            # Needs the model shape to be exact; this uses the recorded
            # bpw and context, which is all I have once the weights are
            # back on a machine I no longer have.
            pct = "see ledger"
        lines.append(
            f"| {e['model']} | {e['quant']} | {e['context']} | "
            f"{e['backend']} {e['backend_version']} | {e['gpu']} | "
            f"{e['decode_tok_s']:.1f} | {pct} | {e['driver']} | "
            f"{e.get('measured_on', '-')} |")
    (ROOT / "docs" / "measured.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote docs/measured.md, {len(rows())} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
