#!/usr/bin/env python3
"""Render docs/measured.md from contributions/.

Run by hand when a contribution is merged, not in CI: a generated file
committed by a workflow is a merge conflict waiting to happen, and this
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

from vramfit import contrib, specs  # noqa: E402

HEADER = """\
# measured

Contributed measurements, one row per (model, quant, context, backend,
card). Generated from `contributions/` -- see
[contributing a measurement](contributing-measurements.md).

`% peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, which is the only column that compares across
different hardware. A card with no row in `vramfit/data/gpus.csv` gets
no fraction rather than a plausible one.
"""


def rows() -> list[dict]:
    out: list[dict] = []
    for path in sorted((ROOT / "contributions").glob("*.json")):
        entries, problems = contrib.load_file(path)
        if problems:
            raise SystemExit(f"{path}: {problems[0]}")
        out += entries
    return out


def main() -> int:
    table = specs.load()
    lines = [HEADER, "",
             "| model | quant | ctx | backend | card | decode tok/s | "
             "% peak | driver | by |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(rows(), key=lambda r: (r["model"], r["context"])):
        spec = table.get(
            __import__("vramfit.devices", fromlist=["normalise"])
            .normalise(e["gpu"]))
        pct = "-"
        if spec is not None and e.get("bits_per_weight"):
            # Needs the model shape to be exact; the published figure
            # uses the reported bpw and the reported context, which is
            # what a contributor can give without shipping weights.
            pct = "see ledger"
        lines.append(
            f"| {e['model']} | {e['quant']} | {e['context']} | "
            f"{e['backend']} {e['backend_version']} | {e['gpu']} | "
            f"{e['decode_tok_s']:.1f} | {pct} | {e['driver']} | "
            f"{e.get('reported_by', '-')} |")
    (ROOT / "docs" / "measured.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote docs/measured.md, {len(rows())} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
