#!/usr/bin/env python3
"""Render docs/measured.md from intake/.

Run by hand after an import, not in the pipeline: a generated file
committed by a job is a merge conflict waiting to happen, and this
takes a second.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vramfit import intake, measured  # noqa: E402

HEADER = """\
# measured

Every measurement I have kept, one row per (model, quant, context,
backend, card). Generated from `intake/` -- see
[recording a measurement off-box](measuring.md).

`% of peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, and it is the only column that compares across
different hardware: decode tok/s on six different cards is six
unrelated numbers. A card or a model with no published shape gets no
fraction rather than a plausible one.
"""


def main() -> int:
    entries: list[dict] = []
    for path in sorted((ROOT / "intake").glob("*.json")):
        found, problems = intake.load_file(path)
        if problems:
            raise SystemExit(f"{path}: {problems[0]}")
        entries += found
    body = measured.markdown(entries)
    (ROOT / "docs" / "measured.md").write_text(
        HEADER + "\n" + body + "\n", encoding="utf-8")
    print(f"wrote docs/measured.md, {len(entries)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
