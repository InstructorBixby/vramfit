#!/usr/bin/env python3
"""Render docs/measured.md from contributions/.

Run by hand when a contribution is merged, not in CI: a generated file
committed by a workflow is a merge conflict waiting to happen, and this
takes a second.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vramfit import contrib, measured  # noqa: E402

HEADER = """\
# measured

Contributed measurements, one row per (model, quant, context, backend,
card). Generated from `contributions/` -- see
[contributing a measurement](contributing-measurements.md).

`% of peak` is the achieved memory bandwidth as a fraction of the card's
advertised peak, and it is the only column that compares across
different hardware: decode tok/s on six different cards is six
unrelated numbers. A card or a model with no published shape gets no
fraction rather than a plausible one.
"""


def main() -> int:
    entries: list[dict] = []
    for path in sorted((ROOT / "contributions").glob("*.json")):
        found, problems = contrib.load_file(path)
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
