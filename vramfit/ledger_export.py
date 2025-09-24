"""Render the ledger as a table I can paste into a report.

Markdown because the numbers get pasted into notes and issues, and CSV
because sometimes the answer is a scatter plot. Both take the latest
measurement per key, since a history of five runs of the same thing is
interesting to me and noise to everyone else.
"""
from __future__ import annotations

import csv
import io

COLUMNS = [
    ("model", "model", "{}"),
    ("quant", "quant", "{}"),
    ("context", "ctx", "{}"),
    ("backend", "backend", "{}"),
    ("gpu", "gpu", "{}"),
    ("prefill_tok_s", "prefill tok/s", "{:.0f}"),
    ("decode_tok_s", "decode tok/s", "{:.1f}"),
    ("decode_frac_peak", "% of bw peak", "{:.0%}"),
    ("achieved_bandwidth_gbs", "GB/s", "{:.0f}"),
    ("driver", "driver", "{}"),
]


def latest_per_key(ledger) -> list:
    out = []
    for key in ledger.keys():
        k = {c: key[c] for c in ("model", "quant", "context",
                                 "backend", "gpu")}
        row = ledger.latest(**k)
        if row:
            out.append(row)
    return out


def _fmt(row, field, spec) -> str:
    val = row.get(field)
    if val is None or val == "":
        return "-"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def markdown(rows: list) -> str:
    head = "| " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(r, f, s) for f, _, s in COLUMNS)
            + " |" for r in rows]
    return "\n".join([head, rule] + body)


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f for f, _, _ in COLUMNS])
    for r in rows:
        w.writerow([r.get(f, "") for f, _, _ in COLUMNS])
    return buf.getvalue()
