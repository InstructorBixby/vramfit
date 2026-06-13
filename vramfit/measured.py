"""Turn ledger entries into a comparable table.

The hard part is the `% of peak` column, and it is the only column worth
having: decode tok/s across six different cards is six unrelated
numbers, while the fraction of each card's own bandwidth says whether
the backend is doing its job there.

Computing it needs bytes-per-token, which needs the model's layer and
head counts. The weights are not coming back from a machine I borrowed
for an afternoon, so the shape comes from a small table of published
architectures keyed on the model name, and an unknown model gets no
fraction. That table is in `data/shapes.csv` and every row is a number
out of a config.json that can be checked against the model itself.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from vramfit import cache, devices, specs
from vramfit.shape import Shape

SHAPES = Path(__file__).parent / "data" / "shapes.csv"


@dataclass
class Row:
    entry: dict
    decode_tok_s: float
    fraction: float | None
    achieved_gbs: float | None
    why_no_fraction: str = ""


def known_shapes(path: str | Path = SHAPES) -> dict[str, Shape]:
    out: dict[str, Shape] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(
                (ln for ln in fh if not ln.startswith("#"))):
            out[row["model"].lower()] = Shape(
                name=row["model"], arch=row["arch"],
                n_layers=int(row["layers"]),
                n_embd=int(row["n_embd"]),
                n_ff=int(row["n_ff"]),
                n_head=int(row["n_head"]),
                n_head_kv=int(row["n_head_kv"]),
                head_dim=int(row["head_dim"]),
                vocab=int(row["vocab"]),
                train_context=int(row["train_context"]))
    return out


def row_for(entry: dict, shapes: dict[str, Shape] | None = None,
            table: dict[str, specs.Spec] | None = None) -> Row:
    from vramfit import roofline

    shapes = known_shapes() if shapes is None else shapes
    table = specs.load() if table is None else table
    decode = float(entry["decode_tok_s"])

    shape = shapes.get(str(entry["model"]).lower())
    spec = table.get(devices.normalise(str(entry["gpu"])))
    if shape is None:
        return Row(entry, decode, None, None,
                   "no published shape for this model: add a row to "
                   "data/shapes.csv")
    if spec is None:
        return Row(entry, decode, None, None,
                   "no spec row for this card")
    bpw = entry.get("bits_per_weight")
    if not bpw:
        return Row(entry, decode, None, None,
                   "no bits_per_weight reported")

    weights = int(shape.params_from_config() * float(bpw) / 8)
    cb = cache.element_bytes(str(entry.get("cache_dtype") or "f16"))
    ceiling = roofline.decode_ceiling(shape, int(entry["context"]),
                                      spec.bandwidth_bps, cb, weights)
    achieved = roofline.achieved_bandwidth(decode,
                                           ceiling.bytes_per_token)
    return Row(entry, decode,
               roofline.fraction_of_peak(decode, ceiling),
               achieved / 1e9)


def markdown(entries: list[dict]) -> str:
    shapes, table = known_shapes(), specs.load()
    lines = ["| model | quant | ctx | backend | card | decode tok/s | "
             "% of peak | GB/s | driver | measured on |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    footnotes: list[str] = []
    notes: list[str] = []
    for entry in sorted(entries,
                        key=lambda e: (e["model"], e["context"])):
        row = row_for(entry, shapes, table)
        if row.fraction is None:
            pct, gbs = "-", "-"
            if row.why_no_fraction not in footnotes:
                footnotes.append(row.why_no_fraction)
        else:
            pct = f"{row.fraction:.0%}"
            gbs = f"{row.achieved_gbs:.0f}"
        if entry.get("notes"):
            notes.append(f"{entry['model']} on {entry['gpu']}: "
                         f"{entry['notes']}")
        lines.append(
            f"| {entry['model']} | {entry['quant']} | "
            f"{entry['context']} | {entry['backend']} "
            f"{entry['backend_version']} | {entry['gpu']} | "
            f"{row.decode_tok_s:.1f} | {pct} | {gbs} | "
            f"{entry['driver']} | {entry.get('measured_on', '-')} |")
    # A fraction below the band with nothing next to it explaining why
    # is one I will re-argue from scratch in six months, so whatever
    # the entry said about the machine comes through to the table.
    if notes:
        lines += ["", "Carried from the entries:"]
        lines += [f"- {n}" for n in notes]
    if footnotes:
        lines += ["", "Missing fractions:"]
        lines += [f"- {f}" for f in footnotes]
    return "\n".join(lines)
