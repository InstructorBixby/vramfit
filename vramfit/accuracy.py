"""Prediction error, from the ledger rather than from memory.

Every benchmark row carries a measured VRAM peak, and the prediction for
that configuration can be recomputed from the model metadata. The
residual is therefore available for every measurement ever taken, which
makes "the predictions are within a couple of percent" a checkable
statement instead of a boast.

Reported signed, never as an absolute value. A systematically low
prediction is a missing term and a fixable bug; a systematically high
one is wasted capacity; errors that average out are noise. Taking the
absolute value first throws away the only thing worth knowing.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median


@dataclass
class Residual:
    row: dict
    predicted: int
    measured: int

    @property
    def error(self) -> float:
        return ((self.measured - self.predicted) / self.predicted
                if self.predicted else 0.0)


@dataclass
class Summary:
    residuals: list
    skipped: int
    skipped_why: list

    @property
    def n(self) -> int:
        return len(self.residuals)

    @property
    def mean_error(self) -> float:
        return mean(r.error for r in self.residuals) if self.n else 0.0

    @property
    def median_error(self) -> float:
        return (median(r.error for r in self.residuals)
                if self.n else 0.0)

    @property
    def worst(self):
        return (max(self.residuals, key=lambda r: abs(r.error))
                if self.n else None)

    def table(self) -> str:
        if not self.n:
            # The skip reasons are the ONLY information in this case,
            # which made dropping them here the worst place to drop
            # them: "nothing to compare" on its own reads like an
            # empty ledger when the real answer is that every row was
            # unreadable, which is a different problem with a
            # different fix.
            out = ["no rows with both a measured peak and a readable "
                   "model: nothing to compare"]
            out += [f"  ! {why}" for why in self.skipped_why]
            return "\n".join(out)
        out = [f"{self.n} comparable measurement(s), "
               f"{self.skipped} skipped", "",
               f"  mean signed error   {self.mean_error:+.1%}",
               f"  median signed error {self.median_error:+.1%}"]
        worst = self.worst
        if worst is not None:
            row = worst.row
            out += ["",
                    f"  worst: {worst.error:+.1%} on {row['model']} "
                    f"{row['quant']} ctx {row['context']} "
                    f"{row['backend']} / {row['gpu']}"]
        if self.mean_error > 0.03:
            out += ["",
                    "  mean error is positive and above 3%: a "
                    "systematically low prediction is a missing term, "
                    "not noise"]
        elif self.mean_error < -0.03:
            out += ["",
                    "  mean error is negative and above 3%: capacity "
                    "is being left unused"]
        for why in self.skipped_why:
            out.append(f"  ! {why}")
        return "\n".join(out)


def residuals(rows: list, predict) -> Summary:
    """`predict(row)` returns predicted bytes, or None if it cannot."""
    out, skipped, why = [], 0, []
    for row in rows:
        measured = row.get("vram_peak") or 0
        if not measured:
            skipped += 1
            continue
        predicted = predict(row)
        if not predicted:
            skipped += 1
            reason = "no readable model metadata for some rows"
            if reason not in why:
                why.append(reason)
            continue
        out.append(Residual(row, predicted, measured))
    return Summary(out, skipped, why)
