"""A table of measurements I have actually taken.

The reason this exists: every answer to "how fast is X on Y" is a number
somebody measured once and then remembered wrongly. Prediction gets the
shape right and measurement gets the number right, and the number is
only useful if it is still there after the next driver upgrade.

The logical key is (model, quant, context, backend, gpu). Everything
else -- driver version, backend build, cache dtype, batch -- is
provenance, and a new measurement under the same key is appended rather
than overwriting, because overwriting is how you lose the ability to
answer "did that upgrade help".

sqlite because it is in the standard library, it survives being opened
by two processes, and a ledger that needs a server is a ledger nobody
keeps.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vramfit.bench.result import Result

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id                     INTEGER PRIMARY KEY,
    model                  TEXT NOT NULL,
    quant                  TEXT NOT NULL,
    context                INTEGER NOT NULL,
    backend                TEXT NOT NULL,
    gpu                    TEXT NOT NULL,
    cache_dtype            TEXT NOT NULL DEFAULT 'f16',
    n_gpu_layers           INTEGER,
    batch                  INTEGER,
    prefill_tok_s          REAL,
    decode_tok_s           REAL,
    prefill_frac_peak      REAL,
    decode_frac_peak       REAL,
    achieved_bandwidth_gbs REAL,
    vram_peak              INTEGER,
    driver                 TEXT,
    backend_version        TEXT,
    notes                  TEXT,
    taken_at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_key
    ON results (model, quant, context, backend, gpu);
"""

KEY_COLUMNS = ("model", "quant", "context", "backend", "gpu")


def default_path() -> Path:
    base = os.environ.get("VRAMFIT_LEDGER")
    if base:
        return Path(base)
    home = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(home).expanduser() / "vramfit" / "ledger.sqlite"


class Ledger:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def record(self, res: Result, taken_at: str = "") -> int:
        cur = self.db.execute(
            """INSERT INTO results
               (model, quant, context, backend, gpu, cache_dtype,
                n_gpu_layers, batch, prefill_tok_s, decode_tok_s,
                prefill_frac_peak, decode_frac_peak,
                achieved_bandwidth_gbs, vram_peak, driver,
                backend_version, notes, taken_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (res.model, res.quant, res.context, res.backend, res.gpu,
             res.cache_dtype, res.n_gpu_layers, res.batch,
             res.prefill_tok_s, res.decode_tok_s,
             res.prefill_frac_peak, res.decode_frac_peak,
             res.achieved_bandwidth_gbs, res.vram_peak, res.driver,
             res.backend_version, json.dumps(res.notes),
             taken_at or datetime.now(timezone.utc).isoformat(
                 timespec="seconds")))
        self.db.commit()
        return cur.lastrowid

    def keys(self) -> list:
        cols = ", ".join(KEY_COLUMNS)
        rows = self.db.execute(
            f"SELECT {cols}, COUNT(*) AS n, MAX(taken_at) AS latest "
            f"FROM results GROUP BY {cols} "
            f"ORDER BY model, quant, context, backend, gpu").fetchall()
        return [dict(r) for r in rows]

    def history(self, **key) -> list:
        where = " AND ".join(f"{c} = :{c}" for c in KEY_COLUMNS
                             if c in key)
        sql = "SELECT * FROM results"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY taken_at, id"
        return [dict(r) for r in self.db.execute(sql, key).fetchall()]

    def latest(self, **key):
        rows = self.history(**key)
        return rows[-1] if rows else None
