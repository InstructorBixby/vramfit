from vramfit import ledger_export
from vramfit.bench.result import Result
from vramfit.ledger import Ledger


def test_latest_per_key_collapses_history(tmp_path):
    with Ledger(tmp_path / "l.sqlite") as led:
        for i, ts in enumerate(("T1", "T2", "T3")):
            led.record(Result(
                model="m", quant="q4_k_m", context=8192,
                backend="llama.cpp", gpu="RTX 3090",
                decode_tok_s=100.0 + i), taken_at=ts)
        rows = ledger_export.latest_per_key(led)
    assert len(rows) == 1
    assert rows[0]["decode_tok_s"] == 102.0


def test_markdown_has_a_header_and_a_rule():
    md = ledger_export.markdown([{"model": "m", "decode_tok_s": 118.4,
                                  "decode_frac_peak": 0.81}])
    lines = md.splitlines()
    assert lines[0].startswith("| model |")
    assert set(lines[1]) <= set("|-")
    assert "81%" in lines[2]


def test_missing_values_render_as_a_dash():
    md = ledger_export.markdown([{"model": "m"}])
    assert "| - |" in md


def test_csv_round_trips_the_columns():
    out = ledger_export.to_csv([{"model": "m", "decode_tok_s": 1.5}])
    assert out.splitlines()[0].startswith("model,quant,context")
