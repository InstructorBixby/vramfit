from vramfit import cli
from vramfit.bench.result import Result
from vramfit.ledger import Ledger


def seed(db, **kw):
    with Ledger(db) as led:
        led.record(Result(
            model="llama-3.1-8b", quant="q4_k_m", context=8192,
            backend="llama.cpp", gpu="NVIDIA GeForce RTX 3090",
            decode_tok_s=kw.get("decode", 118.4),
            decode_frac_peak=0.81,
            driver=kw.get("driver", "550.54.14")),
            taken_at=kw["at"])


def test_list_empty(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    assert cli.main(["ledger", "list", "--db", db]) == 0
    assert "empty" in capsys.readouterr().out


def test_list_shows_the_key(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    seed(db, at="T1")
    assert cli.main(["ledger", "list", "--db", db]) == 0
    assert "q4_k_m" in capsys.readouterr().out


def test_diff_needs_two_rows(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    seed(db, at="T1")
    assert cli.main(["ledger", "diff", "--db", db,
                     "--model", "llama-3.1-8b"]) == 1
    assert "need two" in capsys.readouterr().out


def test_diff_reports_the_driver_change(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    seed(db, at="T1")
    seed(db, at="T2", decode=131.0, driver="550.78")
    assert cli.main(["ledger", "diff", "--db", db,
                     "--model", "llama-3.1-8b"]) == 0
    out = capsys.readouterr().out
    assert "550.78" in out and "%" in out
