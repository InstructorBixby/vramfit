from vramfit import cli
from vramfit.bench.result import Result
from vramfit.ledger import Ledger

GIB = 1024 ** 3


def test_empty_ledger(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    assert cli.main(["accuracy", "--db", db]) == 0
    assert "nothing to compare" in capsys.readouterr().out


def test_reports_error_for_a_published_shape(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    with Ledger(db) as led:
        led.record(Result(
            model="Meta-Llama-3.1-8B-Instruct", quant="q4_k_m",
            context=8192, backend="llama.cpp", gpu="RTX 3090",
            decode_tok_s=118.0,
            vram_peak=int(5.5 * GIB)), taken_at="T1")
    assert cli.main(["accuracy", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "signed error" in out


def test_unknown_model_is_skipped(tmp_path, capsys):
    db = str(tmp_path / "l.sqlite")
    with Ledger(db) as led:
        led.record(Result(
            model="SomeFinetune-9B", quant="q4_k_m", context=8192,
            backend="llama.cpp", gpu="RTX 3090",
            vram_peak=int(5.5 * GIB)), taken_at="T1")
    assert cli.main(["accuracy", "--db", db]) == 0
    assert "metadata" in capsys.readouterr().out
