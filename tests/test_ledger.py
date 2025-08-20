import pytest

from vramfit.bench.result import Result
from vramfit.ledger import Ledger


def res(**kw):
    base = dict(model="llama-3.1-8b", quant="q4_k_m", context=8192,
                backend="llama.cpp", gpu="NVIDIA GeForce RTX 3090",
                decode_tok_s=118.4, decode_frac_peak=0.81,
                driver="550.54.14", backend_version="a1b2c3d")
    base.update(kw)
    return Result(**base)


@pytest.fixture
def led(tmp_path):
    with Ledger(tmp_path / "l.sqlite") as led:
        yield led


def test_record_then_read_back(led):
    led.record(res(), taken_at="T1")
    rows = led.history(model="llama-3.1-8b")
    assert len(rows) == 1
    assert rows[0]["decode_tok_s"] == 118.4


def test_same_key_appends_rather_than_overwrites(led):
    led.record(res(), taken_at="T1")
    led.record(res(decode_tok_s=124.9, driver="550.78"),
               taken_at="T2")
    rows = led.history(model="llama-3.1-8b", quant="q4_k_m",
                       context=8192, backend="llama.cpp",
                       gpu="NVIDIA GeForce RTX 3090")
    assert [r["decode_tok_s"] for r in rows] == [118.4, 124.9]


def test_keys_groups_by_the_logical_key(led):
    led.record(res(), taken_at="T1")
    led.record(res(context=32768), taken_at="T2")
    led.record(res(), taken_at="T3")
    keys = led.keys()
    assert len(keys) == 2
    by_ctx = {k["context"]: k["n"] for k in keys}
    assert by_ctx == {8192: 2, 32768: 1}


def test_latest_is_the_newest(led):
    led.record(res(), taken_at="T1")
    led.record(res(decode_tok_s=130.0), taken_at="T2")
    assert led.latest(model="llama-3.1-8b")["decode_tok_s"] == 130.0


def test_empty_ledger_has_no_latest(led):
    assert led.latest(model="nothing") is None
