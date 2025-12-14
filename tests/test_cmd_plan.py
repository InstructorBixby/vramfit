import pytest

from vramfit import cli, devices

from .conftest import FIXTURES

MODEL = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])


def test_plan_prints_blocks_and_concurrency(capsys):
    assert cli.main(["plan", MODEL, "--vram", "24GiB"]) == 0
    out = capsys.readouterr().out
    assert "blocks" in out
    assert "concurrent sequence" in out


def test_plan_on_a_card_too_small_exits_one(capsys):
    assert cli.main(["plan", MODEL, "--vram", "5GiB"]) == 1
    assert "accept nothing" in capsys.readouterr().out


def test_utilisation_changes_the_block_count(capsys):
    cli.main(["plan", MODEL, "--vram", "24GiB", "--utilisation", "0.9"])
    high = capsys.readouterr().out
    cli.main(["plan", MODEL, "--vram", "24GiB", "--utilisation", "0.6"])
    low = capsys.readouterr().out
    assert high != low
