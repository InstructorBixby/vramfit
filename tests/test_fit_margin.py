import pytest

from vramfit import cli, devices

from .conftest import FIXTURES

MODEL = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])


def test_context_defaults_to_the_trained_length(capsys):
    cli.main(["fit", MODEL, "--vram", "80GiB"])
    assert "ctx 131072" in capsys.readouterr().out


def test_reserve_comes_off_the_top(capsys):
    cli.main(["fit", MODEL, "--context", "8192", "--vram", "16GiB"])
    without = capsys.readouterr().out
    cli.main(["fit", MODEL, "--context", "8192", "--vram", "16GiB",
              "--reserve", "4GiB"])
    with_reserve = capsys.readouterr().out
    assert "fits" in without
    assert with_reserve != without


def test_parallel_multiplies_the_cache(capsys):
    cli.main(["fit", MODEL, "--context", "8192", "--vram", "16GiB",
              "--parallel", "4"])
    out = capsys.readouterr().out
    assert "4 slots" in out
    assert "32768 tokens" in out


def test_thin_margin_is_reported_as_a_maybe(capsys):
    # Chosen so the slack lands under 5% of the card.
    cli.main(["fit", MODEL, "--context", "32768", "--vram", "8.9GiB"])
    out = capsys.readouterr().out
    assert "treat it as a maybe" in out or "does not fit" in out


def test_comfortable_fit_says_spare(capsys):
    cli.main(["fit", MODEL, "--context", "4096", "--vram", "24GiB"])
    assert "spare" in capsys.readouterr().out
