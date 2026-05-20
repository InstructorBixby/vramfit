import pytest

from vramfit import cli, devices

from .conftest import FIXTURES

MODEL = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])


def test_new_spelling_is_quiet(recwarn):
    cli.main(["plan", MODEL, "--vram", "24GiB",
              "--gpu-memory-utilization", "0.8"])
    assert not [w for w in recwarn
                if issubclass(w.category, DeprecationWarning)]


def test_old_spelling_still_works_and_warns():
    with pytest.warns(DeprecationWarning, match="gpu-memory-utilization"):
        rc = cli.main(["plan", MODEL, "--vram", "24GiB",
                       "--utilisation", "0.8"])
    assert rc == 0


def test_the_warning_names_the_removal_release():
    with pytest.warns(DeprecationWarning, match="removed in 2.0"):
        cli.main(["plan", MODEL, "--vram", "24GiB",
                  "--utilisation", "0.8"])


def test_both_spellings_reach_the_same_destination(capsys):
    cli.main(["plan", MODEL, "--vram", "24GiB",
              "--gpu-memory-utilization", "0.5"])
    new = capsys.readouterr().out
    with pytest.warns(DeprecationWarning):
        cli.main(["plan", MODEL, "--vram", "24GiB",
                  "--utilisation", "0.5"])
    assert capsys.readouterr().out == new
