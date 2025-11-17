import pytest

from vramfit import cli, devices

from .conftest import FIXTURES

MODEL = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])


def test_q8_cache_buys_context(capsys):
    cli.main(["fit", MODEL, "--max-context", "--vram", "16GiB"])
    f16 = capsys.readouterr().out
    cli.main(["fit", MODEL, "--max-context", "--vram", "16GiB",
              "--cache-type", "q8_0", "--flash-attn"])
    q8 = capsys.readouterr().out

    def ctx(blob):
        for line in blob.splitlines():
            if line.startswith("largest context"):
                return int(line.split(":")[1].split()[0])
        raise AssertionError(blob)

    assert ctx(q8) > ctx(f16)


def test_quantised_cache_without_flash_attn_warns(capsys):
    cli.main(["fit", MODEL, "--vram", "16GiB", "--cache-type", "q8_0"])
    assert "needs flash attention" in capsys.readouterr().out


def test_unknown_cache_type_is_rejected():
    with pytest.raises(SystemExit, match="unknown cache type"):
        cli.main(["fit", MODEL, "--cache-type", "q3_k_m"])
