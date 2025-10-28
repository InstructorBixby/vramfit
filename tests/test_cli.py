import pytest

from vramfit import cli, devices

from .conftest import FIXTURES

MODEL = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """The CLI must be usable on a machine with no CUDA device."""
    monkeypatch.setattr(devices, "discover", lambda: [])


def test_parse_size_units():
    assert cli.parse_size("16GiB") == 16 * 1024 ** 3
    assert cli.parse_size("16 gib") == 16 * 1024 ** 3
    assert cli.parse_size("8GB") == 8 * 10 ** 9
    assert cli.parse_size("512MiB") == 512 * 1024 ** 2
    assert cli.parse_size("1024") == 1024


def test_shape_prints_the_gqa_ratio(capsys):
    assert cli.main(["shape", MODEL]) == 0
    out = capsys.readouterr().out
    assert "32 attention / 8 kv" in out
    assert "gqa 4x" in out


def test_fit_exits_one_when_it_does_not_fit(capsys):
    rc = cli.main(["fit", MODEL, "--context", "131072",
                   "--vram", "8GiB"])
    assert rc == 1
    assert "does not fit" in capsys.readouterr().out


def test_max_context_reports_a_number(capsys):
    assert cli.main(["fit", MODEL, "--max-context",
                     "--vram", "16GiB"]) == 0
    assert "largest context that fits" in capsys.readouterr().out


def test_for_context_names_a_quant(capsys):
    assert cli.main(["fit", MODEL, "--for-context", "32768",
                     "--vram", "12GiB"]) == 0
    assert "bpw" in capsys.readouterr().out


def test_max_layers_on_a_small_card(capsys):
    assert cli.main(["fit", MODEL, "--max-layers", "--vram",
                     "4GiB", "--context", "8192"]) == 0
    out = capsys.readouterr().out
    assert "largest -ngl that fits" in out
    assert "partial offload" in out


def test_max_layers_on_a_big_card(capsys):
    assert cli.main(["fit", MODEL, "--max-layers", "--vram",
                     "24GiB", "--context", "8192"]) == 0
    assert "fits outright" in capsys.readouterr().out


def test_explicit_ngl_is_respected(capsys):
    assert cli.main(["fit", MODEL, "--n-gpu-layers", "16",
                     "--vram", "24GiB"]) == 0
    assert "offloaded" in capsys.readouterr().out


def test_ladder_has_more_than_one_row(capsys):
    assert cli.main(["ladder", MODEL, "--vram", "16GiB"]) == 0
    assert capsys.readouterr().out.count("\n") > 6


def test_fit_without_a_device_or_vram_is_not_a_crash(capsys):
    assert cli.main(["fit", MODEL, "--context", "4096"]) == 0
