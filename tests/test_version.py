import pytest

from vramfit import __version__, cli


def test_version_flag_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_shape_json_is_machine_readable(capsys):
    import json

    from .conftest import FIXTURES
    model = str(FIXTURES / "llama-3.1-8b-instruct-q4_k_m.header.gguf")
    assert cli.main(["shape", model, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kv_heads"] == 8
    assert data["kv_bytes_per_token"] == 2 * 32 * 8 * 128 * 2
