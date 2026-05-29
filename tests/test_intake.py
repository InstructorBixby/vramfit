import json

import pytest

from vramfit import intake

GOOD = {
    "model": "Meta-Llama-3.1-8B-Instruct",
    "quant": "q4_k_m",
    "context": 8192,
    "backend": "llama.cpp",
    "backend_version": "b4123",
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "560.35.03",
    "cache_dtype": "f16",
    "decode_tok_s": 162.4,
    "nvidia_smi": "name, memory.total [MiB], driver_version\n"
                  "NVIDIA GeForce RTX 4090, 24564 MiB, 560.35.03",
}


def test_a_complete_entry_passes():
    assert intake.validate(GOOD) == []


def test_required_fields_match_the_schema():
    # The hand-written validator and the published schema drift apart
    # if nothing holds them together.
    schema = json.loads(intake.SCHEMA.read_text())
    assert sorted(schema["required"]) == sorted(intake.REQUIRED)


def test_latest_is_not_a_backend_version():
    bad = dict(GOOD, backend_version="latest")
    problems = [str(p) for p in intake.validate(bad)]
    assert any("not a version" in p for p in problems)


def test_nvidia_smi_must_look_like_nvidia_smi():
    bad = dict(GOOD, nvidia_smi="an RTX 4090 I think, 24GB of VRAM")
    assert any("nvidia_smi" in str(p) for p in intake.validate(bad))


def test_missing_driver_is_reported():
    bad = dict(GOOD)
    del bad["driver"]
    assert [str(p) for p in intake.validate(bad)] == \
        ["driver: required"]


def test_unknown_backend_is_rejected():
    bad = dict(GOOD, backend="ollama")
    assert any("not one of" in str(p) for p in intake.validate(bad))


def test_a_field_not_in_the_schema_is_rejected():
    bad = dict(GOOD, tokens_per_watt=3.2)
    assert any("not in the schema" in str(p)
               for p in intake.validate(bad))


def test_load_file_accepts_a_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([GOOD, GOOD]))
    entries, problems = intake.load_file(p)
    assert len(entries) == 2 and problems == []


def test_load_file_prefixes_the_index(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([GOOD, dict(GOOD, backend="ollama")]))
    _, problems = intake.load_file(p)
    assert str(problems[0]).startswith("[1].backend")
