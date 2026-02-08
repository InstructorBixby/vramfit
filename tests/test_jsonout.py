import json

from vramfit import budget, jsonout


def test_every_number_is_bytes(llama8b):
    b = budget.build(llama8b, 8192)
    out = jsonout.budget_dict(llama8b, 8192, b, vram=16 * 1024 ** 3)
    assert out["bytes"]["weights"] == llama8b.weight_bytes
    assert out["bytes"]["total"] == b.total
    assert out["fits"] is True


def test_assumptions_are_in_the_schema(llama8b):
    b = budget.build(llama8b, 8192)
    out = jsonout.budget_dict(llama8b, 8192, b)
    assert out["assumptions"]


def test_round_trips(llama8b):
    b = budget.build(llama8b, 8192)
    blob = jsonout.dump(jsonout.budget_dict(llama8b, 8192, b))
    assert json.loads(blob)["kv_heads"] == 8


def test_no_vram_means_no_verdict(llama8b):
    out = jsonout.budget_dict(llama8b, 8192, budget.build(llama8b, 8192))
    assert "fits" not in out
