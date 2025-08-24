from vramfit import ledger_diff


BEFORE = {"taken_at": "T1", "decode_tok_s": 118.4,
          "decode_frac_peak": 0.81, "achieved_bandwidth_gbs": 758.0,
          "driver": "550.54.14", "backend_version": "a1b2c3d",
          "cache_dtype": "f16", "n_gpu_layers": 999, "batch": 512}


def test_identical_rows_report_no_change():
    out = ledger_diff.render(BEFORE, dict(BEFORE, taken_at="T2"))
    assert "no measured change" in out


def test_noise_is_not_a_change():
    after = dict(BEFORE, taken_at="T2", decode_tok_s=119.5)
    measured, _ = ledger_diff.diff(BEFORE, after)
    assert not any(c.significant for c in measured)


def test_real_change_is_reported_with_a_percentage():
    after = dict(BEFORE, taken_at="T2", decode_tok_s=131.0)
    out = ledger_diff.render(BEFORE, after)
    assert "decode_tok_s" in out
    assert "+10.6%" in out


def test_provenance_change_is_shown_next_to_the_numbers():
    after = dict(BEFORE, taken_at="T2", decode_tok_s=131.0,
                 driver="550.78")
    out = ledger_diff.render(BEFORE, after)
    assert "changed at the same time" in out
    assert "550.78" in out
    assert "attribute the change" in out


def test_all_fields_shows_the_quiet_ones():
    after = dict(BEFORE, taken_at="T2", decode_tok_s=119.0)
    out = ledger_diff.render(BEFORE, after, all_fields=True)
    assert "decode_tok_s" in out
