from vramfit import accuracy

GIB = 1024 ** 3


def row(peak_gib, model="m"):
    return {"model": model, "quant": "q4_k_m", "context": 8192,
            "backend": "llama.cpp", "gpu": "RTX 3090",
            "vram_peak": int(peak_gib * GIB)}


def test_error_is_signed():
    s = accuracy.residuals([row(5.5)], lambda r: int(5.0 * GIB))
    assert s.residuals[0].error > 0
    s = accuracy.residuals([row(4.5)], lambda r: int(5.0 * GIB))
    assert s.residuals[0].error < 0


def test_errors_that_cancel_do_not_look_accurate_by_accident():
    s = accuracy.residuals([row(5.5), row(4.5)],
                           lambda r: int(5.0 * GIB))
    assert abs(s.mean_error) < 1e-9
    # The worst case is still visible even though the mean is zero.
    assert abs(s.worst.error) > 0.09


def test_rows_without_a_measured_peak_are_skipped():
    rows = [row(5.5), {"model": "m", "vram_peak": 0}]
    s = accuracy.residuals(rows, lambda r: int(5.0 * GIB))
    assert s.n == 1 and s.skipped == 1


def test_unreadable_model_is_skipped_with_a_reason():
    s = accuracy.residuals([row(5.5)], lambda r: None)
    assert s.n == 0
    assert any("metadata" in w for w in s.skipped_why)


def test_table_calls_out_a_systematic_underprediction():
    s = accuracy.residuals([row(5.5), row(5.6)],
                           lambda r: int(5.0 * GIB))
    out = s.table()
    assert "missing term" in out


def test_empty_table_says_so():
    assert "nothing to compare" in accuracy.residuals([], lambda r: 0).table()
