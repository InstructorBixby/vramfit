import json

from vramfit import measure


def test_cache_is_keyed_on_device_and_driver(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    monkeypatch.setattr(measure, "cache_path", lambda: path)
    measure.record_cuda_context(304 * 1024 ** 2, "RTX 3090",
                                "550.54.14")
    measure.record_cuda_context(322 * 1024 ** 2, "RTX 3090", "550.78")
    assert measure.cached_cuda_context(0, "RTX 3090", "550.54.14") == \
        304 * 1024 ** 2
    assert measure.cached_cuda_context(0, "RTX 3090", "550.78") == \
        322 * 1024 ** 2


def test_a_new_driver_is_a_miss_not_an_old_answer(tmp_path,
                                                 monkeypatch):
    path = tmp_path / "context.json"
    monkeypatch.setattr(measure, "cache_path", lambda: path)
    measure.record_cuda_context(304 * 1024 ** 2, "RTX 3090",
                                "550.54.14")
    assert measure.cached_cuda_context(0, "RTX 3090", "560.1") is None


def test_stale_entries_are_surfaced_not_deleted(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    monkeypatch.setattr(measure, "cache_path", lambda: path)
    measure.record_cuda_context(304 * 1024 ** 2, "RTX 3090",
                                "550.54.14")
    stale = measure.stale_entries("RTX 3090", "560.1")
    assert stale == ["RTX 3090|550.54.14"]
    assert json.loads(path.read_text())


def test_missing_and_corrupt_caches_are_misses(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    monkeypatch.setattr(measure, "cache_path", lambda: path)
    assert measure.cached_cuda_context(0, "x", "y") is None
    path.write_text("{not json")
    assert measure.cached_cuda_context(0, "x", "y") is None
    assert measure.stale_entries("x", "y") == []
