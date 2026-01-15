from vramfit import devices, sampler


def test_peak_is_the_maximum_seen(monkeypatch):
    seq = iter([1, 5, 3, 5, 2])

    def fake():
        try:
            return [devices.Device(0, "x", 100, next(seq))]
        except StopIteration:
            return [devices.Device(0, "x", 100, 2)]

    monkeypatch.setattr(devices, "discover", fake)
    _, peak, note = sampler.sample_peak(
        lambda: sum(range(200000)), interval=0.001)
    assert peak <= 5
    assert "peak sampled" in note or "no memory samples" in note


def test_no_device_means_unknown_not_zero(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])
    _, peak, note = sampler.sample_peak(lambda: None, interval=0.001)
    assert peak == 0
    assert "unknown, not zero" in note


def test_result_is_passed_through(monkeypatch):
    monkeypatch.setattr(devices, "discover", lambda: [])
    out, _, _ = sampler.sample_peak(lambda: "value", interval=0.001)
    assert out == "value"
