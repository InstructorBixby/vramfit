from vramfit import devices


def test_normalise_strips_vendor_noise():
    assert devices.normalise("NVIDIA GeForce RTX 3090") == "rtx 3090"
    assert devices.normalise("NVIDIA RTX A4000") == "rtx a4000"


def test_free_is_total_minus_used():
    d = devices.Device(0, "RTX 3090", 24 * 1024 ** 3, 1024 ** 3)
    assert d.free == 23 * 1024 ** 3


def test_discover_does_not_raise_without_a_gpu(monkeypatch):
    monkeypatch.setattr(devices.shutil, "which", lambda _: None)
    monkeypatch.setattr(devices, "_from_pynvml",
                        lambda: (_ for _ in ()).throw(ImportError()))
    assert devices.discover() == []
