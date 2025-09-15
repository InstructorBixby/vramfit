from vramfit import devices, specs


def test_normalise_strips_vendor_noise():
    assert devices.normalise("NVIDIA GeForce RTX 3090") == "rtx 3090"
    assert devices.normalise("NVIDIA RTX A4000") == "rtx a4000"


def test_normalise_strips_the_disambiguating_suffix():
    d = devices.Device(0, "NVIDIA GeForce RTX 3090", 0, 0,
                       uuid="GPU-1f2e3d4c5b6a7988")
    assert devices.normalise(d.tag) == "rtx 3090"
    assert specs.lookup(d.tag) is not None


def test_two_identical_cards_get_different_tags():
    a = devices.Device(0, "NVIDIA GeForce RTX 3090", 0, 0,
                       uuid="GPU-aaaabbbbccccdddd")
    b = devices.Device(1, "NVIDIA GeForce RTX 3090", 0, 0,
                       uuid="GPU-11112222333344445")
    assert a.tag != b.tag


def test_tag_falls_back_to_the_bus_id_then_the_name():
    assert "0000:01:00.0" in devices.Device(
        0, "RTX 3090", 0, 0, bus_id="0000:01:00.0").tag
    assert devices.Device(0, "RTX 3090", 0, 0).tag == "RTX 3090"


def test_free_is_total_minus_used():
    d = devices.Device(0, "RTX 3090", 24 * 1024 ** 3, 1024 ** 3)
    assert d.free == 23 * 1024 ** 3


def test_discover_does_not_raise_without_a_gpu(monkeypatch):
    monkeypatch.setattr(devices.shutil, "which", lambda _: None)
    monkeypatch.setattr(devices, "_from_pynvml",
                        lambda: (_ for _ in ()).throw(ImportError()))
    assert devices.discover() == []
