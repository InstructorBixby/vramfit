from vramfit import specs


def test_table_loads_and_skips_comments():
    table = specs.load()
    assert "rtx 3090" in table
    assert all(s.bandwidth_gbs > 0 for s in table.values())


def test_lookup_normalises_the_device_name():
    s = specs.lookup("NVIDIA GeForce RTX 3090")
    assert s is not None
    assert s.vram_mib == 24576


def test_unknown_card_is_none_not_a_guess():
    assert specs.lookup("NVIDIA GeForce RTX 9999") is None


def test_bandwidth_is_decimal_gigabytes():
    s = specs.lookup("RTX 3090")
    assert s.bandwidth_bps == 936.2 * 1e9
