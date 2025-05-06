from vramfit import quants


def test_q4_k_m_is_not_four_point_five():
    assert quants.lookup("q4_k_m").bpw > 4.7


def test_gptq_group_overhead_is_arithmetic():
    q = quants.lookup("gptq-4bit-128g")
    # 4 bits payload plus an fp16 scale and a 4-bit zero per 128.
    assert abs(q.bpw - (4 + 20 / 128)) < 1e-9
    assert q.source == "arithmetic"


def test_exl2_is_named_by_its_bpw():
    assert quants.lookup("exl2-4.65").bpw == 4.65


def test_unknown_scheme_is_none():
    assert quants.lookup("q4_k_xxl") is None


def test_ladder_descends():
    rungs = quants.ladder_at_most(5.0)
    assert rungs[0].bpw <= 5.0
    assert [q.bpw for q in rungs] == sorted(
        (q.bpw for q in rungs), reverse=True)
