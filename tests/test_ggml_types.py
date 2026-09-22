import pytest

from vramfit import ggml_types


def test_every_known_type_has_block_geometry():
    assert ggml_types.check_tables() == []


def test_block_derived_bpw_matches_the_published_figures():
    assert ggml_types.bpw("q4_0") == 4.5
    assert ggml_types.bpw("q4_k") == 4.5
    assert ggml_types.bpw("q6_k") == 6.5625
    assert ggml_types.bpw("q8_0") == 8.5
    assert ggml_types.bpw("iq4_xs") == 4.25
    assert ggml_types.bpw("q2_k") == 2.625


def test_ternary_types_are_known():
    # A contributed file had one of these and the loader refused it.
    assert ggml_types.name(31) == "tq1_0"
    assert ggml_types.bpw("tq1_0") < 2.0


def test_partial_block_rounds_up():
    # 257 weights in blocks of 256 costs two blocks.
    assert ggml_types.tensor_bytes(257, "q4_k") == 288


def test_unknown_id_is_fatal_and_says_what_to_do():
    with pytest.raises(KeyError, match="deliberately"):
        ggml_types.name(199)
