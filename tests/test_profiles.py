import pytest

from vramfit import profiles


def test_llama_cpp_does_not_page():
    assert profiles.LLAMA_CPP.cache_tokens(4097) == 4097


def test_paged_cache_rounds_up_per_sequence():
    # 17 tokens in blocks of 16 costs 32, and ten of them cost 320 --
    # not ceil(170/16)*16.
    assert profiles.VLLM.cache_tokens(17) == 32
    assert profiles.VLLM.cache_tokens(17, sequences=10) == 320


def test_exact_multiples_are_not_rounded():
    assert profiles.VLLM.cache_tokens(4096) == 4096


def test_vllm_utilisation_is_a_fraction():
    assert profiles.VLLM.utilisation == 0.9
    assert profiles.LLAMA_CPP.utilisation == 1.0


def test_every_profile_says_what_it_approximates():
    for p in profiles.PROFILES.values():
        assert p.notes


def test_unknown_backend_lists_the_known_ones():
    with pytest.raises(KeyError, match="no profile for backend"):
        profiles.lookup("tensorrt")
