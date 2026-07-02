from vramfit import budget
from vramfit.shape import Shape


def base(**kw):
    args = dict(name="t", arch="llama", n_layers=32, n_embd=4096,
                n_head=32, n_head_kv=8, head_dim=128, vocab=128256,
                train_context=8192)
    args.update(kw)
    return Shape(**args)


def test_key_and_value_default_to_head_dim():
    sh = base()
    assert sh.key_dim == sh.value_dim == 128
    assert not sh.asymmetric_kv


def test_cache_uses_key_plus_value_width():
    sh = base()
    # The old formula's 2x is key_dim + value_dim when they are equal.
    assert budget.kv_bytes_per_token(sh) == 32 * 8 * 256 * 2


def test_explicit_widths_are_respected():
    sh = base(key_dim=192, value_dim=128)
    assert sh.asymmetric_kv
    assert budget.kv_bytes_per_token(sh) == 32 * 8 * 320 * 2


def test_a_narrower_key_than_head_dim_shrinks_the_cache():
    wide = budget.kv_bytes(base(), 8192)
    narrow = budget.kv_bytes(base(key_dim=64, value_dim=64), 8192)
    assert narrow * 2 == wide


def test_params_from_config_accounts_for_both_widths():
    sym = base(n_ff=14336).params_from_config()
    asym = base(n_ff=14336, key_dim=192).params_from_config()
    assert asym > sym
