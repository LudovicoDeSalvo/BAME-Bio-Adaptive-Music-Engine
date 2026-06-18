import numpy as np
import pytest

from utils.common import (
    l2_normalize, set_seed, load_npz_checked, require_columns, parse_clip_id,
)


def test_parse_clip_id_basic():
    assert parse_clip_id("101_hku1903_5_0") == ("101", "hku1903", "5", 0)


def test_parse_clip_id_sid_with_underscores():
    # song_id containing underscores must survive (rsplit, not split)
    assert parse_clip_id("a_b_c_hku1903_5_2") == ("a_b_c", "hku1903", "5", 2)


def test_parse_clip_id_rejects_nonconforming():
    for bad in ["song_5", "abc", "1_2_3_x"]:
        with pytest.raises(ValueError):
            parse_clip_id(bad)


def test_l2_normalize_unit_norm():
    x = np.random.RandomState(0).randn(5, 1024).astype(np.float32) * 17.0
    y = l2_normalize(x, axis=1)
    norms = np.linalg.norm(y, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_zero_safe():
    x = np.zeros((3, 8), dtype=np.float32)
    y = l2_normalize(x, axis=1)
    assert np.all(np.isfinite(y))  # no div-by-zero


def test_l2_normalize_idempotent():
    x = np.random.RandomState(1).randn(4, 16).astype(np.float32)
    once = l2_normalize(x, axis=1)
    twice = l2_normalize(once, axis=1)
    assert np.allclose(once, twice, atol=1e-6)


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(10)
    import torch
    ta = torch.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    tb = torch.rand(10)
    assert np.allclose(a, b)
    assert torch.allclose(ta, tb)


def test_load_npz_checked(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, a=np.zeros(3), b=np.ones(2))
    data = load_npz_checked(str(p), ["a", "b"])
    assert "a" in data
    with pytest.raises(KeyError):
        load_npz_checked(str(p), ["a", "missing"])


def test_require_columns():
    import pandas as pd
    df = pd.DataFrame({"x": [1], "y": [2]})
    require_columns(df, ["x", "y"], "df")
    with pytest.raises(KeyError):
        require_columns(df, ["x", "z"], "df")
