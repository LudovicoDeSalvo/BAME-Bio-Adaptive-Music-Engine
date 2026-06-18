import numpy as np
import pandas as pd

from user.train_profile import UserDataset


def test_normalizer_roundtrip(tmp_path):
    """The scaler persisted at train time must reproduce train normalization
    exactly at inference time (single source of truth, no /10.0)."""
    csv = tmp_path / "personality.csv"
    df = pd.DataFrame({
        "participant_id": ["u1", "u2", "u3", "u4"],
        "Extroversion_score": [1.0, 5.0, 9.0, 3.0],
        "Openness_score": [2.0, 4.0, 6.0, 8.0],
    })
    df.to_csv(csv, index=False)

    ds = UserDataset(str(csv))
    train_feats = ds.features.numpy()

    # Simulate inference: same persisted min/max, same column order.
    raw = df[ds.feat_cols].values.astype(np.float32)
    infer_feats = (raw - ds.min_vals) / (ds.max_vals - ds.min_vals)

    assert np.allclose(train_feats, infer_feats, atol=1e-6)
    assert train_feats.min() >= 0.0 and train_feats.max() <= 1.0


def test_holdout_excluded(tmp_path):
    csv = tmp_path / "p.csv"
    df = pd.DataFrame({
        "participant_id": ["u1", "u2", "hold"],
        "A_score": [1.0, 2.0, 99.0],
    })
    df.to_csv(csv, index=False)
    ds = UserDataset(str(csv), holdout_user="hold")
    # holdout's extreme value must not move the fitted max
    assert ds.max_vals[0] < 99.0
    assert len(ds) == 2
