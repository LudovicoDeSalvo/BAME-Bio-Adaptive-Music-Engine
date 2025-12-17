import os

import numpy as np

from user.dcn_profile import DCNProfile
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def _load_profile_csv(path: str):
    try:
        import pandas as pd  # type: ignore

        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError("pandas is required to read personality CSV") from exc


def train_user_model(config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    profile_cache = resolve_path(paths.get("profile_cache", "data/processed/profile_cache.csv"))
    model_path = resolve_path(paths.get("profile_model_path", "models/profile_model.json"))
    embeddings_path = resolve_path("data/processed/user_embeddings.npz")

    if not os.path.exists(profile_cache):
        raise FileNotFoundError("Profile cache missing. Run data processing first.")

    df = _load_profile_csv(profile_cache)

    numeric_cols = [c for c in df.columns if c.endswith("_score")]
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if c != "participant_id"]

    features = df[numeric_cols].to_numpy(dtype=float)
    participant_ids = df["participant_id"].to_numpy(dtype=object)

    model = DCNProfile(features.shape[1], int(model_cfg.get("profile_embedding_dim", 16)))
    model.fit(features)

    ensure_dir(os.path.dirname(model_path))
    model.save(model_path)

    embeddings = model.encode(features)
    np.savez(embeddings_path, embeddings=embeddings, participant_ids=participant_ids)
    return model_path
