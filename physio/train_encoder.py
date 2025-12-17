import os

import numpy as np

from physio.encoder import PhysioEncoder
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def train_physio_model(epochs: int = 20, config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    physio_cache = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    model_path = resolve_path(paths.get("physio_model_path", "models/physio_encoder.json"))
    embeddings_path = resolve_path("data/processed/physio_embeddings.npz")

    if not os.path.exists(physio_cache):
        raise FileNotFoundError("Physio cache missing. Run data processing first.")

    payload = np.load(physio_cache, allow_pickle=True)
    features = payload["features"]

    encoder = PhysioEncoder(features.shape[1], int(model_cfg.get("physio_embedding_dim", 64)))
    encoder.fit(features)

    ensure_dir(os.path.dirname(model_path))
    encoder.save(model_path)

    embeddings = encoder.encode(features)
    np.savez(embeddings_path, embeddings=embeddings)
    return model_path
