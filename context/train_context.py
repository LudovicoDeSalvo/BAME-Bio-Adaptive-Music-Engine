import os

import numpy as np

from context.sequence_model import SequenceContextEncoder
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def train_context_model(config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    embeddings_path = resolve_path(paths.get("embeddings_path", "data/processed/audio_embeddings.npz"))
    model_path = resolve_path(paths.get("context_model_path", "models/context_model.json"))

    if os.path.exists(embeddings_path):
        payload = np.load(embeddings_path, allow_pickle=True)
        embeddings = payload["embeddings"]
        sequences = embeddings[: min(len(embeddings), 100)].reshape(-1, 1, embeddings.shape[-1])
    else:
        input_dim = int(model_cfg.get("audio_embedding_dim", 128))
        sequences = np.random.default_rng(0).normal(size=(32, 1, input_dim))

    input_dim = sequences.shape[-1]
    model = SequenceContextEncoder(input_dim, int(model_cfg.get("context_embedding_dim", 32)))
    model.fit(sequences)

    ensure_dir(os.path.dirname(model_path))
    model.save(model_path)
    return model_path
