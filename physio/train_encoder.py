import os

import numpy as np

from physio.encoder import PhysioEncoder
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def train_physio_model(epochs: int = 20, config_path: str = "configs/config.yaml") -> tuple[str, str]:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    physio_cache = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    model_path = resolve_path(paths.get("physio_model_path", "models/physio_encoder.json"))
    embeddings_path = resolve_path(paths.get("physio_embeddings", "data/processed/physio_embeddings.npz"))

    if not os.path.exists(physio_cache):
        raise FileNotFoundError("Physio cache missing. Run data processing first.")

    payload = np.load(physio_cache, allow_pickle=True)

    for k in ["features", "participant_ids", "song_ids"]:
        if k not in payload:
            raise KeyError(f"physio_cache missing key '{k}'")

    # Baseline: engineered record-level features (non-sequential)
    features = np.asarray(payload["features"], dtype=np.float32)
    n = features.shape[0]
    if n < 2:
        raise RuntimeError(f"Not enough records to train physio encoder: N={n}")
    for key in ["participant_ids", "song_ids", "valence", "arousal", "song_no"]:
        if key in payload and len(payload[key]) != n:
            raise ValueError(f"Length mismatch: {key} has {len(payload[key])} rows, expected {n}")

    physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
    encoder = PhysioEncoder(features.shape[1], physio_dim)

    try:
        encoder.fit(features, epochs=epochs)
    except TypeError:
        encoder.fit(features)

    ensure_dir(os.path.dirname(model_path))
    encoder.save(model_path)

    embeddings = np.asarray(encoder.encode(features), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != features.shape[0] or embeddings.shape[1] != physio_dim:
        raise ValueError(
            f"Bad embeddings shape: {embeddings.shape}, expected ({features.shape[0]}, {physio_dim})"
        )

    ensure_dir(os.path.dirname(embeddings_path))
    save_payload = {
        "embeddings": embeddings,
        "participant_ids": np.char.strip(payload["participant_ids"].astype(str)),
        "song_ids": np.asarray(payload["song_ids"]).astype(int),
        "physio_dim": physio_dim,
        "input_dim": features.shape[1],
    }
    if "valence" in payload:
        save_payload["valence"] = payload["valence"]
    if "arousal" in payload:
        save_payload["arousal"] = payload["arousal"]
    if "song_no" in payload:
        save_payload["song_no"] = payload["song_no"]
    if "feature_names" in payload:
        save_payload["feature_names"] = payload["feature_names"]
    np.savez(embeddings_path, **save_payload)
    return model_path, embeddings_path


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train physio encoder and export embeddings.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (if supported).")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    model_path, embeddings_path = train_physio_model(epochs=args.epochs, config_path=args.config)
    print(f">> Saved physio model to: {model_path}")
    print(f">> Saved physio embeddings to: {embeddings_path}")


if __name__ == "__main__":
    _main()
