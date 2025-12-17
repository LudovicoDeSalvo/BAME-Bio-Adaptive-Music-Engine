import hashlib
import os
from typing import List, Tuple

import numpy as np

from utils.common import ensure_dir, load_config, resolve_path


class MERTEmbedder:
    def __init__(self, embedding_dim: int, prefer_librosa: bool = True) -> None:
        self.embedding_dim = int(embedding_dim)
        self._librosa = None
        if prefer_librosa:
            try:
                import librosa  # type: ignore

                self._librosa = librosa
            except Exception:
                self._librosa = None

    def _hash_embedding(self, token: bytes) -> np.ndarray:
        seed = int.from_bytes(token[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=self.embedding_dim).astype(np.float32)

    def embed_file(self, path: str) -> np.ndarray:
        if self._librosa is None:
            token = hashlib.sha256(path.encode("utf-8")).digest()
            return self._hash_embedding(token)

        try:
            y, sr = self._librosa.load(path, sr=22050, mono=True, duration=30)
            mel = self._librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            feat = np.mean(mel, axis=1)
            if feat.shape[0] < self.embedding_dim:
                pad = np.zeros(self.embedding_dim - feat.shape[0], dtype=float)
                feat = np.concatenate([feat, pad], axis=0)
            return feat[: self.embedding_dim].astype(np.float32)
        except Exception:
            token = hashlib.sha256(path.encode("utf-8")).digest()
            return self._hash_embedding(token)


def _load_audio_csv(path: str) -> List[Tuple[int, str]]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        return [(int(row["song_id"]), str(row.get("link", ""))) for _, row in df.iterrows()]
    except Exception as exc:
        raise RuntimeError("pandas is required to read audio CSV") from exc


def extract_all_embeddings(config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})

    audio_csv = resolve_path(paths.get("audio_csv", ""))
    audio_dir = resolve_path(paths.get("audio_dir", ""))
    embeddings_path = resolve_path(paths.get("embeddings_path", "data/processed/audio_embeddings.npz"))

    embedding_dim = int(model_cfg.get("audio_embedding_dim", 128))
    embedder = MERTEmbedder(embedding_dim)

    ensure_dir(os.path.dirname(embeddings_path))

    rows = _load_audio_csv(audio_csv)
    song_ids: List[int] = []
    embeddings: List[np.ndarray] = []

    for song_id, _ in rows:
        file_path = os.path.join(audio_dir, f"{song_id}.mp3")
        if not os.path.exists(file_path):
            continue
        emb = embedder.embed_file(file_path)
        song_ids.append(song_id)
        embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No audio files found to embed. Check audio_dir in config.")

    np.savez(
        embeddings_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        song_ids=np.asarray(song_ids, dtype=int),
    )
    return embeddings_path
