from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


class AudioIndex:
    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None
        self.song_ids: np.ndarray | None = None
        self._faiss = None
        self._index = None

    def build(self, embeddings: np.ndarray, song_ids: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        song_ids = np.asarray(song_ids, dtype=int)
        self.embeddings = embeddings
        self.song_ids = song_ids
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = faiss.IndexFlatL2(embeddings.shape[1])
            self._index.add(embeddings)
        except Exception:
            self._faiss = None
            self._index = None

    def query(self, vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self.embeddings is None or self.song_ids is None:
            raise RuntimeError("Index not built")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        k = min(k, self.embeddings.shape[0])
        if self._index is not None:
            distances, indices = self._index.search(vector, k)
            return [(int(self.song_ids[i]), float(distances[0, idx])) for idx, i in enumerate(indices[0])]
        diffs = self.embeddings - vector
        distances = np.sum(diffs * diffs, axis=1)
        best = np.argsort(distances)[:k]
        return [(int(self.song_ids[i]), float(distances[i])) for i in best]

    def save(self, path: str) -> None:
        if self.embeddings is None or self.song_ids is None:
            raise RuntimeError("Index not built")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, embeddings=self.embeddings, song_ids=self.song_ids)

    @classmethod
    def load(cls, path: str) -> "AudioIndex":
        payload = np.load(path, allow_pickle=True)
        index = cls()
        index.build(payload["embeddings"], payload["song_ids"])
        return index
