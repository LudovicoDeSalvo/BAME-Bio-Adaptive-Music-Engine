import os
from typing import Optional, Tuple

import numpy as np

from audio.faiss_index import AudioIndex


class WolpertingerPolicy:
    def __init__(self, index: AudioIndex, k: int = 5) -> None:
        self.index = index
        self.k = k

    def select(self, action_vector: np.ndarray) -> Tuple[int, float]:
        candidates = self.index.query(action_vector, k=self.k)
        if not candidates:
            raise RuntimeError("No candidates returned from index")
        best_song_id, best_dist = min(candidates, key=lambda item: item[1])
        return best_song_id, best_dist


def load_index(embeddings_path: str, index_path: Optional[str] = None) -> AudioIndex:
    if index_path and os.path.exists(index_path):
        return AudioIndex.load(index_path)
    payload = np.load(embeddings_path, allow_pickle=True)
    index = AudioIndex()
    index.build(payload["embeddings"], payload["song_ids"])
    if index_path:
        index.save(index_path)
    return index
