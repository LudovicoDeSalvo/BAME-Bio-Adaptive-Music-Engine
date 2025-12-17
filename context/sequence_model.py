import json
from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (np.sum(exp) + 1e-8)


@dataclass
class SequenceContextEncoder:
    input_dim: int
    embedding_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.proj = rng.normal(scale=0.1, size=(self.input_dim, self.embedding_dim))
        self.mean = np.zeros(self.input_dim, dtype=float)
        self.std = np.ones(self.input_dim, dtype=float)

    def fit(self, sequences: np.ndarray) -> None:
        flat = sequences.reshape(-1, sequences.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-6

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=float)
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        seq = (seq - self.mean) / self.std
        if seq.shape[0] == 1:
            pooled = seq[0]
        else:
            last = seq[-1]
            scores = seq @ last
            weights = _softmax(scores)
            pooled = weights @ seq
        return pooled @ self.proj

    def save(self, path: str) -> None:
        payload = {
            "input_dim": int(self.input_dim),
            "embedding_dim": int(self.embedding_dim),
            "seed": int(self.seed),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "proj": self.proj.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SequenceContextEncoder":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(payload["input_dim"], payload["embedding_dim"], payload.get("seed", 42))
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.std = np.asarray(payload["std"], dtype=float)
        model.proj = np.asarray(payload["proj"], dtype=float)
        return model
