import json
from dataclasses import dataclass

import numpy as np


@dataclass
class PhysioEncoder:
    input_dim: int
    embedding_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(scale=0.1, size=(self.input_dim, self.embedding_dim))
        self.mean = np.zeros(self.input_dim, dtype=float)
        self.std = np.ones(self.input_dim, dtype=float)

    def fit(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=float)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-6

    def encode(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        normed = (features - self.mean) / self.std
        return normed @ self.weights

    def save(self, path: str) -> None:
        payload = {
            "input_dim": int(self.input_dim),
            "embedding_dim": int(self.embedding_dim),
            "seed": int(self.seed),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhysioEncoder":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(payload["input_dim"], payload["embedding_dim"], payload.get("seed", 42))
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.std = np.asarray(payload["std"], dtype=float)
        model.weights = np.asarray(payload["weights"], dtype=float)
        return model
