import json
from dataclasses import dataclass

import numpy as np


@dataclass
class DCNProfile:
    input_dim: int
    embedding_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.cross_w = rng.normal(scale=0.1, size=(self.input_dim,))
        self.cross_b = rng.normal(scale=0.1, size=(self.input_dim,))
        self.proj = rng.normal(scale=0.1, size=(self.input_dim, self.embedding_dim))
        self.mean = np.zeros(self.input_dim, dtype=float)
        self.std = np.ones(self.input_dim, dtype=float)

    def fit(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=float)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-6

    def _cross(self, x: np.ndarray) -> np.ndarray:
        xw = x @ self.cross_w
        return x * xw[:, None] + self.cross_b + x

    def encode(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        x = (features - self.mean) / self.std
        x = self._cross(x)
        return x @ self.proj

    def save(self, path: str) -> None:
        payload = {
            "input_dim": int(self.input_dim),
            "embedding_dim": int(self.embedding_dim),
            "seed": int(self.seed),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "cross_w": self.cross_w.tolist(),
            "cross_b": self.cross_b.tolist(),
            "proj": self.proj.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DCNProfile":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(payload["input_dim"], payload["embedding_dim"], payload.get("seed", 42))
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.std = np.asarray(payload["std"], dtype=float)
        model.cross_w = np.asarray(payload["cross_w"], dtype=float)
        model.cross_b = np.asarray(payload["cross_b"], dtype=float)
        model.proj = np.asarray(payload["proj"], dtype=float)
        return model
