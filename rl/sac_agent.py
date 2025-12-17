import json
from dataclasses import dataclass

import numpy as np


@dataclass
class SACAgent:
    state_dim: int
    action_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(scale=0.1, size=(self.state_dim, self.action_dim))
        self.b = np.zeros(self.action_dim, dtype=float)

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        action = state @ self.W + self.b
        return np.tanh(action)

    def update(self, batch: dict) -> None:
        _ = batch

    def save(self, path: str) -> None:
        payload = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "seed": int(self.seed),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SACAgent":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        agent = cls(payload["state_dim"], payload["action_dim"], payload.get("seed", 42))
        agent.W = np.asarray(payload["W"], dtype=float)
        agent.b = np.asarray(payload["b"], dtype=float)
        return agent
