import json
from dataclasses import dataclass

import numpy as np


@dataclass
class WorldModel:
    state_dim: int
    action_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.A = rng.normal(scale=0.05, size=(self.state_dim, self.state_dim))
        self.B = rng.normal(scale=0.05, size=(self.action_dim, self.state_dim))
        self.bias = np.zeros(self.state_dim, dtype=float)

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        ones = np.ones((states.shape[0], 1), dtype=float)
        X = np.concatenate([states, actions, ones], axis=1)
        W, *_ = np.linalg.lstsq(X, next_states, rcond=None)
        self.A = W[: self.state_dim].T
        self.B = W[self.state_dim : self.state_dim + self.action_dim].T
        self.bias = W[-1]

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        return state @ self.A + action @ self.B + self.bias

    def reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        next_state = np.asarray(next_state, dtype=float)
        return float(-np.linalg.norm(next_state - state))

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state = self.predict_next(state, action)
        reward = self.reward(state, next_state)
        return next_state, reward

    def save(self, path: str) -> None:
        payload = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "seed": int(self.seed),
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "bias": self.bias.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(payload["state_dim"], payload["action_dim"], payload.get("seed", 42))
        model.A = np.asarray(payload["A"], dtype=float)
        model.B = np.asarray(payload["B"], dtype=float)
        model.bias = np.asarray(payload["bias"], dtype=float)
        return model
