import json
from dataclasses import dataclass

import numpy as np


@dataclass
class WorldModel:
    state_dim: int
    action_dim: int
    seed: int = 42
    noise_sigma: float = 0.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.A = self.rng.normal(scale=0.05, size=(self.state_dim, self.state_dim))
        self.B = self.rng.normal(scale=0.05, size=(self.action_dim, self.state_dim))
        self.bias = np.zeros(self.state_dim, dtype=float)

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
        return arr

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> None:
        states = self._ensure_2d(states)
        actions = self._ensure_2d(actions)
        next_states = self._ensure_2d(next_states)
        if states.shape[0] != actions.shape[0] or states.shape[0] != next_states.shape[0]:
            raise ValueError("states, actions, next_states must have the same batch size")
        if states.shape[1] != self.state_dim or next_states.shape[1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {states.shape[1]} and {next_states.shape[1]}")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {actions.shape[1]}")
        ones = np.ones((states.shape[0], 1), dtype=float)
        X = np.concatenate([states, actions, ones], axis=1)
        W, *_ = np.linalg.lstsq(X, next_states, rcond=None)
        self.A = W[: self.state_dim].T
        self.B = W[self.state_dim : self.state_dim + self.action_dim]  # keep (action_dim, state_dim)
        self.bias = W[-1]

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state_b = self._ensure_2d(state)
        action_b = self._ensure_2d(action)
        if state_b.shape[0] != action_b.shape[0]:
            raise ValueError("state and action batch sizes must match")
        next_state = state_b @ self.A + action_b @ self.B + self.bias[None, :]
        if self.noise_sigma > 0:
            noise = self.rng.normal(scale=self.noise_sigma, size=next_state.shape)
            next_state = next_state + noise
        return next_state[0] if next_state.shape[0] == 1 else next_state

    def reward(self, next_state: np.ndarray, goal_state: np.ndarray | None = None) -> float | np.ndarray:
        ns = self._ensure_2d(next_state)
        if goal_state is None:
            gs = np.zeros_like(ns)
        else:
            gs = self._ensure_2d(goal_state)
            if gs.shape[0] == 1 and ns.shape[0] > 1:
                gs = np.repeat(gs, ns.shape[0], axis=0)
            elif gs.shape[0] not in (1, ns.shape[0]):
                raise ValueError("goal_state batch size must be 1 or match next_state batch size")
        r = -np.linalg.norm(ns - gs, axis=1)
        return r[0] if r.shape[0] == 1 else r

    def step(self, state: np.ndarray, action: np.ndarray, goal_state: np.ndarray | None = None) -> tuple[np.ndarray, float | np.ndarray]:
        next_state = self.predict_next(state, action)
        reward = self.reward(next_state, goal_state)
        return next_state, reward

    def save(self, path: str) -> None:
        payload = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "seed": int(self.seed),
            "noise_sigma": float(self.noise_sigma),
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
        model = cls(
            payload["state_dim"],
            payload["action_dim"],
            payload.get("seed", 42),
            payload.get("noise_sigma", 0.0),
        )
        model.A = np.asarray(payload["A"], dtype=float)
        model.B = np.asarray(payload["B"], dtype=float)
        model.bias = np.asarray(payload["bias"], dtype=float)
        return model
