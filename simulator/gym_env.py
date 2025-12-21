import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise RuntimeError("Install gymnasium to use MusicEnv as an environment") from exc

from simulator.world_model import WorldModel
from utils.common import load_config, resolve_path


@dataclass
class EnvCfg:
    seed: int = 42
    max_steps: int = 50
    action_clip: float = 1.0
    goal_state: Optional[np.ndarray] = None


class MusicEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str = "configs/config.yaml", env_cfg: EnvCfg | None = None) -> None:
        super().__init__()
        config = load_config(config_path)
        self.paths = config.get("paths", {})
        self.model_cfg = config.get("model", {})
        training = config.get("training", {})

        self.cfg = env_cfg or EnvCfg(seed=int(training.get("seed", 42)))

        self.physio_dim = int(self.model_cfg.get("physio_embedding_dim", 64))
        self.profile_dim = int(self.model_cfg.get("profile_embedding_dim", 16))
        self.context_dim = int(self.model_cfg.get("context_embedding_dim", 32))
        self.action_dim = int(self.model_cfg.get("action_dim", 128))
        self.state_dim = self.physio_dim + self.profile_dim + self.context_dim

        self._rng = np.random.default_rng(self.cfg.seed)

        processed_dir = self.paths.get("processed_dir", "data/processed")
        physio_emb_path = resolve_path(
            self.paths.get("physio_embeddings", os.path.join(processed_dir, "physio_embeddings.npz"))
        )
        user_emb_path = resolve_path(
            self.paths.get("user_embeddings", os.path.join(processed_dir, "user_embeddings.npz"))
        )

        self.physio_pool = self._load_pool(physio_emb_path, self.physio_dim)
        self.profile_pool = self._load_pool(user_emb_path, self.profile_dim)

        world_path = resolve_path(self.paths.get("world_model_path", "models/world_model.json"))
        if os.path.exists(world_path):
            self.world_model = WorldModel.load(world_path)
        else:
            self.world_model = WorldModel(self.state_dim, self.action_dim)

        if self.world_model.state_dim != self.state_dim or self.world_model.action_dim != self.action_dim:
            raise ValueError(
                f"WorldModel dims mismatch: wm(state_dim={self.world_model.state_dim}, action_dim={self.world_model.action_dim}) "
                f"vs env(state_dim={self.state_dim}, action_dim={self.action_dim}). "
                "Retrain or align configs."
            )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.cfg.action_clip, high=self.cfg.action_clip, shape=(self.action_dim,), dtype=np.float32
        )

        self._t = 0
        self.state = np.zeros(self.state_dim, dtype=float)

        if self.cfg.goal_state is None:
            self.goal_state = None
        else:
            g = np.asarray(self.cfg.goal_state, dtype=np.float32).reshape(-1)
            if g.size != self.state_dim:
                raise ValueError(f"goal_state must have shape ({self.state_dim},)")
            self.goal_state = g
        self._default_goal_state = None if self.goal_state is None else self.goal_state.copy()

    def _load_pool(self, path: str, dim: int) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding pool not found at {path}. Run preprocessing first.")
        payload = np.load(path, allow_pickle=True)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        if embeddings.shape[1] < dim:
            pad = np.zeros((embeddings.shape[0], dim - embeddings.shape[1]), dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)
        elif embeddings.shape[1] > dim:
            embeddings = embeddings[:, :dim]
        return embeddings

    def _compose_state(self) -> np.ndarray:
        physio = self.physio_pool[int(self._rng.integers(0, len(self.physio_pool)))]
        profile = self.profile_pool[int(self._rng.integers(0, len(self.profile_pool)))]
        context = np.zeros(self.context_dim, dtype=float)
        return np.concatenate([physio, profile, context], axis=0)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "goal_state" in options and options["goal_state"] is not None:
            g = np.asarray(options["goal_state"], dtype=np.float32).reshape(-1)
            if g.size != self.state_dim:
                raise ValueError(f"goal_state must have shape ({self.state_dim},)")
            self.goal_state = g
        else:
            self.goal_state = None if self._default_goal_state is None else self._default_goal_state.copy()

        self._t = 0
        self.state = self._compose_state().astype(np.float32)
        info = {"t": self._t}
        return self.state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.action_dim:
            if a.size < self.action_dim:
                out = np.zeros(self.action_dim, dtype=np.float32)
                out[: a.size] = a
                a = out
            else:
                a = a[: self.action_dim]

        a = np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

        next_state = self.world_model.predict_next(self.state, a).astype(np.float32)
        reward = float(self.world_model.reward(next_state, self.goal_state))

        self.state = next_state

        terminated = False
        if self.goal_state is not None:
            terminated = bool(np.linalg.norm(self.state - self.goal_state) < 1e-2)
        truncated = self._t >= int(self.cfg.max_steps)
        info = {"t": self._t}

        return self.state, reward, terminated, truncated, info
