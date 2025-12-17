import os

import numpy as np

from simulator.world_model import WorldModel
from utils.common import load_config, resolve_path


class MusicEnv:
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        config = load_config(config_path)
        self.paths = config.get("paths", {})
        self.model_cfg = config.get("model", {})

        self.physio_dim = int(self.model_cfg.get("physio_embedding_dim", 64))
        self.profile_dim = int(self.model_cfg.get("profile_embedding_dim", 16))
        self.context_dim = int(self.model_cfg.get("context_embedding_dim", 32))
        self.action_dim = int(self.model_cfg.get("action_dim", 128))
        self.state_dim = self.physio_dim + self.profile_dim + self.context_dim

        self._rng = np.random.default_rng(42)
        self.physio_pool = self._load_pool(resolve_path("data/processed/physio_embeddings.npz"), self.physio_dim)
        self.profile_pool = self._load_pool(resolve_path("data/processed/user_embeddings.npz"), self.profile_dim)

        world_path = resolve_path(self.paths.get("world_model_path", "models/world_model.json"))
        if os.path.exists(world_path):
            self.world_model = WorldModel.load(world_path)
        else:
            self.world_model = WorldModel(self.state_dim, self.action_dim)

        self.state = self.reset()

    def _load_pool(self, path: str, dim: int) -> np.ndarray:
        if not os.path.exists(path):
            return self._rng.normal(size=(32, dim))
        payload = np.load(path, allow_pickle=True)
        embeddings = payload["embeddings"]
        if embeddings.shape[1] != dim:
            if embeddings.shape[1] < dim:
                pad = np.zeros((embeddings.shape[0], dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, pad], axis=1)
            else:
                embeddings = embeddings[:, :dim]
        return embeddings

    def reset(self) -> np.ndarray:
        physio = self.physio_pool[self._rng.integers(0, len(self.physio_pool))]
        profile = self.profile_pool[self._rng.integers(0, len(self.profile_pool))]
        context = np.zeros(self.context_dim, dtype=float)
        self.state = np.concatenate([physio, profile, context], axis=0)
        return self.state

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != self.action_dim:
            if action.shape[0] < self.action_dim:
                pad = np.zeros(self.action_dim - action.shape[0], dtype=float)
                action = np.concatenate([action, pad], axis=0)
            else:
                action = action[: self.action_dim]
        next_state, reward = self.world_model.step(self.state, action)
        self.state = next_state
        done = False
        info = {}
        return next_state, reward, done, info
