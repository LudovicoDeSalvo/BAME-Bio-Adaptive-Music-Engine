import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from simulator.world_model import WorldModel
from context.sequence_model import ContextTransformer
from utils.common import load_config, resolve_path

class MusicEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/config.yaml"):
        super().__init__()

        self.config = load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # dims
        model_cfg = self.config.get("model", {})
        self.physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
        self.user_dim = int(model_cfg.get("profile_embedding_dim", 32))
        self.context_dim = int(model_cfg.get("context_embedding_dim", 128))
        self.action_dim = int(model_cfg.get("action_dim", 1024))
        
        self.state_dim = self.physio_dim + self.user_dim + self.context_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # load world model
        wm_path = resolve_path(self.config['paths']['world_model_path'])
        if os.path.exists(wm_path):
            self.world_model = WorldModel.load(wm_path, device=self.device, 
                                             state_dim=self.state_dim, 
                                             action_dim=self.action_dim,
                                             physio_dim=self.physio_dim)
        else:
            print("!!! World model not found. Using random")
            self.world_model = WorldModel(self.state_dim, self.action_dim, self.physio_dim).to(self.device)

        # load context model-
        ctx_path = resolve_path("context/checkpoints/context_model.pth")
        self.context_model = ContextTransformer(input_dim=self.action_dim, hidden_dim=self.context_dim).to(self.device)
        if os.path.exists(ctx_path):
            self.context_model.load_state_dict(torch.load(ctx_path, map_location=self.device))
            self.context_model.eval()
        else:
            print("!!! Context model not found")

        self._load_pools()
        
        self.max_steps = 50
        self.history_window = 5 
        self.history_buffer = deque(maxlen=self.history_window)

    def _load_pools(self):
        try:
            p_path = resolve_path(self.config['paths']['physio_embeddings'])
            u_path = resolve_path(self.config['paths']['user_embeddings'])
            self.physio_pool = np.load(p_path, allow_pickle=True)['embeddings']
            self.user_pool = np.load(u_path, allow_pickle=True)['embeddings']
        except:
            self.physio_pool = np.zeros((10, self.physio_dim))
            self.user_pool = np.zeros((10, self.user_dim))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rng = np.random.default_rng(seed)
        
        u_idx = self.rng.integers(0, len(self.user_pool))
        p_idx = self.rng.integers(0, len(self.physio_pool))
        
        self.current_user = self.user_pool[u_idx]
        self.current_physio = self.physio_pool[p_idx]
        
        self.history_buffer.clear()
        self.current_context = np.zeros(self.context_dim, dtype=np.float32)
        
        target_idx = self.rng.integers(0, len(self.physio_pool))
        self.target_physio = self.physio_pool[target_idx]
        
        self._t = 0
        self.state = np.concatenate([self.current_physio, self.current_user, self.current_context])
        
        return self.state.astype(np.float32), {}

    def step(self, action):

        self._t += 1
        
        self.history_buffer.append(action)
        self.current_context = self._compute_context()
        
        prev_state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_physio_tensor = self.world_model(prev_state_tensor, action_tensor)
            self.current_physio = next_physio_tensor.cpu().numpy()[0]
            
        self.state = np.concatenate([self.current_physio, self.current_user, self.current_context])
        
        reward = self._calculate_reward()
        
        terminated = False
        truncated = self._t >= self.max_steps
        
        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def _compute_context(self):

        if len(self.history_buffer) == 0:
            return np.zeros(self.context_dim, dtype=np.float32)
        
        seq = np.array(self.history_buffer)
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ctx = self.context_model(inp) 

        return ctx.cpu().numpy()[0]

    def _calculate_reward(self):
            """
            Range: (-inf, 0]
            """
            diff = self.current_physio - self.target_physio
            dist = np.linalg.norm(diff)

            reward = -(dist / 100.0)
            
            return float(reward)