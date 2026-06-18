import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from simulator.world_model import WorldModel
from context.sequence_model import ContextTransformer
from utils.common import load_config, resolve_path, get_device

class MusicEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/config.yaml", overrides=None):
        super().__init__()

        self.config = load_config(config_path)
        # Sweep support: shallow-merge override values into the training section
        # so reward_scale / max_steps can vary per trial without rewriting the
        # on-disk config. Only the training block is overridable; model dims must
        # stay fixed (they're tied to the trained component checkpoints).
        if overrides:
            self.config.setdefault("training", {}).update(overrides.get("training", {}))
        self.device = get_device()
        
        # dims
        model_cfg = self.config.get("model", {})
        self.physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
        self.user_dim = int(model_cfg.get("profile_embedding_dim", 32))
        self.context_dim = int(model_cfg.get("context_embedding_dim", 128))
        self.action_dim = int(model_cfg.get("action_dim", 1024))
        
        # World-model input = [physio, user, context]. The world model predicts
        # next physio from these alone; it must NOT see the goal.
        self.wm_state_dim = self.physio_dim + self.user_dim + self.context_dim

        # Agent OBSERVATION = world-model state + target_physio. The reward is
        # goal-relative, so the policy MUST be able to condition on the target;
        # otherwise identical observations map to different rewards and the
        # goal-conditioned control problem is unlearnable (it can only collapse
        # to a fixed attractor). The target is appended as the LAST physio_dim.
        self.state_dim = self.wm_state_dim + self.physio_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # load world model
        wm_path = resolve_path(self.config['paths']['world_model_path'])
        if os.path.exists(wm_path):
            self.world_model = WorldModel.load(wm_path, device=self.device,
                                             state_dim=self.wm_state_dim,
                                             action_dim=self.action_dim,
                                             physio_dim=self.physio_dim)
        else:
            print("!!! World model not found. Using random")
            self.world_model = WorldModel(self.wm_state_dim, self.action_dim, self.physio_dim).to(self.device)
        # Always eval: dropout/other stochastic layers must be off during
        # env rollouts so dynamics are deterministic given (state, action).
        self.world_model.eval()

        # load context model-
        ctx_path = resolve_path("context/checkpoints/context_model.pth")
        self.context_model = ContextTransformer(input_dim=self.action_dim, hidden_dim=self.context_dim).to(self.device)
        if os.path.exists(ctx_path):
            self.context_model.load_state_dict(torch.load(ctx_path, map_location=self.device))
        else:
            print("!!! Context model not found")
        self.context_model.eval()

        self._load_pools()

        # Episode horizon comes from config (single source of truth) so the env,
        # training, and eval all agree instead of three hardcoded constants.
        self.max_steps = int(self.config.get("training", {}).get("max_steps", 50))
        # Reward scale (see config.yaml): lifts the O(1) RMS reward above the
        # 1024-dim entropy term so SAC actually optimizes control.
        self.reward_scale = float(self.config.get("training", {}).get("reward_scale", 10.0))
        self.history_window = 5
        self.history_buffer = deque(maxlen=self.history_window)

    def _load_pools(self):
        # Fail loudly: silently training on np.zeros pools produces a model
        # that looks trained but learned nothing. A missing pool is a bug.
        p_path = resolve_path(self.config['paths']['physio_embeddings'])
        u_path = resolve_path(self.config['paths']['user_embeddings'])
        try:
            p_npz = np.load(p_path, allow_pickle=True)
            u_npz = np.load(u_path, allow_pickle=True)
            self.physio_pool = p_npz['embeddings']
            self.user_pool = u_npz['embeddings']
        except (FileNotFoundError, KeyError, OSError) as e:
            raise RuntimeError(
                f"Cannot load embedding pools ({p_path}, {u_path}): {e}. "
                f"Run preprocessing + physio/user training first."
            )

        # Per-physio-row participant id + a participant -> user_vec map so reset()
        # can pair a sampled physio state with ITS OWN user's profile. The world
        # model was trained only on matched (physio, user) pairs; sampling the two
        # independently would feed it pairs that never co-occur, i.e. off its
        # training distribution. Pairing keeps agent-training states on-manifold.
        if 'participant_ids' in p_npz and 'participant_ids' in u_npz:
            self._physio_pids = np.array([str(p) for p in p_npz['participant_ids']])
            self._user_by_pid = {
                str(pid): vec
                for pid, vec in zip(u_npz['participant_ids'], self.user_pool)
            }

        # Per-dimension physio std used to scale the reward. Prefer the std
        # persisted by the encoder (single source of truth, M3) so training and
        # eval share the exact reward/eval geometry; fall back to recomputing it
        # from the pool for legacy embeddings without the key.
        if 'physio_std' in p_npz:
            self._physio_std = np.asarray(p_npz['physio_std'], dtype=np.float32)
        else:
            self._physio_std = np.std(self.physio_pool, axis=0).astype(np.float32)

        # Per-dim physio bounds for the inference clamp. Even with a bounded-delta,
        # rollout-trained world model, a long episode can still walk physio off the
        # observed manifold; clamping each predicted state to the pool's observed
        # range (with a small margin) is the hard safety net that prevents the
        # ~20-std explosions the controllability probe exposed.
        pmin = self.physio_pool.min(axis=0).astype(np.float32)
        pmax = self.physio_pool.max(axis=0).astype(np.float32)
        margin = 0.5 * self._physio_std  # allow modest extrapolation past extremes
        self._physio_min = pmin - margin
        self._physio_max = pmax + margin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Seed the env RNG only when a seed is explicitly given; otherwise keep
        # the existing stream so episode-to-episode resets stay reproducible
        # (a fresh default_rng(None) every episode would reintroduce entropy).
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)

        p_idx = int(self.rng.integers(0, len(self.physio_pool)))
        self.current_physio = self.physio_pool[p_idx]

        # Pair the physio state with its own participant's profile when the
        # linkage is available (real pools); otherwise fall back to an
        # independent user draw (e.g. unit-test pools without participant ids).
        pids = getattr(self, "_physio_pids", None)
        umap = getattr(self, "_user_by_pid", None)
        if pids is not None and umap is not None and umap.get(pids[p_idx]) is not None:
            self.current_user = umap[pids[p_idx]]
        else:
            u_idx = int(self.rng.integers(0, len(self.user_pool)))
            self.current_user = self.user_pool[u_idx]

        self.history_buffer.clear()
        self.current_context = np.zeros(self.context_dim, dtype=np.float32)

        # Goal must differ from the start state, else the episode is degenerate
        # (zero initial distance, nothing to learn).
        target_idx = int(self.rng.integers(0, len(self.physio_pool)))
        if len(self.physio_pool) > 1:
            while target_idx == p_idx:
                target_idx = int(self.rng.integers(0, len(self.physio_pool)))
        self.target_physio = self.physio_pool[target_idx]

        self._t = 0
        self.state = self._build_obs()

        return self.state.astype(np.float32), {}

    def _build_obs(self):
        """Agent observation = [physio, user, context, target_physio]."""
        return np.concatenate([
            self.current_physio, self.current_user,
            self.current_context, self.target_physio,
        ])

    def step(self, action):

        self._t += 1

        action = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        self.history_buffer.append(action)
        self.current_context = self._compute_context()

        # World-model input is the [physio, user, context] prefix only — the
        # target appended for the agent is sliced off here.
        wm_state = self.state[:self.wm_state_dim]
        prev_state_tensor = torch.FloatTensor(wm_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_physio_tensor = self.world_model(prev_state_tensor, action_tensor)
            self.current_physio = next_physio_tensor.cpu().numpy()[0]

        # Hard safety net: keep predicted physio within the observed pool range so
        # autoregressive drift cannot explode the state off-manifold (see probe).
        # Guarded: a custom pool loader may not set bounds, in which case skip.
        pmin = getattr(self, "_physio_min", None)
        pmax = getattr(self, "_physio_max", None)
        if pmin is not None and pmax is not None:
            self.current_physio = np.clip(self.current_physio, pmin, pmax)

        self.state = self._build_obs()
        
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

    def distance(self, a, b):
        """Scale-aware RMS distance between two physio latents.

        Per-dimension normalized by the physio-pool std (O(1) scale), then mean
        over dims so the value is invariant to physio_dim. This is the SINGLE
        distance geometry used by both the reward and the evaluation
        `improvement` metric — the agent is scored on exactly what it optimizes.
        NaN/inf guarded.
        """
        diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)

        std = getattr(self, "_physio_std", None)
        if std is None or not np.all(np.isfinite(std)):
            std = np.ones_like(diff)

        norm_diff = diff / np.maximum(std, 1e-6)
        rmse = float(np.sqrt(np.mean(norm_diff ** 2)))
        return rmse if np.isfinite(rmse) else 1e3

    def _calculate_reward(self):
            """Negative scale-aware RMS distance to the target physio state,
            lifted by reward_scale. Range: (-inf, 0], NaN/inf guarded."""
            reward = -self.distance(self.current_physio, self.target_physio) * self.reward_scale

            if not np.isfinite(reward):
                reward = -1e3  # finite penalty, never NaN/inf into the buffer

            return reward