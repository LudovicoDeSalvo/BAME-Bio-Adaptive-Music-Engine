import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from collections import defaultdict

from simulator.world_model import WorldModel
from context.sequence_model import ContextTransformer
from utils.common import ensure_dir, load_config, resolve_path, get_device, set_seed, l2_normalize, parse_clip_id

class TransitionDataset(Dataset):
    def __init__(self, physio_path, audio_emb_path, audio_id_path, user_pool_path, ratings_path, context_model_path, device, holdout_user=None, action_dim=1024, context_dim=128):
        print(">> [Simulator] Loading datasets...")
        self.context_dim = context_dim
        
        # data loading
        p_data = np.load(physio_path, allow_pickle=True)
        audio_embs = np.load(audio_emb_path, allow_pickle=True)
        audio_ids = np.load(audio_id_path, allow_pickle=True)
        u_data = np.load(user_pool_path, allow_pickle=True)
        ratings = pd.read_csv(ratings_path)

        if holdout_user:
            print(f">> [Dataset] Holding out user: {holdout_user}")
            ratings = ratings[ratings['participant_id'].astype(str) != str(holdout_user)]
        
        # audio
        self.audio_map = {str(k): v for k, v in zip(audio_ids, audio_embs)}
        
        # user
        self.user_map = {str(k): v for k, v in zip(u_data['participant_ids'], u_data['embeddings'])}
        
        # physio
        physio_lookup = defaultdict(list)
        p_clip_ids = p_data['clip_ids']

        p_features = p_data['embeddings'] if 'embeddings' in p_data else p_data['features']
        p_song_ids = p_data['song_ids']
        p_participants = p_data['participant_ids']
        # (participant, song_no) is the unique playback/trial key. A participant
        # may hear the same song_id twice (different session position + physio);
        # grouping by song_id alone merges those distinct trials and fabricates
        # cross-playback physio transitions. Fall back to song_id for legacy
        # caches generated before song_nos was added.
        self._has_song_nos = 'song_nos' in p_data
        p_song_nos = p_data['song_nos'] if self._has_song_nos else None

        for idx, (cid, pid) in enumerate(zip(p_clip_ids, p_participants)):
            try:
                sid = str(p_song_ids[idx])
                _sid, _pid, _sno, chunk_idx = parse_clip_id(cid)

                key = (str(pid), sid, str(p_song_nos[idx])) if self._has_song_nos else (str(pid), sid)
                physio_lookup[key].append({
                    'chunk_idx': chunk_idx,
                    'physio': p_features[idx],
                    'clip_id': str(cid)
                })
            except (ValueError, IndexError, KeyError):
                continue

        # context
        self.ctx_model = ContextTransformer(input_dim=action_dim, hidden_dim=context_dim).to(device)
        self.ctx_model.load_state_dict(torch.load(context_model_path, map_location=device))
        self.ctx_model.eval()
        self.device = device
        
        self.transitions = []
        # Rollout trajectories: each is a list of CONTIGUOUS (state, action, target)
        # transitions within one trial. Rollout training samples fixed-length
        # windows from these to feed predictions back in (see train_world_model).
        self.trajectories = []

        user_groups = ratings.groupby('participant_id')

        for pid, group in tqdm(user_groups):
            pid = str(pid)
            if pid not in self.user_map: continue

            u_vec = self.user_map[pid]
            sorted_songs = group.sort_values('song_no')
            history_buffer = []

            for _, row in sorted_songs.iterrows():
                sid = str(row['song_id'])

                key = (pid, sid, str(row['song_no'])) if self._has_song_nos else (pid, sid)
                if key not in physio_lookup:
                    continue

                clips = physio_lookup[key]
                clips.sort(key=lambda x: x['chunk_idx'])

                # A run of contiguous-chunk transitions forms one rollout
                # trajectory; a chunk-continuity break or a missing action ends it.
                current_run = []
                for i in range(len(clips) - 1):
                    curr = clips[i]
                    nxt = clips[i+1]

                    if nxt['chunk_idx'] != curr['chunk_idx'] + 1:
                        if current_run:
                            self.trajectories.append(current_run)
                            current_run = []
                        continue

                    # causality check
                    action_id = curr['clip_id']
                    if action_id not in self.audio_map:
                        if current_run:
                            self.trajectories.append(current_run)
                            current_run = []
                        continue
                    # L2-normalize: the world model must train on the same unit-sphere
                    # action manifold that FAISS returns at inference (real_action).
                    action_vec = np.asarray(l2_normalize(self.audio_map[action_id]), dtype=np.float32)

                    with torch.no_grad():
                        if len(history_buffer) == 0:
                            context_vec = np.zeros(self.context_dim, dtype=np.float32)
                        else:
                            seq = np.array(history_buffer[-5:])
                            inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                            c_out = self.ctx_model(inp)
                            context_vec = c_out.cpu().numpy()[0]

                    # construct state
                    state_t = np.concatenate([curr['physio'], u_vec, context_vec]).astype(np.float32)
                    target_physio = np.asarray(nxt['physio'], dtype=np.float32)

                    trans = (state_t, action_vec, target_physio)
                    self.transitions.append(trans)
                    current_run.append(trans)

                    history_buffer.append(action_vec)

                if current_run:
                    self.trajectories.append(current_run)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, np_ = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(np_, dtype=torch.float32)
        )


class RolloutWindows(Dataset):
    """Fixed-length contiguous windows sampled from rollout trajectories."""
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        s, a, t = self.windows[i]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
        )


def _build_windows(trajectories, depth):
    """All contiguous length-`depth` windows across trajectories.

    Each window is (states[depth, state_dim], actions[depth, action_dim],
    targets[depth, physio_dim]). depth=1 reproduces the original one-step set.
    """
    windows = []
    for traj in trajectories:
        if len(traj) < depth:
            continue
        for st in range(len(traj) - depth + 1):
            seg = traj[st:st + depth]
            states = np.stack([s for (s, a, t) in seg])
            actions = np.stack([a for (s, a, t) in seg])
            targets = np.stack([t for (s, a, t) in seg])
            windows.append((states, actions, targets))
    return windows


def _rollout_loss(model, states, actions, targets, physio_dim, criterion):
    """k-step rollout MSE.

    Feeds the model's OWN physio prediction forward as the next state's physio
    slice while keeping each step's REAL user+context tail (context depends only
    on the real action history, so it is valid to reuse). This is what penalizes
    compounding error that one-step training never sees. depth=1 == one-step MSE.
    """
    depth = states.shape[1]
    pred_phys = None
    total = 0.0
    for k in range(depth):
        if k == 0:
            cur_state = states[:, 0, :]
        else:
            cur_state = torch.cat([pred_phys, states[:, k, physio_dim:]], dim=1)
        pred = model(cur_state, actions[:, k, :])
        total = total + criterion(pred, targets[:, k, :])
        pred_phys = pred
    return total / depth

def train_world_model(epochs=70, config_path="configs/config.yaml", holdout_user=None):
    config = load_config(config_path)
    paths = config.get("paths", {})
    set_seed(int(config.get("training", {}).get("seed", 42)))

    # dims from config (single source of truth)
    mc = config.get("model", {})
    physio_dim = int(mc.get("physio_embedding_dim", 64))
    user_dim = int(mc.get("profile_embedding_dim", 32))
    context_dim = int(mc.get("context_embedding_dim", 128))
    action_dim = int(mc.get("action_dim", 1024))
    state_dim = physio_dim + user_dim + context_dim

    # load embeddings
    physio_path = resolve_path(paths.get("physio_embeddings", "data/processed/physio_embeddings.npz"))
    audio_emb = resolve_path("data/processed/song_embeddings.npy")
    audio_ids = resolve_path("data/processed/song_id_map.npy")
    user_pool = resolve_path(paths.get("user_embeddings"))
    ratings = resolve_path(paths.get("ratings_csv"))
    ctx_model_path = resolve_path("context/checkpoints/context_model.pth")
    
    save_path = resolve_path("simulator/checkpoints/world_model.pth")
    ensure_dir(os.path.dirname(save_path))
    
    device = get_device()

    if not os.path.exists(ctx_model_path):
        print(" !!! Error: context model not found")
        return

    try:
        # check for holdout user config (explicit per-fold LOSO overrides it)
        holdout = holdout_user if holdout_user is not None else config.get('training', {}).get('holdout_user', None)

        dataset = TransitionDataset(physio_path, audio_emb, audio_ids, user_pool, ratings, ctx_model_path, device, holdout,
                                    action_dim=action_dim, context_dim=context_dim)
    except Exception as e:
        print(f" !!! Dataset Error: {e}")
        return
    
    if len(dataset) == 0:
        print(" !!! No transitions created. Check data alignment")
        return

    # Trajectory-level 85/15 split (NOT transition-level) so rollout windows
    # never leak across train/val.
    seed = int(config.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    trajs = list(dataset.trajectories)
    rng.shuffle(trajs)
    n_val = max(1, int(round(0.15 * len(trajs)))) if len(trajs) > 1 else 0
    val_trajs = trajs[:n_val]
    train_trajs = trajs[n_val:]

    # Validation is one-step MSE over held-out trajectories: a stable metric for
    # best-checkpoint selection independent of the training rollout depth.
    val_windows = _build_windows(val_trajs, 1)
    val_loader = (DataLoader(RolloutWindows(val_windows), batch_size=64, shuffle=False)
                  if val_windows else None)

    sim_cfg = config.get("simulator", {})
    max_depth = max(1, int(sim_cfg.get("rollout_depth", 3)))
    max_delta_scale = float(sim_cfg.get("max_delta_scale", 1.0))

    # Per-dim physio std for the bounded-delta head; persisted in the checkpoint.
    physio_std = None
    pdata = np.load(physio_path, allow_pickle=True)
    if 'physio_std' in pdata:
        physio_std = np.asarray(pdata['physio_std'], dtype=np.float32)

    model = WorldModel(state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim,
                       max_delta_scale=max_delta_scale, physio_std=physio_std).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f">> [Simulator] Training World Model: {len(train_trajs)} train / {len(val_trajs)} val "
          f"trajectories | rollout depth up to {max_depth} | delta_scale {max_delta_scale}")

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    current_depth = None
    loader = None

    for epoch in range(epochs):
        # Curriculum: ramp rollout depth 1..max_depth across training. Higher
        # depth applies stronger rollout-stability pressure but has fewer eligible
        # windows (most trials are short), so fall back to depth 1 if a depth is
        # empty.
        depth = min(max_depth, 1 + (epoch * max_depth) // max(1, epochs))
        if depth != current_depth:
            windows = _build_windows(train_trajs, depth)
            if not windows:
                depth = 1
                windows = _build_windows(train_trajs, 1)
            loader = DataLoader(RolloutWindows(windows), batch_size=64, shuffle=True)
            current_depth = depth

        model.train()
        total_loss = 0.0
        for states, actions, targets in tqdm(loader, desc=f"Epoch {epoch+1} (k={depth})", leave=False):
            states, actions, targets = states.to(device), actions.to(device), targets.to(device)

            optimizer.zero_grad()
            loss = _rollout_loss(model, states, actions, targets, physio_dim, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = None
        if val_loader is not None:
            model.eval()
            v = 0.0
            with torch.no_grad():
                for states, actions, targets in val_loader:
                    states, actions, targets = states.to(device), actions.to(device), targets.to(device)
                    v += _rollout_loss(model, states, actions, targets, physio_dim, criterion).item()
            val_loss = v / max(1, len(val_loader))
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            msg = f"   Epoch {epoch+1}/{epochs} (k={depth}) | train: {total_loss/max(1,len(loader)):.5f}"
            if val_loss is not None:
                msg += f" | val(1-step): {val_loss:.5f} (best {best_val:.5f})"
            print(msg)

    model.load_state_dict(best_state)
    model.save(save_path)
    print(f">> [Simulator] Model saved to {save_path}")

if __name__ == "__main__":
    train_world_model()