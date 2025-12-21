import os
from typing import Tuple

import numpy as np

from simulator.world_model import WorldModel
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def _pad_or_truncate_1d(vec: np.ndarray, target_dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size == target_dim:
        return v
    if v.size < target_dim:
        out = np.zeros(target_dim, dtype=float)
        out[: v.size] = v
        return out
    return v[:target_dim]


def _pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == target_dim:
        return arr
    if arr.shape[1] < target_dim:
        pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=float)
        return np.concatenate([arr, pad], axis=1)
    return arr[:, :target_dim]


def _align_actions(
    song_ids: np.ndarray,
    action_dim: int,
    embeddings_path: str,
    fallback_actions: np.ndarray | None = None,
) -> Tuple[np.ndarray, str]:
    rng = np.random.default_rng(0)
    used_source = "audio_embeddings"
    if os.path.exists(embeddings_path):
        payload = np.load(embeddings_path, allow_pickle=True)
        embeddings = payload["embeddings"]
        embed_song_ids = payload["song_ids"]
        lookup = {int(sid): np.asarray(embeddings[idx], dtype=float).reshape(-1) for idx, sid in enumerate(embed_song_ids)}
        actions = []
        had_missing = False
        used_va_fill = False
        used_noise_fill = False
        warned_shape = False
        for i, sid in enumerate(song_ids):
            vec = lookup.get(int(sid))
            if vec is None:
                had_missing = True
                if fallback_actions is not None and i < len(fallback_actions):
                    vec = fallback_actions[i]
                    used_va_fill = True
                else:
                    vec = rng.normal(scale=0.1, size=action_dim)
                    used_noise_fill = True
            if not warned_shape and np.asarray(vec).ndim != 1:
                print(">> Warning: embedding/fallback with unexpected ndim; flattening.", flush=True)
                warned_shape = True
            vec = _pad_or_truncate_1d(vec, action_dim)
            actions.append(vec)
        actions = np.stack(actions, axis=0)
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions / (norms + 1e-8)
        if had_missing and used_noise_fill and used_va_fill:
            used_source = "audio_embeddings+va_fill+noise_fill"
        elif had_missing and used_noise_fill:
            used_source = "audio_embeddings+noise_fill"
        elif had_missing and used_va_fill:
            used_source = "audio_embeddings+va_fill"
    else:
        if fallback_actions is None:
            actions = rng.normal(scale=0.1, size=(len(song_ids), action_dim))
            used_source = "random_noise"
        else:
            actions = fallback_actions
            used_source = "valence_arousal"
        actions = _pad_or_truncate(actions, action_dim)
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions / (norms + 1e-8)

    actions = _pad_or_truncate(actions, action_dim)
    return actions, used_source


def _load_physio_states(physio_cache: str, physio_dim: int) -> np.ndarray:
    payload = np.load(physio_cache, allow_pickle=True)
    if os.path.exists(resolve_path("data/processed/physio_embeddings.npz")):
        emb_payload = np.load(resolve_path("data/processed/physio_embeddings.npz"), allow_pickle=True)
        states = emb_payload["embeddings"]
    else:
        states = payload["features"]
    states = _pad_or_truncate(states, physio_dim)
    return states


def train_world_model(config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    physio_cache = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    embeddings_path = resolve_path(paths.get("embeddings_path", "data/processed/audio_embeddings.npz"))
    world_path = resolve_path(paths.get("world_model_path", "models/world_model.json"))

    if not os.path.exists(physio_cache):
        raise FileNotFoundError("Physio cache missing. Run data processing first.")

    physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
    action_dim = int(model_cfg.get("action_dim", 128))

    payload = np.load(physio_cache, allow_pickle=True)
    song_ids = payload["song_ids"]
    if "song_no" in payload:
        song_nos = payload["song_no"].astype(int)
    else:
        song_nos = np.arange(len(song_ids))
        print(">> Warning: 'song_no' not found in cache. Using dataset order as sequence index.")
    participant_ids = np.char.strip(payload["participant_ids"].astype(str))
    valence = payload["valence"] if "valence" in payload else None
    arousal = payload["arousal"] if "arousal" in payload else None
    physio_states = _load_physio_states(physio_cache, physio_dim)

    states = physio_states
    state_dim = states.shape[1]

    fallback_actions = None
    if valence is not None and arousal is not None:
        fallback_actions = np.stack([valence, arousal], axis=1)

    actions, action_source = _align_actions(song_ids, action_dim, embeddings_path, fallback_actions=fallback_actions)

    traj_states = []
    traj_actions = []
    traj_next_states = []
    unique_users = np.unique(participant_ids)
    used_users = 0
    for uid in unique_users:
        idx = np.where(participant_ids == uid)[0]
        if idx.size < 2:
            continue
        used_users += 1
        order = np.argsort(song_nos[idx])
        idx = idx[order]
        s = states[idx]
        a = actions[idx]
        traj_states.append(s[:-1])
        traj_actions.append(a[:-1])
        traj_next_states.append(s[1:])

    if not traj_states:
        raise RuntimeError("No trajectories found to train the world model.")

    train_states = np.concatenate(traj_states, axis=0)
    train_actions = np.concatenate(traj_actions, axis=0)
    train_next_states = np.concatenate(traj_next_states, axis=0)

    print(f">> Training world model on {train_states.shape[0]} transitions from {used_users} users")
    print(f">> Action source: {action_source}, state_dim={state_dim}, action_dim={action_dim}")

    model = WorldModel(state_dim, action_dim)
    model.fit(train_states, train_actions, train_next_states)

    ensure_dir(os.path.dirname(world_path))
    model.save(world_path)
    return world_path
