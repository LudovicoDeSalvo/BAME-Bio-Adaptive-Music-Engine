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


def _load_physio_states(physio_cache: str, physio_dim: int, embeddings_path: str) -> np.ndarray:
    payload = np.load(physio_cache, allow_pickle=True)
    if os.path.exists(embeddings_path):
        emb_payload = np.load(embeddings_path, allow_pickle=True)
        states = emb_payload["embeddings"]
    else:
        states = payload["features"]
    states = _pad_or_truncate(states, physio_dim)
    return states


def _load_profile_states(
    embeddings_path: str,
    profile_dim: int,
    participant_ids: np.ndarray,
    fallback_profile_cache: str | None,
) -> np.ndarray:
    if profile_dim <= 0:
        return np.zeros((len(participant_ids), 0), dtype=float)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"User embeddings not found at {embeddings_path}")
    payload = np.load(embeddings_path, allow_pickle=True)
    if "embeddings" not in payload:
        raise KeyError("user_embeddings.npz missing 'embeddings'")
    embeddings = _pad_or_truncate(payload["embeddings"], profile_dim)
    if "participant_ids" in payload:
        emb_ids = np.char.strip(np.asarray(payload["participant_ids"], dtype=str))
    else:
        if fallback_profile_cache is None or not os.path.exists(fallback_profile_cache):
            raise KeyError("user_embeddings.npz missing 'participant_ids' and no profile_cache.csv available")
        import pandas as pd  # type: ignore

        df = pd.read_csv(fallback_profile_cache)
        if "participant_id" not in df.columns:
            raise KeyError("profile_cache.csv missing 'participant_id' column")
        emb_ids = np.char.strip(np.asarray(df["participant_id"].to_numpy(), dtype=str))
    if len(emb_ids) != embeddings.shape[0]:
        min_len = min(len(emb_ids), embeddings.shape[0])
        print(
            f">> Warning: profile embeddings length mismatch (ids={len(emb_ids)}, emb={embeddings.shape[0]}). "
            f"Truncating to {min_len}."
        )
        emb_ids = emb_ids[:min_len]
        embeddings = embeddings[:min_len]
    lookup = {pid: embeddings[idx] for idx, pid in enumerate(emb_ids)}
    profiles = []
    missing = 0
    for pid in participant_ids:
        vec = lookup.get(pid)
        if vec is None:
            missing += 1
            vec = np.zeros(profile_dim, dtype=float)
        profiles.append(vec)
    if missing:
        print(f">> Warning: missing {missing} profile embeddings; filled with zeros.")
    return np.asarray(profiles, dtype=float)


def train_world_model(config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    physio_cache = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    embeddings_path = resolve_path(paths.get("embeddings_path", "data/processed/audio_embeddings.npz"))
    world_path = resolve_path(paths.get("world_model_path", "models/world_model.json"))
    processed_dir = paths.get("processed_dir", "data/processed")
    physio_emb_path = resolve_path(paths.get("physio_embeddings", os.path.join(processed_dir, "physio_embeddings.npz")))
    user_emb_path = resolve_path(paths.get("user_embeddings", os.path.join(processed_dir, "user_embeddings.npz")))
    profile_cache = resolve_path(paths.get("profile_cache", os.path.join(processed_dir, "profile_cache.csv")))

    if not os.path.exists(physio_cache):
        raise FileNotFoundError("Physio cache missing. Run data processing first.")

    physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
    profile_dim = int(model_cfg.get("profile_embedding_dim", 16))
    context_dim = int(model_cfg.get("context_embedding_dim", 32))
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
    physio_states = _load_physio_states(physio_cache, physio_dim, physio_emb_path)
    profile_states = _load_profile_states(user_emb_path, profile_dim, participant_ids, profile_cache)
    context_states = np.zeros((physio_states.shape[0], context_dim), dtype=float)

    states = np.concatenate([physio_states, profile_states, context_states], axis=1)
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


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train world model from physio cache.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    world_path = train_world_model(config_path=args.config)
    print(f">> Saved world model to: {world_path}")


if __name__ == "__main__":
    _main()
