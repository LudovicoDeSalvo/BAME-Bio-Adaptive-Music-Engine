import os

import numpy as np

from simulator.world_model import WorldModel
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def _align_actions(song_ids: np.ndarray, action_dim: int, embeddings_path: str) -> np.ndarray:
    if not os.path.exists(embeddings_path):
        return np.random.default_rng(0).normal(size=(len(song_ids), action_dim))
    payload = np.load(embeddings_path, allow_pickle=True)
    embeddings = payload["embeddings"]
    embed_song_ids = payload["song_ids"]
    lookup = {int(sid): embeddings[idx] for idx, sid in enumerate(embed_song_ids)}
    actions = []
    for sid in song_ids:
        vec = lookup.get(int(sid))
        if vec is None:
            vec = np.zeros(action_dim, dtype=float)
        actions.append(vec)
    actions = np.asarray(actions, dtype=float)
    if actions.shape[1] != action_dim:
        if actions.shape[1] < action_dim:
            pad = np.zeros((actions.shape[0], action_dim - actions.shape[1]))
            actions = np.concatenate([actions, pad], axis=1)
        else:
            actions = actions[:, :action_dim]
    return actions


def _load_physio_states(physio_cache: str, physio_dim: int) -> np.ndarray:
    payload = np.load(physio_cache, allow_pickle=True)
    if os.path.exists(resolve_path("data/processed/physio_embeddings.npz")):
        emb_payload = np.load(resolve_path("data/processed/physio_embeddings.npz"), allow_pickle=True)
        states = emb_payload["embeddings"]
    else:
        states = payload["features"]
    if states.shape[1] != physio_dim:
        if states.shape[1] < physio_dim:
            pad = np.zeros((states.shape[0], physio_dim - states.shape[1]))
            states = np.concatenate([states, pad], axis=1)
        else:
            rng = np.random.default_rng(0)
            proj = rng.normal(scale=0.1, size=(states.shape[1], physio_dim))
            states = states @ proj
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
    profile_dim = int(model_cfg.get("profile_embedding_dim", 16))
    context_dim = int(model_cfg.get("context_embedding_dim", 32))
    state_dim = physio_dim + profile_dim + context_dim
    action_dim = int(model_cfg.get("action_dim", 128))

    payload = np.load(physio_cache, allow_pickle=True)
    song_ids = payload["song_ids"]
    physio_states = _load_physio_states(physio_cache, physio_dim)

    profile_stub = np.zeros((physio_states.shape[0], profile_dim), dtype=float)
    context_stub = np.zeros((physio_states.shape[0], context_dim), dtype=float)
    states = np.concatenate([physio_states, profile_stub, context_stub], axis=1)

    actions = _align_actions(song_ids, action_dim, embeddings_path)

    states = states[:-1]
    next_states = states[1:]
    actions = actions[:-1]

    model = WorldModel(state_dim, action_dim)
    model.fit(states, actions, next_states)

    ensure_dir(os.path.dirname(world_path))
    model.save(world_path)
    return world_path
