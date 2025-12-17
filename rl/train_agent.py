import os

import numpy as np

from audio.faiss_index import AudioIndex
from rl.sac_agent import SACAgent
from rl.wolpertinger import WolpertingerPolicy
from simulator.gym_env import MusicEnv
from utils.common import ensure_dir, load_config, resolve_path, set_seed


def _load_index(paths: dict) -> AudioIndex | None:
    embeddings_path = resolve_path(paths.get("embeddings_path", "data/processed/audio_embeddings.npz"))
    index_path = resolve_path(paths.get("index_path", "data/processed/audio_index.npz"))
    if not os.path.exists(embeddings_path):
        return None
    if os.path.exists(index_path):
        return AudioIndex.load(index_path)
    payload = np.load(embeddings_path, allow_pickle=True)
    index = AudioIndex()
    index.build(payload["embeddings"], payload["song_ids"])
    ensure_dir(os.path.dirname(index_path))
    index.save(index_path)
    return index


def train_sac_agent(steps: int = 10000, config_path: str = "configs/config.yaml") -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})

    set_seed(int(training.get("seed", 42)))

    env = MusicEnv(config_path)
    agent = SACAgent(env.state_dim, int(model_cfg.get("action_dim", 128)))

    index = _load_index(paths)
    wolpertinger = WolpertingerPolicy(index, k=5) if index else None
    embeddings_lookup = None
    if index:
        embeddings_lookup = {int(sid): index.embeddings[idx] for idx, sid in enumerate(index.song_ids)}

    state = env.reset()
    for _ in range(int(steps)):
        action = agent.act(state)
        if wolpertinger and embeddings_lookup:
            song_id, _ = wolpertinger.select(action)
            action_vec = embeddings_lookup.get(song_id, action)
        else:
            action_vec = action

        next_state, reward, done, _ = env.step(action_vec)
        agent.update({"state": state, "action": action_vec, "reward": reward, "next_state": next_state})
        state = env.reset() if done else next_state

    model_path = resolve_path(paths.get("sac_model_path", "models/sac_agent.json"))
    ensure_dir(os.path.dirname(model_path))
    agent.save(model_path)
    return model_path
