# scripts/smoke_test_env.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.common import ensure_dir, load_config, resolve_path  # noqa: E402
from simulator.gym_env import MusicEnv  # noqa: E402


def _ensure_dummy_embeddings(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    processed_dir = paths.get("processed_dir", "data/processed")
    physio_path = resolve_path(paths.get("physio_embeddings", f"{processed_dir}/physio_embeddings.npz"))
    user_path = resolve_path(paths.get("user_embeddings", f"{processed_dir}/user_embeddings.npz"))

    physio_dim = int(model_cfg.get("physio_embedding_dim", 64))
    profile_dim = int(model_cfg.get("profile_embedding_dim", 16))

    if not Path(physio_path).exists():
        ensure_dir(Path(physio_path).parent.as_posix())
        np.savez(physio_path, embeddings=np.random.normal(size=(8, physio_dim)).astype(np.float32))
        print(f">> Warning: created dummy physio embeddings at {physio_path}")
    if not Path(user_path).exists():
        ensure_dir(Path(user_path).parent.as_posix())
        np.savez(user_path, embeddings=np.random.normal(size=(8, profile_dim)).astype(np.float32))
        print(f">> Warning: created dummy user embeddings at {user_path}")


def main():
    config_path = "configs/config.yaml"
    _ensure_dummy_embeddings(config_path)

    env = MusicEnv(config_path)

    obs, info = env.reset()
    print("reset obs:", obs.shape, obs.dtype, "info:", info)

    for t in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        assert np.isfinite(r), f"Non-finite reward at t={t}: {r}"
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            print("done at t=", t, "terminated=", terminated, "truncated=", truncated)
            break

    # reset con goal_state
    goal = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    obs, info = env.reset(options={"goal_state": goal})
    print("reset with goal obs:", obs.shape, obs.dtype, "info:", info)

    for t in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        if terminated:
            print("Reached goal at t=", t, "reward=", r)
            break

    env.close()
    print("OK ✅")


if __name__ == "__main__":
    main()
