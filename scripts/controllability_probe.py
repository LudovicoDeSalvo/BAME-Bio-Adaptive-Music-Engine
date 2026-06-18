"""Controllability probe for the music-control environment.

The SAC sweep showed the agent's mean per-step normalized distance to target
pinned at ~5.8 std across every hyperparameter setting — i.e. no learning. That
is either (a) an RL problem the agent could in principle solve, or (b) an env
ceiling: no song actually moves physio toward the target. This probe settles it
WITHOUT any RL, by querying the world model directly.

Two measurements, averaged over N random (start, target) pairs:

  1. ONE-STEP: from a fixed start state, apply every catalog song once and record
     the resulting normalized distance to target. Reports the random-song mean
     (blind baseline) and the oracle best-song min (best achievable in one step).

  2. GREEDY ORACLE ROLLOUT: from the start, at each step pick the song that most
     reduces immediate distance, commit it via the real env dynamics, repeat for
     max_steps. Reports start vs final distance — the practical controllability
     ceiling, the closest non-RL analog to the agent's task.

Interpretation:
  oracle/greedy << start (and << ~5.8)  -> env IS controllable; the failure is in
                                           the RL agent (proto-action reach, critic
                                           re-rank, or signal strength).
  oracle/greedy ~ random ~ start        -> ENV CEILING; the world model has no
                                           action sensitivity (or music->physio
                                           signal is absent in the data). Fixing
                                           the agent cannot help.

Run:
    python -m scripts.controllability_probe --targets 20 --max-steps 50
"""

import os
import sys
import argparse

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulator.gym_env import MusicEnv          # noqa: E402
from rl.wolpertinger import WolpertingerPolicy  # noqa: E402
from utils.common import get_device, set_seed   # noqa: E402

# Agent reference: the flat mean per-step normalized distance the SAC sweep
# reached regardless of hyperparameters. Used only for side-by-side context.
AGENT_REFERENCE_NORM_DIST = 5.8


def _batch_norm_dist(next_physios, target, std):
    """Row-wise scale-aware RMS distance — vectorized MusicEnv.distance."""
    diff = next_physios - target[None, :]
    nd = diff / np.maximum(std, 1e-6)[None, :]
    return np.sqrt(np.mean(nd ** 2, axis=1))


def _predict_all_songs(env, wm_state, catalog_t):
    """next_physio for EVERY catalog song from a single fixed wm_state.

    Mirrors MusicEnv.step's physio update exactly: next_physio =
    world_model(wm_state, song). wm_state is the [physio,user,context] prefix and
    is identical across songs, so we broadcast it and run one batched forward.
    Returns (n_songs, physio_dim) numpy.
    """
    n = catalog_t.shape[0]
    state_b = torch.as_tensor(wm_state, dtype=torch.float32, device=env.device)
    state_b = state_b.unsqueeze(0).expand(n, -1)
    with torch.no_grad():
        nxt = env.world_model(state_b, catalog_t)
    return nxt.cpu().numpy()


def run_controllability_probe(targets=20, max_steps=None, seed=42,
                              config_path="configs/config.yaml"):
    set_seed(seed)
    device = get_device()

    env = MusicEnv(config_path=config_path)
    if max_steps is None:
        max_steps = env.max_steps
    std = env._physio_std

    # Catalog = the exact song vectors the agent applies as actions.
    try:
        wolp = WolpertingerPolicy(k_neighbors=1, device=device)
    except Exception as e:
        print(f" !!! FAISS error: {e}")
        return None
    catalog = np.asarray(wolp.retriever.embeddings, dtype=np.float32)
    catalog_t = torch.as_tensor(catalog, dtype=torch.float32, device=device)
    n_songs = catalog.shape[0]

    print(f">> [Probe] {n_songs} songs | {targets} targets | max_steps {max_steps}")

    start_d, rand_d, oracle1_d = [], [], []
    greedy_final_d = []

    for t in range(targets):
        # Deterministic reset: same (start, target) for every song this trial.
        env.reset(seed=seed + t)
        target = env.target_physio.copy()
        start = float(env.distance(env.current_physio, target))

        # --- one-step: all songs from the start state ---
        wm_state = env.state[:env.wm_state_dim].copy()
        nxt = _predict_all_songs(env, wm_state, catalog_t)
        d = _batch_norm_dist(nxt, target, std)
        start_d.append(start)
        rand_d.append(float(np.mean(d)))
        oracle1_d.append(float(np.min(d)))

        # --- greedy oracle rollout: commit best song each step via real env ---
        env.reset(seed=seed + t)  # re-reset to identical start
        for _ in range(max_steps):
            wm_state = env.state[:env.wm_state_dim].copy()
            nxt = _predict_all_songs(env, wm_state, catalog_t)
            d = _batch_norm_dist(nxt, target, std)
            best = int(np.argmin(d))
            env.step(catalog[best])
        greedy_final_d.append(float(env.distance(env.current_physio, target)))

    _report(np.mean(start_d), np.mean(rand_d), np.mean(oracle1_d),
            np.mean(greedy_final_d), max_steps)
    return {
        "start_dist": float(np.mean(start_d)),
        "random_song_mean": float(np.mean(rand_d)),
        "oracle_one_step": float(np.mean(oracle1_d)),
        "greedy_final": float(np.mean(greedy_final_d)),
        "agent_reference": AGENT_REFERENCE_NORM_DIST,
    }


def _report(start, rand, oracle1, greedy, max_steps):
    print("\n" + "=" * 64)
    print(" CONTROLLABILITY PROBE")
    print("=" * 64)
    print(f" start distance              : {start:6.3f}")
    print(f" random-song mean (1 step)   : {rand:6.3f}")
    print(f" oracle best song (1 step)   : {oracle1:6.3f}")
    print(f" greedy oracle ({max_steps:>2} steps)    : {greedy:6.3f}")
    print(f" agent (sweep reference)     : {AGENT_REFERENCE_NORM_DIST:6.3f}")
    print("-" * 64)

    # Relative improvement the oracle achieves over the blind baseline.
    gain_1 = (rand - oracle1) / rand if rand else 0.0
    gain_greedy = (start - greedy) / start if start else 0.0
    print(f" oracle 1-step gain vs random : {gain_1 * 100:5.1f}%")
    print(f" greedy gain vs start         : {gain_greedy * 100:5.1f}%")

    # Regime priority: rollout divergence dominates. If the greedy oracle — which
    # at every step picks the locally optimal song — still ends up FAR above the
    # start distance, the dynamics themselves are unstable under autoregressive
    # rollout, and no agent can train in them. This must be checked before the
    # one-step gain, which can look fine while the rollout silently blows up.
    rollout_diverged = greedy > 2.0 * start

    if rollout_diverged:
        print("\n VERDICT: WORLD-MODEL DRIFT. One-step control works (oracle <")
        print("          random < start), but the greedy rollout EXPLODES away")
        print(f"          from the target ({start:.2f} -> {greedy:.2f}). The world")
        print("          model is trained on one-step transitions from REAL")
        print("          physio; feeding its own predictions back compounds error")
        print("          and leaves the training manifold. The agent cannot learn")
        print("          in a env that diverges within an episode. Fix the world")
        print("          model rollout stability (multi-step/rollout training loss,")
        print("          residual/decayed prediction, or clamp physio to the pool")
        print("          range each step) BEFORE any further agent tuning.")
    elif gain_1 > 0.15 or gain_greedy > 0.15:
        print("\n VERDICT: env IS controllable. The oracle steers physio toward")
        print("          the target and the rollout stays bounded, so the flat")
        print("          agent result is an RL failure (proto-action reach / critic")
        print("          re-rank / weak signal), not an env ceiling. Fix the agent.")
    else:
        print("\n VERDICT: ENV CEILING. No song meaningfully reduces distance —")
        print("          the world model has little/no action sensitivity (or the")
        print("          music->physio signal is absent in the data). Tuning the")
        print("          agent cannot help; fix the world model / data upstream.")


def main():
    p = argparse.ArgumentParser(description="Env controllability probe")
    p.add_argument("--targets", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None,
                   help="greedy rollout length (default: env max_steps)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_controllability_probe(targets=args.targets, max_steps=args.max_steps,
                              seed=args.seed)


if __name__ == "__main__":
    main()
