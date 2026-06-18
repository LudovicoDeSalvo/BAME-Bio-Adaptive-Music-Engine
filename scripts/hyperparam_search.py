"""Hyperparameter sweep for the SAC agent.

Runs `rl.train_agent.train_sac_agent` repeatedly under different training-section
overrides, scores each run on learning progress + stability, and writes a ranked
leaderboard. Designed around the failure mode observed in the BAME runs: an
unreachable entropy target (`target_entropy_scale`) drives `alpha` to run away,
which inflates the Q target and diverges the critic while the policy collapses to
a single song. The default search space therefore centers on the levers that
control that loop: target_entropy_scale, reward_scale, lr.

Each trial trains from scratch with `save=False` (never clobbers the real
checkpoint) and `quiet=True`, returning a metrics dict consumed here.

Run:
    python -m scripts.hyperparam_search --steps 6000 --method grid
    python -m scripts.hyperparam_search --steps 6000 --method random --trials 12

Scoring (higher = better):
    diverged critic                      -> disqualified (-inf)
    otherwise   -mean_norm_distance - COLLAPSE_PENALTY * collapse_fraction

`mean_norm_distance` (the achieved mean per-step normalized distance to the
target, in std units) is used instead of raw reward because raw reward is in
units of reward_scale * max_steps and is therefore NOT comparable across trials
that vary those. Lower distance is better, so the score negates it.
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import itertools

# Allow `python scripts/hyperparam_search.py` from the repo root in addition to
# `python -m scripts.hyperparam_search`.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl.train_agent import train_sac_agent  # noqa: E402

# --- Configuration ---
DEFAULT_OUT_DIR = "rl/sweeps"
COLLAPSE_PENALTY = 2.0  # subtracted per unit collapse_fraction (0..1); on the
# same O(1) scale as mean_norm_distance so a fully collapsed policy is penalized
# comparably to a couple of std of residual distance.

# Grid search space. Keys must be training-section hyperparameters understood by
# train_sac_agent's overrides (target_entropy_scale, lr, gamma, tau, hidden_dim,
# batch_size, k_neighbors, reward_scale, max_steps).
SEARCH_SPACE = {
    # Auto-tuned alpha + low target_entropy_scale was insufficient: every trial
    # still diverged the critic because the summed-over-1024-dims entropy bonus
    # inflates the Q target even at alpha~1. So fix alpha small (0 disables the
    # entropy bonus) and add gradient clipping to cap the loss blow-up.
    "alpha": [0.0, 1e-3, 1e-2],
    "max_grad_norm": [1.0, 10.0],
    "reward_scale": [1.0, 10.0],
}


def score_trial(m):
    """Map a metrics dict to a scalar to maximize. -inf disqualifies."""
    if m is None:
        return -math.inf
    if m.get("critic_diverged"):
        return -math.inf
    mnd = m.get("mean_norm_distance")
    if mnd is None or not math.isfinite(mnd):
        return -math.inf
    collapse = m.get("collapse_fraction") or 0.0
    if not math.isfinite(collapse):
        collapse = 1.0
    return -mnd - COLLAPSE_PENALTY * collapse


def _grid(space):
    keys = list(space.keys())
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def _random(space, n, rng):
    keys = list(space.keys())
    seen = set()
    # Cap attempts so a small space (fewer unique combos than n) still terminates.
    max_combos = 1
    for k in keys:
        max_combos *= len(space[k])
    n = min(n, max_combos)
    while len(seen) < n:
        combo = tuple(rng.choice(space[k]) for k in keys)
        if combo in seen:
            continue
        seen.add(combo)
        yield dict(zip(keys, combo))


def run_hyperparam_search(steps=6000, method="grid", trials=12, seed=42,
                          out_dir=DEFAULT_OUT_DIR, space=None):
    """Execute the sweep and return the sorted list of result records."""
    space = space or SEARCH_SPACE
    os.makedirs(os.path.join(ROOT_DIR, out_dir), exist_ok=True)
    rng = random.Random(seed)

    if method == "grid":
        configs = list(_grid(space))
    elif method == "random":
        configs = list(_random(space, trials, rng))
    else:
        raise ValueError(f"unknown method {method!r} (expected 'grid' or 'random')")

    print(f">> [Sweep] {len(configs)} trials x {steps} steps "
          f"(method={method}, seed={seed})")
    print(f">> [Sweep] search space: {space}")

    results = []
    t0 = time.time()
    for i, overrides in enumerate(configs, 1):
        print(f"\n>> [Sweep] trial {i}/{len(configs)}: {overrides}")
        ts = time.time()
        try:
            metrics = train_sac_agent(
                steps=steps, overrides=overrides,
                quiet=True, return_metrics=True, save=False,
            )
        except Exception as e:  # one bad trial must not kill the whole sweep
            print(f"   !!! trial failed: {e}")
            metrics = None

        sc = score_trial(metrics)
        rec = {"trial": i, "overrides": overrides, "score": sc, "metrics": metrics}
        results.append(rec)

        if metrics is None:
            print(f"   -> FAILED  ({time.time() - ts:.0f}s)")
        else:
            print(f"   -> score={sc:.3f}  norm_dist={metrics['mean_norm_distance']:.3f}  "
                  f"final_alpha={metrics['final_alpha']:.3f}  "
                  f"diverged={metrics['critic_diverged']}  "
                  f"collapse={metrics['collapse_fraction']:.2f}  "
                  f"({time.time() - ts:.0f}s)")

    results.sort(key=lambda r: r["score"], reverse=True)
    _write_outputs(results, steps, method, seed, out_dir)
    _print_leaderboard(results)
    print(f"\n>> [Sweep] done in {time.time() - t0:.0f}s")
    return results


def _write_outputs(results, steps, method, seed, out_dir):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(ROOT_DIR, out_dir, f"sweep_{stamp}")

    with open(base + ".json", "w") as f:
        json.dump(
            {"steps": steps, "method": method, "seed": seed, "results": results},
            f, indent=2,
        )

    # Flat CSV (drops the reward_ma_history list) for quick spreadsheet triage.
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "trial", "score", "mean_norm_distance", "best_reward_ma",
            "final_reward_ma", "final_critic_loss", "final_alpha", "max_alpha",
            "critic_diverged", "collapse_fraction", "overrides",
        ])
        for rank, r in enumerate(results, 1):
            m = r["metrics"] or {}
            w.writerow([
                rank, r["trial"], f"{r['score']:.4f}",
                _fmt(m.get("mean_norm_distance")), _fmt(m.get("best_reward_ma")),
                _fmt(m.get("final_reward_ma")),
                _fmt(m.get("final_critic_loss")), _fmt(m.get("final_alpha")),
                _fmt(m.get("max_alpha")), m.get("critic_diverged"),
                _fmt(m.get("collapse_fraction")), json.dumps(r["overrides"]),
            ])

    # Best config alone, ready to paste into config.yaml's training block.
    if results and results[0]["metrics"] is not None:
        with open(base + "_best.json", "w") as f:
            json.dump(results[0]["overrides"], f, indent=2)

    print(f"\n>> [Sweep] wrote {base}.json / .csv")


def _fmt(x):
    return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else x)


def _print_leaderboard(results, top=10):
    print("\n" + "=" * 78)
    print(" SWEEP LEADERBOARD (top {})".format(min(top, len(results))))
    print("=" * 78)
    print(f" {'#':>2} {'score':>8} {'normdist':>9} {'alpha':>7} {'div':>4} "
          f"{'coll':>5}  overrides")
    for rank, r in enumerate(results[:top], 1):
        m = r["metrics"]
        if m is None:
            print(f" {rank:>2} {'FAILED':>8}   - - - -  {r['overrides']}")
            continue
        print(f" {rank:>2} {r['score']:>8.3f} {m['mean_norm_distance']:>9.3f} "
              f"{m['final_alpha']:>7.2f} {str(m['critic_diverged'])[:4]:>4} "
              f"{m['collapse_fraction']:>5.2f}  {r['overrides']}")
    if results and results[0]["metrics"] is not None:
        print("\n Best overrides -> paste into config.yaml `training`:")
        print("  " + json.dumps(results[0]["overrides"]))


def main():
    p = argparse.ArgumentParser(description="SAC hyperparameter sweep")
    p.add_argument("--steps", type=int, default=6000,
                   help="training steps per trial (default 6000)")
    p.add_argument("--method", choices=["grid", "random"], default="grid")
    p.add_argument("--trials", type=int, default=12,
                   help="number of trials for --method random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    run_hyperparam_search(steps=args.steps, method=args.method,
                          trials=args.trials, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
