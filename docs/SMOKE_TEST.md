# BAME — Manual Smoke-Test Guide

How to verify the pipeline end-to-end by hand: the fast **CPU unit suite** (Part A)
and the **full GPU pipeline run** (Part B). Run every command from the project root
(`BAME-Bio-Adaptive-Music-Engine/`).

---

## 0. Prerequisites

```bash
# Python 3.10+, then install deps
pip install -r requirements.txt
pip install pytest        # test-only, not a runtime dep
```

- `faiss-gpu` may fail to resolve on some CUDA setups — install `faiss-cpu` instead.
- GPU part assumes an NVIDIA card (developed on RTX 4070 12 GB). Check with `nvidia-smi`.
- Dataset archive `HKU956.zip` must be in the project root.

---

## Part A — CPU unit suite (fast, no GPU, no data)

The suite is **CPU-only by design**: `tests/conftest.py` sets
`CUDA_VISIBLE_DEVICES=""` before torch loads, so it is deterministic and needs no GPU.

```bash
python -m pytest tests/ -q
```

**Expected:** `43 passed` (a few harmless SwigPy `DeprecationWarning`s from faiss).

What it covers:

| File | Checks |
|------|--------|
| `test_common.py` | `l2_normalize` unit-norm/zero-safe/idempotent, `set_seed` reproducibility, npz/column validators, `parse_clip_id` (rsplit + sid-with-underscores + reject non-conforming) |
| `test_windows.py` | `extract_stats`, full-window shape, **truncated partial windows zeroed** |
| `test_normalizer.py` | user min-max scaler **train==inference round-trip**, holdout exclusion |
| `test_sac_actor.py` | tanh action bounded `|a|≤1`, `log_prob` shape `[batch]` + finite |
| `test_world_model.py` | delta output shape == `physio_dim` |
| `test_gym_env.py` | reset/step contract, **seed determinism**, reward NaN guard, non-finite action guard |
| `test_faiss.py` | index normalized (cosine), retrieval shapes, unit-norm candidates |
| `test_imports.py` | import-smoke of all modules |

Run a single file / test:

```bash
python -m pytest tests/test_sac_actor.py -q
python -m pytest tests/test_gym_env.py::test_seed_determinism -q
```

If `pytest` reports device-mismatch errors, your shell exported a GPU — force CPU:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

---

## Part B — Full GPU pipeline smoke run

Goal: run preprocessing → MERT → all training stages → inference without crashing,
on the GPU, with seeded reproducibility. Use a **small subset** to keep it minutes,
not hours.

The interactive driver is `python main.py` (menu options `[0]`–`[9]`). For a scripted
smoke run, call each module entrypoint directly (shown below).

### B.0 — Audio: the one thing the dataset is missing

`HKU956.zip` contains physiological signals + 3 CSVs but **no audio**. Audio is
referenced in `2. original_song_audio.csv` (Jamendo URLs). You need audio files in
`data/raw/HKU956/2. audio_files/{song_id}.wav` (or `.mp3`) before `align_and_slice`
will produce any clips.

Two ways to get audio:

**Option 1 — Real audio (Jamendo).** Requires a Jamendo API client id (credentials are
no longer hardcoded; the downloader reads the env var) and network. The downloader
writes straight to `2. audio_files`, which is exactly where `align_and_slice` reads —
no move/symlink needed:

```bash
export JAMENDO_CLIENT_ID=<your_client_id>
python -m scripts.download_songs
```

**Option 2 — Synthetic audio (fast, offline; metrics are NOT scientifically meaningful).**
Generates short tones for the songs of a few participants so the pipeline runs:

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import os, numpy as np, pandas as pd, torch, torchaudio
ROOT="data/raw/HKU956"
out=os.path.join(ROOT,"2. audio_files"); os.makedirs(out, exist_ok=True)
r=pd.read_csv(os.path.join(ROOT,"3. AV_ratings.csv"))
parts=["hku1903","hku1904","hku1905"]      # holdout user hku1903 + 2 train users
sids=sorted(set(r[r.participant_id.astype(str).isin(parts)].song_id.astype(str)))
sr, dur, rng = 24000, 95, np.random.default_rng(0)
for sid in sids:
    t=np.linspace(0,dur,sr*dur,endpoint=False,dtype=np.float32)
    wav=0.2*np.sin(2*np.pi*(110+int(sid)%400)*t)+0.02*rng.standard_normal(t.shape).astype(np.float32)
    torchaudio.save(os.path.join(out,f"{sid}.wav"), torch.from_numpy(wav).unsqueeze(0), sr)
print("wrote", len(sids), "wav files to", out)
PY
```

### B.1 — Unzip dataset

```bash
mkdir -p data/raw
unzip -q -o HKU956.zip -d data/raw/
# expect: data/raw/HKU956/{1. physiological_signals, 2.*, 3. AV_ratings.csv, 4. participant_personality.csv}
```

### B.2 — Align & slice (option [1])

```bash
python -m scripts.align_and_slice
```

**Expected:** `Generated matched clips: <N>` (>0) and
`Saved aligned dataset to data/processed/physio_cache.npz` + `data/processed/audio_clips/`.
Only songs that have audio produce clips; everything else is skipped (logged).

### B.3 — MERT audio embeddings on GPU (option [2])

```bash
python -m audio.mert_embedder
```

First run downloads `m-a-p/MERT-v1-330M` (~1.3 GB) — slow once, cached after.
**Expected:** `SUCCESS! Saved <N> embeddings to data/processed/song_embeddings.npy`.
Embeddings are L2-normalized at save (the shared action manifold).

### B.4 — Verify alignment (option [3], optional)

```bash
python -c "import numpy as np; d=np.load('data/processed/physio_cache.npz', allow_pickle=True); print('clips', d['clip_ids'].shape, 'window tensor', d['window_features'].shape)"
```

> The cache no longer stores the legacy summary `features` array (dead path removed).
> The encoder consumes `window_features` (`[N, T, 30]`) directly; verify against that.

### B.5 — Train each stage (reduced scale)

Seeding: every `train_*` entrypoint calls `set_seed(config.training.seed)` (default 42),
so two runs reproduce. Use small epochs/steps for the smoke run.

```bash
# [4] physio encoder  -- emits physio_embeddings.npz WITH song_nos, so the world
#     model keys transitions by the unique (participant, song_id, song_no) trial
#     and never merges distinct replays. Also fits the per-window z-score
#     normalizer on the holdout-EXCLUDED split and persists it (win_norm_mean/std)
#     into the same .npz, so inference normalizes the holdout with stats that
#     never saw it. Now also persists physio_std (the env's reward/eval scale,
#     single source of truth), trains an auxiliary reconstruction head, and uses
#     an 85/15 val split to keep the best-val checkpoint. Regenerate if it
#     predates any of these.
python -c "from physio.train_encoder import train_physio_model; train_physio_model(epochs=2)"

# [5] user profiler  (persists the min-max scaler into user_embeddings.npz)
python -c "from user.train_profile import train_user_model; train_user_model()"

# [6] context transformer  -- MUST be regenerated: the manifold change means any
#     old checkpoint was trained on raw (un-normalized) MERT vectors. Now also
#     excludes the holdout user (config.training.holdout_user), like every other
#     stage, so the LOSO eval subject never leaks into the context model.
python -c "from context.train_context import train_context_model; train_context_model()"

# [7] world model  (trains on L2-normalized actions)
python -c "from simulator.train_simulator import train_world_model; train_world_model(epochs=2)"

# [8] SAC agent  (~500 steps for smoke)
python -c "from rl.train_agent import train_sac_agent; train_sac_agent(steps=500)"
```

**Expected:** each prints loss/progress and saves a checkpoint
(`physio/checkpoints/`, `data/processed/user_embeddings.npz`,
`context/checkpoints/context_model.pth`, `simulator/checkpoints/world_model.pth`,
`rl/checkpoints/sac_final_actor.pth` + `_critic.pth`). No crash = pass.

> If `MusicEnv` raises `Cannot load embedding pools ...`, a required `.npz` is missing —
> run the stage that produces it (this is intentional: the env now fails loudly instead
> of silently training on zeros).

### B.6 — Inference & LOSO (option [9])

```bash
# single held-out user (config training.holdout_user, default hku1903)
python -m scripts.inference

# QUICK LOSO (use this for the smoke run): loops every participant against the
# ONE checkpoint set you just trained. Fast, but leaky — only the config holdout
# is truly excluded, every other subject was in training. Tagged "quick_leaky"
# in loso_summary.json. Smoke test only, NOT a generalization number.
python -m scripts.inference loso-quick
```

> **Do NOT run `python -m scripts.inference loso` as a smoke test.** The bare
> `loso` is *true* Leave-One-Subject-Out: it **retrains the full pipeline**
> (user → physio → context → world → agent) once per participant and restores the
> default checkpoints at the end — hours of compute, the real generalization
> protocol. Use `loso-quick` here; reserve `loso` for an actual evaluation run.

**Expected:** reports written to `reports/evaluation/`:
`summary.json`, `physio_prediction.png`, `agent_control_trajectory.png`,
`agent_metrics.csv`; LOSO adds `loso_per_user.csv` + `loso_summary.json`.
Look for finite world-model MSE and an agent improvement over the **true random
baseline** (a random real song applied each step, no critic re-ranking). The
agent and baseline improvements use the same scale-aware RMS distance the reward
optimizes.

`summary.json` / `loso_summary.json` now also report `persistence_mse` — the MSE
of a trivial "next = current" predictor over the same transitions. Because chunks
overlap 50% (30 s @ 15 s stride) the two latents are autocorrelated, so the world
model is only meaningful if its MSE is **clearly below** the persistence MSE; a WM
MSE near or above persistence means it is just copying the current state.

### B.7 — Reproducibility check

Run inference twice and confirm identical numbers (seeding works):

```bash
python -m scripts.inference && cp reports/evaluation/summary.json /tmp/run1.json
python -m scripts.inference && diff /tmp/run1.json reports/evaluation/summary.json && echo "REPRODUCIBLE"
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Generated matched clips: 0` | No audio in `2. audio_files/` (see B.0) |
| `Cannot load embedding pools` | Missing `physio_embeddings.npz` / `user_embeddings.npz` — run B.5 stages in order |
| `KeyError: ... norm_min/norm_max` in inference | `user_embeddings.npz` predates the scaler-persistence fix — re-run `[5]` |
| World-model trains but merges song replays | `physio_embeddings.npz` predates the `song_nos` fix — re-run `[4]` so transitions key by `(participant, song_id, song_no)` |
| Inference normalizes with cache stats (holdout leak) | `physio_embeddings.npz` predates the encoder-fit `win_norm_mean/std` — re-run `[4]`; until then inference falls back to the (legacy, holdout-aware) cache normalizer |
| device-mismatch in tests | shell exported a GPU — prefix with `CUDA_VISIBLE_DEVICES=""` |
| `faiss-gpu` install fails | `pip install faiss-cpu` |
| MERT download stalls | first run pulls ~1.3 GB; re-run, it resumes from HF cache |

> **Caveat on synthetic audio (Option 2):** MERT embeddings of synthetic tones are not
> musically meaningful, so resulting MSE/control numbers only prove the pipeline *runs*.
> For real results, use real audio (Option 1) and full epochs.
